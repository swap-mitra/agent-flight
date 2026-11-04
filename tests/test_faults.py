"""Fault injection for the failure table (FR-R9, SDD Failure handling).

Every fault is crossed with every interception path. A faulted run must give
the agent exactly what an unfaulted run gives it (results, stream chunks,
exceptions), and each faulted run logs one warning. The OTel row is tested
with the exporter (row 1.13); the policy, budget and gate rows with theirs.
"""

import asyncio
import errno
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from test_anthropic import ASK
from test_anthropic import Server as AnthropicServer
from test_openai import Server as OpenAIServer

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.export.otel import OtelSink
from flight_recorder.intercept import _model
from flight_recorder.policy import pricing
from flight_recorder.store.cassette import CassetteStore

FAILED = "the agent continues unaffected"  # both failure warnings end so


def openai_client(server: OpenAIServer | None = None) -> openai.OpenAI:
    http = httpx2.Client(transport=httpx2.MockTransport(server or OpenAIServer()))
    return fr.wrap(openai.OpenAI(api_key="sk-test", http_client=http, max_retries=0))


def anthropic_client() -> anthropic.Anthropic:
    http = httpx2.Client(transport=httpx2.MockTransport(AnthropicServer()))
    return fr.wrap(
        anthropic.Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    )


def async_anthropic_client() -> anthropic.AsyncAnthropic:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(AnthropicServer()))
    return fr.wrap(
        anthropic.AsyncAnthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    )


@fr.tool(effect="read")
def lookup(order_id: str) -> dict[str, str]:
    return {"order_id": order_id, "status": "shipped"}


@fr.tool(effect="read")
async def alookup(order_id: str) -> dict[str, str]:
    return {"order_id": order_id, "status": "shipped"}


@fr.tool(effect="write")
def broken(order_id: str) -> None:
    raise ValueError(f"no order {order_id}")


# Each path runs one agent run and returns everything the agent saw.


def tool_sync() -> Any:
    with fr.run("faults"):
        return lookup("o1")


def tool_async() -> Any:
    async def agent() -> Any:
        async with fr.run("faults"):
            return await alookup("o1")

    return asyncio.run(agent())


def tool_error() -> Any:
    with fr.run("faults"):
        try:
            broken("o1")
        except ValueError as exc:
            return repr(exc)


def openai_create() -> Any:
    with fr.run("faults"):
        return openai_client().chat.completions.create(model="gpt-x", messages=[])


def openai_stream() -> Any:
    llm = openai_client()
    with fr.run("faults"):
        stream = llm.chat.completions.create(model="gpt-x", messages=[], stream=True)
        return [chunk.model_dump() for chunk in stream]


def openai_raw() -> Any:
    llm = openai_client()
    with fr.run("faults"):
        raw = llm.chat.completions.with_raw_response.create(model="gpt-x", messages=[])
        return raw.parse()


def openai_error() -> Any:
    llm = openai_client(OpenAIServer(fail=True))
    with fr.run("faults"):
        try:
            llm.chat.completions.create(model="gpt-x", messages=[])
        except openai.InternalServerError as exc:
            return type(exc), str(exc)


def anthropic_create() -> Any:
    messages: Any = anthropic_client().messages
    with fr.run("faults"):
        return messages.create(**ASK)


def anthropic_stream() -> Any:
    llm = anthropic_client()
    with fr.run("faults"), llm.messages.stream(**ASK) as stream:
        return list(stream.text_stream), stream.get_final_message()


def anthropic_beta() -> Any:
    messages: Any = anthropic_client().beta.messages
    with fr.run("faults"):
        return messages.create(**ASK)


def anthropic_async_stream() -> Any:
    llm = async_anthropic_client()

    async def agent() -> Any:
        async with fr.run("faults"):
            stream = await llm.messages.create(**ASK, stream=True)
            return [event.model_dump() async for event in stream]

    return asyncio.run(agent())


TOOL_PATHS = [tool_sync, tool_async, tool_error]
MODEL_PATHS = [
    openai_create,
    openai_stream,
    openai_raw,
    openai_error,
    anthropic_create,
    anthropic_stream,
    anthropic_beta,
    anthropic_async_stream,
]


def boom(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("injected fault")


# Each fault configures recording under root and breaks one part of it.
Fault = Callable[[Path, pytest.MonkeyPatch], None]


def store_dir_unusable(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Creating the store fails with a real OS error, on every platform.
    blocker = root / "not_a_dir"
    blocker.write_text("x")
    fr.configure(dir=blocker / "fr")


def store_dir_read_only(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform == "win32" or os.geteuid() == 0:
        pytest.skip("the OS does not enforce a read-only directory here")
    root.chmod(0o555)  # the test restores it so tmp_path cleanup can remove it
    fr.configure(dir=root)


def disk_full_mid_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fr.configure(dir=root)
    write = CassetteStore._write  # type: ignore[reportPrivateUsage]

    def full_after_header(self: CassetteStore, item: Any) -> None:
        if item[2] != "header":
            raise OSError(errno.ENOSPC, "No space left on device")
        write(self, item)

    monkeypatch.setattr(CassetteStore, "_write", full_after_header)


def index_unwritable(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (root / "index.sqlite").mkdir()  # sqlite cannot open a directory
    fr.configure(dir=root)


def redaction_raises(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fr.configure(dir=root, redact=[boom])


def step_assembly_raises(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fr.configure(dir=root)
    monkeypatch.setattr(recorder, "Step", boom)


def otel_export_raises(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Export runs on its own thread; the cassette is still written.
    fr.configure(dir=root, otel=True)
    monkeypatch.setattr(OtelSink, "_export", boom)


def response_unreadable(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # As if an SDK release changed its response shape.
    fr.configure(dir=root)
    monkeypatch.setattr(_model.Request, "call", boom)


def pricing_raises(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fr.configure(dir=root)
    monkeypatch.setattr(pricing, "cost", boom)


EVERY_PATH = [
    store_dir_unusable,
    store_dir_read_only,
    disk_full_mid_run,
    index_unwritable,
    redaction_raises,
    step_assembly_raises,
    otel_export_raises,
]
MODEL_ONLY = [response_unreadable, pricing_raises]

CASES = [
    pytest.param(fault, path, id=f"{fault.__name__}-{path.__name__}")
    for fault in EVERY_PATH
    for path in TOOL_PATHS + MODEL_PATHS
] + [
    pytest.param(fault, path, id=f"{fault.__name__}-{path.__name__}")
    for fault in MODEL_ONLY
    for path in MODEL_PATHS
    # A rejected request costs 0 without consulting the pricing table.
    if (fault, path) != (pricing_raises, openai_error)
]


@pytest.fixture(autouse=True)
def _close_stores() -> Iterator[None]:
    yield
    for sink in recorder.sinks:
        if isinstance(sink, CassetteStore | OtelSink):
            sink.close()


def flushed() -> None:
    for sink in recorder.sinks:
        if isinstance(sink, CassetteStore | OtelSink):
            assert sink.flush(timeout=10), "a writer thread died or hung"


def failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if FAILED in r.getMessage()]


@pytest.mark.parametrize(("fault", "path"), CASES)
def test_fault_never_changes_what_the_agent_sees(
    fault: Fault,
    path: Callable[[], Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fr.configure(dir=tmp_path / "clean")
    clean = path()
    flushed()
    assert failures(caplog) == []

    root = tmp_path / "faulted"
    root.mkdir()
    try:
        fault(root, monkeypatch)
        first, second = path(), path()
        flushed()
    finally:
        root.chmod(0o755)  # let tmp_path cleanup remove it

    assert first == second == clean
    logged = failures(caplog)
    assert len(logged) == 2  # once per run
    assert logged[0] != logged[1]  # each names its own run
