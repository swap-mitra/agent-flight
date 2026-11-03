"""Contract tests: the real Anthropic SDK against a mock transport (FR-R1, FR-R2)."""

import asyncio
import json
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.intercept import anthropic as fr_anthropic
from flight_recorder.model import Run, Step

MODEL = "claude-x-2026-01-01"
# 10 uncached + 2 written to cache + 100 read from cache = a 112-token prompt.
USAGE: dict[str, Any] = {
    "input_tokens": 10,
    "cache_creation_input_tokens": 2,
    "cache_read_input_tokens": 100,
    "output_tokens": 3,
}
MESSAGE: dict[str, Any] = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [{"type": "text", "text": "shipped"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": USAGE,
}


def events(error_mid_stream: bool = False) -> list[dict[str, Any]]:
    start: dict[str, Any] = {
        **MESSAGE,
        "content": [],
        "stop_reason": None,
        "usage": {**USAGE, "output_tokens": 1},
    }
    out: list[dict[str, Any]] = [
        {"type": "message_start", "message": start},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ship"},
        },
    ]
    if error_mid_stream:
        out.append(
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            }
        )
    return [
        *out,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ped"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]


class Server:
    """Answers like the Anthropic API and counts requests."""

    def __init__(self, fail: bool = False, error_mid_stream: bool = False) -> None:
        self.requests = 0
        self.fail = fail
        self.error_mid_stream = error_mid_stream

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests += 1
        if self.fail:
            return httpx2.Response(
                500,
                json={
                    "type": "error",
                    "error": {"type": "api_error", "message": "down"},
                },
            )
        if not json.loads(request.content).get("stream"):
            return httpx2.Response(200, json=MESSAGE)
        body = "".join(
            f"event: {e['type']}\ndata: {json.dumps(e)}\n\n"
            for e in events(self.error_mid_stream)
        )
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=body.encode()
        )


class Sink:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.steps: list[Step] = []

    def run_started(self, run: Run) -> None:
        self.events.append("start")

    def step_closed(self, run_id: str, step: Step) -> None:
        self.events.append("step")
        self.steps.append(step)

    def run_finished(self, run: Run) -> None:
        self.events.append("finish")


@pytest.fixture
def sink() -> Sink:
    sink = Sink()
    recorder.sinks = [sink]
    return sink


def client(server: Server) -> Anthropic:
    http = httpx2.Client(transport=httpx2.MockTransport(server))
    return fr.wrap(Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0))


def async_client(server: Server) -> AsyncAnthropic:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(server))
    return fr.wrap(
        AsyncAnthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    )


def model_of(step: Step) -> Any:
    assert step.model is not None
    return step.model


ASK: dict[str, Any] = {
    "model": "claude-x",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


def test_create_records_every_field_with_cached_tokens_in_the_prompt(
    sink: Sink,
) -> None:
    llm = client(Server())
    assert isinstance(llm, Anthropic)
    with fr.run("triage"):
        result = llm.messages.create(
            model="claude-x",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            system="be brief",
            extra_headers={"X-Trace": "t"},
        )

    assert isinstance(result, Message)
    (step,) = sink.steps
    call = model_of(step)
    assert (step.status, call.provider) == ("ok", "anthropic")
    assert (call.model_requested, call.model_used) == ("claude-x", MODEL)
    assert call.params == {"max_tokens": 100, "system": "be brief"}
    assert call.messages == [{"role": "user", "content": "hi"}]
    assert call.finish_reason == "end_turn"
    assert (call.usage.input_tokens, call.usage.output_tokens) == (112, 3)
    assert call.response["usage"]["cache_read_input_tokens"] == 100  # breakdown kept


def test_api_error_is_recorded_and_reraised_unchanged(sink: Sink) -> None:
    llm = client(Server(fail=True))
    with fr.run("triage"), pytest.raises(anthropic.InternalServerError):
        llm.messages.create(**ASK)

    (step,) = sink.steps
    assert step.status == "error"
    assert model_of(step).error.type == "anthropic.InternalServerError"


def test_create_stream_passes_events_through_and_records_usage(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        stream = llm.messages.create(**ASK, stream=True)
        assert isinstance(stream, anthropic.Stream)
        types = [e.type for e in stream]

    assert types == [e["type"] for e in events()]
    call = model_of(sink.steps[0])
    assert (call.model_used, call.finish_reason) == (MODEL, "end_turn")
    assert (call.usage.input_tokens, call.usage.output_tokens) == (112, 3)
    assert len(call.response) == 7


def test_messages_stream_records_however_the_stream_is_read(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        with llm.messages.stream(**ASK) as stream:
            text = list(stream.text_stream)
        with llm.messages.stream(**ASK) as stream:
            final = stream.get_final_message()

    assert text == ["ship", "ped"]
    assert final.content[0].type == "text" and final.content[0].text == "shipped"
    for step in sink.steps:
        call = model_of(step)
        assert (
            call.finish_reason,
            call.usage.input_tokens,
            call.usage.output_tokens,
        ) == (
            "end_turn",
            112,
            3,
        )
    assert len(sink.steps) == 2


def test_stream_manager_never_entered_sends_nothing_and_records_nothing(
    sink: Sink,
) -> None:
    server = Server()
    with fr.run("triage"):
        client(server).messages.stream(**ASK)
    assert (server.requests, sink.steps) == (0, [])


def test_stream_left_early_is_abandoned_before_the_footer(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"), llm.messages.stream(**ASK) as stream:
        next(iter(stream))

    assert model_of(sink.steps[0]).finish_reason == "abandoned"
    assert sink.events == ["start", "step", "finish"]


def test_error_mid_stream_is_recorded_and_reraised(sink: Sink) -> None:
    llm = client(Server(error_mid_stream=True))
    with (
        fr.run("triage"),
        pytest.raises(anthropic.APIStatusError),
        llm.messages.stream(**ASK) as stream,
    ):
        stream.until_done()

    (step,) = sink.steps
    assert step.status == "error"
    assert "Overloaded" in model_of(step).error.message


@fr.tool(effect="read")
def lookup(order_id: str) -> str:
    return "shipped"


def test_tool_called_while_reading_a_stream_is_not_its_child(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"), llm.messages.stream(**ASK) as stream:
        for _ in stream.text_stream:
            lookup("o1")

    assert [s.parent_seq for s in sink.steps if s.kind == "tool"] == [None, None]


def test_async_client_create_create_stream_and_messages_stream(sink: Sink) -> None:
    llm = async_client(Server())

    async def agent() -> list[str]:
        async with fr.run("triage"):
            await llm.messages.create(**ASK)
            stream = await llm.messages.create(**ASK, stream=True)
            async for _ in stream:
                pass
            async with llm.messages.stream(**ASK) as managed:
                return [t async for t in managed.text_stream]

    assert asyncio.run(agent()) == ["ship", "ped"]
    calls = [model_of(s) for s in sink.steps]
    assert [c.finish_reason for c in calls] == ["end_turn"] * 3
    assert [c.usage.input_tokens for c in calls] == [112] * 3


def test_wrapping_twice_records_once(sink: Sink) -> None:
    llm = fr.wrap(client(Server()))
    with fr.run("triage"):
        llm.messages.create(**ASK)
        with llm.messages.stream(**ASK) as stream:
            stream.until_done()
    assert len(sink.steps) == 2


def test_renamed_stream_hook_fails_open(
    sink: Sink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # As if a new SDK version renamed the hook: the call works, unrecorded.
    monkeypatch.setitem(fr_anthropic._STREAM_HOOK, False, "_renamed")  # type: ignore[reportPrivateUsage]
    llm = client(Server())
    with fr.run("triage"), llm.messages.stream(**ASK) as stream:
        assert list(stream.text_stream) == ["ship", "ped"]

    assert sink.steps == []
    assert "stream manager has no _renamed" in caplog.text
