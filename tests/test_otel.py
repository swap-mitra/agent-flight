"""OpenTelemetry export (FR-R7, SDD OpenTelemetry mapping).

Uses the real OpenTelemetry SDK with an in-memory exporter, and the OpenAI
mock server for real model calls.
"""

import sys
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

import httpx2
import openai
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode
from test_openai import Server

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.export import otel
from flight_recorder.export.otel import OtelSink


@pytest.fixture
def spans() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(spans: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    return provider


@pytest.fixture
def sink(provider: TracerProvider) -> Iterator[OtelSink]:
    sink = OtelSink(provider)
    recorder.sinks = [sink]
    yield sink
    sink.close()


def exported(sink: OtelSink, spans: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    assert sink.flush(timeout=10)
    return {s.name: s for s in spans.get_finished_spans()}


def llm() -> openai.OpenAI:
    http = httpx2.Client(transport=httpx2.MockTransport(Server()))
    return fr.wrap(openai.OpenAI(api_key="sk-test", http_client=http, max_retries=0))


@fr.tool(effect="read")
def lookup(order_id: str) -> str:
    return "shipped"


@fr.tool(effect="write")
def summarise(order_id: str) -> str:
    # A model call made inside a tool nests under the tool's span.
    llm().chat.completions.create(model="gpt-x", messages=[])
    return "summary"


@fr.tool(effect="irreversible")
def refund(order_id: str) -> None:
    raise ValueError("refunds are closed")


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def parent_id(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent else None


def span_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.span_id


def test_a_run_is_one_trace_of_genai_spans(
    sink: OtelSink, spans: InMemorySpanExporter
) -> None:
    client = llm()
    with fr.run("triage", inputs={"ticket": "t1"}) as run:
        client.chat.completions.create(model="gpt-x", messages=[], temperature=0.2)
        lookup("o1")

    by_name = exported(sink, spans)
    agent = by_name["invoke_agent triage"]
    chat, tool = by_name["chat gpt-x"], by_name["execute_tool lookup"]
    assert {s.context.trace_id for s in by_name.values() if s.context} == {
        agent.context.trace_id if agent.context else 0
    }
    assert parent_id(chat) == parent_id(tool) == span_id(agent)
    assert (agent.kind, chat.kind, tool.kind) == (
        SpanKind.INTERNAL,
        SpanKind.CLIENT,
        SpanKind.INTERNAL,
    )
    assert attrs(agent) == {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "triage",
        "gen_ai.provider.name": "openai",
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 3,
        "fr.run.id": run.run_id,
        "fr.run.status": "ok",
    }
    assert attrs(chat) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-x",
        "gen_ai.response.model": "gpt-x-2026-01-01",
        "gen_ai.request.temperature": 0.2,
        "gen_ai.response.finish_reasons": ("stop",),
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 3,
        "gen_ai.usage.cache_read.input_tokens": 4,
        "fr.step.seq": 0,
        "fr.step.source": "live",
    }
    assert attrs(tool) == {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "lookup",
        "gen_ai.tool.type": "function",
        "fr.tool.effect": "read",
        "fr.step.seq": 1,
        "fr.step.source": "live",
    }


def test_spans_keep_real_timing_and_nest_by_parent_step(
    sink: OtelSink, spans: InMemorySpanExporter
) -> None:
    with fr.run("triage"):
        summarise("o1")

    by_name = exported(sink, spans)
    agent, tool = by_name["invoke_agent triage"], by_name["execute_tool summarise"]
    chat = by_name["chat gpt-x"]
    assert parent_id(chat) == span_id(tool)
    # Start times are recorded to the microsecond, so an end can round past
    # its parent's by less than that.
    for outer, inner in ((agent, tool), (tool, chat)):
        assert outer.start_time and outer.end_time and inner.start_time
        assert outer.start_time <= inner.start_time
        assert inner.end_time is not None
        assert inner.end_time <= outer.end_time + 1000


def test_a_child_run_nests_under_its_parent_in_one_trace(
    sink: OtelSink, spans: InMemorySpanExporter
) -> None:
    with fr.run("triage"), fr.run("lookup-agent") as child:
        lookup("o1")

    by_name = exported(sink, spans)
    parent, nested = (
        by_name["invoke_agent triage"],
        by_name["invoke_agent lookup-agent"],
    )
    assert parent_id(nested) == span_id(parent)
    assert attrs(nested)["fr.parent_run.id"] == child.parent_run_id
    assert parent_id(by_name["execute_tool lookup"]) == span_id(nested)


def test_a_run_joins_the_applications_current_trace(
    sink: OtelSink, spans: InMemorySpanExporter, provider: TracerProvider
) -> None:
    app = provider.get_tracer("app")
    with app.start_as_current_span("POST /tickets") as request, fr.run("triage"):
        lookup("o1")

    agent = exported(sink, spans)["invoke_agent triage"]
    assert parent_id(agent) == request.get_span_context().span_id


def test_failures_set_error_status_and_type(
    sink: OtelSink, spans: InMemorySpanExporter
) -> None:
    with pytest.raises(ValueError), fr.run("triage"):
        refund("o1")

    by_name = exported(sink, spans)
    tool, agent = by_name["execute_tool refund"], by_name["invoke_agent triage"]
    assert (tool.status.status_code, tool.status.description) == (
        StatusCode.ERROR,
        "refunds are closed",
    )
    assert attrs(tool)["error.type"] == "ValueError"
    assert agent.status.status_code == StatusCode.ERROR
    assert attrs(agent)["error.type"] == "_OTHER"


def test_content_is_off_by_default_and_redacted_when_on(
    provider: TracerProvider,
    spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for capture in ("", "SPAN_ONLY"):
        monkeypatch.setenv(otel.CONTENT_ENV, capture)
        spans.clear()
        sink = OtelSink(provider)
        recorder.sinks = [sink]
        with fr.run("triage"):
            lookup("sk-ant-abcdefghijklmnopqrstuvwxyz0123")
            llm().chat.completions.create(model="gpt-x", messages=[])
        by_name = exported(sink, spans)
        tool, chat = attrs(by_name["execute_tool lookup"]), attrs(by_name["chat gpt-x"])
        if not capture:
            assert not {k for k in tool | chat if "arguments" in k or "fr.model." in k}
            continue
        assert "sk-ant-" not in tool["gen_ai.tool.call.arguments"]  # redacted first
        assert tool["gen_ai.tool.call.result"] == '"shipped"'
        assert chat["fr.model.messages"] == "[]"
        assert '"id": "c1"' in chat["fr.model.response"]


class BlockedExporter(SpanExporter):
    """A synchronous exporter whose backend never answers."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.release.wait()
        return SpanExportResult.SUCCESS


def test_an_unreachable_backend_never_blocks_the_agent() -> None:
    blocked = BlockedExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(blocked))
    sink = OtelSink(provider)
    recorder.sinks = [sink]

    results: list[str] = []

    def agent() -> None:
        for _ in range(3):
            with fr.run("triage"):
                results.append(lookup("o1"))

    # In a thread, so a regression fails here instead of hanging the suite.
    thread = threading.Thread(target=fr.copy_context(agent), daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "the agent was blocked by the exporter"
    assert time.monotonic() - started < 2
    assert results == ["shipped"] * 3
    assert sink.flush(timeout=0.2) is False
    blocked.release.set()
    assert sink.flush(timeout=10)


def test_configure_otel_exports_through_the_global_provider(
    provider: TracerProvider,
    spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    fr.configure(dir=tmp_path, otel=True)
    with fr.run("triage"):
        lookup("o1")

    (sink,) = [s for s in recorder.sinks if isinstance(s, OtelSink)]
    assert "invoke_agent triage" in exported(sink, spans)
    fr.configure(dir=tmp_path)  # closes the exporter's worker
    assert not any(isinstance(s, OtelSink) for s in recorder.sinks)


def test_configure_otel_without_the_extra_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.delitem(sys.modules, "flight_recorder.export.otel")
    with pytest.raises(ModuleNotFoundError, match=r"\[otel\]"):
        fr.configure(dir=tmp_path, otel=True)
