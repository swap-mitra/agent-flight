"""OpenTelemetry export (FR-R7): each run as one trace of GenAI spans.

Spans follow the GenAI semantic conventions at ``SEMCONV``; that repository
publishes no releases, so the exporter pins a commit and the tests assert the
attribute names, making a move to a newer commit a reviewed change.

Spans are built when a run closes, from its redacted record, because a step
closes after the steps nested inside it. Each span keeps its step's real
start and end time. The run span is a child of the application's span that
was current when the run started, so a run joins an existing trace. A child
run is held until its parent closes and nests under it. Building happens on a
background thread, so a slow or unreachable backend never blocks the agent.
"""

import atexit
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode, TracerProvider

from flight_recorder import recorder
from flight_recorder.model import ModelCall, Run, Step, ToolCall
from flight_recorder.store.cassette import EXIT_FLUSH_TIMEOUT

logger = logging.getLogger("flight_recorder")

SEMCONV = "open-telemetry/semantic-conventions-genai@633cbb0 (2026-11-04)"
# The conventions' switch for message content; spans carry it for these values.
CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT"
_CONTENT_ON = ("SPAN_ONLY", "SPAN_AND_EVENT")

# Recorded request parameters with a GenAI attribute of their own.
_PARAMS = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "max_tokens": "gen_ai.request.max_tokens",
    "max_completion_tokens": "gen_ai.request.max_tokens",
    "max_output_tokens": "gen_ai.request.max_tokens",
    "stream": "gen_ai.request.stream",
}

Attributes = dict[str, Any]


@dataclass
class _Pending:
    run: Run
    context: otel_context.Context  # the application's, when the run started
    start_ns: int
    end_ns: int = 0
    steps: list[tuple[Step, int]] = field(default_factory=list[tuple[Step, int]])
    children: list["_Pending"] = field(default_factory=list["_Pending"])


class OtelSink:
    """A sink that exports each finished run to OpenTelemetry.

    Uses ``provider``, or else the application's global TracerProvider at
    export time, so spans go wherever the application already sends them.
    """

    def __init__(self, provider: TracerProvider | None = None) -> None:
        self._provider = provider
        self._content = os.environ.get(CONTENT_ENV, "").upper() in _CONTENT_ON
        # ponytail: a run's steps are held in memory until it closes; a run
        # with very many large steps holds them all.
        self._runs: dict[str, _Pending] = {}
        self._lock = threading.Lock()
        self._worker = _Worker()

    def run_started(self, run: Run) -> None:
        # The steps' clock, so a step never starts before its run.
        started = _ns(recorder.now())
        pending = _Pending(run, otel_context.get_current(), started)
        with self._lock:
            self._runs[run.run_id] = pending

    def step_closed(self, run_id: str, step: Step) -> None:
        with self._lock:
            self._runs[run_id].steps.append((step, _ns(recorder.now())))

    def run_finished(self, run: Run) -> None:
        with self._lock:
            pending = self._runs.pop(run.run_id)
            pending.run, pending.end_ns = run, _ns(recorder.now())
            parent = self._runs.get(run.parent_run_id or "")
            if parent is not None:
                parent.children.append(pending)
                return
        self._worker.submit(run.run_id, lambda: self._export(pending))

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until every finished run is exported; False on timeout."""
        return self._worker.flush(timeout)

    def close(self) -> None:
        self._worker.flush(EXIT_FLUSH_TIMEOUT)

    def _export(self, pending: _Pending) -> None:
        provider = self._provider or trace.get_tracer_provider()
        self._export_run(provider.get_tracer("flight_recorder"), pending, None)

    def _export_run(
        self,
        tracer: trace.Tracer,
        pending: _Pending,
        parent: otel_context.Context | None,
    ) -> None:
        run = pending.run
        span = tracer.start_span(
            f"invoke_agent {run.name}",
            context=parent or pending.context,
            kind=SpanKind.INTERNAL,
            start_time=pending.start_ns,
            attributes=_run_attributes(run, [s for s, _ in pending.steps]),
        )
        if run.status == "error":
            span.set_attribute("error.type", "_OTHER")  # the run records no type
            span.set_status(Status(StatusCode.ERROR))
        run_context = trace.set_span_in_context(span)
        # Parents have lower seq numbers, so they exist before their children.
        contexts: dict[int, otel_context.Context] = {}
        for step, closed_ns in sorted(pending.steps, key=lambda s: s[0].seq):
            start_ns, end_ns = _interval(step, closed_ns)
            step_span = tracer.start_span(
                _step_name(step),
                context=contexts.get(step.parent_seq, run_context)
                if step.parent_seq is not None
                else run_context,
                kind=SpanKind.CLIENT if step.model else SpanKind.INTERNAL,
                start_time=start_ns,
                attributes=_step_attributes(step, self._content),
            )
            error = (step.model or step.tool).error  # type: ignore[union-attr]
            if error is not None:
                step_span.set_status(Status(StatusCode.ERROR, error.message))
            step_span.end(end_time=end_ns)
            contexts[step.seq] = trace.set_span_in_context(step_span)
        for child in pending.children:
            self._export_run(tracer, child, run_context)
        span.end(end_time=pending.end_ns)


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _interval(step: Step, closed_ns: int) -> tuple[int, int]:
    """A step's start and end in ns, from its recorded start and latency."""
    latency_ns = int(step.latency_ms * 1_000_000)
    if step.started_at is None:  # recorded before Step.started_at existed
        return closed_ns - latency_ns, closed_ns
    start_ns = _ns(step.started_at)
    return start_ns, start_ns + latency_ns


def _ns(moment: datetime) -> int:
    return (moment - _EPOCH) // timedelta(microseconds=1) * 1000


def _step_name(step: Step) -> str:
    if step.model is not None:
        return f"chat {step.model.model_requested}"
    assert step.tool is not None
    return f"execute_tool {step.tool.name}"


def _run_attributes(run: Run, steps: list[Step]) -> Attributes:
    attributes: Attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": run.name,
        "gen_ai.usage.input_tokens": run.totals.input_tokens,
        "gen_ai.usage.output_tokens": run.totals.output_tokens,
        "fr.run.id": run.run_id,
        "fr.run.status": run.status,
    }
    # Required by the conventions, but a run has a provider only when every
    # model call in it used the same one.
    providers = {s.model.provider for s in steps if s.model is not None}
    if len(providers) == 1:
        attributes["gen_ai.provider.name"] = providers.pop()
    if run.parent_run_id is not None:
        attributes["fr.parent_run.id"] = run.parent_run_id
    if run.totals.cost_usd is not None:
        attributes["fr.cost_usd"] = run.totals.cost_usd
    return attributes


def _step_attributes(step: Step, content: bool) -> Attributes:
    attributes: Attributes = {"fr.step.seq": step.seq, "fr.step.source": step.source}
    if step.model is not None:
        attributes |= _model_attributes(step.model, content)
    if step.tool is not None:
        attributes |= _tool_attributes(step.tool, content)
    return attributes


def _model_attributes(model: ModelCall, content: bool) -> Attributes:
    attributes: Attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": model.provider,
        "gen_ai.request.model": model.model_requested,
        "gen_ai.response.model": model.model_used,
    }
    for param, name in _PARAMS.items():
        value = model.params.get(param)
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            or name == "gen_ai.request.stream"
            and isinstance(value, bool)
        ):
            attributes[name] = value
    if model.finish_reason is not None:
        attributes["gen_ai.response.finish_reasons"] = [model.finish_reason]
    if model.usage is not None:
        usage = model.usage
        attributes["gen_ai.usage.input_tokens"] = usage.input_tokens
        attributes["gen_ai.usage.output_tokens"] = usage.output_tokens
        if usage.cache_read_tokens:
            attributes["gen_ai.usage.cache_read.input_tokens"] = usage.cache_read_tokens
        if usage.cache_write_tokens:
            attributes["gen_ai.usage.cache_write.input_tokens"] = (
                usage.cache_write_tokens
            )
    if model.cost_usd is not None:
        attributes["fr.cost_usd"] = model.cost_usd
    if model.error is not None:
        attributes["error.type"] = model.error.type
    if content:
        # Provider-native JSON: the conventions' gen_ai.input.messages schema
        # differs from each provider's, so these are not given its names.
        attributes["fr.model.messages"] = json.dumps(model.messages)
        if model.response is not None:
            attributes["fr.model.response"] = json.dumps(model.response)
    return attributes


def _tool_attributes(tool: ToolCall, content: bool) -> Attributes:
    attributes: Attributes = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool.name,
        "gen_ai.tool.type": "function",
        "fr.tool.effect": tool.effect,
    }
    if tool.call_id is not None:
        attributes["gen_ai.tool.call.id"] = tool.call_id
    if tool.error is not None:
        attributes["error.type"] = tool.error.type
    if content:
        attributes["gen_ai.tool.call.arguments"] = json.dumps(tool.args)
        if tool.error is None and tool.result is not None:
            attributes["gen_ai.tool.call.result"] = json.dumps(tool.result)
    return attributes


class _Worker:
    """One daemon thread running export jobs; failures are logged, not raised."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[str, Callable[[], None]]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, run_id: str, job: Callable[[], None]) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._drain, name="flight_recorder-otel", daemon=True
                )
                self._thread.start()
                atexit.register(self._flush_at_exit)
        self._jobs.put((run_id, job))

    def flush(self, timeout: float | None) -> bool:
        # ponytail: the same bounded wait as CassetteStore.flush; share it if
        # a third writer appears.
        deadline = None if timeout is None else time.monotonic() + timeout
        done = self._jobs.all_tasks_done
        with done:
            while self._jobs.unfinished_tasks:
                if self._thread is None or not self._thread.is_alive():
                    return False
                left = 0.1 if deadline is None else deadline - time.monotonic()
                if left <= 0:
                    return False
                done.wait(min(left, 0.1))
        return True

    def _flush_at_exit(self) -> None:
        if not self.flush(EXIT_FLUSH_TIMEOUT):
            logger.warning(
                "%d runs were not exported to OpenTelemetry before exit",
                self._jobs.unfinished_tasks,
            )

    def _drain(self) -> None:
        while True:
            run_id, job = self._jobs.get()
            try:
                job()
            except Exception:
                logger.warning(
                    "exporting run %s to OpenTelemetry failed; the agent"
                    " continues unaffected",
                    run_id,
                    exc_info=True,
                )
            finally:
                self._jobs.task_done()
