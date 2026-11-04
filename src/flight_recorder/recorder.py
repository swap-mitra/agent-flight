"""Step assembly, totals and sinks: interception pipeline stages 1 and 5.

Recording fails open (FR-R9): a failure here is logged once per run and never
reaches the agent. Interceptors open a step before the real call and close it
after, then return or re-raise exactly what the call produced.
"""

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, JsonValue
from pydantic_core import to_jsonable_python

from flight_recorder import redact
from flight_recorder.model import ErrorInfo, ModelCall, Run, Step, ToolCall

logger = logging.getLogger("flight_recorder")

M = TypeVar("M", bound=BaseModel)


class Sink(Protocol):
    """Receives a run as it is recorded: the cassette store and the OTel exporter.

    Every record a sink receives is a redacted copy.
    """

    def run_started(self, run: Run) -> None: ...
    def step_closed(self, run_id: str, step: Step) -> None: ...
    def run_finished(self, run: Run) -> None: ...


# Set by fr.configure() once the store and exporter exist (rows 1.7, 1.13).
sinks: list[Sink] = []

# Called as each step opens, in the caller's context, for that step's tags;
# framework adapters add theirs (the LangChain adapter: the LangGraph node).
tag_sources: list[Callable[[], list[str]]] = []


def jsonable(value: Any) -> JsonValue:
    """Convert any value to JSON data without raising; unknown types become repr()."""
    try:
        return to_jsonable_python(value, fallback=repr, bytes_mode="base64")
    except Exception:  # noqa: BLE001  recording fails open (FR-R9)
        return repr(value)


# The innermost open step in this context, as (run_id, seq). Keyed by run so a
# child run opened inside a tool does not adopt its parent run's step.
_open_step: ContextVar[tuple[str, int] | None] = ContextVar(
    "flight_recorder_step", default=None
)


def error_info(exc: BaseException) -> ErrorInfo:
    """Record an exception by type and message; builtins keep their short name."""
    cls = type(exc)
    name = cls.__qualname__
    if cls.__module__ != "builtins":
        name = f"{cls.__module__}.{name}"
    return ErrorInfo(type=name, message=str(exc))


# The system clock can advance in ticks of a millisecond or more, so wall-clock
# times are one reading of it plus perf_counter: as precise as latency, and
# consistent with each other. ponytail: drifts from the system clock by
# perf_counter's error over the process's life (parts per million).
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ANCHOR_NS = time.time_ns() - time.perf_counter_ns()


def now(perf_ns: int | None = None) -> datetime:
    """UTC wall-clock time on the recorder's clock (of ``perf_ns``, if given)."""
    perf_ns = time.perf_counter_ns() if perf_ns is None else perf_ns
    return _EPOCH + timedelta(microseconds=(_ANCHOR_NS + perf_ns) // 1000)


@dataclass
class PendingStep:
    seq: int
    kind: Literal["model", "tool"]
    started: float  # perf_counter, for latency
    started_at: datetime  # wall clock, for placing the step in time
    parent_seq: int | None
    token: Token[tuple[str, int] | None] | None
    tags: list[str] = field(default_factory=list[str])


@dataclass
class ActiveRun:
    """A run being recorded, with the state its steps share across threads."""

    run: Run
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next_seq: int = 0
    _error_logged: bool = False
    _on_finish: list[Callable[[], None]] = field(
        default_factory=list[Callable[[], None]]
    )

    def start(self) -> None:
        if (run := self._redacted(self.run)) is not None:
            self._emit(lambda sink: sink.run_started(run))

    def finish(self, failed: bool) -> None:
        for callback in self._on_finish:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
                self.log_failure(exc)
        self.run.status = "error" if failed else "ok"
        self.run.output = jsonable(self.run.output)
        if (run := self._redacted(self.run)) is not None:
            self._emit(lambda sink: sink.run_finished(run))

    def open_step(self, kind: Literal["model", "tool"]) -> PendingStep:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        outer = _open_step.get()
        parent_seq = outer[1] if outer and outer[0] == self.run.run_id else None
        tags: list[str] = []
        try:
            for source in tag_sources:
                tags += source()
        except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
            self.log_failure(exc)
        token = _open_step.set((self.run.run_id, seq))
        started = time.perf_counter_ns()
        return PendingStep(
            seq, kind, started / 1e9, now(started), parent_seq, token, tags
        )

    def release(self, pending: PendingStep) -> None:
        """Stop treating ``pending`` as the open step while it stays pending.

        A stream's step closes only when the caller finishes reading it; calls
        made meanwhile are the caller's, not children of the stream.
        """
        if pending.token is not None:
            # ValueError: released in another context (a stream read elsewhere),
            # whose open step was never this one, so there is nothing to restore.
            with contextlib.suppress(ValueError):
                _open_step.reset(pending.token)
            pending.token = None

    def on_finish(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` when the run ends, before its footer is written."""
        with self._lock:
            self._on_finish.append(callback)

    def close_step(
        self,
        pending: PendingStep,
        status: Literal["ok", "error"],
        *,
        model: ModelCall | None = None,
        tool: ToolCall | None = None,
    ) -> None:
        self.release(pending)
        try:
            step = Step(
                seq=pending.seq,
                kind=pending.kind,
                parent_seq=pending.parent_seq,
                tags=pending.tags,
                started_at=pending.started_at,
                latency_ms=(time.perf_counter() - pending.started) * 1000,
                status=status,
                model=model,
                tool=tool,
            )
            self._add_to_totals(step)
        except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
            self.log_failure(exc)
            return
        if (redacted := self._redacted(step)) is not None:
            run_id = self.run.run_id
            self._emit(lambda sink: sink.step_closed(run_id, redacted))

    def _add_to_totals(self, step: Step) -> None:
        with self._lock:
            totals = self.run.totals
            totals.steps += 1
            if step.model is None:
                return
            if step.model.usage is not None:
                totals.input_tokens += step.model.usage.input_tokens
                totals.output_tokens += step.model.usage.output_tokens
            # One unpriced call makes the total unknown; a partial sum would mislead.
            if totals.cost_usd is not None and step.model.cost_usd is not None:
                totals.cost_usd += step.model.cost_usd
            else:
                totals.cost_usd = None

    def _redacted(self, record: M) -> M | None:
        """A redacted copy, or None (logged) when it cannot be redacted safely."""
        try:
            return type(record).model_validate(
                redact.apply(record.model_dump(mode="json"))
            )
        except Exception as exc:  # noqa: BLE001  never write what was not redacted
            self.log_failure(exc)
            return None

    def _emit(self, call: Callable[[Sink], None]) -> None:
        for sink in sinks:
            try:
                call(sink)
            except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
                self.log_failure(exc)

    def log_failure(self, exc: Exception) -> None:
        """Log a recording failure; only the first per run is logged."""
        with self._lock:
            if self._error_logged:
                return
            self._error_logged = True
        logger.warning(
            "recording failed for run %s; the agent continues unaffected",
            self.run.run_id,
            exc_info=exc,
        )
