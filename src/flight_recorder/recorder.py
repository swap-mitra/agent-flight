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
from typing import Any, Literal, Protocol

from pydantic import JsonValue
from pydantic_core import to_jsonable_python

from flight_recorder.model import ErrorInfo, ModelCall, Run, Step, ToolCall

logger = logging.getLogger("flight_recorder")


class Sink(Protocol):
    """Receives a run as it is recorded: the cassette store and the OTel exporter."""

    def run_started(self, run: Run) -> None: ...
    def step_closed(self, run: Run, step: Step) -> None: ...
    def run_finished(self, run: Run) -> None: ...


# Set by fr.configure() once the store and exporter exist (rows 1.7, 1.13).
sinks: list[Sink] = []


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


@dataclass
class PendingStep:
    seq: int
    kind: Literal["model", "tool"]
    started: float
    parent_seq: int | None
    token: Token[tuple[str, int] | None]


@dataclass
class ActiveRun:
    """A run being recorded, with the state its steps share across threads."""

    run: Run
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next_seq: int = 0
    _error_logged: bool = False

    def start(self) -> None:
        self._emit(lambda sink: sink.run_started(self.run))

    def finish(self, failed: bool) -> None:
        self.run.status = "error" if failed else "ok"
        self.run.output = jsonable(self.run.output)
        self._emit(lambda sink: sink.run_finished(self.run))

    def open_step(self, kind: Literal["model", "tool"]) -> PendingStep:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        outer = _open_step.get()
        parent_seq = outer[1] if outer and outer[0] == self.run.run_id else None
        token = _open_step.set((self.run.run_id, seq))
        return PendingStep(seq, kind, time.perf_counter(), parent_seq, token)

    def close_step(
        self,
        pending: PendingStep,
        status: Literal["ok", "error"],
        *,
        model: ModelCall | None = None,
        tool: ToolCall | None = None,
    ) -> None:
        # ValueError: closed in another context (e.g. a stream consumed elsewhere),
        # whose open step was never this one, so there is nothing to restore.
        with contextlib.suppress(ValueError):
            _open_step.reset(pending.token)
        try:
            step = Step(
                seq=pending.seq,
                kind=pending.kind,
                parent_seq=pending.parent_seq,
                latency_ms=(time.perf_counter() - pending.started) * 1000,
                status=status,
                model=model,
                tool=tool,
            )
            self._add_to_totals(step)
        except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
            self._log_failure(exc)
            return
        self._emit(lambda sink: sink.step_closed(self.run, step))

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

    def _emit(self, call: Callable[[Sink], None]) -> None:
        for sink in sinks:
            try:
                call(sink)
            except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
                self._log_failure(exc)

    def _log_failure(self, exc: Exception) -> None:
        with self._lock:
            if self._error_logged:
                return
            self._error_logged = True
        logger.warning(
            "recording failed for run %s; the agent continues unaffected",
            self.run.run_id,
            exc_info=exc,
        )
