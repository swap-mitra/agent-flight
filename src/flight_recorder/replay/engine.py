"""Replay: serve recorded steps in place of live calls (FR-P1 to FR-P4, FR-P10).

A replay ``Session`` holds a baseline (a case run and its child runs) and a
cursor over each run's recorded steps. The tool and model interceptors ask it
for the next matching recorded step; a match is served instead of the live
call, and anything else is a divergence handled by policy. In tools-recorded
mode only tool steps are served and models run live; in fully-recorded mode
model responses are served too, through the HTTP shim ``fr.wrap`` installs, so
nothing reaches the network.
"""

import builtins
import json
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import JsonValue

from flight_recorder import redact
from flight_recorder.errors import PolicyError, ReplayDivergence
from flight_recorder.model import Effect, ErrorInfo, ModelCall, Run, Step
from flight_recorder.recorder import ActiveRun

Mode = Literal["tools", "full"]
Policy = Literal["strict", "stub", "live"]
Compare = Callable[[JsonValue, JsonValue], bool]


@dataclass
class _Rule:
    compare: Compare | None = None
    ignore: tuple[str, ...] = ()
    on_divergence: Policy | None = None
    allow_write: bool = False


_rules: dict[str, _Rule] = {}
# Each tool's side-effect class, registered by @fr.tool and wrap_tools.
_effects: dict[str, Effect] = {}


def matcher(
    tool: str,
    compare: Compare | None = None,
    *,
    ignore: tuple[str, ...] | list[str] = (),
    on_divergence: Policy | None = None,
    allow_write: bool = False,
) -> None:
    """Configure how replay matches and serves calls of ``tool``.

    ``ignore``: JSON paths into the arguments (``$.request_id``) left out of
    the comparison. ``compare(recorded_args, live_args)``: replaces the
    comparison. ``on_divergence``: this tool's policy when no recorded call
    matches: strict, stub or live. Live is refused for an irreversible tool,
    and for a write tool unless ``allow_write``; PolicyError is raised here,
    or when the tool is defined, never mid-run.
    """
    if on_divergence is not None and on_divergence not in get_args(Policy):
        raise PolicyError(f"on_divergence must be one of {get_args(Policy)}")
    rule = _Rule(compare, tuple(ignore), on_divergence, allow_write)
    for path in rule.ignore:
        redact.path(path)  # raises ValueError for a bad path, now
    _check_live(tool, rule, _effects.get(tool))
    _rules[tool] = rule


def register(tool: str, effect: Effect) -> None:
    """Record a tool's effect; checks any replay rule already set for it."""
    _effects[tool] = effect
    if (rule := _rules.get(tool)) is not None:
        _check_live(tool, rule, effect)


def _check_live(tool: str, rule: _Rule, effect: Effect | None) -> None:
    if rule.on_divergence != "live" or effect is None or effect == "read":
        return
    if effect == "irreversible" or not rule.allow_write:
        raise PolicyError(
            f"replay can't call the {effect} tool {tool!r} live on divergence"
            + ("" if effect == "irreversible" else "; pass allow_write=True")
        )


@dataclass
class Divergence:
    """A live call with no matching recorded step (FR-P4)."""

    run: str
    index: int  # the live call's position among its run's served steps
    expected: dict[str, Any] | None  # the recorded step at the cursor
    actual: dict[str, Any]
    preceding: dict[str, Any] | None  # the last recorded step matched
    policy: Policy

    def __str__(self) -> str:
        def show(call: dict[str, Any] | None) -> str:
            return "nothing (the recording ended)" if call is None else _dumps(call)

        return (
            f"replay of run {self.run!r} diverged at step {self.index} ({self.policy}):\n"
            f"  expected: {show(self.expected)}\n"
            f"  actual:   {show(self.actual)}\n"
            f"  after:    {show(self.preceding) if self.preceding else 'the start of the run'}"
        )


@dataclass
class Served:
    """What an interceptor does instead of (or as) the live call."""

    source: Literal["replayed", "stubbed", "live"]
    result: Any = None
    error: BaseException | None = None
    response: Any = None  # a recorded model response or a stream's chunks


Recording = tuple[Run, list[Step]]


@dataclass
class _Cursor:
    run: Run
    steps: list[Step]
    next: int = 0
    calls: int = 0
    last: Step | None = None

    def expected(self) -> Step | None:
        return self.steps[self.next] if self.next < len(self.steps) else None

    def take(self, matches: Callable[[Step], bool], lookahead: int) -> Step | None:
        """The next recorded step if it matches, skipping up to ``lookahead``
        recorded read steps to find one; writes are never skipped."""
        self.calls += 1
        for i in range(self.next, min(self.next + lookahead + 1, len(self.steps))):
            step = self.steps[i]
            if matches(step):
                self.next, self.last = i + 1, step
                return step
            if step.tool is None or step.tool.effect != "read":
                return None  # only reads are skipped
        return None


@dataclass
class Session:
    """One replayed test case: a sink that pairs each live run with its recording."""

    baseline: Recording
    children: list[Recording] = field(default_factory=list[Recording])
    mode: Mode = "tools"
    policy: Policy = "strict"
    lookahead: int = 0
    divergences: list[Divergence] = field(default_factory=list[Divergence])
    _cursors: dict[str, _Cursor] = field(default_factory=dict[str, _Cursor])
    _unpaired: deque[Recording] = field(default_factory=deque[Recording])

    def __post_init__(self) -> None:
        self._unpaired.extend(self.children)

    # Sink: the session sees each run start in the context that starts it.
    def run_started(self, run: Run) -> None:
        if _session.get() is not self:
            return
        recording: Recording
        if not self._cursors:
            recording = self.baseline
        elif self._unpaired and self._unpaired[0][0].name == run.name:
            recording = self._unpaired.popleft()  # child runs pair by start order
        else:
            recording = (run, [])  # an extra child run: every call diverges
        kept = [s for s in recording[1] if s.tool is not None or self.mode == "full"]
        self._cursors[run.run_id] = _Cursor(recording[0], kept)

    def step_closed(self, run_id: str, step: Step) -> None:
        pass

    def run_finished(self, run: Run) -> None:
        pass

    @property
    def failed(self) -> bool:
        """A strict divergence fails the case even if the agent caught it."""
        return any(d.policy == "strict" for d in self.divergences)

    def _diverge(
        self,
        cursor: _Cursor,
        actual: dict[str, Any],
        policy: Policy,
        expected: dict[str, Any] | None = None,
    ) -> Divergence:
        if expected is None and (step := cursor.expected()) is not None:
            expected = _summary(step)
        divergence = Divergence(
            run=cursor.run.name,
            index=cursor.calls - 1,
            expected=expected,
            actual=actual,
            preceding=_summary(cursor.last) if cursor.last else None,
            policy=policy,
        )
        self.divergences.append(divergence)
        return divergence


_session: ContextVar[Session | None] = ContextVar(
    "flight_recorder_replay", default=None
)
# The recorded response the HTTP shim serves for the call being sent.
served_response: ContextVar[Any] = ContextVar("flight_recorder_served", default=None)


@contextmanager
def replaying(session: Session) -> Generator[Session]:
    """Serve ``session``'s recording to the runs started inside this block."""
    from flight_recorder import recorder

    token = _session.set(session)
    recorder.sinks.append(session)
    try:
        yield session
    finally:
        _session.reset(token)
        recorder.sinks.remove(session)


def replaying_now() -> bool:
    """True inside a replayed case: one context lookup, for the hot path."""
    return _session.get() is not None


def _cursor(active: ActiveRun) -> tuple[Session, _Cursor] | None:
    session = _session.get()
    if session is None:
        return None
    cursor = session._cursors.get(active.run.run_id)  # pyright: ignore[reportPrivateUsage]
    return None if cursor is None else (session, cursor)


def serve_tool(
    active: ActiveRun, name: str, effect: Effect, args: dict[str, JsonValue]
) -> Served | None:
    """The recorded result of this tool call, or None to run it unrecorded by
    replay (no session, or a run outside the case)."""
    found = _cursor(active)
    if found is None:
        return None
    session, cursor = found
    rule = _rules.get(name, _Rule())
    live = _tool_key(name, args, rule)

    def matches(step: Step) -> bool:
        if step.tool is None or step.tool.name != name:
            return False
        recorded = _tool_key(name, step.tool.args, rule, redacted=True)
        if rule.compare is not None:
            return rule.compare(recorded, live)
        return _dumps(recorded) == _dumps(live)

    step = cursor.take(matches, session.lookahead)
    if step is not None and step.tool is not None:
        if step.tool.error is not None:
            return Served("replayed", error=rebuild_error(step.tool.error))
        return Served("replayed", result=step.tool.result)
    policy = rule.on_divergence or session.policy
    if policy == "live" and rule.on_divergence is None and effect != "read":
        policy = "strict"  # a run-wide live policy only ever re-runs reads
    divergence = session._diverge(cursor, {"tool": name, "args": live}, policy)  # pyright: ignore[reportPrivateUsage]
    if policy == "live":
        return Served("live")
    return Served("stubbed", error=ReplayDivergence(divergence))


def serve_model(active: ActiveRun, call: ModelCall) -> Served | None:
    """In fully-recorded mode, the recorded response to this model call."""
    found = _cursor(active)
    if found is None or found[0].mode != "full":
        return None
    session, cursor = found
    live = _model_key(call)

    def matches(step: Step) -> bool:
        return step.model is not None and _model_key(step.model) == live

    step = cursor.take(matches, 0)
    if step is not None and step.model is not None:
        if step.model.response is None:  # the recorded call failed
            error = step.model.error or ErrorInfo(type="Exception", message="")
            return Served("replayed", error=rebuild_error(error))
        return Served("replayed", response=step.model.response)
    policy = "live" if session.policy == "live" else "strict"
    expected, actual = _model_divergence(cursor.expected(), call)
    divergence = session._diverge(cursor, actual, policy, expected)  # pyright: ignore[reportPrivateUsage]
    if policy == "live":
        return Served("live")
    raise ReplayDivergence(divergence)


class ReplayedError(Exception):
    """A recorded exception whose type replay can't recreate; its message
    names the original type."""


def rebuild_error(error: ErrorInfo) -> BaseException:
    """The recorded exception: builtins and LangChain's ToolException by type,
    anything else as ReplayedError."""
    cls = getattr(builtins, error.type, None)
    if error.type == "langchain_core.tools.base.ToolException":
        try:
            from langchain_core.tools import ToolException
        except ModuleNotFoundError:
            pass
        else:
            return ToolException(error.message)
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            return cls(error.message)
        except TypeError:
            pass
    return ReplayedError(f"{error.type}: {error.message}")


def _tool_key(
    name: str, args: dict[str, JsonValue], rule: _Rule, redacted: bool = False
) -> JsonValue:
    """Arguments as compared: redacted like the recording, ignored paths out."""
    if not redacted:  # the recorded side was redacted before it was written
        record = redact.apply({"kind": "tool", "tool": {"name": name, "args": args}})
        args = _dict(_dict(record.get("tool")).get("args"))
    record = dict(args)
    for path in rule.ignore:
        record = redact.path(path)(record)
    return record


def _model_parts(call: ModelCall) -> dict[str, str]:
    """What a model call matches on, redacted, one JSON string per part."""
    record = redact.apply(
        {"kind": "model", "model": {"messages": call.messages, "params": call.params}}
    )
    model = _dict(record.get("model"))
    parts = {"model": f"{call.provider}/{call.model_requested}"}
    parts["messages"] = _dumps(model.get("messages"))
    for name, value in _dict(model.get("params")).items():
        parts[f"params.{name}"] = _dumps(value)
    return parts


def _model_key(call: ModelCall) -> str:
    return _dumps(_model_parts(call))


def _model_divergence(
    expected: Step | None, call: ModelCall
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The model, and only the parts that differ, of each side, so the report
    names what changed (the system prompt, a message, a parameter)."""
    live = _model_parts(call)
    if expected is None or expected.model is None:
        return None, {"model": live["model"]}
    recorded = _model_parts(expected.model)
    differ = [
        k for k in recorded.keys() | live.keys() if recorded.get(k) != live.get(k)
    ]
    differ = sorted(differ, key=lambda k: (k != "model", k))[:2]

    def side(parts: dict[str, str]) -> dict[str, Any]:
        shown = {k: _short(_plain(parts.get(k))) for k in differ if k != "model"}
        return {"model": parts["model"], **shown}

    return side(recorded), side(live)


def _plain(part: str | None) -> str:
    """A part as text: a JSON string unquoted, anything else as JSON."""
    if part is None:
        return "(absent)"
    value = json.loads(part)
    return value if isinstance(value, str) else part


def _short(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summary(step: Step) -> dict[str, Any]:
    if step.tool is not None:
        return {"tool": step.tool.name, "args": step.tool.args}
    assert step.model is not None
    return {"model": step.model.model_requested}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
