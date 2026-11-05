"""LangChain and LangGraph adapter (FR-R1): it wraps objects, not callbacks.

Callbacks only observe, so they could never serve a replayed result or block a
call. ``wrap_model`` hands a chat model's own SDK clients to ``fr.wrap``, so
its steps are exactly those of direct SDK calls; ``wrap_tools`` records each
tool's own call. A step opened inside a LangGraph node is tagged
``node:<name>``. LangChain is imported only when these are called.
"""

import functools
import types
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Literal, TypeVar, get_args

from flight_recorder import recorder
from flight_recorder.context import active_run
from flight_recorder.intercept import wrap
from flight_recorder.model import Effect, ToolCall
from flight_recorder.policy import gate
from flight_recorder.recorder import ActiveRun, PendingStep, error_info, jsonable
from flight_recorder.replay import engine

M = TypeVar("M")
T = TypeVar("T")

# What LangChain passes a tool besides its arguments.
_INJECTED = ("run_manager", "config", "callbacks")

# The id of the model's tool call being run, set by BaseTool.run/arun, which
# receive it, for _run/_arun, which record the call.
_call_id: ContextVar[str | None] = ContextVar("flight_recorder_call_id", default=None)
# Set while a wrapped tool records, so a tool whose _arun calls its own _run
# records the call once.
_recording: ContextVar[bool] = ContextVar("flight_recorder_lc_tool", default=False)


def wrap_model(model: M) -> M:
    """Record the calls a LangChain chat model makes inside ``fr.run()``.

    Supports the ``langchain-openai`` and ``langchain-anthropic`` chat models
    (and their subclasses) by wrapping the SDK clients they hold, which keeps
    provider-native messages, usage and cost. Returns the same model; any other
    model raises ``TypeError``.
    """
    packages = {cls.__module__.split(".")[0] for cls in type(model).__mro__}
    if "langchain_openai" in packages:
        names = ("root_client", "root_async_client")
    elif "langchain_anthropic" in packages:
        names = ("_client", "_async_client")  # built on first access
    else:
        raise TypeError(
            f"fr.langchain.wrap_model does not support {type(model).__qualname__}"
        )
    for name in names:
        if (client := getattr(model, name, None)) is not None:
            wrap(client)
    _tag_nodes()
    return model


def wrap_tools(
    tools: Sequence[T], effects: Mapping[str, Effect] | None = None
) -> list[T]:
    """Record each call of these LangChain tools as a tool step.

    ``effects`` maps tool names to read, write or irreversible; a tool not
    named is irreversible (FR-G1), unless the policy's ``[gate.effects]``
    classifies it. With a ``[gate]`` in the policy, each call is gated. Returns wrapped copies, in order; the tools
    given are unchanged. Calls made outside ``fr.run()`` run unrecorded.
    """
    from langchain_core.tools import BaseTool

    effects = dict(effects or {})
    for item in tools:
        if not isinstance(item, BaseTool):
            raise TypeError(f"not a LangChain tool: {item!r}")
    if unknown := effects.keys() - {t.name for t in tools}:  # type: ignore[attr-defined]
        raise ValueError(f"effects names tools not given: {sorted(unknown)}")
    for name, effect in effects.items():
        if effect not in get_args(Effect):
            raise ValueError(f"effect for {name!r} must be one of {get_args(Effect)}")
    _tag_nodes()
    return [_wrap_tool(t, effects.get(t.name)) for t in tools]  # type: ignore[attr-defined]


def _wrap_tool(tool: Any, declared: Effect | None) -> Any:
    copy = tool.model_copy()
    name: str = copy.name
    engine.register(name, declared or "irreversible")
    # The class's own methods, so re-wrapping replaces an earlier wrap (and its
    # effect) instead of nesting inside it.
    cls: type[Any] = copy.__class__
    run, arun, sync, async_ = (
        types.MethodType(getattr(cls, method), copy)
        for method in ("run", "arun", "_run", "_arun")
    )

    def arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        recorded = {k: jsonable(v) for k, v in kwargs.items() if k not in _INJECTED}
        if args:  # a single-input tool passes its input positionally
            recorded["*args"] = jsonable(list(args))
        return recorded

    def close(
        active: ActiveRun,
        pending: PendingStep,
        call: tuple[Effect, dict[str, Any]],
        result: Any = None,
        exc: BaseException | None = None,
        status: Literal["ok", "error", "denied"] | None = None,
    ) -> None:
        effect, recorded = call
        active.close_step(
            pending,
            status or ("ok" if exc is None else "error"),
            tool=ToolCall(
                name=name,
                effect=effect,
                args=recorded,
                result=jsonable(result),
                error=error_info(exc) if exc is not None else None,
                call_id=_call_id.get(),
            ),
        )

    def start(
        active: ActiveRun, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[PendingStep, tuple[Effect, dict[str, Any]], engine.Served | None]:
        effect = gate.effect_of(name, declared)
        recorded = arguments(args, kwargs)
        served = None
        if engine.replaying_now():
            served = engine.serve_tool(active, name, effect, recorded)
            if served is not None and served.source == "live":
                served = None
        pending = active.open_step("tool")
        if served is not None:
            pending.gate = gate.replayed(active, name, effect, recorded, served.gate)
        return pending, (effect, recorded), served

    def replay(
        active: ActiveRun,
        pending: PendingStep,
        call: tuple[Effect, dict[str, Any]],
        served: engine.Served,
    ) -> Any:
        # run() wraps the result in a ToolMessage with the live call id, so
        # the model's conversation stays consistent.
        pending.source = served.source
        recorded = served.gate
        denied = recorded is not None and recorded.outcome not in (
            "allowed",
            "approved",
        )
        if denied:
            active.denied = True
        status = "denied" if denied else None
        if served.error is not None:
            close(active, pending, call, exc=served.error, status=status)
            raise served.error
        close(active, pending, call, served.result, status=status)
        return served.result

    def deny(
        active: ActiveRun,
        pending: PendingStep,
        call: tuple[Effect, dict[str, Any]],
        verdict: gate.Verdict,
    ) -> Any:
        exc, returned = gate.denial(verdict, name)
        active.denied = True
        close(active, pending, call, returned, exc, status="denied")
        if returned is not None:
            return returned  # becomes the ToolMessage the model reads
        raise exc

    # functools.wraps keeps the signature LangChain reads to decide whether to
    # pass run_manager and config.
    @functools.wraps(sync)
    def recorded_run(*args: Any, **kwargs: Any) -> Any:
        active = active_run()
        if active is None or _recording.get():
            return sync(*args, **kwargs)
        pending, call, served = start(active, args, kwargs)
        if served is not None:
            return replay(active, pending, call, served)
        verdict = gate.check(active, name, *call)
        if verdict is not None:
            pending.gate = verdict.decision
            if not verdict.allowed:
                return deny(active, pending, call, verdict)
        token = _recording.set(True)
        try:
            result = sync(*args, **kwargs)
        except BaseException as exc:
            close(active, pending, call, exc=exc)
            raise
        finally:
            _recording.reset(token)
        close(active, pending, call, result)
        return result

    @functools.wraps(async_)
    async def recorded_arun(*args: Any, **kwargs: Any) -> Any:
        active = active_run()
        if active is None:
            return await async_(*args, **kwargs)
        pending, call, served = start(active, args, kwargs)
        if served is not None:
            return replay(active, pending, call, served)
        verdict = await gate.acheck(active, name, *call)
        if verdict is not None:
            pending.gate = verdict.decision
            if not verdict.allowed:
                return deny(active, pending, call, verdict)
        token = _recording.set(True)
        try:
            result = await async_(*args, **kwargs)
        except BaseException as exc:
            close(active, pending, call, exc=exc)
            raise
        finally:
            _recording.reset(token)
        close(active, pending, call, result)
        return result

    # The instance attributes shadow the class's methods; pydantic's own
    # setattr would reject names that are not fields.
    for attribute, method in (
        ("run", _with_call_id(run)),
        ("arun", _with_call_id_async(arun)),
        ("_run", recorded_run),
        ("_arun", recorded_arun),
    ):
        object.__setattr__(copy, attribute, method)
    return copy


def _with_call_id(run: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(run)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _call_id.set(kwargs.get("tool_call_id"))
        try:
            return run(*args, **kwargs)
        finally:
            _call_id.reset(token)

    return wrapper


def _with_call_id_async(arun: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(arun)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _call_id.set(kwargs.get("tool_call_id"))
        try:
            return await arun(*args, **kwargs)
        finally:
            _call_id.reset(token)

    return wrapper


def _node_tags() -> list[str]:
    from langchain_core.runnables.config import var_child_runnable_config

    config = var_child_runnable_config.get() or {}
    node = config.get("metadata", {}).get("langgraph_node")
    return [f"node:{node}"] if isinstance(node, str) else []


def _tag_nodes() -> None:
    if _node_tags not in recorder.tag_sources:
        recorder.tag_sources.append(_node_tags)
