"""@fr.tool: record each call of a tool as a step (FR-R3), gated (FR-G1 to G6)."""

import functools
import inspect
from collections.abc import Callable
from typing import Any, Literal, ParamSpec, TypeVar, cast, get_args

from pydantic import JsonValue

from flight_recorder.context import active_run
from flight_recorder.model import Effect, ToolCall
from flight_recorder.policy import gate
from flight_recorder.recorder import ActiveRun, PendingStep, error_info, jsonable
from flight_recorder.replay import engine

P = ParamSpec("P")
T = TypeVar("T")

# Bound instances repr differently on every run, which would break replay matching.
_IGNORED_ARGS = ("self", "cls")
# Gate outcomes after which the tool ran.
_RAN = ("allowed", "approved")


def tool(*, effect: Effect | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Record calls to a sync or async function as tool steps.

    Use as ``@fr.tool(effect="read")``, or wrap an existing callable with
    ``fr.tool(effect="read")(fn)``. ``effect`` is the side-effect class: read,
    write or irreversible. A tool that doesn't declare one is irreversible
    (FR-G1), unless the policy's ``[gate.effects]`` classifies it; the policy
    wins over the code. With a ``[gate]`` in the policy, each call is gated.
    Calls made outside ``fr.run()`` run unrecorded and ungated. The function's
    signature and docstring are kept, since frameworks read them.
    """
    if effect is not None and effect not in get_args(Effect):
        raise ValueError(f"effect must be one of {get_args(Effect)}, not {effect!r}")

    def decorate(fn: Callable[P, T]) -> Callable[P, T]:
        sig = inspect.signature(fn)
        name = fn.__name__
        engine.register(name, effect or "irreversible")

        def close(
            active: ActiveRun,
            pending: PendingStep,
            effect: Effect,
            args: dict[str, JsonValue],
            result: Any = None,
            exc: BaseException | None = None,
            status: Literal["ok", "error", "denied"] | None = None,
        ) -> None:
            active.close_step(
                pending,
                status or ("ok" if exc is None else "error"),
                tool=ToolCall(
                    name=name,
                    effect=effect,
                    args=args,
                    result=jsonable(result),
                    error=error_info(exc) if exc is not None else None,
                ),
            )

        def start(
            active: ActiveRun, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[PendingStep, Effect, dict[str, JsonValue], engine.Served | None]:
            resolved = gate.effect_of(name, effect)
            bound = _bind(sig, args, kwargs)
            served = None
            if engine.replaying_now():
                served = engine.serve_tool(active, name, resolved, bound)
                if served is not None and served.source == "live":
                    served = None
            pending = active.open_step("tool")
            if served is not None:
                pending.gate = gate.replayed(active, name, resolved, bound, served.gate)
            return pending, resolved, bound, served

        def replay(
            active: ActiveRun,
            pending: PendingStep,
            effect: Effect,
            bound: dict[str, JsonValue],
            served: engine.Served,
        ) -> Any:
            pending.source = served.source
            recorded = served.gate
            denied = recorded is not None and recorded.outcome not in _RAN
            if denied:
                active.denied = True
            status = "denied" if denied else None
            if served.error is not None:
                close(active, pending, effect, bound, exc=served.error, status=status)
                raise served.error
            close(active, pending, effect, bound, served.result, status=status)
            return served.result

        def deny(
            active: ActiveRun,
            pending: PendingStep,
            effect: Effect,
            bound: dict[str, JsonValue],
            verdict: gate.Verdict,
        ) -> Any:
            exc, returned = gate.denial(verdict, name)
            active.denied = True
            close(active, pending, effect, bound, returned, exc, status="denied")
            if returned is not None:
                return returned
            raise exc

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                active = active_run()
                if active is None:
                    return await fn(*args, **kwargs)
                pending, resolved, bound, served = start(active, args, kwargs)
                if served is not None:
                    return replay(active, pending, resolved, bound, served)
                verdict = await gate.acheck(active, name, resolved, bound)
                if verdict is not None:
                    pending.gate = verdict.decision
                    if not verdict.allowed:
                        return deny(active, pending, resolved, bound, verdict)
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as exc:
                    close(active, pending, resolved, bound, exc=exc)
                    raise
                close(active, pending, resolved, bound, result)
                return result

            return cast(Callable[P, T], async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            active = active_run()
            if active is None:
                return fn(*args, **kwargs)
            pending, resolved, bound, served = start(active, args, kwargs)
            if served is not None:
                return replay(active, pending, resolved, bound, served)
            verdict = gate.check(active, name, resolved, bound)
            if verdict is not None:
                pending.gate = verdict.decision
                if not verdict.allowed:
                    return deny(active, pending, resolved, bound, verdict)
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                close(active, pending, resolved, bound, exc=exc)
                raise
            close(active, pending, resolved, bound, result)
            return result

        return wrapper

    return decorate


def _bind(
    sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, JsonValue]:
    """Arguments by parameter name, defaults included, so matching ignores order."""
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        # The real call raises the same TypeError; keep what was passed.
        return {"*args": jsonable(args), "**kwargs": jsonable(kwargs)}
    bound.apply_defaults()
    return {
        name: jsonable(value)
        for name, value in bound.arguments.items()
        if name not in _IGNORED_ARGS
    }
