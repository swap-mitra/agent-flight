"""@fr.tool: record each call of a tool as a step (FR-R3)."""

import functools
import inspect
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast, get_args

from pydantic import JsonValue

from flight_recorder.context import active_run
from flight_recorder.model import Effect, ToolCall
from flight_recorder.recorder import ActiveRun, PendingStep, error_info, jsonable
from flight_recorder.replay import engine

P = ParamSpec("P")
T = TypeVar("T")

# Bound instances repr differently on every run, which would break replay matching.
_IGNORED_ARGS = ("self", "cls")


def tool(
    *, effect: Effect = "irreversible"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Record calls to a sync or async function as tool steps.

    Use as ``@fr.tool(effect="read")``, or wrap an existing callable with
    ``fr.tool(effect="read")(fn)``. ``effect`` is the side-effect class: read,
    write or irreversible; an unclassified tool is irreversible (FR-G1).
    Calls made outside ``fr.run()`` run unrecorded. The function's signature
    and docstring are kept, since frameworks read them to build tool schemas.
    """
    if effect not in get_args(Effect):
        raise ValueError(f"effect must be one of {get_args(Effect)}, not {effect!r}")

    def decorate(fn: Callable[P, T]) -> Callable[P, T]:
        sig = inspect.signature(fn)

        engine.register(fn.__name__, effect)

        def close(
            active: ActiveRun,
            pending: PendingStep,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            result: Any = None,
            exc: BaseException | None = None,
        ) -> None:
            active.close_step(
                pending,
                "ok" if exc is None else "error",
                tool=ToolCall(
                    name=fn.__name__,
                    effect=effect,
                    args=_bind(sig, args, kwargs),
                    result=jsonable(result) if exc is None else None,
                    error=error_info(exc) if exc is not None else None,
                ),
            )

        def _served(
            active: ActiveRun, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> engine.Served | None:
            if not engine.replaying_now():
                return None
            bound = _bind(sig, args, kwargs)
            served = engine.serve_tool(active, fn.__name__, effect, bound)
            return None if served is None or served.source == "live" else served

        def _replay(
            active: ActiveRun,
            pending: PendingStep,
            served: engine.Served,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            pending.source = served.source
            if served.error is not None:
                close(active, pending, args, kwargs, exc=served.error)
                raise served.error
            close(active, pending, args, kwargs, served.result)
            return served.result

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                active = active_run()
                if active is None:
                    return await fn(*args, **kwargs)
                served = _served(active, args, kwargs)
                pending = active.open_step("tool")
                if served is not None:
                    return _replay(active, pending, served, args, kwargs)
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as exc:
                    close(active, pending, args, kwargs, exc=exc)
                    raise
                close(active, pending, args, kwargs, result)
                return result

            return cast(Callable[P, T], async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            active = active_run()
            if active is None:
                return fn(*args, **kwargs)
            served = _served(active, args, kwargs)
            pending = active.open_step("tool")
            if served is not None:
                return _replay(active, pending, served, args, kwargs)
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                close(active, pending, args, kwargs, exc=exc)
                raise
            close(active, pending, args, kwargs, result)
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
