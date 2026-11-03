"""The current run, tracked in a ContextVar (ADR-007).

asyncio tasks inherit the run automatically. Threads do not, so work sent to a
thread goes through copy_context() or thread_executor().
"""

import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar, Token
from contextvars import copy_context as _copy_context
from types import TracebackType
from typing import Any, ParamSpec, TypeVar

from flight_recorder.model import Run
from flight_recorder.recorder import ActiveRun, jsonable

P = ParamSpec("P")
T = TypeVar("T")

_current: ContextVar[ActiveRun | None] = ContextVar("flight_recorder_run", default=None)


def active_run() -> ActiveRun | None:
    """The run being recorded in this context, or None outside any run."""
    return _current.get()


def uuid7() -> str:
    """A UUIDv7 string: time-sortable at millisecond resolution (RFC 9562).

    stdlib uuid.uuid7 only arrives in Python 3.14.
    """
    ms = time.time_ns() // 1_000_000
    value = (ms & (2**48 - 1)) << 80 | int.from_bytes(os.urandom(10))
    value = value & ~(0xF << 76) | 0x7 << 76  # version 7
    value = value & ~(0b11 << 62) | 0b10 << 62  # RFC variant
    return str(uuid.UUID(int=value))


class run:
    """Open a run: ``with fr.run("triage", inputs=...) as run:``, or ``async with``.

    A run opened inside another becomes its child. On exit the status is ``ok``,
    or ``error`` if the block raised; the exception propagates unchanged.
    """

    def __init__(self, name: str, inputs: Any = None) -> None:
        parent = _current.get()
        self._active = ActiveRun(
            Run(
                run_id=uuid7(),
                name=name,
                parent_run_id=parent.run.run_id if parent else None,
                status="incomplete",
                inputs=jsonable(inputs),
            )
        )
        self._token: Token[ActiveRun | None] | None = None

    def __enter__(self) -> Run:
        self._token = _current.set(self._active)
        self._active.start()
        return self._active.run

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._token is not None
        _current.reset(self._token)
        self._active.finish(failed=exc is not None)

    async def __aenter__(self) -> Run:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)


def copy_context(fn: Callable[P, T]) -> Callable[P, T]:
    """Wrap fn to run in a copy of the caller's context, e.g. as a Thread target."""
    ctx = _copy_context()

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return ctx.run(fn, *args, **kwargs)

    return wrapper


class _ContextThreadPoolExecutor(ThreadPoolExecutor):
    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        # Copy per task, at submit time: each task sees the run that submitted it.
        return super().submit(copy_context(fn), *args, **kwargs)


def thread_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    """A ThreadPoolExecutor whose tasks see the run that submitted them."""
    return _ContextThreadPoolExecutor(max_workers=max_workers)
