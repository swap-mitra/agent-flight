"""Model-call recording shared by the provider interceptors (FR-R1, FR-R2).

A provider module subclasses ``Request`` to say how its requests and
responses read, then routes each SDK call through ``call`` or ``acall``.
"""

import inspect
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, NamedTuple

from flight_recorder.context import active_run
from flight_recorder.model import ModelCall, Usage
from flight_recorder.policy import pricing
from flight_recorder.recorder import ActiveRun, PendingStep, error_info, jsonable

# SDK transport options, not model parameters.
TRANSPORT = ("extra_headers", "extra_query", "timeout")


class Read(NamedTuple):
    """What a provider reads from a response or a stream's chunks."""

    model: str  # the model used
    finish_reason: str | None
    usage: Usage | None
    cache_write_1h: int = 0  # of usage.cache_write_tokens, priced apart
    unpriced: str | None = None  # a pricing modifier the table can't express


class Request:
    """One call's request as recorded, and the kwargs actually sent."""

    provider: str
    messages_key = "messages"
    # The SDK's "argument not given" sentinels, which its own helpers pass on.
    omitted: tuple[type, ...] = ()
    # Errors meaning the provider rejected the request, which it doesn't bill.
    rejected: tuple[type[BaseException], ...] = ()
    # Raw HTTP responses, from with_raw_response and with_streaming_response.
    raw: tuple[type, ...] = ()

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.model = str(kwargs.get("model", ""))
        self.messages = jsonable(kwargs.get(self.messages_key))
        self.params = {
            k: jsonable(self._format(v) if isinstance(v, type) else v)
            for k, v in kwargs.items()
            if k not in (self.messages_key, "model", *TRANSPORT)
            and not isinstance(v, self.omitted)
            and not (k == "extra_body" and v is None)  # SDK helpers pass None
        }
        self.kwargs = kwargs

    def given(self, value: Any) -> bool:
        return value is not None and not isinstance(value, self.omitted)

    def format(self, value: type) -> Any:
        """The JSON schema the SDK sends for a parse() output type."""
        return value

    def _format(self, value: type) -> Any:
        try:
            return self.format(value)
        except Exception:  # noqa: BLE001  the SDK raises its own error on send
            return value

    def read_response(self, response: Any) -> Read:
        raise NotImplementedError

    def read_chunks(self, chunks: list[Any]) -> Read:
        raise NotImplementedError

    def hidden(self, chunk: Any) -> bool:
        """True for a chunk recorded but not shown to the caller."""
        return False

    def call(
        self,
        response: Any = None,
        error: BaseException | None = None,
        chunks: list[Any] | None = None,
        abandoned: bool = False,
    ) -> ModelCall:
        read, recorded = Read(self.model, None, None), None
        if chunks is not None:
            read = self.read_chunks(chunks)
            recorded = [c.model_dump(mode="json", warnings=False) for c in chunks]
        elif response is not None:
            read = self.read_response(response)
            recorded = response.model_dump(mode="json", warnings=False)
        model_used = read.model or self.model
        if error is not None and recorded is None:
            # A rejected request isn't billed; after any other failure (a
            # timeout, a client-side parse error) the tokens spent are unknown.
            cost = 0.0 if isinstance(error, self.rejected) else None
        elif abandoned or (error is not None and chunks is not None):
            cost = None  # a stream cut short: tokens spent, count unknown
        elif read.usage is None:
            cost = None
        else:
            cost = pricing.cost(
                self.provider,
                model_used,
                read.usage,
                cache_write_1h=read.cache_write_1h,
                unpriced=read.unpriced,
            )
        return ModelCall(
            provider=self.provider,
            model_requested=self.model,
            model_used=model_used,
            params=self.params,
            messages=self.messages,
            response=recorded,
            finish_reason="abandoned" if abandoned else read.finish_reason,
            usage=read.usage,
            cost_usd=cost,
            error=error_info(error) if error is not None else None,
        )


def mark(fn: Callable[..., Any]) -> Callable[..., Any]:
    fn._fr_wrapped = True  # type: ignore[attr-defined]
    return fn


def wrapped(fn: Any) -> bool:
    return getattr(fn, "_fr_wrapped", False)


def call(
    request_type: type[Request],
    kwargs: dict[str, Any],
    send: Callable[[dict[str, Any]], Any],
    stream_type: type,
) -> Any:
    """Send ``kwargs`` and record the call as a model step of the active run."""
    active = active_run()
    if active is None:
        return send(kwargs)
    request = request_type(kwargs)
    pending = active.open_step("model")
    try:
        result = send(request.kwargs)
    except BaseException as exc:
        _close(active, pending, request, error=exc)
        raise
    _record(active, pending, request, result, stream_type, _sync_stream)
    return result


async def acall(
    request_type: type[Request],
    kwargs: dict[str, Any],
    send: Callable[[dict[str, Any]], Awaitable[Any]],
    stream_type: type,
) -> Any:
    """``call`` for async clients."""
    active = active_run()
    if active is None:
        return await send(kwargs)
    request = request_type(kwargs)
    pending = active.open_step("model")
    try:
        result = await send(request.kwargs)
    except BaseException as exc:
        _close(active, pending, request, error=exc)
        raise
    _record(active, pending, request, result, stream_type, _async_stream)
    return result


def _record(
    active: ActiveRun,
    pending: PendingStep,
    request: Request,
    result: Any,
    stream_type: type,
    record_stream: Callable[["_LateStep", Any], None],
) -> None:
    if not isinstance(result, (stream_type, *request.raw)):
        _close(active, pending, request, response=result)
        return
    try:
        step = _LateStep(active, pending, request)
        if isinstance(result, stream_type):
            record_stream(step, result)
        else:
            _record_raw(step, result, stream_type, record_stream)
    except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
        active.log_failure(exc)
        active.release(pending)


def _record_raw(
    step: "_LateStep",
    raw: Any,
    stream_type: type,
    record_stream: Callable[["_LateStep", Any], None],
) -> None:
    """Record a raw response when the caller first parses it.

    The SDK caches the parse, so the caller gets the object recorded here. Never
    parsed, the step closes as abandoned when the run ends; no body is read
    early, so with_streaming_response stays lazy.
    """
    parse = raw.parse
    parsed_once = False

    def record(parsed: Any) -> None:
        nonlocal parsed_once
        if parsed_once:
            return
        parsed_once = True
        try:
            if isinstance(parsed, stream_type):
                record_stream(step, parsed)
            else:
                step.close(response=parsed)
        except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
            step.active.log_failure(exc)

    if inspect.iscoroutinefunction(parse):

        async def aparse(*args: Any, **kwargs: Any) -> Any:
            parsed = await parse(*args, **kwargs)
            record(parsed)
            return parsed

        raw.parse = aparse
    else:

        def sparse(*args: Any, **kwargs: Any) -> Any:
            parsed = parse(*args, **kwargs)
            record(parsed)
            return parsed

        raw.parse = sparse


def _close(
    active: ActiveRun, pending: PendingStep, request: Request, **call: Any
) -> None:
    """Close the step from ``request.call(**call)``, failing open."""
    try:
        model = request.call(**call)
    except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
        active.log_failure(exc)
        active.release(pending)
        return
    active.close_step(pending, "error" if model.error else "ok", model=model)


class _LateStep:
    """A step that closes after its call returns (a stream, or a raw response).

    Closes exactly once, from any thread.
    """

    def __init__(
        self, active: ActiveRun, pending: PendingStep, request: Request
    ) -> None:
        self.active, self.pending, self.request = active, pending, request
        self.chunks: list[Any] | None = None  # set once a stream is recorded
        self._lock = threading.Lock()
        self._closed = False
        # The caller's calls while reading the result are not its children.
        active.release(pending)
        active.on_finish(self.abandon)

    def shown(self, chunk: Any) -> bool:
        """Record ``chunk``; False if the caller must not see it."""
        assert self.chunks is not None
        self.chunks.append(chunk)
        return not self.request.hidden(chunk)

    def close(
        self,
        response: Any = None,
        error: BaseException | None = None,
        abandoned: bool = False,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        _close(
            self.active,
            self.pending,
            self.request,
            response=response,
            chunks=None if self.chunks is None else list(self.chunks),
            error=error,
            abandoned=abandoned,
        )

    def abandon(self) -> None:
        self.close(abandoned=True)


# ponytail: both SDKs' Stream classes read from a private _iterator; replacing
# it keeps the stream's type. The contract tests fail loudly if an SDK
# version renames it, and production recording then fails open.


def _sync_stream(step: _LateStep, stream: Any) -> None:
    step.chunks = []

    def chunks(inner: Iterator[Any]) -> Iterator[Any]:
        try:
            for chunk in inner:
                if step.shown(chunk):
                    yield chunk
        except Exception as exc:
            step.close(error=exc)
            raise
        else:
            step.close()
        finally:
            step.abandon()  # no-op once closed; else the stream was dropped

    stream._iterator = chunks(stream._iterator)
    close = stream.close

    def close_early() -> None:
        step.abandon()
        close()

    stream.close = close_early


def _async_stream(step: _LateStep, stream: Any) -> None:
    step.chunks = []

    async def chunks(inner: AsyncIterator[Any]) -> AsyncIterator[Any]:
        try:
            async for chunk in inner:
                if step.shown(chunk):
                    yield chunk
        except Exception as exc:
            step.close(error=exc)
            raise
        else:
            step.close()
        finally:
            step.abandon()  # no-op once closed; else the stream was dropped

    stream._iterator = chunks(stream._iterator)
    close = stream.close

    async def close_early() -> None:
        step.abandon()
        await close()

    stream.close = close_early
