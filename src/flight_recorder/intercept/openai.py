"""OpenAI client interception (FR-R1, FR-R2).

``wrap`` replaces ``create`` on one client's ``chat.completions`` and
``responses`` resources, so the client keeps its type and every other method.
"""

import functools
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Literal, cast

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from flight_recorder.context import active_run
from flight_recorder.model import ModelCall, Usage
from flight_recorder.recorder import ActiveRun, PendingStep, error_info, jsonable

Api = Literal["chat", "responses"]

# SDK transport options, not model parameters.
_TRANSPORT = ("extra_headers", "extra_query", "timeout")
_MESSAGES: dict[Api, str] = {"chat": "messages", "responses": "input"}


def wrap(client: OpenAI | AsyncOpenAI) -> None:
    """Record every ``create`` call this client makes inside ``fr.run()``."""
    is_async = isinstance(client, AsyncOpenAI)
    resources: list[tuple[Api, Any]] = [
        ("chat", client.chat.completions),
        ("responses", client.responses),
    ]
    for api, resource in resources:
        original = resource.create
        if getattr(original, "_fr_wrapped", False):
            continue
        make = _async_create if is_async else _sync_create
        create = make(api, original)
        create._fr_wrapped = True  # type: ignore[attr-defined]
        setattr(resource, "create", create)  # noqa: B010  shadows the bound method


class _Request:
    """One call's request as recorded, and the kwargs actually sent."""

    def __init__(self, api: Api, kwargs: dict[str, Any]) -> None:
        self.api: Api = api
        self.model = str(kwargs.get("model", ""))
        messages = _MESSAGES[api]
        self.messages = jsonable(kwargs.get(messages))
        self.params = {
            k: jsonable(v)
            for k, v in kwargs.items()
            if k not in (messages, "model", *_TRANSPORT)
        }
        self.kwargs = kwargs
        # A chat stream reports usage only when asked. Ask, and hide the extra
        # usage-only chunk, so the caller sees the chunks it would have seen.
        self.hide_usage_chunk = (
            api == "chat"
            and kwargs.get("stream") is True
            and "stream_options" not in kwargs
        )
        if self.hide_usage_chunk:
            self.kwargs = {**kwargs, "stream_options": {"include_usage": True}}

    def call(
        self,
        response: Any = None,
        error: BaseException | None = None,
        chunks: list[Any] | None = None,
        abandoned: bool = False,
    ) -> ModelCall:
        model_used, finish_reason, usage = self.model, None, None
        if chunks is not None:
            model_used, finish_reason, usage = _from_chunks(
                self.api, chunks, self.model
            )
            response = [c.model_dump(mode="json") for c in chunks]
        elif response is not None:
            model_used, finish_reason, usage = _from_response(self.api, response)
            response = response.model_dump(mode="json")
        return ModelCall(
            provider="openai",
            model_requested=self.model,
            model_used=model_used,
            params=self.params,
            messages=self.messages,
            response=response,
            finish_reason="abandoned" if abandoned else finish_reason,
            usage=usage,
            error=error_info(error) if error is not None else None,
        )


def _from_response(api: Api, response: Any) -> tuple[str, str | None, Usage | None]:
    usage = None
    if api == "chat":
        finish = response.choices[0].finish_reason if response.choices else None
        if response.usage is not None:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
    else:
        details = response.incomplete_details
        finish = details.reason if details is not None else response.status
        if response.usage is not None:
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
    return response.model, finish, usage


def _from_chunks(
    api: Api, chunks: list[Any], model: str
) -> tuple[str, str | None, Usage | None]:
    if api == "responses":
        # The final event carries the whole response.
        for event in reversed(chunks):
            if isinstance(getattr(event, "response", None), BaseModel):
                return _from_response(api, event.response)
        return model, None, None
    finish, usage = None, None
    for chunk in chunks:
        model = chunk.model or model
        if chunk.choices and chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
        if chunk.usage is not None:
            usage = Usage(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
            )
    return model, finish, usage


def _sync_create(api: Api, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def create(**kwargs: Any) -> Any:
        active = active_run()
        if active is None:
            return original(**kwargs)
        request = _Request(api, kwargs)
        pending = active.open_step("model")
        try:
            result: Any = original(**request.kwargs)
        except BaseException as exc:
            _close(active, pending, request, error=exc)
            raise
        if isinstance(result, Stream):
            try:
                _record_stream(active, pending, request, result)  # type: ignore[reportUnknownArgumentType]
            except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
                active.log_failure(exc)
                active.release(pending)
        else:
            _close(active, pending, request, response=result)
        return cast(Any, result)  # isinstance narrowed it to Stream[Unknown]

    return create


def _async_create(api: Api, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def create(**kwargs: Any) -> Any:
        active = active_run()
        if active is None:
            return await original(**kwargs)
        request = _Request(api, kwargs)
        pending = active.open_step("model")
        try:
            result: Any = await original(**request.kwargs)
        except BaseException as exc:
            _close(active, pending, request, error=exc)
            raise
        if isinstance(result, AsyncStream):
            try:
                _record_async_stream(active, pending, request, result)  # type: ignore[reportUnknownArgumentType]
            except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
                active.log_failure(exc)
                active.release(pending)
        else:
            _close(active, pending, request, response=result)
        return cast(Any, result)  # isinstance narrowed it to Stream[Unknown]

    return create


def _close(
    active: ActiveRun, pending: PendingStep, request: _Request, **call: Any
) -> None:
    """Close the step from ``request.call(**call)``, failing open."""
    try:
        model = request.call(**call)
    except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
        active.log_failure(exc)
        active.release(pending)
        return
    active.close_step(pending, "error" if model.error else "ok", model=model)


class _StreamStep:
    """The pending step of one stream; closes exactly once, from any thread."""

    def __init__(
        self, active: ActiveRun, pending: PendingStep, request: _Request
    ) -> None:
        self.active, self.pending, self.request = active, pending, request
        self.chunks: list[Any] = []
        self._lock = threading.Lock()
        self._closed = False
        active.release(pending)
        active.on_finish(self.abandon)

    def shown(self, chunk: Any) -> bool:
        """Record ``chunk``; False if the caller must not see it."""
        self.chunks.append(chunk)
        return not (
            self.request.hide_usage_chunk
            and not chunk.choices
            and chunk.usage is not None
        )

    def close(
        self, error: BaseException | None = None, abandoned: bool = False
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        _close(
            self.active,
            self.pending,
            self.request,
            chunks=list(self.chunks),
            error=error,
            abandoned=abandoned,
        )

    def abandon(self) -> None:
        self.close(abandoned=True)


def _record_stream(
    active: ActiveRun, pending: PendingStep, request: _Request, stream: Stream[Any]
) -> None:
    step = _StreamStep(active, pending, request)

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

    # ponytail: replaces the SDK's private _iterator so the stream keeps its
    # type; the contract tests fail loudly if a new SDK version renames it.
    stream._iterator = chunks(stream._iterator)  # type: ignore[reportPrivateUsage]
    close = stream.close

    def close_early() -> None:
        step.abandon()
        close()

    setattr(stream, "close", close_early)  # noqa: B010


def _record_async_stream(
    active: ActiveRun, pending: PendingStep, request: _Request, stream: AsyncStream[Any]
) -> None:
    step = _StreamStep(active, pending, request)

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

    stream._iterator = chunks(stream._iterator)  # type: ignore[reportPrivateUsage]
    close = stream.close

    async def close_early() -> None:
        step.abandon()
        await close()

    setattr(stream, "close", close_early)  # noqa: B010
