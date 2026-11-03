"""OpenAI client interception (FR-R1, FR-R2).

``wrap`` replaces ``create`` on one client's ``chat.completions`` and
``responses`` resources, so the client keeps its type and every other method.
"""

import functools
from typing import Any

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from flight_recorder.intercept._model import Read, Request, acall, call, mark, wrapped
from flight_recorder.model import Usage


class _Chat(Request):
    provider = "openai"

    def __init__(self, kwargs: dict[str, Any]) -> None:
        super().__init__(kwargs)
        # A chat stream reports usage only when asked. Ask, and hide the extra
        # usage-only chunk, so the caller sees the chunks it would have seen.
        self.ask_usage = kwargs.get("stream") is True and "stream_options" not in kwargs
        if self.ask_usage:
            self.kwargs = {**kwargs, "stream_options": {"include_usage": True}}

    def hidden(self, chunk: Any) -> bool:
        return self.ask_usage and not chunk.choices and chunk.usage is not None

    def read_response(self, response: Any) -> Read:
        finish = response.choices[0].finish_reason if response.choices else None
        return response.model, finish, _usage(response.usage)

    def read_chunks(self, chunks: list[Any]) -> Read:
        model, finish, usage = self.model, None, None
        for chunk in chunks:
            model = chunk.model or model
            if chunk.choices and chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
            usage = _usage(chunk.usage) or usage
        return model, finish, usage


class _Responses(Request):
    provider = "openai"
    messages_key = "input"

    def read_response(self, response: Any) -> Read:
        details = response.incomplete_details
        finish = details.reason if details is not None else response.status
        usage = response.usage
        if usage is not None:
            usage = Usage(
                input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
            )
        return response.model, finish, usage

    def read_chunks(self, chunks: list[Any]) -> Read:
        # The final event carries the whole response.
        for event in reversed(chunks):
            if isinstance(getattr(event, "response", None), BaseModel):
                return self.read_response(event.response)
        return self.model, None, None


def _usage(usage: Any) -> Usage | None:
    if usage is None:
        return None
    return Usage(
        input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens
    )


def wrap(client: OpenAI | AsyncOpenAI) -> None:
    """Record every ``create`` call this client makes inside ``fr.run()``."""
    resources: list[tuple[type[Request], Any]] = [
        (_Chat, client.chat.completions),
        (_Responses, client.responses),
    ]
    for request_type, resource in resources:
        original = resource.create
        if wrapped(original):
            continue
        if isinstance(client, AsyncOpenAI):
            create = _async_create(request_type, original)
        else:
            create = _sync_create(request_type, original)
        setattr(resource, "create", mark(create))  # noqa: B010  shadows the method


def _sync_create(request_type: type[Request], original: Any) -> Any:
    @functools.wraps(original)
    def create(**kwargs: Any) -> Any:
        return call(request_type, kwargs, lambda kw: original(**kw), Stream)

    return create


def _async_create(request_type: type[Request], original: Any) -> Any:
    @functools.wraps(original)
    async def create(**kwargs: Any) -> Any:
        return await acall(request_type, kwargs, lambda kw: original(**kw), AsyncStream)

    return create
