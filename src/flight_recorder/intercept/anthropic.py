"""Anthropic client interception (FR-R1, FR-R2).

``wrap`` replaces ``create``, ``stream`` and ``parse`` on one client's
``messages`` resource, so the client keeps its type and every other method.
``with_raw_response`` and ``with_streaming_response`` call ``create``, so they
are recorded too.
"""

import functools
from typing import Any, cast

import anthropic
from anthropic import Anthropic, AsyncAnthropic, AsyncStream, Stream
from anthropic.lib._parse._transform import (  # pyright: ignore[reportMissingTypeStubs]
    transform_schema,
)
from pydantic import TypeAdapter

from flight_recorder.context import active_run
from flight_recorder.intercept._model import Read, Request, acall, call, mark, wrapped
from flight_recorder.model import Usage

# The private attribute each stream manager sends its request through.
_STREAM_HOOK = {
    False: "_MessageStreamManager__api_request",
    True: "_AsyncMessageStreamManager__api_request",
}


class _Messages(Request):
    provider = "anthropic"
    omitted = (anthropic.Omit, anthropic.NotGiven)
    rejected = (anthropic.APIStatusError,)
    raw = (anthropic.APIResponse, anthropic.AsyncAPIResponse)

    def format(self, value: type) -> Any:
        # As messages.parse() does. ponytail: transform_schema is an SDK internal.
        schema = TypeAdapter[Any](value).json_schema()
        return {"schema": transform_schema(schema), "type": "json_schema"}

    def read_response(self, response: Any) -> Read:
        return _read(response.model, response.stop_reason, _fields(response.usage))

    def read_chunks(self, chunks: list[Any]) -> Read:
        model, finish = self.model, None
        fields: dict[str, Any] | None = None
        for event in chunks:
            if event.type == "message_start":
                model, fields = event.message.model, _fields(event.message.usage)
            elif event.type == "message_delta":
                finish = event.delta.stop_reason or finish
                # Cumulative; a field the delta omits keeps its start value.
                if fields is not None:
                    latest = _fields(event.usage).items()
                    fields.update({k: v for k, v in latest if v is not None})
        if fields is None:
            return Read(model, finish, None)
        return _read(model, finish, fields)


# The usage fields read, from a response or a stream's start and deltas.
_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
    "cache_creation",
    "server_tool_use",
    "service_tier",
    "speed",
    "inference_geo",
)


def _fields(usage: Any) -> dict[str, Any]:
    return {name: getattr(usage, name, None) for name in _FIELDS}


def _read(model: str, finish: str | None, fields: dict[str, Any]) -> Read:
    # Anthropic counts cached prompt tokens apart from input_tokens; Usage's
    # input_tokens is the whole prompt, comparable with OpenAI.
    read = fields["cache_read_input_tokens"] or 0
    write = fields["cache_creation_input_tokens"] or 0
    usage = Usage(
        input_tokens=(fields["input_tokens"] or 0) + read + write,
        output_tokens=fields["output_tokens"] or 0,
        cache_read_tokens=read,
        cache_write_tokens=write,
    )
    cache, tools = fields["cache_creation"], fields["server_tool_use"]
    modifiers = [
        f"{name} {fields[name]!r}"
        for name, standard in (
            ("service_tier", "standard"),
            ("speed", "standard"),
            ("inference_geo", "global"),
        )
        if fields[name] not in (None, standard)
    ]
    return Read(
        model,
        finish,
        usage,
        cache_write_1h=cache.ephemeral_1h_input_tokens if cache is not None else 0,
        unpriced=", ".join(modifiers) or None,
        # Web fetch is free; code execution hours are billed per organisation.
        web_searches=tools.web_search_requests if tools is not None else 0,
    )


def wrap(client: Anthropic | AsyncAnthropic) -> None:
    """Record every model call this client's ``messages`` resource makes."""
    _wrap_messages(client.messages, isinstance(client, AsyncAnthropic), _STREAM_HOOK)


def _wrap_messages(messages: Any, is_async: bool, hooks: dict[bool, str]) -> None:
    if wrapped(messages.create):
        return
    stream = messages.stream
    stream_type = AsyncStream if is_async else Stream

    def recorded(method: Any) -> Any:
        if is_async:

            @functools.wraps(method)
            async def arecorded(**kwargs: Any) -> Any:
                return await acall(
                    _Messages, kwargs, lambda kw: method(**kw), stream_type
                )

            return arecorded

        @functools.wraps(method)
        def srecorded(**kwargs: Any) -> Any:
            return call(_Messages, kwargs, lambda kw: method(**kw), stream_type)

        return srecorded

    # ponytail: stream() returns a manager that sends its request on entry
    # through a private hook; replacing the hook records every way of reading
    # the stream (events, text_stream, get_final_message). Contract tests fail
    # loudly if an SDK version renames it; production then records nothing
    # for the call and logs once.
    hook = hooks[is_async]

    @functools.wraps(stream)
    def new_stream(**kwargs: Any) -> Any:
        manager = stream(**kwargs)
        send = getattr(manager, hook, None)
        if send is None:
            if (active := active_run()) is not None:
                active.log_failure(AttributeError(f"stream manager has no {hook}"))
            return manager
        if is_async:
            sent = acall(_Messages, kwargs, lambda _: send, stream_type)
        else:
            sent = lambda: call(_Messages, kwargs, lambda _: send(), stream_type)
        setattr(manager, hook, sent)
        return manager

    messages.create = mark(recorded(messages.create))
    messages.parse = mark(recorded(messages.parse))
    messages.stream = mark(new_stream)
    # Built on first use around create; drop any built before.
    for cached in ("with_raw_response", "with_streaming_response"):
        cast(dict[str, Any], vars(messages)).pop(cached, None)
