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
        usage = response.usage
        return Read(
            response.model,
            response.stop_reason,
            _usage(_counts(usage)),
            *_billing(usage),
        )

    def read_chunks(self, chunks: list[Any]) -> Read:
        model, finish, start = self.model, None, None
        counts: dict[str, int | None] = {}
        for event in chunks:
            if event.type == "message_start":
                model, start = event.message.model, event.message.usage
                counts = _counts(start)
            elif event.type == "message_delta":
                finish = event.delta.stop_reason or finish
                # Cumulative counts; a field the delta omits keeps its start value.
                latest = _counts(event.usage)
                counts.update({k: v for k, v in latest.items() if v is not None})
        if start is None:
            return Read(model, finish, None)
        return Read(model, finish, _usage(counts), *_billing(start))


_COUNTS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _counts(usage: Any) -> dict[str, int | None]:
    return {name: getattr(usage, name, None) for name in _COUNTS}


def _usage(counts: dict[str, int | None]) -> Usage:
    # Anthropic counts cached prompt tokens apart from input_tokens; Usage's
    # input_tokens is the whole prompt, comparable with OpenAI.
    read = counts["cache_read_input_tokens"] or 0
    write = counts["cache_creation_input_tokens"] or 0
    return Usage(
        input_tokens=(counts["input_tokens"] or 0) + read + write,
        output_tokens=counts["output_tokens"] or 0,
        cache_read_tokens=read,
        cache_write_tokens=write,
    )


def _billing(usage: Any) -> tuple[int, str | None]:
    """One-hour cache writes, and any modifier the pricing table doesn't price."""
    cache = getattr(usage, "cache_creation", None)
    one_hour = cache.ephemeral_1h_input_tokens if cache is not None else 0
    modifiers = [
        f"{name} {value!r}"
        for name, standard in (
            ("service_tier", "standard"),
            ("speed", "standard"),
            ("inference_geo", "global"),
        )
        if (value := getattr(usage, name, None)) not in (None, standard)
    ]
    return one_hour, ", ".join(modifiers) or None


def wrap(client: Anthropic | AsyncAnthropic) -> None:
    """Record every model call this client's ``messages`` resource makes."""
    messages: Any = client.messages
    if wrapped(messages.create):
        return
    is_async = isinstance(client, AsyncAnthropic)
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
    hook = _STREAM_HOOK[is_async]

    @functools.wraps(stream)
    def new_stream(**kwargs: Any) -> Any:
        manager = stream(**kwargs)
        send = getattr(manager, hook, None)
        if send is None:
            if (active := active_run()) is not None:
                active.log_failure(AttributeError(f"stream manager has no {hook}"))
            return manager
        if is_async:
            recorded = acall(_Messages, kwargs, lambda _: send, stream_type)
        else:
            recorded = lambda: call(_Messages, kwargs, lambda _: send(), stream_type)
        setattr(manager, hook, recorded)
        return manager

    messages.create = mark(recorded(messages.create))
    messages.parse = mark(recorded(messages.parse))
    messages.stream = mark(new_stream)
    # Built on first use around create; drop any built before.
    for cached in ("with_raw_response", "with_streaming_response"):
        cast(dict[str, Any], vars(messages)).pop(cached, None)
