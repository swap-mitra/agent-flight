"""Anthropic client interception (FR-R1, FR-R2).

``wrap`` replaces ``create`` and ``stream`` on one client's ``messages``
resource, so the client keeps its type and every other method.
"""

import functools
from typing import Any

from anthropic import Anthropic, AsyncAnthropic, AsyncStream, Stream

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

    def read_response(self, response: Any) -> Read:
        return response.model, response.stop_reason, _usage(response.usage)

    def read_chunks(self, chunks: list[Any]) -> Read:
        model, finish = self.model, None
        tokens_in: int | None = None
        tokens_out: int | None = None
        for event in chunks:
            if event.type == "message_start":
                model = event.message.model
                start = _usage(event.message.usage)
                tokens_in, tokens_out = start.input_tokens, start.output_tokens
            elif event.type == "message_delta":
                finish = event.delta.stop_reason or finish
                tokens_out = event.usage.output_tokens  # cumulative
                if event.usage.input_tokens is not None:
                    tokens_in = _usage(event.usage).input_tokens
        if tokens_in is None or tokens_out is None:
            return model, finish, None
        return model, finish, Usage(input_tokens=tokens_in, output_tokens=tokens_out)


def _usage(usage: Any) -> Usage:
    # Anthropic counts cached prompt tokens separately; input_tokens is the
    # whole prompt, comparable with OpenAI. The breakdown stays in the response.
    return Usage(
        input_tokens=(usage.input_tokens or 0)
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0),
        output_tokens=usage.output_tokens,
    )


def wrap(client: Anthropic | AsyncAnthropic) -> None:
    """Record every ``messages.create`` and ``messages.stream`` call."""
    messages: Any = client.messages
    if wrapped(messages.create):
        return
    is_async = isinstance(client, AsyncAnthropic)
    create, stream = messages.create, messages.stream
    stream_type = AsyncStream if is_async else Stream

    if is_async:

        @functools.wraps(create)
        async def acreate(**kwargs: Any) -> Any:
            return await acall(_Messages, kwargs, lambda kw: create(**kw), stream_type)

        new_create = acreate
    else:

        @functools.wraps(create)
        def screate(**kwargs: Any) -> Any:
            return call(_Messages, kwargs, lambda kw: create(**kw), stream_type)

        new_create = screate

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

    messages.create = mark(new_create)
    messages.stream = mark(new_stream)
