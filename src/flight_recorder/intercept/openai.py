"""OpenAI client interception (FR-R1, FR-R2).

``wrap`` replaces ``create`` and ``parse`` on one client's ``chat.completions``
and ``responses`` resources, so the client keeps its type and every other
method. The SDK's ``stream()`` helpers call ``create`` on the same resource,
and ``with_raw_response`` and ``with_streaming_response`` call these methods,
so they are recorded too.
"""

import functools
from typing import Any, cast

import openai
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai._legacy_response import LegacyAPIResponse
from openai.lib._parsing import type_to_response_format_param
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel

from flight_recorder.intercept._model import Read, Request, acall, call, mark, wrapped
from flight_recorder.model import Usage
from flight_recorder.replay import http


# ponytail: the raw response class and the two format converters are SDK
# internals. The contract tests fail loudly if a version moves them.
class _OpenAI(Request):
    provider = "openai"
    omitted = (openai.Omit, openai.NotGiven)
    rejected = (openai.APIStatusError,)
    raw = (LegacyAPIResponse, openai.APIResponse, openai.AsyncAPIResponse)


class _Chat(_OpenAI):
    def __init__(self, kwargs: dict[str, Any]) -> None:
        super().__init__(kwargs)
        # A chat stream reports usage only when asked. Ask, and hide the extra
        # usage-only chunk, so the caller sees the chunks it would have seen.
        self.ask_usage = kwargs.get("stream") is True and not self.given(
            kwargs.get("stream_options")
        )
        if self.ask_usage:
            self.kwargs = {**kwargs, "stream_options": {"include_usage": True}}

    def format(self, value: type) -> Any:
        return type_to_response_format_param(value)

    def hidden(self, chunk: Any) -> bool:
        return self.ask_usage and not chunk.choices and chunk.usage is not None

    def read_response(self, response: Any) -> Read:
        finish = response.choices[0].finish_reason if response.choices else None
        unpriced = _join(_tier(response), self._searched())
        return Read(response.model, finish, _usage(response.usage), unpriced=unpriced)

    def read_chunks(self, chunks: list[Any]) -> Read:
        model, finish, usage, tier = self.model, None, None, None
        for chunk in chunks:
            model = chunk.model or model
            if chunk.choices and chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
            usage = _usage(chunk.usage) or usage
            tier = _tier(chunk) or tier
        return Read(model, finish, usage, unpriced=_join(tier, self._searched()))

    def _searched(self) -> str | None:
        # Chat web search bills on top of tokens, at a fee OpenAI does not
        # publish for its search models, so it is never guessed.
        return "web_search_options" if "web_search_options" in self.params else None


class _Responses(_OpenAI):
    messages_key = "input"

    def format(self, value: type) -> Any:
        return type_to_text_format_param(value)

    def read_response(self, response: Any) -> Read:
        details = response.incomplete_details
        finish = details.reason if details is not None else response.status
        usage = response.usage
        if usage is not None:
            cached = usage.input_tokens_details
            usage = Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=(cached and cached.cached_tokens) or 0,
            )
        types = [item.type for item in response.output]
        search, unpriced = self._search_fee()
        searches = types.count("web_search_call")
        return Read(
            response.model,
            finish,
            usage,
            unpriced=_join(_tier(response), unpriced if searches else None),
            tool_calls={
                search: searches,
                "file_search": types.count("file_search_call"),
            },
        )

    def _search_fee(self) -> tuple[str, str | None]:
        """The fee a web search call is billed at, from the tools the request
        declared (web_search_call items don't say which tool made them), and
        why it can't be priced if both were declared."""
        tools = self.params.get("tools")
        kinds = [
            str(tool.get("type"))
            for tool in (tools if isinstance(tools, list) else [])
            if isinstance(tool, dict)
        ]
        declared = {
            "web_search_preview" if k.startswith("web_search_preview") else "web_search"
            for k in kinds
            if k.startswith("web_search")
        }
        if len(declared) > 1:
            return "web_search", "both web_search and web_search_preview"
        return (declared.pop() if declared else "web_search"), None

    def read_chunks(self, chunks: list[Any]) -> Read:
        # The final event carries the whole response.
        for event in reversed(chunks):
            if isinstance(getattr(event, "response", None), BaseModel):
                return self.read_response(event.response)
        return Read(self.model, None, None)


def _usage(usage: Any) -> Usage | None:
    if usage is None:
        return None
    cached = usage.prompt_tokens_details
    return Usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_read_tokens=(cached and cached.cached_tokens) or 0,
    )


def _join(*reasons: str | None) -> str | None:
    return ", ".join(r for r in reasons if r) or None


def _tier(response: Any) -> str | None:
    """The service tier, if the pricing table (standard tier) doesn't price it."""
    tier = getattr(response, "service_tier", None)
    return None if tier in (None, "auto", "default") else f"service_tier {tier!r}"


def wrap(client: OpenAI | AsyncOpenAI) -> None:
    """Record every model call this client makes inside ``fr.run()``."""
    sdk: Any = client
    http.install(sdk._client)  # serves fully-recorded replay
    resources: list[tuple[type[Request], Any]] = [
        (_Chat, client.chat.completions),
        (_Responses, client.responses),
    ]
    for request_type, resource in resources:
        if wrapped(resource.create):
            continue
        for name in ("create", "parse"):
            original = getattr(resource, name)
            if isinstance(client, AsyncOpenAI):
                method = _async_create(request_type, original)
            else:
                method = _sync_create(request_type, original)
            setattr(resource, name, mark(method))
        # Built on first use around the methods above; drop any built before.
        for cached in ("with_raw_response", "with_streaming_response"):
            cast(dict[str, Any], vars(resource)).pop(cached, None)


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
