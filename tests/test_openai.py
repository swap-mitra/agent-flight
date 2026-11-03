"""Contract tests: the real OpenAI SDK against a mock transport (FR-R1, FR-R2)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.model import Run, Step
from flight_recorder.policy import pricing

MODEL = "gpt-x-2026-01-01"
# 12 prompt tokens, 4 of them read from the prompt cache.
USAGE_CHAT = {
    "prompt_tokens": 12,
    "completion_tokens": 3,
    "total_tokens": 15,
    "prompt_tokens_details": {"cached_tokens": 4},
}
COMPLETION: dict[str, Any] = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "shipped"},
            "finish_reason": "stop",
        }
    ],
    "usage": USAGE_CHAT,
}
RESPONSE: dict[str, Any] = {
    "id": "r1",
    "object": "response",
    "created_at": 0,
    "model": MODEL,
    "status": "completed",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 4},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


class Order(BaseModel):
    status: str


ORDER_JSON = '{"status": "shipped"}'
STRUCTURED: dict[str, Any] = {
    **COMPLETION,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": ORDER_JSON},
            "finish_reason": "stop",
        }
    ],
}


def chunk(content: str | None, finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": finish}
        ],
    }


def sse(events: list[dict[str, Any]], named: bool = False) -> bytes:
    lines = [
        (f"event: {e['type']}\n" if named else "") + f"data: {json.dumps(e)}\n\n"
        for e in events
    ]
    return ("".join(lines) + ("" if named else "data: [DONE]\n\n")).encode()


class Server:
    """Answers like the OpenAI API and keeps every request body."""

    def __init__(
        self,
        fail: bool = False,
        error_mid_stream: bool = False,
        tier: str = "default",
        structured: bool = True,
    ) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.fail = fail
        self.error_mid_stream = error_mid_stream
        self.tier = tier
        self.structured = structured

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail:
            return httpx2.Response(500, json={"error": {"message": "overloaded"}})
        if not body.get("stream"):
            payload = (
                (
                    STRUCTURED
                    if "response_format" in body and self.structured
                    else COMPLETION
                )
                if request.url.path.endswith("/chat/completions")
                else RESPONSE
            )
            return httpx2.Response(200, json={**payload, "service_tier": self.tier})
        headers = {"content-type": "text/event-stream"}
        if request.url.path.endswith("/responses"):
            events = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**RESPONSE, "status": "in_progress", "usage": None},
                },
                {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": RESPONSE,
                },
            ]
            return httpx2.Response(
                200, headers=headers, content=sse(events, named=True)
            )
        # Structured output streams JSON, in two pieces like the plain text.
        first, rest = (
            ('{"status": ', '"shipped"}')
            if "response_format" in body
            else ("ship", "ped")
        )
        events = [chunk(first)]
        if self.error_mid_stream:
            events.append({"error": {"message": "stream broke"}})
        events.append(chunk(rest, "stop"))
        if body.get("stream_options", {}).get("include_usage"):
            events.append({**chunk(None), "choices": [], "usage": USAGE_CHAT})
        events = [{**e, "service_tier": self.tier} for e in events]
        return httpx2.Response(200, headers=headers, content=sse(events))


class Sink:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.steps: list[Step] = []

    def run_started(self, run: Run) -> None:
        self.events.append("start")

    def step_closed(self, run_id: str, step: Step) -> None:
        self.events.append("step")
        self.steps.append(step)

    def run_finished(self, run: Run) -> None:
        self.events.append("finish")


@pytest.fixture
def sink() -> Sink:
    sink = Sink()
    recorder.sinks = [sink]
    return sink


def client(server: Server) -> OpenAI:
    http = httpx2.Client(transport=httpx2.MockTransport(server))
    return fr.wrap(OpenAI(api_key="sk-test", http_client=http, max_retries=0))


def async_client(server: Server) -> AsyncOpenAI:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(server))
    return fr.wrap(AsyncOpenAI(api_key="sk-test", http_client=http, max_retries=0))


def model_of(step: Step) -> Any:
    assert step.model is not None
    return step.model


def test_chat_create_records_every_field_and_returns_the_sdk_object(sink: Sink) -> None:
    llm = client(Server())
    assert isinstance(llm, OpenAI)
    with fr.run("triage"):
        result = llm.chat.completions.create(
            model="gpt-x",
            messages=[{"role": "user", "content": "where is my order?"}],
            temperature=0,
            extra_headers={"X-Trace": "t1"},
        )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "shipped"
    (step,) = sink.steps
    call = model_of(step)
    assert (step.kind, step.status, call.provider) == ("model", "ok", "openai")
    assert (call.model_requested, call.model_used) == ("gpt-x", MODEL)
    assert call.params == {"temperature": 0}
    assert call.messages == [{"role": "user", "content": "where is my order?"}]
    assert call.finish_reason == "stop"
    assert (call.usage.input_tokens, call.usage.output_tokens) == (12, 3)
    assert call.response["id"] == "c1"


def test_outside_a_run_calls_are_not_recorded(sink: Sink) -> None:
    result = client(Server()).chat.completions.create(model="gpt-x", messages=[])
    assert result.id == "c1"
    assert sink.steps == []


def test_api_error_is_recorded_and_reraised_unchanged(sink: Sink) -> None:
    llm = client(Server(fail=True))
    with fr.run("triage"), pytest.raises(openai.InternalServerError):
        llm.chat.completions.create(model="gpt-x", messages=[])

    (step,) = sink.steps
    call = model_of(step)
    assert step.status == "error"
    assert call.error.type == "openai.InternalServerError"
    assert (call.model_used, call.response, call.usage) == ("gpt-x", None, None)
    assert call.cost_usd == 0.0  # rejected requests are not billed


def test_chat_stream_yields_the_callers_chunks_and_records_usage(sink: Sink) -> None:
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        stream = llm.chat.completions.create(model="gpt-x", messages=[], stream=True)
        assert isinstance(stream, openai.Stream)
        seen = [c.choices[0].delta.content for c in stream]

    assert seen == ["ship", "ped"]  # the usage-only chunk is not shown
    assert server.bodies[0]["stream_options"] == {"include_usage": True}
    (step,) = sink.steps
    call = model_of(step)
    assert call.params == {"stream": True}  # recorded as the caller sent it
    assert (call.finish_reason, call.usage.input_tokens, call.model_used) == (
        "stop",
        12,
        MODEL,
    )
    assert len(call.response) == 3


def test_caller_requested_usage_chunk_is_passed_through(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        stream = llm.chat.completions.create(
            model="gpt-x",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = list(stream)

    assert len(chunks) == 3 and chunks[-1].usage is not None
    assert model_of(sink.steps[0]).usage.output_tokens == 3


def test_stream_closed_early_or_left_open_is_abandoned_before_the_footer(
    sink: Sink,
) -> None:
    llm = client(Server())
    with fr.run("triage"):
        with llm.chat.completions.create(
            model="gpt-x", messages=[], stream=True
        ) as stream:
            next(stream)
        llm.chat.completions.create(
            model="gpt-x", messages=[], stream=True
        )  # never read

    assert [model_of(s).finish_reason for s in sink.steps] == ["abandoned", "abandoned"]
    assert [model_of(s).cost_usd for s in sink.steps] == [None, None]
    assert [s.status for s in sink.steps] == ["ok", "ok"]
    assert sink.events == ["start", "step", "step", "finish"]


def test_error_mid_stream_is_recorded_and_reraised(sink: Sink) -> None:
    llm = client(Server(error_mid_stream=True))
    seen: list[str | None] = []
    with fr.run("triage"), pytest.raises(openai.APIError):
        for c in llm.chat.completions.create(model="gpt-x", messages=[], stream=True):
            seen.append(c.choices[0].delta.content)

    assert seen == ["ship"]
    (step,) = sink.steps
    assert step.status == "error"
    assert model_of(step).error.message == "stream broke"


@fr.tool(effect="read")
def lookup(order_id: str) -> str:
    return "shipped"


def test_tool_called_while_reading_a_stream_is_not_its_child(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        for _ in llm.chat.completions.create(model="gpt-x", messages=[], stream=True):
            lookup("o1")

    tools = [s for s in sink.steps if s.kind == "tool"]
    assert [t.parent_seq for t in tools] == [None, None]


def test_responses_create_and_stream_are_recorded(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        llm.responses.create(
            model="gpt-x", input="where is my order?", instructions="be brief"
        )
        events = list(llm.responses.create(model="gpt-x", input="hi", stream=True))

    assert [e.type for e in events] == ["response.created", "response.completed"]
    plain, streamed = (model_of(s) for s in sink.steps)
    assert plain.messages == "where is my order?"
    assert plain.params == {"instructions": "be brief"}
    for call in (plain, streamed):
        assert (call.model_used, call.finish_reason) == (MODEL, "completed")
        assert (call.usage.input_tokens, call.usage.output_tokens) == (12, 3)


def test_async_client_create_and_stream(sink: Sink) -> None:
    llm = async_client(Server())

    async def agent() -> list[str | None]:
        async with fr.run("triage"):
            await llm.chat.completions.create(model="gpt-x", messages=[])
            stream = await llm.chat.completions.create(
                model="gpt-x", messages=[], stream=True
            )
            return [c.choices[0].delta.content async for c in stream]

    assert asyncio.run(agent()) == ["ship", "ped"]
    plain, streamed = (model_of(s) for s in sink.steps)
    assert plain.finish_reason == streamed.finish_reason == "stop"
    assert streamed.usage.input_tokens == 12


def test_wrapping_twice_records_once(sink: Sink) -> None:
    llm = fr.wrap(client(Server()))
    with fr.run("triage"):
        llm.chat.completions.create(model="gpt-x", messages=[])
    assert len(sink.steps) == 1


def test_unsupported_client_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="does not support object"):
        fr.wrap(object())


@pytest.fixture
def priced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prices = tmp_path / "prices.toml"
    prices.write_text(
        '[openai]\n"gpt-x" = { input = 1.0, cache_read = 0.5, output = 2.0 }\n'
    )
    monkeypatch.setattr(pricing, "table", pricing.load(prices))


def test_cost_prices_cached_tokens_on_every_api(sink: Sink, priced: None) -> None:
    llm = client(Server())
    with fr.run("triage") as run:
        llm.chat.completions.create(model="gpt-x", messages=[])
        list(llm.chat.completions.create(model="gpt-x", messages=[], stream=True))
        llm.responses.create(model="gpt-x", input="hi")
        list(llm.responses.create(model="gpt-x", input="hi", stream=True))

    # The dated model used falls back to gpt-x: 8 uncached x $1, 4 cached x
    # $0.50, 3 out x $2 per million.
    for step in sink.steps:
        assert model_of(step).usage.cache_read_tokens == 4
        assert model_of(step).cost_usd == pytest.approx(16 / 1_000_000)
    assert run.totals.cost_usd == pytest.approx(4 * 16 / 1_000_000)


def test_unpriced_service_tier_records_null_cost(
    sink: Sink, priced: None, caplog: pytest.LogCaptureFixture
) -> None:
    llm = client(Server(tier="flex"))
    with fr.run("triage"):
        llm.chat.completions.create(model="gpt-x", messages=[])
        list(llm.chat.completions.create(model="gpt-x", messages=[], stream=True))

    assert [model_of(s).cost_usd for s in sink.steps] == [None, None]
    assert caplog.text.count("used service_tier 'flex'") == 1


def test_chat_parse_records_the_schema_it_sends(sink: Sink) -> None:
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        result = llm.chat.completions.parse(
            model="gpt-x", messages=[], response_format=Order
        )

    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed == Order(status="shipped")
    (step,) = sink.steps
    call = model_of(step)
    assert call.params["response_format"] == server.bodies[0]["response_format"]
    assert call.params["response_format"]["json_schema"]["name"] == "Order"
    assert call.response["choices"][0]["message"]["parsed"] == {"status": "shipped"}
    assert call.usage.input_tokens == 12


def test_chat_stream_helper_is_recorded_like_create(sink: Sink) -> None:
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        llm.chat.completions.parse(model="gpt-x", messages=[], response_format=Order)
        with llm.chat.completions.stream(
            model="gpt-x", messages=[], response_format=Order
        ) as stream:
            text = [e.delta for e in stream if e.type == "content.delta"]

    assert "".join(text) == ORDER_JSON
    parsed, streamed = (model_of(s) for s in sink.steps)
    # The SDK's own "not given" markers are not recorded as parameters.
    assert set(streamed.params) == {"stream", "response_format"}
    assert streamed.params["response_format"] == parsed.params["response_format"]
    assert server.bodies[1]["stream_options"] == {"include_usage": True}
    assert (streamed.finish_reason, streamed.usage.output_tokens) == ("stop", 3)


def test_responses_parse_and_stream_helper_are_recorded(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        llm.responses.parse(model="gpt-x", input="hi", text_format=Order)
        with llm.responses.stream(model="gpt-x", input="hi") as stream:
            stream.until_done()

    parsed, streamed = (model_of(s) for s in sink.steps)
    assert parsed.params["text_format"]["name"] == "Order"
    for call in (parsed, streamed):
        assert (call.finish_reason, call.usage.input_tokens) == ("completed", 12)


def test_raw_response_is_recorded_when_parsed(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        raw = llm.chat.completions.with_raw_response.create(model="gpt-x", messages=[])
        assert sink.steps == []  # not parsed yet
        first, second = raw.parse(), raw.parse()

    assert first is second and first.id == "c1"
    (step,) = sink.steps
    assert model_of(step).usage.input_tokens == 12
    assert model_of(step).response["id"] == "c1"


def test_raw_response_never_parsed_is_abandoned_before_the_footer(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        raw = llm.chat.completions.with_raw_response.create(model="gpt-x", messages=[])
        assert raw.http_response.status_code == 200

    (step,) = sink.steps
    call = model_of(step)
    assert (call.finish_reason, call.response, call.cost_usd) == (
        "abandoned",
        None,
        None,
    )
    assert sink.events == ["start", "step", "finish"]


def test_streaming_response_stays_lazy_and_records_the_parsed_stream(
    sink: Sink,
) -> None:
    llm = client(Server())
    with (
        fr.run("triage"),
        llm.chat.completions.with_streaming_response.create(
            model="gpt-x", messages=[], stream=True
        ) as response,
    ):
        seen = [c.choices[0].delta.content for c in response.parse()]

    assert seen == ["ship", "ped"]
    (step,) = sink.steps
    assert (model_of(step).finish_reason, model_of(step).usage.input_tokens) == (
        "stop",
        12,
    )


def test_raw_wrapper_built_before_wrap_is_recorded(sink: Sink) -> None:
    http = httpx2.Client(transport=httpx2.MockTransport(Server()))
    llm = OpenAI(api_key="sk-test", http_client=http, max_retries=0)
    raw_create = llm.chat.completions.with_raw_response.create  # cached by the SDK
    fr.wrap(llm)
    with fr.run("triage"):
        llm.chat.completions.with_raw_response.create(
            model="gpt-x", messages=[]
        ).parse()

    assert raw_create is not llm.chat.completions.with_raw_response.create
    assert len(sink.steps) == 1


def test_failure_after_a_response_is_not_priced_as_rejected(sink: Sink) -> None:
    # The reply isn't valid JSON for Order: the SDK raises after the request
    # was served and billed, so the cost is unknown, not 0.
    llm = client(Server(structured=False))
    with fr.run("triage"), pytest.raises(Exception) as caught:
        llm.chat.completions.parse(model="gpt-x", messages=[], response_format=Order)

    assert not isinstance(caught.value, openai.APIStatusError)
    (step,) = sink.steps
    assert (step.status, model_of(step).cost_usd) == ("error", None)


def test_async_parse_and_raw_response(sink: Sink) -> None:
    llm = async_client(Server())

    async def agent() -> Order | None:
        async with fr.run("triage"):
            result = await llm.chat.completions.parse(
                model="gpt-x", messages=[], response_format=Order
            )
            raw = await llm.chat.completions.with_raw_response.create(
                model="gpt-x", messages=[]
            )
            raw.parse()
            async with llm.chat.completions.with_streaming_response.create(
                model="gpt-x", messages=[]
            ) as response:
                await response.parse()
            return result.choices[0].message.parsed

    assert asyncio.run(agent()) == Order(status="shipped")
    assert [model_of(s).usage.input_tokens for s in sink.steps] == [12, 12, 12]
