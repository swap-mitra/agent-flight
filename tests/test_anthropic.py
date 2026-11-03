"""Contract tests: the real Anthropic SDK against a mock transport (FR-R1, FR-R2)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, ParsedMessage
from pydantic import BaseModel

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.intercept import anthropic as fr_anthropic
from flight_recorder.model import Run, Step
from flight_recorder.policy import pricing

MODEL = "claude-x-2026-01-01"
# 10 uncached + 2 written to cache (one 5-minute, one 1-hour) + 100 read from
# cache = a 112-token prompt.
USAGE: dict[str, Any] = {
    "input_tokens": 10,
    "cache_creation_input_tokens": 2,
    "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 1},
    "cache_read_input_tokens": 100,
    "output_tokens": 3,
    "service_tier": "standard",
}
MESSAGE: dict[str, Any] = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [{"type": "text", "text": "shipped"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": USAGE,
}


def events(
    error_mid_stream: bool = False, usage: dict[str, Any] = USAGE
) -> list[dict[str, Any]]:
    start: dict[str, Any] = {
        **MESSAGE,
        "content": [],
        "stop_reason": None,
        "usage": {**usage, "output_tokens": 1},
    }
    out: list[dict[str, Any]] = [
        {"type": "message_start", "message": start},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ship"},
        },
    ]
    if error_mid_stream:
        out.append(
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            }
        )
    return [
        *out,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ped"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]


class Order(BaseModel):
    status: str


ORDER_JSON = '{"status": "shipped"}'


class Server:
    """Answers like the Anthropic API and keeps every request body."""

    def __init__(
        self,
        fail: bool = False,
        error_mid_stream: bool = False,
        usage: dict[str, Any] = USAGE,
        replies: list[dict[str, Any]] | None = None,
    ) -> None:
        self.replies = replies or []  # message fields for each plain reply, in turn
        self.requests = 0
        self.bodies: list[dict[str, Any]] = []
        self.fail = fail
        self.error_mid_stream = error_mid_stream
        self.usage = usage

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests += 1
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail:
            return httpx2.Response(
                500,
                json={
                    "type": "error",
                    "error": {"type": "api_error", "message": "down"},
                },
            )
        if not body.get("stream"):
            structured = "output_config" in body or "output_format" in body
            text = ORDER_JSON if structured else "shipped"
            content = [{"type": "text", "text": text}]
            reply = self.replies.pop(0) if self.replies else {}
            return httpx2.Response(
                200,
                json={**MESSAGE, "content": content, "usage": self.usage, **reply},
            )
        sse = "".join(
            f"event: {e['type']}\ndata: {json.dumps(e)}\n\n"
            for e in events(self.error_mid_stream, self.usage)
        )
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse.encode()
        )


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


def client(server: Server) -> Anthropic:
    http = httpx2.Client(transport=httpx2.MockTransport(server))
    return fr.wrap(Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0))


def async_client(server: Server) -> AsyncAnthropic:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(server))
    return fr.wrap(
        AsyncAnthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    )


def model_of(step: Step) -> Any:
    assert step.model is not None
    return step.model


ASK: dict[str, Any] = {
    "model": "claude-x",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


def test_create_records_every_field_with_cached_tokens_in_the_prompt(
    sink: Sink,
) -> None:
    llm = client(Server())
    assert isinstance(llm, Anthropic)
    with fr.run("triage"):
        result = llm.messages.create(
            model="claude-x",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            system="be brief",
            extra_headers={"X-Trace": "t"},
        )

    assert isinstance(result, Message)
    (step,) = sink.steps
    call = model_of(step)
    assert (step.status, call.provider) == ("ok", "anthropic")
    assert (call.model_requested, call.model_used) == ("claude-x", MODEL)
    assert call.params == {"max_tokens": 100, "system": "be brief"}
    assert call.messages == [{"role": "user", "content": "hi"}]
    assert call.finish_reason == "end_turn"
    assert (call.usage.input_tokens, call.usage.output_tokens) == (112, 3)
    assert call.response["usage"]["cache_read_input_tokens"] == 100  # breakdown kept


def test_api_error_is_recorded_and_reraised_unchanged(sink: Sink) -> None:
    llm = client(Server(fail=True))
    with fr.run("triage"), pytest.raises(anthropic.InternalServerError):
        llm.messages.create(**ASK)

    (step,) = sink.steps
    assert step.status == "error"
    assert model_of(step).error.type == "anthropic.InternalServerError"
    assert model_of(step).cost_usd == 0.0  # rejected requests are not billed


def test_create_stream_passes_events_through_and_records_usage(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        stream = llm.messages.create(**ASK, stream=True)
        assert isinstance(stream, anthropic.Stream)
        types = [e.type for e in stream]

    assert types == [e["type"] for e in events()]
    call = model_of(sink.steps[0])
    assert (call.model_used, call.finish_reason) == (MODEL, "end_turn")
    assert (call.usage.input_tokens, call.usage.output_tokens) == (112, 3)
    assert len(call.response) == 7


def test_messages_stream_records_however_the_stream_is_read(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        with llm.messages.stream(**ASK) as stream:
            text = list(stream.text_stream)
        with llm.messages.stream(**ASK) as stream:
            final = stream.get_final_message()

    assert text == ["ship", "ped"]
    assert final.content[0].type == "text" and final.content[0].text == "shipped"
    for step in sink.steps:
        call = model_of(step)
        assert (
            call.finish_reason,
            call.usage.input_tokens,
            call.usage.output_tokens,
        ) == (
            "end_turn",
            112,
            3,
        )
    assert len(sink.steps) == 2


def test_stream_manager_never_entered_sends_nothing_and_records_nothing(
    sink: Sink,
) -> None:
    server = Server()
    with fr.run("triage"):
        client(server).messages.stream(**ASK)
    assert (server.requests, sink.steps) == (0, [])


def test_stream_left_early_is_abandoned_before_the_footer(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"), llm.messages.stream(**ASK) as stream:
        next(iter(stream))

    assert model_of(sink.steps[0]).finish_reason == "abandoned"
    assert model_of(sink.steps[0]).cost_usd is None
    assert sink.events == ["start", "step", "finish"]


def test_error_mid_stream_is_recorded_and_reraised(sink: Sink) -> None:
    llm = client(Server(error_mid_stream=True))
    with (
        fr.run("triage"),
        pytest.raises(anthropic.APIStatusError),
        llm.messages.stream(**ASK) as stream,
    ):
        stream.until_done()

    (step,) = sink.steps
    assert step.status == "error"
    assert "Overloaded" in model_of(step).error.message


@fr.tool(effect="read")
def lookup(order_id: str) -> str:
    return "shipped"


def test_tool_called_while_reading_a_stream_is_not_its_child(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"), llm.messages.stream(**ASK) as stream:
        for _ in stream.text_stream:
            lookup("o1")

    assert [s.parent_seq for s in sink.steps if s.kind == "tool"] == [None, None]


def test_async_client_create_create_stream_and_messages_stream(sink: Sink) -> None:
    llm = async_client(Server())

    async def agent() -> list[str]:
        async with fr.run("triage"):
            await llm.messages.create(**ASK)
            stream = await llm.messages.create(**ASK, stream=True)
            async for _ in stream:
                pass
            async with llm.messages.stream(**ASK) as managed:
                return [t async for t in managed.text_stream]

    assert asyncio.run(agent()) == ["ship", "ped"]
    calls = [model_of(s) for s in sink.steps]
    assert [c.finish_reason for c in calls] == ["end_turn"] * 3
    assert [c.usage.input_tokens for c in calls] == [112] * 3


def test_wrapping_twice_records_once(sink: Sink) -> None:
    llm = fr.wrap(client(Server()))
    with fr.run("triage"):
        llm.messages.create(**ASK)
        with llm.messages.stream(**ASK) as stream:
            stream.until_done()
    assert len(sink.steps) == 2


def test_renamed_stream_hook_fails_open(
    sink: Sink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # As if a new SDK version renamed the hook: the call works, unrecorded.
    monkeypatch.setitem(fr_anthropic._STREAM_HOOK, False, "_renamed")  # type: ignore[reportPrivateUsage]
    llm = client(Server())
    with fr.run("triage"), llm.messages.stream(**ASK) as stream:
        assert list(stream.text_stream) == ["ship", "ped"]

    assert sink.steps == []
    assert "stream manager has no _renamed" in caplog.text


@pytest.fixture
def priced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prices = tmp_path / "prices.toml"
    prices.write_text(
        '[anthropic]\n"claude-x" = { input = 1.0, output = 2.0, cache_read = 0.1,'
        " cache_write = 1.25, cache_write_1h = 2.0, web_search = 10.0 }\n"
    )
    monkeypatch.setattr(pricing, "table", pricing.load(prices))


def test_cost_prices_each_cache_rate_however_the_call_is_made(
    sink: Sink, priced: None
) -> None:
    llm = client(Server())
    with fr.run("triage"):
        llm.messages.create(**ASK)
        list(llm.messages.create(**ASK, stream=True))
        with llm.messages.stream(**ASK) as stream:
            stream.until_done()

    # Per million: 10 uncached x $1, 100 read x $0.10, one 5m write x $1.25,
    # one 1h write x $2, 3 out x $2.
    assert len(sink.steps) == 3
    for step in sink.steps:
        usage = model_of(step).usage
        assert (usage.cache_read_tokens, usage.cache_write_tokens) == (100, 2)
        assert model_of(step).cost_usd == pytest.approx(29.25 / 1_000_000)


def test_us_inference_geo_records_null_cost(
    sink: Sink, priced: None, caplog: pytest.LogCaptureFixture
) -> None:
    llm = client(Server(usage={**USAGE, "inference_geo": "us"}))
    with fr.run("triage"):
        llm.messages.create(**ASK)
        list(llm.messages.create(**ASK, stream=True))

    assert [model_of(s).cost_usd for s in sink.steps] == [None, None]
    assert caplog.text.count("used inference_geo 'us'") == 1


def test_run_on_an_unpriced_model_has_null_cost_and_one_warning(
    sink: Sink, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FR-B8 acceptance. claude-x is not in the shipped table.
    monkeypatch.setattr(pricing, "_warned", set[str]())
    llm = client(Server())
    with fr.run("triage") as run:
        llm.messages.create(**ASK)
        llm.messages.create(**ASK)

    assert [model_of(s).cost_usd for s in sink.steps] == [None, None]
    assert run.totals.cost_usd is None
    assert caplog.text.count("no price for anthropic model 'claude-x-2026-01-01'") == 1


def test_parse_records_the_schema_it_sends(sink: Sink) -> None:
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        result = llm.messages.parse(**ASK, output_format=Order)

    assert isinstance(result, ParsedMessage)
    assert result.parsed_output == Order(status="shipped")
    (step,) = sink.steps
    call = model_of(step)
    assert call.params["output_format"] == server.bodies[0]["output_config"]["format"]
    assert call.usage.input_tokens == 112


def test_raw_and_streaming_responses_are_recorded_when_parsed(sink: Sink) -> None:
    llm = client(Server())
    with fr.run("triage"):
        raw_api: Any = llm.messages.with_raw_response
        streaming_api: Any = llm.messages.with_streaming_response
        raw = raw_api.create(**ASK)
        assert raw.parse() is raw.parse()
        with streaming_api.create(**ASK, stream=True) as response:
            types = [e.type for e in response.parse()]
        raw_api.create(**ASK)  # never parsed

    assert types == [e["type"] for e in events()]
    plain, streamed, unparsed = (model_of(s) for s in sink.steps)
    assert plain.usage.input_tokens == streamed.usage.input_tokens == 112
    assert (streamed.finish_reason, len(streamed.response)) == ("end_turn", 7)
    assert (unparsed.finish_reason, unparsed.response) == ("abandoned", None)


def test_async_parse_and_streaming_response(sink: Sink) -> None:
    llm = async_client(Server())

    async def agent() -> Order | None:
        async with fr.run("triage"):
            result = await llm.messages.parse(**ASK, output_format=Order)
            streaming_api: Any = llm.messages.with_streaming_response
            async with streaming_api.create(**ASK) as response:
                await response.parse()
            return result.parsed_output

    assert asyncio.run(agent()) == Order(status="shipped")
    assert [model_of(s).usage.input_tokens for s in sink.steps] == [112, 112]


def test_web_searches_add_their_fee_to_the_token_cost(sink: Sink, priced: None) -> None:
    tools = {"web_search_requests": 2, "web_fetch_requests": 1}  # fetch is free
    llm = client(Server(usage={**USAGE, "server_tool_use": tools}))
    with fr.run("triage"):
        llm.messages.create(**ASK)
        list(llm.messages.create(**ASK, stream=True))

    fee = 2 * 10.0 / 1000
    for step in sink.steps:
        assert model_of(step).cost_usd == pytest.approx(29.25 / 1_000_000 + fee)
    assert len(sink.steps) == 2


def test_beta_create_parse_and_stream_are_recorded(sink: Sink) -> None:
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        llm.beta.messages.create(**ASK)
        llm.beta.messages.parse(**ASK, output_format=Order)
        with llm.beta.messages.stream(**ASK) as stream:
            stream.until_done()

    created, parsed, streamed = (model_of(s) for s in sink.steps)
    sent = server.bodies[1]
    sent_format = sent["output_config"]["format"] if "output_config" in sent else None
    assert parsed.params["output_format"] == (sent_format or sent["output_format"])
    for call in (created, parsed, streamed):
        assert (call.finish_reason, call.usage.input_tokens) == ("end_turn", 112)


@anthropic.beta_tool
@fr.tool(effect="read")
def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return "shipped"


def test_tool_runner_records_each_model_turn_and_tool_call(sink: Sink) -> None:
    use = {"type": "tool_use", "id": "t1", "name": "lookup_order"}
    use = {**use, "input": {"order_id": "o1"}}
    server = Server(replies=[{"content": [use], "stop_reason": "tool_use"}])
    llm = client(server)
    with fr.run("triage"):
        beta: Any = llm.beta.messages  # tool_runner's overloads need literal kwargs
        runner = beta.tool_runner(**ASK, tools=[lookup_order])
        final = runner.until_done()

    assert final.stop_reason == "end_turn"
    assert [s.kind for s in sink.steps] == ["model", "tool", "model"]
    assert [model_of(s).finish_reason for s in sink.steps if s.model] == [
        "tool_use",
        "end_turn",
    ]


@pytest.fixture
def two_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prices = tmp_path / "prices.toml"
    prices.write_text(
        '[anthropic]\n"claude-x" = { input = 1.0, output = 2.0 }\n'
        '"claude-advisor" = { input = 10.0, output = 20.0 }\n'
    )
    monkeypatch.setattr(pricing, "table", pricing.load(prices))


def iteration(kind: str, tokens_in: int, tokens_out: int, **extra: Any) -> Any:
    return {
        "type": kind,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        **extra,
    }


def test_iterations_are_priced_at_each_models_rates(
    sink: Sink, two_models: None
) -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 3,
        "iterations": [
            iteration("message", 10, 3, model=MODEL),
            iteration("advisor_message", 100, 5, model="claude-advisor"),
            # Not part of the top-level usage, billed all the same.
            iteration("compaction", 20, 2),
        ],
    }
    llm = client(Server(usage=usage))
    with fr.run("triage"):
        llm.beta.messages.create(**ASK)
        list(llm.beta.messages.create(**ASK, stream=True))

    # Per million: 10 x $1 + 3 x $2, then 100 x $10 + 5 x $20, then 20 x $1 + 2 x $2.
    for step in sink.steps:
        assert model_of(step).usage.input_tokens == 10  # as the API reports it
        assert model_of(step).cost_usd == pytest.approx(1140 / 1_000_000)
    assert len(sink.steps) == 2


def test_refusal_fallback_iteration_records_null_cost(
    sink: Sink, two_models: None, caplog: pytest.LogCaptureFixture
) -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 3,
        "iterations": [
            iteration("message", 10, 0, model=MODEL),
            iteration("fallback_message", 10, 3, model="claude-advisor"),
        ],
    }
    llm = client(Server(usage=usage))
    with fr.run("triage"):
        llm.beta.messages.create(**ASK)

    assert model_of(sink.steps[0]).cost_usd is None
    assert "used a fallback_message iteration" in caplog.text


def test_async_beta_create_and_stream(sink: Sink) -> None:
    llm = async_client(Server())

    async def agent() -> None:
        async with fr.run("triage"):
            await llm.beta.messages.create(**ASK)
            async with llm.beta.messages.stream(**ASK) as stream:
                await stream.until_done()

    asyncio.run(agent())
    assert [model_of(s).usage.input_tokens for s in sink.steps] == [112, 112]
