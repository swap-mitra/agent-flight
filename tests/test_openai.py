"""Contract tests: the real OpenAI SDK against a mock transport (FR-R1, FR-R2)."""

import asyncio
import json
from typing import Any

import httpx2
import openai
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.model import Run, Step

MODEL = "gpt-x-2026-01-01"
USAGE_CHAT = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
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
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
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

    def __init__(self, fail: bool = False, error_mid_stream: bool = False) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.fail = fail
        self.error_mid_stream = error_mid_stream

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail:
            return httpx2.Response(500, json={"error": {"message": "overloaded"}})
        if not body.get("stream"):
            payload = (
                COMPLETION
                if request.url.path.endswith("/chat/completions")
                else RESPONSE
            )
            return httpx2.Response(200, json=payload)
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
        events = [chunk("ship")]
        if self.error_mid_stream:
            events.append({"error": {"message": "stream broke"}})
        events.append(chunk("ped", "stop"))
        if body.get("stream_options", {}).get("include_usage"):
            events.append({**chunk(None), "choices": [], "usage": USAGE_CHAT})
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
