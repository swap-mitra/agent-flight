"""Model content on OTel spans in the GenAI conventions' message schema (FR-R7).

Real OpenAI and Anthropic SDK calls, answered by mock transports with tool
calls, media and reasoning, recorded and exported with content capture on.
"""

import json
from collections.abc import Callable, Iterator
from typing import Any, cast

import anthropic
import httpx2
import openai
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.export import otel
from flight_recorder.export.otel import OtelSink

CONTENT = (
    "gen_ai.system_instructions",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
)
Spans = Callable[[], list[dict[str, Any]]]


@pytest.fixture
def chats(monkeypatch: pytest.MonkeyPatch) -> Iterator[Spans]:
    """The content attributes of each exported chat span, JSON-decoded."""
    monkeypatch.setenv(otel.CONTENT_ENV, "SPAN_ONLY")
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    sink = OtelSink(provider)
    recorder.sinks = [sink]

    def read() -> list[dict[str, Any]]:
        assert sink.flush(timeout=10)
        chat = [s for s in spans.get_finished_spans() if s.name.startswith("chat")]
        chat.sort(key=lambda s: s.start_time or 0)
        return [
            {
                k: json.loads(str(v))
                for k, v in (s.attributes or {}).items()
                if k in CONTENT
            }
            for s in chat
        ]

    yield read
    sink.close()


def replying(
    body: Any = None, events: list[dict[str, Any]] | None = None, status: int = 200
) -> httpx2.Client:
    """An HTTP client whose every request gets this JSON body or these SSE events."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        if events is None:
            return httpx2.Response(status, json=body)
        named = "chat.completion.chunk" not in json.dumps(events)
        sse = "".join(
            (f"event: {e['type']}\n" if named else "") + f"data: {json.dumps(e)}\n\n"
            for e in events
        ) + ("" if named else "data: [DONE]\n\n")
        headers = {"content-type": "text/event-stream"}
        return httpx2.Response(200, headers=headers, content=sse.encode())

    return httpx2.Client(transport=httpx2.MockTransport(handle))


# Any: the SDKs' overloads cannot match arguments unpacked from a dict.
def openai_client(http: httpx2.Client) -> Any:
    return fr.wrap(openai.OpenAI(api_key="sk-test", http_client=http, max_retries=0))


def anthropic_client(http: httpx2.Client) -> Any:
    client = anthropic.Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    return fr.wrap(client)


def text(content: str) -> dict[str, Any]:
    return {"type": "text", "content": content}


def tool_call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    return {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}


def tool_response(call_id: str, response: Any) -> dict[str, Any]:
    return {"type": "tool_call_response", "id": call_id, "response": response}


PNG = "iVBORw0KGgo="


class Has(dict[str, Any]):
    """Equal to any dict holding these items, whatever else it holds."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict):
            return False
        found = cast(dict[str, Any], other)
        return all(found.get(k) == v for k, v in self.items())

    __hash__ = None  # type: ignore[assignment]


# OpenAI Chat Completions

CHAT_INPUT: list[dict[str, Any]] = [
    {"role": "developer", "content": "You are a support agent."},
    {
        "role": "user",
        "name": "ada",
        "content": [
            {"type": "text", "text": "Where is my order? Photo:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
            {"type": "image_url", "image_url": {"url": "https://x.test/box.png"}},
        ],
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"order_id": "o1"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "shipped"},
]
CHAT_EXPECTED_INPUT = [
    {"role": "system", "parts": [text("You are a support agent.")]},
    {
        "role": "user",
        "parts": [
            text("Where is my order? Photo:"),
            {
                "type": "blob",
                "modality": "image",
                "mime_type": "image/png",
                "content": PNG,
            },
            {
                "type": "uri",
                "modality": "image",
                "mime_type": None,
                "uri": "https://x.test/box.png",
            },
        ],
        "name": "ada",
    },
    {"role": "assistant", "parts": [tool_call("call_1", "lookup", {"order_id": "o1"})]},
    {"role": "tool", "parts": [tool_response("call_1", "shipped")]},
]
CHAT_EXPECTED_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            text("Refunding now."),
            tool_call("call_2", "refund", {"order_id": "o1", "amount": 5}),
        ],
    }
]


def completion(**fields: Any) -> dict[str, Any]:
    return {"id": "c1", "created": 0, "model": "gpt-x", **fields}


def test_openai_chat_and_its_stream_in_the_message_schema(chats: Spans) -> None:
    reply = completion(
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Refunding now.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "refund",
                                "arguments": '{"order_id": "o1", "amount": 5}',
                            },
                        }
                    ],
                },
            }
        ],
    )

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        choice = {"index": 0, "delta": delta, "finish_reason": finish}
        return completion(object="chat.completion.chunk", choices=[choice])

    def call_delta(arguments: str, **head: Any) -> dict[str, Any]:
        function = {"arguments": arguments, **({"name": head["name"]} if head else {})}
        call = {"index": 0, "function": function}
        if head:
            call |= {"id": head["id"], "type": "function"}
        return {"tool_calls": [call]}

    # Text and the tool call's arguments arrive in pieces.
    chunks = [
        chunk({"role": "assistant", "content": "Refunding "}),
        chunk({"content": "now."}),
        chunk(call_delta('{"order_id": ', id="call_2", name="refund")),
        chunk(call_delta('"o1", "amount": 5}')),
        chunk({}, "tool_calls"),
    ]
    with fr.run("triage"):
        openai_client(replying(reply)).chat.completions.create(
            model="gpt-x", messages=CHAT_INPUT
        )
        list(
            openai_client(replying(events=chunks)).chat.completions.create(
                model="gpt-x", messages=CHAT_INPUT, stream=True
            )
        )

    expected = {
        "gen_ai.input.messages": CHAT_EXPECTED_INPUT,
        "gen_ai.output.messages": CHAT_EXPECTED_OUTPUT,
    }
    assert chats() == [expected, expected]


# OpenAI Responses

RESPONSE_OUTPUT: list[dict[str, Any]] = [
    {
        "type": "reasoning",
        "id": "rs1",
        "summary": [{"type": "summary_text", "text": "Check the order first."}],
    },
    {
        "type": "web_search_call",
        "id": "ws1",
        "status": "completed",
        "action": {"type": "search", "query": "carrier delays"},
    },
    {
        "type": "function_call",
        "id": "fc1",
        "call_id": "call_3",
        "name": "lookup",
        "arguments": '{"order_id": "o1"}',
        "status": "completed",
    },
    {
        "type": "message",
        "id": "m1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "Looking it up.", "annotations": []}
        ],
    },
]
RESPONSES_EXPECTED_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            {"type": "reasoning", "content": "Check the order first."},
            # No part type of its own: kept as the SDK recorded it.
            Has(type="web_search_call", id="ws1", status="completed"),
            tool_call("call_3", "lookup", {"order_id": "o1"}),
            text("Looking it up."),
        ],
    }
]


def response(output: list[dict[str, Any]], status: str = "completed") -> dict[str, Any]:
    return {
        "id": "r1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-x",
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def test_openai_responses_and_its_streams_in_the_message_schema(chats: Spans) -> None:
    created = {"type": "response.created", "sequence_number": 0}
    events = [
        {**created, "response": response([], "in_progress")},
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": response(RESPONSE_OUTPUT),
        },
    ]
    # A stream that ends early: only the items that finished are output.
    cut_short = [
        {**created, "response": response([], "in_progress")},
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": RESPONSE_OUTPUT[0],
        },
    ]
    history: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "input_text", "text": "Where is o1?"}]},
        RESPONSE_OUTPUT[2],
        {"type": "function_call_output", "call_id": "call_3", "output": "shipped"},
    ]
    with fr.run("triage"):
        ask: dict[str, Any] = {"model": "gpt-x", "instructions": "Be brief."}
        openai_client(replying(response(RESPONSE_OUTPUT))).responses.create(
            input="Where is o1?", **ask
        )
        stream = openai_client(replying(events=events)).responses.create(
            input=history, stream=True, **ask
        )
        list(stream)
        list(
            openai_client(replying(events=cut_short)).responses.create(
                input="Where is o1?", stream=True, **ask
            )
        )

    first, streamed, short = chats()
    assert first == {
        "gen_ai.system_instructions": [text("Be brief.")],
        "gen_ai.input.messages": [{"role": "user", "parts": [text("Where is o1?")]}],
        "gen_ai.output.messages": RESPONSES_EXPECTED_OUTPUT,
    }
    assert streamed["gen_ai.input.messages"] == [
        {"role": "user", "parts": [text("Where is o1?")]},
        {
            "role": "assistant",
            "parts": [tool_call("call_3", "lookup", {"order_id": "o1"})],
        },
        {"role": "tool", "parts": [tool_response("call_3", "shipped")]},
    ]
    assert streamed["gen_ai.output.messages"] == RESPONSES_EXPECTED_OUTPUT
    assert short["gen_ai.output.messages"] == [
        {
            "role": "assistant",
            "parts": [{"type": "reasoning", "content": "Check the order first."}],
        }
    ]


# Anthropic

ANTHROPIC_INPUT: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Is this box damaged?"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": PNG},
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tu1",
                "name": "lookup",
                "input": {"order_id": "o1"},
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "shipped"}
        ],
    },
]
ANTHROPIC_EXPECTED_INPUT = [
    {
        "role": "user",
        "parts": [
            text("Is this box damaged?"),
            {
                "type": "blob",
                "modality": "image",
                "mime_type": "image/png",
                "content": PNG,
            },
        ],
    },
    {"role": "assistant", "parts": [tool_call("tu1", "lookup", {"order_id": "o1"})]},
    # A user message holding only tool results is the tool's turn.
    {"role": "tool", "parts": [tool_response("tu1", "shipped")]},
]
ANTHROPIC_CONTENT: list[dict[str, Any]] = [
    {"type": "thinking", "thinking": "The photo shows a dent.", "signature": "sig"},
    {"type": "text", "text": "Yes, refunding."},
    {"type": "tool_use", "id": "tu2", "name": "refund", "input": {"order_id": "o1"}},
]
ANTHROPIC_EXPECTED_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            {"type": "reasoning", "content": "The photo shows a dent."},
            text("Yes, refunding."),
            tool_call("tu2", "refund", {"order_id": "o1"}),
        ],
    }
]
USAGE = {"input_tokens": 10, "output_tokens": 5}


def message(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "msg1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": content,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": USAGE,
    }


def anthropic_events() -> list[dict[str, Any]]:
    def start(index: int, block: dict[str, Any]) -> dict[str, Any]:
        return {"type": "content_block_start", "index": index, "content_block": block}

    def delta(index: int, **fields: Any) -> dict[str, Any]:
        return {"type": "content_block_delta", "index": index, "delta": fields}

    def stop(index: int) -> dict[str, Any]:
        return {"type": "content_block_stop", "index": index}

    return [
        {"type": "message_start", "message": message([])},
        start(0, {"type": "thinking", "thinking": "", "signature": ""}),
        delta(0, type="thinking_delta", thinking="The photo "),
        delta(0, type="thinking_delta", thinking="shows a dent."),
        delta(0, type="signature_delta", signature="sig"),
        stop(0),
        start(1, {"type": "text", "text": ""}),
        delta(1, type="text_delta", text="Yes, "),
        delta(1, type="text_delta", text="refunding."),
        stop(1),
        start(2, {"type": "tool_use", "id": "tu2", "name": "refund", "input": {}}),
        delta(2, type="input_json_delta", partial_json='{"order_'),
        delta(2, type="input_json_delta", partial_json='id": "o1"}'),
        stop(2),
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]


def test_anthropic_and_its_streams_in_the_message_schema(chats: Spans) -> None:
    system = [{"type": "text", "text": "You are a support agent."}]
    ask: dict[str, Any] = {
        "model": "claude-x",
        "max_tokens": 100,
        "system": system,
        "messages": ANTHROPIC_INPUT,
    }
    with fr.run("triage"):
        anthropic_client(replying(message(ANTHROPIC_CONTENT))).messages.create(**ask)
        list(
            anthropic_client(replying(events=anthropic_events())).messages.create(
                **ask, stream=True
            )
        )
        with anthropic_client(replying(events=anthropic_events())).messages.stream(
            **ask
        ) as stream:
            stream.until_done()

    expected = {
        "gen_ai.system_instructions": [text("You are a support agent.")],
        "gen_ai.input.messages": ANTHROPIC_EXPECTED_INPUT,
        "gen_ai.output.messages": ANTHROPIC_EXPECTED_OUTPUT,
    }
    assert chats() == [expected, expected, expected]


def test_a_failed_call_exports_its_input_and_no_output(chats: Spans) -> None:
    failing = replying({"error": {"message": "overloaded"}}, status=500)
    with fr.run("triage"), pytest.raises(openai.InternalServerError):
        openai_client(failing).chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}]
        )

    assert chats() == [
        {"gen_ai.input.messages": [{"role": "user", "parts": [text("hi")]}]}
    ]


def test_content_of_unknown_shape_is_kept_and_never_breaks_the_export(
    chats: Spans,
) -> None:
    odd: Any = [
        "not a message",
        {"role": "user", "content": [{"type": "hologram", "beam": 3}, 7]},
        {"role": "user", "content": 42},
    ]
    with fr.run("triage"):
        openai_client(
            replying(completion(object="chat.completion", choices=[]))
        ).chat.completions.create(model="gpt-x", messages=odd)
        anthropic_client(replying(message([]))).messages.create(
            model="claude-x", max_tokens=1, messages=odd
        )

    openai_span, anthropic_span = chats()
    hologram = {"type": "hologram", "beam": 3}
    assert openai_span == {
        "gen_ai.input.messages": [
            {"role": "", "parts": []},
            {"role": "user", "parts": [hologram, {"type": "unknown", "value": 7}]},
            {"role": "user", "parts": []},
        ]
    }
    assert anthropic_span["gen_ai.input.messages"][1] == {
        "role": "user",
        "parts": [hologram, {"type": "unknown", "value": 7}],
    }
