"""LangChain adapter (FR-R1, SDD Framework adapters).

Real LangChain chat models over the real SDKs, served by the same mock
transports as the SDK contract tests.
"""

import asyncio
import logging
from typing import Any, TypedDict

import anthropic
import httpx2
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from test_anthropic import Server as AnthropicServer
from test_openai import Server as OpenAIServer
from test_openai import Sink

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.model import Step


@pytest.fixture
def sink() -> Sink:
    sink = Sink()
    recorder.sinks = [sink]
    return sink


def chat_openai(server: OpenAIServer | None = None) -> ChatOpenAI:
    transport = httpx2.MockTransport(server or OpenAIServer())
    return ChatOpenAI(
        model="gpt-x",
        api_key="sk-test",  # type: ignore[arg-type]
        max_retries=0,
        http_client=httpx2.Client(transport=transport),
        http_async_client=httpx2.AsyncClient(transport=transport),
    )


def chat_anthropic() -> ChatAnthropic:
    model = ChatAnthropic(model="claude-x", api_key="sk-ant-test", max_retries=0)  # type: ignore[call-arg, arg-type]
    http = httpx2.Client(transport=httpx2.MockTransport(AnthropicServer()))
    # ChatAnthropic builds its client itself; this is the one it would build,
    # on the mock transport.
    client = anthropic.Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    object.__setattr__(model, "_client", client)
    return model


def tools_of(sink: Sink) -> list[Any]:
    return [s.tool for s in sink.steps if s.tool is not None]


def models_of(sink: Sink) -> list[Any]:
    return [s.model for s in sink.steps if s.model is not None]


ASK = [HumanMessage("where is my order?")]


def seen(message: Any) -> dict[str, Any]:
    """What the agent sees, less the run id LangChain mints for each call."""
    return message.model_dump(exclude={"id"})


def test_wrap_model_records_chat_openai_like_a_direct_sdk_call(sink: Sink) -> None:
    plain = chat_openai()
    expected = (
        seen(plain.invoke(ASK)),
        [c.content for c in plain.stream(ASK)],
        seen(asyncio.run(plain.ainvoke(ASK))),
    )
    model = chat_openai()
    assert fr.langchain.wrap_model(model) is model

    with fr.run("triage"):
        got = (
            seen(model.invoke(ASK)),
            [c.content for c in model.stream(ASK)],
            seen(asyncio.run(model.ainvoke(ASK))),
        )

    assert got == expected
    calls = models_of(sink)
    assert len(calls) == 3
    for call in calls:
        assert (call.provider, call.model_requested, call.model_used) == (
            "openai",
            "gpt-x",
            "gpt-x-2026-01-01",
        )
        assert call.messages == [{"role": "user", "content": "where is my order?"}]
        assert (call.usage.input_tokens, call.usage.output_tokens) == (12, 3)
    assert calls[1].params["stream"] is True
    assert isinstance(calls[1].response, list)  # every chunk, as for the SDK


def test_wrap_model_records_chat_anthropic_like_a_direct_sdk_call(sink: Sink) -> None:
    expected = seen(chat_anthropic().invoke(ASK))
    model = fr.langchain.wrap_model(chat_anthropic())

    with fr.run("triage"):
        got = seen(model.invoke(ASK))
        streamed = [c.content for c in model.stream(ASK)]

    assert got == expected
    assert streamed
    calls = models_of(sink)
    assert [(c.provider, c.model_requested) for c in calls] == [
        ("anthropic", "claude-x"),
        ("anthropic", "claude-x"),
    ]
    assert calls[0].messages == [{"role": "user", "content": "where is my order?"}]
    assert calls[0].finish_reason is not None
    assert calls[0].usage is not None


def test_wrap_model_twice_records_once_and_other_models_raise(sink: Sink) -> None:
    model = fr.langchain.wrap_model(fr.langchain.wrap_model(chat_openai()))
    with fr.run("triage"):
        model.invoke(ASK)
    assert len(sink.steps) == 1

    with pytest.raises(TypeError, match="GenericFakeChatModel"):
        fr.langchain.wrap_model(GenericFakeChatModel(messages=iter([])))


@tool
def lookup(order_id: str) -> str:
    """Look an order up."""
    return f"{order_id} shipped"


@tool
async def alookup(order_id: str) -> str:
    """Look an order up."""
    return f"{order_id} shipped"


@tool
def refund(order_id: str) -> str:
    """Refund an order."""
    raise ValueError(f"no refunds for {order_id}")


@tool
def cancel(order_id: str) -> str:
    """Cancel an order."""
    raise ToolException(f"{order_id} already shipped")


cancel.handle_tool_error = True


def call(name: str, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "name": name,
        "args": {"order_id": "o1"},
        "id": call_id,
        "type": "tool_call",
    }


def test_wrap_tools_records_each_call_with_its_effect_and_call_id(sink: Sink) -> None:
    expected = lookup.invoke(call("lookup"))
    (wrapped,) = fr.langchain.wrap_tools([lookup], effects={"lookup": "read"})
    assert isinstance(wrapped, BaseTool)
    assert (wrapped.name, wrapped.description) == (lookup.name, lookup.description)

    with fr.run("triage"):
        got = wrapped.invoke(call("lookup"))
        plain = wrapped.invoke({"order_id": "o2"})
        lookup.invoke({"order_id": "o3"})  # the original stays unrecorded

    assert got == expected
    assert isinstance(got, ToolMessage)
    assert plain == "o2 shipped"
    first, second = tools_of(sink)
    assert (first.name, first.effect, first.args, first.result, first.call_id) == (
        "lookup",
        "read",
        {"order_id": "o1"},
        "o1 shipped",
        "call_1",
    )
    assert (second.args, second.call_id) == ({"order_id": "o2"}, None)


def test_tool_errors_reach_the_agent_unchanged_and_are_recorded(sink: Sink) -> None:
    expected = cancel.invoke(call("cancel", "call_2"))
    wrapped_refund, wrapped_cancel = fr.langchain.wrap_tools([refund, cancel])

    with fr.run("triage"):
        with pytest.raises(ValueError, match="no refunds for o1"):
            wrapped_refund.invoke(call("refund"))
        handled = wrapped_cancel.invoke(call("cancel", "call_2"))

    assert handled == expected  # LangChain turned the ToolException into content
    raised, recorded = tools_of(sink)
    assert (raised.effect, raised.error.type) == ("irreversible", "ValueError")
    assert recorded.error.type == "langchain_core.tools.base.ToolException"
    assert recorded.call_id == "call_2"
    assert [s.status for s in sink.steps] == ["error", "error"]


def test_async_tools_record_once(sink: Sink) -> None:
    async_tool, sync_tool = fr.langchain.wrap_tools([alookup, lookup])

    async def agent() -> list[Any]:
        async with fr.run("triage"):
            # lookup has no coroutine, so LangChain runs it in a thread.
            return [
                await async_tool.ainvoke(call("alookup")),
                await sync_tool.ainvoke(call("lookup", "call_2")),
            ]

    results = asyncio.run(agent())
    assert [r.content for r in results] == ["o1 shipped", "o1 shipped"]
    assert [(t.name, t.call_id) for t in tools_of(sink)] == [
        ("alookup", "call_1"),
        ("lookup", "call_2"),
    ]


def test_wrap_tools_rejects_unknown_names_effects_and_objects() -> None:
    with pytest.raises(ValueError, match="nope"):
        fr.langchain.wrap_tools([lookup], effects={"nope": "read"})
    with pytest.raises(ValueError, match="lookup"):
        fr.langchain.wrap_tools([lookup], effects={"lookup": "safe"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="not a LangChain tool"):
        fr.langchain.wrap_tools([lambda: None])


def test_rewrapping_a_tool_records_once_with_the_new_effect(sink: Sink) -> None:
    (once,) = fr.langchain.wrap_tools([lookup])
    (twice,) = fr.langchain.wrap_tools([once], effects={"lookup": "read"})
    with fr.run("triage"):
        twice.invoke({"order_id": "o1"})
    assert [t.effect for t in tools_of(sink)] == ["read"]


class Legacy(BaseTool):
    """A hand-written tool whose async path reuses its sync one."""

    name: str = "legacy"
    description: str = "Look an order up."

    def _run(self, order_id: str) -> str:
        return f"{order_id} shipped"

    async def _arun(self, order_id: str) -> str:
        return self._run(order_id)


def test_an_arun_that_calls_run_records_once(sink: Sink) -> None:
    (wrapped,) = fr.langchain.wrap_tools([Legacy()])

    async def agent() -> Any:
        async with fr.run("triage"):
            return await wrapped.ainvoke({"order_id": "o1"})

    assert asyncio.run(agent()) == "o1 shipped"
    assert [t.args for t in tools_of(sink)] == [{"order_id": "o1"}]


class State(TypedDict):
    messages: list[Any]


def test_steps_inside_a_langgraph_node_are_tagged_with_it(sink: Sink) -> None:
    model = fr.langchain.wrap_model(chat_openai())
    (wrapped,) = fr.langchain.wrap_tools([lookup], effects={"lookup": "read"})

    def agent(state: State) -> State:
        return {"messages": [*state["messages"], model.invoke(state["messages"])]}

    def act(state: State) -> State:
        return {"messages": [*state["messages"], wrapped.invoke(call("lookup"))]}

    graph: Any = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("act", act)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "act")
    graph.add_edge("act", END)

    with fr.run("triage"):
        out = graph.compile().invoke({"messages": ASK})
        model.invoke(ASK)  # outside any node: no tag

    assert isinstance(out["messages"][1], AIMessage)
    assert [s.tags for s in sink.steps] == [["node:agent"], ["node:act"], []]


def test_a_failing_tag_source_logs_once_and_the_step_is_still_recorded(
    sink: Sink, caplog: pytest.LogCaptureFixture
) -> None:
    def broken() -> list[str]:
        raise RuntimeError("tags unavailable")

    recorder.tag_sources.append(broken)
    (wrapped,) = fr.langchain.wrap_tools([lookup])
    with caplog.at_level(logging.WARNING, logger="flight_recorder"), fr.run("t"):
        assert wrapped.invoke({"order_id": "o1"}) == "o1 shipped"
        wrapped.invoke({"order_id": "o2"})

    assert [s.tags for s in sink.steps] == [[], []]
    assert len(caplog.records) == 1
    assert all(isinstance(s, Step) for s in sink.steps)
