"""The reference agent: a small LangGraph agent over Heptapod's tools (row 1.14).

One model node and one tool node in a loop, over the tools of a Heptapod
Registry (one per backend: CRM, ticketing, inventory). ``build(...,
record=True)`` is the only change recording needs (FR-R1); the agent is
otherwise the same code either way.
"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import (  # pyright: ignore[reportMissingTypeStubs]
    ToolNode,
    tools_condition,
)

import flight_recorder as fr
from flight_recorder.model import Effect

# Heptapod's own agent prompt.
SYSTEM = (
    "You operate systems. Each tool is one system; choose what it does "
    "with the `operation` argument. Rely on tool results, not assumptions. When a tool returns "
    "an error, decide whether to correct the call, try another route, or report the problem. "
    "Finish with a short summary of what you found and what you changed."
)
# Each Heptapod tool is a whole backend, which can create and update records.
EFFECTS: dict[str, Effect] = {
    "crm": "write",
    "ticketing": "write",
    "inventory": "write",
}


def heptapod_tools(registry: Any) -> list[BaseTool]:
    """One LangChain tool per adapter in a Heptapod ``Registry``.

    A backend failure reaches the model as an error result, as in Heptapod's
    own runtime: JSON ``{"error": ...}`` with the tool message marked failed.
    """

    def tool_for(spec: dict[str, Any]) -> BaseTool:
        def call(**input: Any) -> str:
            result = registry.call(spec["name"], input)
            if not result.success:
                raise ToolException(json.dumps({"error": result.error}))
            return json.dumps(result.data)

        return StructuredTool.from_function(
            func=call,
            name=spec["name"],
            description=spec["description"],
            args_schema=spec["input_schema"],
            handle_tool_error=True,
        )

    return [tool_for(spec) for spec in registry.tools()]


def build(model: BaseChatModel, tools: list[BaseTool], record: bool = False) -> Any:
    """The compiled graph; with ``record``, its model and tool calls are recorded."""
    if record:
        model = fr.langchain.wrap_model(model)
        tools = fr.langchain.wrap_tools(tools, effects=EFFECTS)
    bound = model.bind_tools(tools)

    def agent(state: MessagesState) -> dict[str, Any]:
        reply = bound.invoke([SystemMessage(SYSTEM), *state["messages"]])
        return {"messages": [reply]}

    graph: Any = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def answer(graph: Any, task: str) -> str:
    """Run the agent on one task and return its final answer."""
    final = graph.invoke({"messages": [HumanMessage(task)]})
    return final["messages"][-1].text
