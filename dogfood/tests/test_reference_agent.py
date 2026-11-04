"""The reference agent's tests, run as written and again with recording on.

FR-R1 acceptance: the same tests pass unchanged once the agent is wrapped.
FR-R2 acceptance: every model step of the recorded run has every field.
"""

import contextlib
import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import MODEL, ScriptedClaude, chat_model, text, tool_use
from reference_agent import answer, build, heptapod_tools

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.model import Step
from flight_recorder.store.cassette import CassetteStore, load

TASK = "Ada called: find her ticket, check the part, update her phone."


class Agent:
    """Runs the reference agent, recorded or not; tests use it the same way."""

    def __init__(self, record: bool, root: Path) -> None:
        self.record = record
        self.root = root
        if record:
            fr.configure(dir=root)

    def run(self, registry: Any, script: ScriptedClaude) -> str:
        graph = build(chat_model(script), heptapod_tools(registry), self.record)
        recording = (
            fr.run("reference-agent", inputs={"task": TASK})
            if self.record
            else contextlib.nullcontext()
        )
        with recording:
            return answer(graph, TASK)

    def steps(self) -> list[Step]:
        (store,) = [s for s in recorder.sinks if isinstance(s, CassetteStore)]
        assert store.flush(timeout=10)
        (file,) = (self.root / "runs").rglob("*.jsonl")
        _, steps = load(file)
        return steps


@pytest.fixture(params=["as written", "recorded"])
def agent(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Agent]:
    saved = list(recorder.sinks)
    yield Agent(request.param == "recorded", tmp_path)
    for sink in recorder.sinks:
        if isinstance(sink, CassetteStore):
            sink.close()
    recorder.sinks = saved


def test_demo_task_across_all_three_backends(agent: Agent, registry: Any) -> None:
    # The mocks are fresh per test; a new value still proves this run's update.
    phone = f"+1-555-{uuid.uuid4().int % 10000:04d}"
    script = ScriptedClaude(
        [
            [tool_use("t1", "crm", {"operation": "list_customers"})],
            [
                tool_use(
                    "t2",
                    "ticketing",
                    {"operation": "get_ticket", "ticket_no": "T-9001"},
                )
            ],
            [
                tool_use(
                    "t3",
                    "inventory",
                    {"operation": "get_stock", "partNumber": "BP-310"},
                )
            ],
            [
                tool_use(
                    "t4",
                    "crm",
                    {
                        "operation": "update_customer",
                        "customer_id": "C-1001",
                        "phone": phone,
                    },
                )
            ],
            [text("done")],
        ]
    )

    assert agent.run(registry, script) == "done"

    # What the model was shown links up across systems: customer, ticket, part.
    results = script.tool_results()
    assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3", "t4"]
    customers, ticket, stock, updated = (json.loads(r["content"]) for r in results)
    ada = next(c for c in customers["customers"] if c["name"] == "Ada Lindqvist")
    assert "T-9001" in ada["open_ticket_ids"]
    assert (ticket["cust_ref"], ticket["state"], ticket["part_ref"]) == (
        "C-1001",
        "open",
        "BP-310",
    )
    assert sum(int(loc["quantityOnHand"]) for loc in stock["locations"]) > 0
    assert updated["phone"] == phone
    stored = registry.call(
        "crm", {"operation": "get_customer", "customer_id": "C-1001"}
    )
    assert stored.data["phone"] == phone

    if not agent.record:
        return
    steps = agent.steps()
    models = [s for s in steps if s.model is not None]
    tools = [s for s in steps if s.tool is not None]
    assert len(models) == 5
    for step in models:  # FR-R2: every field, for every model step
        call = step.model
        assert call is not None
        assert (call.provider, call.model_requested, call.model_used) == (
            "anthropic",
            MODEL,
            MODEL,
        )
        assert call.messages and call.params["tools"] and call.response
        assert call.finish_reason in ("tool_use", "end_turn")
        assert call.usage is not None and call.usage.input_tokens == 1200
        assert call.cost_usd == pytest.approx((1200 * 5.00 + 80 * 25.00) / 1e6)
        assert step.latency_ms > 0
        assert step.tags == ["node:agent"]
    assert [(t.tool.name, t.tool.call_id) for t in tools if t.tool] == [
        ("crm", "t1"),
        ("ticketing", "t2"),
        ("inventory", "t3"),
        ("crm", "t4"),
    ]
    for step, shown in zip(tools, results, strict=True):
        assert step.tool is not None
        assert (step.tool.effect, step.status, step.tags) == (
            "write",
            "ok",
            ["node:tools"],
        )
        assert step.tool.result == shown["content"]  # what the model saw


def test_a_backend_error_reaches_the_model_as_an_error_result(
    agent: Agent, registry: Any
) -> None:
    script = ScriptedClaude(
        [
            [
                tool_use(
                    "t1", "ticketing", {"operation": "get_ticket", "ticket_no": "T-0"}
                )
            ],
            [text("no such ticket")],
        ]
    )

    assert agent.run(registry, script) == "no such ticket"

    (result,) = script.tool_results()
    assert result["is_error"] is True
    assert "error" in json.loads(result["content"])

    if not agent.record:
        return
    (tool,) = [s for s in agent.steps() if s.tool is not None]
    assert tool.status == "error"
    assert tool.tool is not None and tool.tool.error is not None
    assert tool.tool.error.message == result["content"]
