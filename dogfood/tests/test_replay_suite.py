"""The reference agent as a replayed regression test (plan row 2.9, FR-P1 to P10).

The demo task's baseline lives in ``dogfood/.fr/cassettes/reference/``. CI
replays it fully recorded: model responses and tool results both come from the
cassette, so the run needs no network, and any change to what the agent sends
the model (its prompt, its tools, its messages) is a divergence that fails
the case. CI also edits the system prompt on purpose and checks that the
replay fails, then posts the report on the pull request.

Re-record after a deliberate change with
``pytest dogfood/tests/test_replay_suite.py --fr-mode=record``.
"""

from typing import Any

import pytest
from conftest import ScriptedClaude, chat_model, text, tool_use
from reference_agent import answer, build, heptapod_tools

TASK = "Ada called: find her ticket, check the part, update her phone."


@pytest.mark.fr(cassette="reference/demo_task", replay="full")
def test_demo_task(fr_run: Any, registry: Any) -> None:
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
                        "phone": "+1-555-0142",
                    },
                )
            ],
            [
                text(
                    "Ada's ticket T-9001 needs part BP-310, which is in stock; phone updated."
                )
            ],
        ]
    )
    graph = build(chat_model(script), heptapod_tools(registry), record=True)
    fr_run.output = answer(graph, TASK)
