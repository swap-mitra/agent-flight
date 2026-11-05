"""Gate decisions in replay (FR-G9, row 4.10).

A case is recorded through the pytest plugin under one gate policy, then
replayed under another: the recorded decisions stand (no one approves in CI),
the current rules are evaluated, and a difference is a gate change.
"""

import json

import pytest

AGENT = """
import os

import pytest

import flight_recorder as fr
from flight_recorder.errors import ToolDenied

fr.configure(policy=os.environ["POLICY_FILE"])


@fr.tool(effect="irreversible")
def refund(order_id: str, amount: float) -> str:
    return "refunded"


@pytest.mark.fr(cassette="desk/refund")
def test_refund(fr_run):
    try:
        fr_run.output = refund("o1", float(os.environ.get("AMOUNT", "50")))
    except ToolDenied as exc:
        fr_run.output = f"denied: {exc}"
"""

ALLOW = '[[gate.rules]]\ntool = "refund"\naction = "allow"\n'
DENY = '[[gate.rules]]\ntool = "refund"\naction = "deny"\n'
APPROVE = '[gate]\napproval_timeout = "5s"\n[[gate.rules]]\ntool = "refund"\naction = "approve"\n'


@pytest.fixture
def agent(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> pytest.Pytester:
    monkeypatch.setenv("FR_DIR", str(pytester.path / "store"))
    pytester.makepyfile(test_agent=AGENT)
    return pytester


def use(agent: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    file = agent.path / "fr.toml"
    file.write_text(text, encoding="utf-8")
    monkeypatch.setenv("POLICY_FILE", str(file))


def replay(agent: pytest.Pytester) -> dict[str, object]:
    agent.runpytest_subprocess("--fr-report=out")
    (result,) = json.loads((agent.path / "out" / "report.json").read_text("utf-8"))
    return result


def test_an_unchanged_rule_replays_the_same(
    agent: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(agent, monkeypatch, ALLOW)
    agent.runpytest_subprocess("--fr-mode=record").assert_outcomes(passed=1)
    assert replay(agent)["verdict"] == "same"


def test_a_tightened_rule_is_a_gate_change(
    agent: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(agent, monkeypatch, ALLOW)
    agent.runpytest_subprocess("--fr-mode=record").assert_outcomes(passed=1)
    use(agent, monkeypatch, APPROVE)  # would wait for a person if it were live
    result = replay(agent)
    assert result["verdict"] == "changed"
    (diff,) = result["diffs"]  # type: ignore[misc]
    assert "gate tool refund #0: allow -> approve" in diff["reasons"]
    assert diff["output_equal"] is True  # the recorded decision stood


def test_a_recorded_denial_replays_as_a_denial(
    agent: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(agent, monkeypatch, DENY)
    agent.runpytest_subprocess("--fr-mode=record").assert_outcomes(passed=1)
    assert replay(agent)["verdict"] == "same"
    use(agent, monkeypatch, ALLOW)  # loosened: the recorded denial still stands
    result = replay(agent)
    assert result["verdict"] == "changed"
    (diff,) = result["diffs"]  # type: ignore[misc]
    assert "gate tool refund #0: deny -> allow" in diff["reasons"]
    assert diff["output_equal"] is True


def test_cassettes_keep_the_gate_decision(
    agent: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(agent, monkeypatch, DENY)
    agent.runpytest_subprocess("--fr-mode=record").assert_outcomes(passed=1)
    lines = (
        (agent.path / "store" / "cassettes" / "desk" / "refund.jsonl")
        .read_text("utf-8")
        .splitlines()
    )
    step = json.loads(lines[1])
    assert step["status"] == "denied"
    assert step["gate"]["action"] == "deny" and step["gate"]["outcome"] == "denied"
