"""fr eval-route: a candidate routing policy judged by replay (FR-B7, row 3.10).

A small agent (the real OpenAI SDK on a mock transport that answers as the
requested model) is recorded with the pytest plugin, then replayed by
``fr eval-route`` under a policy that routes it to a cheaper model.
"""

import json
from pathlib import Path

import pytest

from flight_recorder import cli

AGENT = """
import json
import os

import httpx2
import openai
import pytest

import flight_recorder as fr


def handle(request):
    body = json.loads(request.content)
    # A cheaper model gets a different answer when ANSWER_SMALL says so.
    small = body["model"] == "gpt-5-mini" and os.environ.get("ANSWER_SMALL")
    choice = {"index": 0, "finish_reason": "stop",
              "message": {"role": "assistant", "content": "lost" if small else "shipped"}}
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
    return httpx2.Response(200, json={
        "id": "c1", "created": 0, "model": body["model"],
        "object": "chat.completion", "choices": [choice], "usage": usage,
    })


llm = fr.wrap(openai.OpenAI(
    api_key="sk-test", max_retries=0,
    http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
))


@fr.tool(effect="read")
def lookup(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}


def agent() -> str:
    order = lookup("o1")
    with fr.step_tag("extract"):
        reply = llm.chat.completions.create(
            model="gpt-5", messages=[{"role": "user", "content": json.dumps(order)}]
        )
    return reply.choices[0].message.content


@pytest.mark.fr(cassette="desk/triage", replay="full")
def test_triage(fr_run):
    fr_run.output = agent()


@pytest.mark.fr(cassette="desk/report")
def test_report(fr_run):
    fr_run.output = agent()
"""

CHEAPER = """
[[routes]]
match = { tag = "extract", run_name = "test_triage" }
model = "openai/gpt-5-mini"
"""


@pytest.fixture
def recorded(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FR_DIR", str(pytester.path / "store"))
    pytester.makepyfile(test_agent=AGENT)
    pytester.runpytest_subprocess("--fr-mode=record").assert_outcomes(passed=2)
    policy = pytester.path / "cheaper.toml"
    policy.write_text(CHEAPER, encoding="utf-8")
    return policy


def test_eval_route_reports_same_verdict_rate_and_cost_delta(
    recorded: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    out = recorded.parent / "out"
    code = cli.main(["eval-route", "--policy", str(recorded), "--report", str(out)])
    printed = capfd.readouterr().out
    assert code == 0
    # gpt-5: 1,000 x $1.25 + 100 x $10 per million = $0.00225;
    # gpt-5-mini: 1,000 x $0.25 + 100 x $2 per million = $0.00045.
    assert (
        "| `test_triage` | 100% (1/1) | $0.0022 | $0.0004 | -$0.0018 (-80.0%) |"
        in printed
    )
    assert (
        "| `test_report` | 100% (1/1) | $0.0022 | $0.0022 | +$0.0000 (+0.0%) |"
        in printed
    )
    assert "model calls ran live" in printed
    data = json.loads((out / "eval-route.json").read_text("utf-8"))
    triage = next(r for r in data["runs"] if r["run"] == "test_triage")
    assert triage["same_rate"] == 1.0
    assert triage["delta_usd"] == pytest.approx(0.00045 - 0.00225)
    assert (out / "eval-route.md").read_text("utf-8") == printed[
        printed.index("## Flight") :
    ]


def test_a_cheaper_model_that_changes_the_outcome_lowers_the_rate(
    recorded: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWER_SMALL", "1")
    code = cli.main(["eval-route", "--policy", str(recorded), "--", "-q"])
    printed = capfd.readouterr().out
    assert code == 1
    assert "| `test_triage` | 0% (0/1) |" in printed
    assert "| `test_report` | 100% (1/1) |" in printed


def test_eval_route_fails_clearly_without_a_policy_or_a_report(
    recorded: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["eval-route", "--policy", "missing.toml"]) == 1
    assert "no policy file" in capfd.readouterr().err
    bad = recorded.parent / "bad.toml"
    bad.write_text("[budgets]\nbogus = 1\n", encoding="utf-8")
    assert cli.main(["eval-route", "--policy", str(bad)]) == 1
    assert "without a replay report" in capfd.readouterr().err
