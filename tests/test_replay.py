"""Replay, diff, verdicts, repeats, reports and baseline updates (FR-P1 to FR-P10).

Each test records a small agent (a real OpenAI SDK client on a mock transport,
two tools, an optional sub-agent run) with the pytest plugin in a subprocess,
then replays it with something changed through environment variables.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.errors import PolicyError, ReplayDivergence
from flight_recorder.model import Run, Step
from flight_recorder.replay.engine import Session, replaying

AGENT = """
import json
import os
from pathlib import Path

import httpx2
import openai
import pytest

import flight_recorder as fr

env = os.environ.get
COUNTER = Path("model_calls.txt")


def answer() -> str:
    n = int(COUNTER.read_text()) if COUNTER.exists() else 0
    COUNTER.write_text(str(n + 1))
    if env("ALTERNATE"):  # the 2nd model call of a session answers differently
        return "lost" if n == 1 else "shipped"
    return env("ANSWER", "shipped")


def handle(request):
    if env("NO_NETWORK"):
        raise RuntimeError("network call during fully-recorded replay")
    body = json.loads(request.content)
    text = answer()
    base = {"id": "c1", "created": 0, "model": "gpt-x"}
    if not body.get("stream"):
        choice = {"index": 0, "finish_reason": "stop",
                  "message": {"role": "assistant", "content": text}}
        return httpx2.Response(200, json={**base, "object": "chat.completion", "choices": [choice]})
    chunks = [
        {**base, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
        for piece in (text[:3], text[3:])
    ]
    sse = "".join(f"data: {json.dumps(c)}\\n\\n" for c in chunks) + "data: [DONE]\\n\\n"
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())


llm = fr.wrap(openai.OpenAI(
    api_key="sk-test", max_retries=0,
    http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
))
CALLS = Path("tool_calls.txt")

if env("IGNORE_STAMP"):
    fr.matcher("lookup", ignore=["$.stamp"])


@fr.tool(effect="read")
def lookup(order_id: str, stamp: str = "1") -> dict:
    with CALLS.open("a") as f:
        f.write(f"lookup {order_id}\\n")
    return {"order_id": order_id, "status": "shipped"}


@fr.tool(effect="irreversible")
def refund(order_id: str, amount: float) -> str:
    with CALLS.open("a") as f:
        f.write(f"refund {order_id} {amount}\\n")
    return "refunded"


def ask(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if env("STREAM"):
        stream = llm.chat.completions.create(model="gpt-x", messages=messages, stream=True)
        return "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    reply = llm.chat.completions.create(model="gpt-x", messages=messages)
    return reply.choices[0].message.content


def helper(order_id: str) -> dict:
    with fr.run("helper"):
        return lookup(order_id)


def agent(order_id: str) -> str:
    if env("SKIP_LOOKUP"):
        order = {"order_id": order_id, "status": "shipped"}
    elif env("HELPER"):
        order = helper(order_id)
    else:
        order = lookup(order_id, env("STAMP", "1"))
    text = ask(env("PROMPT", "Where is my order?") + json.dumps(order))
    if env("EXTRA_CALL"):
        ask("Anything else?")
    if amount := env("REFUND"):
        try:
            refund(order_id, float(amount))
        except Exception as exc:
            if not env("CATCH"):
                raise
            text += f" ({type(exc).__name__})"
    return text


@pytest.mark.fr(
    cassette="desk/case",
    on_divergence=env("POLICY", "strict"),
    lookahead=int(env("LOOKAHEAD", "0")),
)
def test_case(fr_run):
    fr_run.output = agent(env("ORDER", "o1"))


@pytest.mark.fr(cassette="desk/flaky", repeats=3, min_pass=int(env("MIN_PASS", "2")))
def test_flaky(fr_run):
    fr_run.output = agent("o1")
"""


@pytest.fixture
def agent(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> pytest.Pytester:
    monkeypatch.setenv("FR_DIR", str(pytester.path / "store"))
    pytester.makepyfile(test_agent=AGENT)
    return pytester


def run(pytester: pytest.Pytester, *args: str, **env: str) -> pytest.RunResult:
    """One pytest session over the agent's test_case, with ``env`` set."""
    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)
        return pytester.runpytest_subprocess("-k", "test_case", *args)


def record(pytester: pytest.Pytester, **env: str) -> None:
    run(pytester, "--fr-mode=record", **env).assert_outcomes(passed=1)
    (pytester.path / "tool_calls.txt").unlink()


def tool_calls(pytester: pytest.Pytester) -> list[str]:
    calls = pytester.path / "tool_calls.txt"
    return calls.read_text().splitlines() if calls.exists() else []


def test_tools_mode_serves_tools_and_runs_models_live(agent: pytest.Pytester) -> None:
    record(agent, REFUND="5", HELPER="1")

    result = run(agent, REFUND="5", HELPER="1")

    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*flight recorder*", "same     desk/case"])
    assert tool_calls(agent) == []  # every tool call, the sub-agent's too, served
    assert (agent.path / "model_calls.txt").read_text() == "2"  # the model ran live


@pytest.mark.parametrize("stream", ["", "1"])
def test_full_mode_serves_models_too_with_no_network(
    agent: pytest.Pytester, stream: str
) -> None:
    record(agent, STREAM=stream)

    result = run(agent, "--fr-replay=full", STREAM=stream, NO_NETWORK="1")

    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["same     desk/case"])


def test_a_strict_divergence_fails_and_pinpoints_the_step(
    agent: pytest.Pytester,
) -> None:
    record(agent, REFUND="5")

    # The agent catches the divergence; strict still fails the case.
    result = run(agent, REFUND="50", CATCH="1")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*diverged at step 1 (strict)*",
            '*expected: {"args":{"amount":5.0,"order_id":"o1"},"tool":"refund"}*',
            '*actual:   {"args":{"amount":50.0,"order_id":"o1"},"tool":"refund"}*',
            '*after:    {"args":{"order_id":"o1","stamp":"1"},"tool":"lookup"}*',
        ]
    )
    assert tool_calls(agent) == []  # the irreversible tool never ran


def test_a_broken_prompt_fails_fully_recorded_replay(agent: pytest.Pytester) -> None:
    record(agent)

    result = run(agent, "--fr-replay=full", PROMPT="Ignore the order.", NO_NETWORK="1")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*ReplayDivergence*diverged at step 1*"])


def test_changed_steps_warn_by_default_and_fail_when_asked(
    agent: pytest.Pytester,
) -> None:
    record(agent)

    warned = run(agent, EXTRA_CALL="1")
    failed = run(agent, "--fr-on-change=fail", EXTRA_CALL="1")

    warned.assert_outcomes(passed=1)
    warned.stdout.fnmatch_lines(["changed  desk/case", "*added model openai/gpt-x #*"])
    failed.assert_outcomes(failed=1)
    failed.stdout.fnmatch_lines(["*accept it with: fr accept desk/case*"])


def test_a_different_final_output_fails(agent: pytest.Pytester) -> None:
    record(agent)

    result = run(agent, ANSWER="lost")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*final output differs*"])


def test_ignored_argument_paths_still_match(agent: pytest.Pytester) -> None:
    record(agent, STAMP="100")

    run(agent, STAMP="200", IGNORE_STAMP="1").assert_outcomes(passed=1)
    run(agent, STAMP="200").assert_outcomes(failed=1)


def test_live_policy_runs_a_read_tool_that_diverged(agent: pytest.Pytester) -> None:
    record(agent)

    result = run(agent, POLICY="live", ORDER="o2")

    result.assert_outcomes(passed=1)  # same output; the changed argument warns
    result.stdout.fnmatch_lines(
        ["changed  desk/case", "*changed tool lookup #0 (args)*"]
    )
    assert tool_calls(agent) == ["lookup o2"]


def test_repeats_pass_on_k_of_n(agent: pytest.Pytester) -> None:
    for min_pass, outcome in (("2", "passed"), ("3", "failed")):
        (agent.path / "model_calls.txt").unlink(missing_ok=True)
        args = ("-k", "test_flaky")
        env = {"ALTERNATE": "1", "MIN_PASS": min_pass}
        with pytest.MonkeyPatch.context() as mp:
            for key, value in env.items():
                mp.setenv(key, value)
            agent.runpytest_subprocess("--fr-mode=record", *args)
            result = agent.runpytest_subprocess(*args)
        result.assert_outcomes(**{outcome: 1})
        result.stdout.fnmatch_lines(["*desk/flaky (2/3 passed)*"])


def test_update_and_accept_make_the_replay_the_new_baseline(
    agent: pytest.Pytester,
) -> None:
    record(agent)
    run(agent, "--fr-update", EXTRA_CALL="1").stdout.fnmatch_lines(
        ["updated  desk/case"]
    )
    run(agent, EXTRA_CALL="1").stdout.fnmatch_lines(["same     desk/case"])

    run(agent).stdout.fnmatch_lines(["changed  desk/case"])  # back to one call
    accepted = agent.run("fr", "accept", "desk/case")
    assert accepted.ret == 0
    run(agent).stdout.fnmatch_lines(["same     desk/case"])


def test_reports_and_fr_diff(agent: pytest.Pytester) -> None:
    record(agent)

    run(agent, "--fr-report=reports", EXTRA_CALL="1").assert_outcomes(passed=1)

    (entry,) = json.loads((agent.path / "reports" / "report.json").read_text())
    assert (entry["case"], entry["verdict"], entry["passed"]) == (
        "desk/case",
        "changed",
        1,
    )
    markdown = (agent.path / "reports" / "report.md").read_text()
    assert "| `desk/case` | changed | 1/1 |" in markdown
    store = agent.path / "store"
    result = agent.run(
        "fr",
        "diff",
        str(store / "cassettes/desk/case.jsonl"),
        str(store / "replays/desk/case.jsonl"),
    )
    assert result.ret == 0
    result.stdout.fnmatch_lines(["changed  test_case", "*added model openai/gpt-x*"])


def test_live_on_divergence_is_refused_for_irreversible_and_unopted_write_tools() -> (
    None
):
    @fr.tool(effect="irreversible")
    def wire_money(account: str) -> str:
        return "sent"

    @fr.tool(effect="write")
    def update_crm(record: str) -> str:
        return "ok"

    with pytest.raises(PolicyError, match="irreversible tool 'wire_money'"):
        fr.matcher("wire_money", on_divergence="live")
    with pytest.raises(PolicyError, match="allow_write=True"):
        fr.matcher("update_crm", on_divergence="live")
    fr.matcher("update_crm", on_divergence="live", allow_write=True)
    # A rule set first is checked when the tool is defined.
    fr.matcher("close_account", on_divergence="live")
    with pytest.raises(PolicyError):

        @fr.tool(effect="irreversible")
        def close_account(account: str) -> str:
            return "closed"


def test_store_path_shape(agent: pytest.Pytester) -> None:
    record(agent, HELPER="1")
    run(agent, HELPER="1").assert_outcomes(passed=1)

    store = agent.path / "store"
    assert (store / "cassettes/desk/case/1-helper.jsonl").is_file()
    assert (store / "replays/desk/case.jsonl").is_file()
    assert (store / "replays/desk/case/1-helper.jsonl").is_file()
    assert isinstance(Path, type)


def test_lookahead_skips_recorded_reads_never_writes(agent: pytest.Pytester) -> None:
    record(agent, REFUND="5")

    skipped = run(agent, REFUND="5", SKIP_LOOKUP="1", LOOKAHEAD="1")
    strict = run(agent, REFUND="5", SKIP_LOOKUP="1")

    skipped.assert_outcomes(passed=1)
    skipped.stdout.fnmatch_lines(["changed  desk/case", "*removed tool lookup #0*"])
    strict.assert_outcomes(failed=1)


def test_a_run_wide_live_policy_never_calls_an_irreversible_tool(
    agent: pytest.Pytester,
) -> None:
    record(agent, REFUND="5")

    result = run(agent, POLICY="live", REFUND="50")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*diverged at step 1 (strict)*"])
    assert tool_calls(agent) == []


class Keep:
    """A sink that keeps the steps of the runs it sees."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def run_started(self, run: Run) -> None:
        pass

    def step_closed(self, run_id: str, step: Step) -> None:
        self.steps.append(step)

    def run_finished(self, run: Run) -> None:
        pass


@fr.tool(effect="write")
def note(text: str) -> str:
    return "noted"


@fr.tool(effect="read")
def status(order_id: str) -> str:
    return "shipped"


@pytest.mark.parametrize(
    ("skipped", "served"), [(lambda: status("o1"), True), (lambda: note("hi"), False)]
)
def test_lookahead_skips_a_recorded_read_never_a_write(
    skipped: Callable[[], str], served: bool
) -> None:
    keep = Keep()
    recorder.sinks = [keep]
    with fr.run("case") as baseline:
        skipped()
        status("o2")
    session = Session((baseline, keep.steps), lookahead=1)

    with replaying(session), fr.run("case"):
        if served:
            assert status("o2") == "shipped"
        else:
            with pytest.raises(ReplayDivergence):
                status("o2")
