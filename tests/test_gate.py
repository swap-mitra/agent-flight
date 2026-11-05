"""The action gate and approvals (FR-G1 to G9, rows 4.1 to 4.11)."""

import asyncio
import hashlib
import hmac
import io
import json
import logging
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

import flight_recorder as fr
from flight_recorder import approvals, cli, recorder
from flight_recorder.approvals import webhook
from flight_recorder.errors import PolicyError, ToolDenied
from flight_recorder.model import Run, Step
from flight_recorder.policy import conditions, gate


class Sink:
    def __init__(self) -> None:
        self.runs: list[Run] = []
        self.steps: list[Step] = []

    def run_started(self, run: Run) -> None:
        pass

    def step_closed(self, run_id: str, step: Step) -> None:
        self.steps.append(step)

    def run_finished(self, run: Run) -> None:
        self.runs.append(run)


@pytest.fixture
def sink() -> Sink:
    sink = Sink()
    recorder.sinks = [sink]
    return sink


def policy(tmp_path: Path, text: str) -> None:
    file = tmp_path / "fr.toml"
    file.write_text(text, encoding="utf-8")
    kept = [s for s in recorder.sinks if isinstance(s, Sink)]
    fr.configure(dir=tmp_path / ".fr", policy=file)
    recorder.sinks += kept


@fr.tool(effect="read")
def lookup(order_id: str) -> str:
    return f"{order_id} shipped"


@fr.tool(effect="irreversible")
def refund(order_id: str, amount: float) -> str:
    return f"refunded {amount}"


@fr.tool(effect="write")
def delete_note(note_id: str) -> str:
    return "deleted"


@fr.tool()
def undeclared(x: int) -> int:
    return x


def until(ready: Any, seconds: float = 10) -> None:
    """Wait for ``ready()``; fail rather than hang if it never comes."""
    deadline = time.monotonic() + seconds
    while not ready():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


def decide_when_asked(decision: str, approver: str = "ada") -> threading.Thread:
    """Resolve the first approval request that shows up, from another thread."""

    def run() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            waiting = approvals.pending()
            if waiting:
                approvals.resolve(waiting[0].id, decision, approver, "checked")  # type: ignore[arg-type]
                return
            time.sleep(0.02)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# Row 4.1: gate rules in the policy file


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (
            '[[gate.rules]]\naction = "allow"\nwhen = "__import__(\'os\')"\n',
            "Call is not allowed",
        ),
        (
            '[[gate.rules]]\naction = "allow"\nwhen = "secrets.key == 1"\n',
            "unknown name",
        ),
        ('[[gate.rules]]\naction = "allow"\nwhen = "args[x] > 1"\n', "literal key"),
        (
            '[[gate.rules]]\naction = "allow"\nwhen = "args.amount >"\n',
            "not an expression",
        ),
        (
            '[[gate.rules]]\naction = "allow"\nwhen_fn = "nowhere:fn"\n',
            "can't be imported",
        ),
        ('[[gate.rules]]\naction = "allow"\nwhen_fn = "json"\n', "module:function"),
        ('[[gate.rules]]\naction = "maybe"\n', "action"),
        ('[gate]\napproval_timeout = "soon"\n', "approval_timeout"),
        ('[gate]\napproval_timeout = "0s"\n', "positive"),
        ('[gate.effects]\nrefund = "dangerous"\n', "effects"),
        ('[gate.webhook]\nurl = "ftp://x"\n', "http"),
        (
            '[gate.webhook]\nurl = "https://x"\nsecret_env = "FR_TEST_NO_SECRET"\n',
            "FR_TEST_NO_SECRET",
        ),
    ],
)
def test_an_invalid_gate_raises_at_configure(
    tmp_path: Path, text: str, error: str
) -> None:
    file = tmp_path / "fr.toml"
    file.write_text(text, encoding="utf-8")
    with pytest.raises(PolicyError, match=error):
        fr.configure(dir=tmp_path / ".fr", policy=file)


def test_conditions_compare_fields_without_eval() -> None:
    names = {
        "args": {"amount": 600, "tags": ["vip"], "to": {"domain": "x.com"}},
        "run": {"name": "triage"},
        "env": {"name": "prod"},
    }
    assert conditions.evaluate("args.amount > 500 and env.name == 'prod'", names)
    assert conditions.evaluate("'vip' in args.tags or args.amount < 0", names)
    assert conditions.evaluate("not args['to'].domain == 'y.com'", names)
    assert conditions.evaluate("500 < args.amount <= 600", names)
    assert not conditions.evaluate("args.amount > -1 and run.name in ['report']", names)
    with pytest.raises(KeyError, match="no field 'missing'"):
        conditions.evaluate("args.missing == 1", names)


# Rows 4.2 and 4.3: classes and rules


def test_without_a_gate_section_tools_are_not_gated(tmp_path: Path, sink: Sink) -> None:
    policy(tmp_path, "[budgets]\nper_run_usd = 1\n")
    with fr.run("triage"):
        assert refund("o1", 5.0) == "refunded 5.0"
    assert sink.steps[0].gate is None


def test_rules_first_match_wins_and_the_default_gates_irreversible(
    tmp_path: Path, sink: Sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy(
        tmp_path,
        """
[gate]
approval_timeout = "5s"

[[gate.rules]]
id = "small-refunds"
tool = "refund"
when = "args.amount <= 100"
action = "allow"

[[gate.rules]]
tool = "delete_*"
env = "prod"
action = "deny"
reason = "no deletes in production"
""",
    )
    monkeypatch.setenv("FR_ENV", "prod")
    with fr.run("triage"):
        assert refund("o1", 50.0) == "refunded 50.0"
        assert lookup("o1") == "o1 shipped"
        with pytest.raises(ToolDenied, match="no deletes in production") as info:
            delete_note("n1")
        decide_when_asked("approve")
        assert refund("o1", 600.0) == "refunded 600.0"
    small, read, deleted, big = sink.steps
    assert small.gate is not None and (small.gate.action, small.gate.rule) == (
        "allow",
        "small-refunds",
    )
    assert read.gate is not None and (read.gate.outcome, read.gate.rule) == (
        "allowed",
        None,
    )
    assert deleted.status == "denied"
    assert deleted.gate is not None and (deleted.gate.outcome, deleted.gate.rule) == (
        "denied",
        "rules[1]",
    )
    assert info.value.decision == deleted.gate
    assert big.gate is not None
    assert (big.gate.action, big.gate.outcome, big.gate.approver) == (
        "approve",
        "approved",
        "ada",
    )
    assert big.gate.approval_id is not None
    assert sink.runs[0].status == "denied"  # a call was denied


def test_a_rule_can_match_on_the_tools_class(tmp_path: Path) -> None:
    policy(
        tmp_path,
        '[[gate.rules]]\neffect = "write"\naction = "deny"\n'
        '[[gate.rules]]\naction = "allow"\n',
    )
    with fr.run("triage"):
        assert refund("o1", 5.0) == "refunded 5.0"  # irreversible: not matched
        with pytest.raises(ToolDenied):
            delete_note("n1")  # write: matched


def test_a_replayed_call_keeps_its_recorded_outcome_under_the_current_rules(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from flight_recorder.model import GateDecision
    from flight_recorder.recorder import ActiveRun

    policy(tmp_path, '[[gate.rules]]\ntool = "refund"\naction = "deny"\n')
    recorded = GateDecision(
        action="approve",
        outcome="approved",
        approver="ada",
        decided_at=datetime.now(UTC),
        approval_id="a1",
    )
    active = ActiveRun(
        Run(run_id="r1", name="triage", status="incomplete", inputs=None)
    )
    decision = gate.replayed(active, "refund", "irreversible", {"amount": 5}, recorded)
    assert decision is not None
    assert (decision.action, decision.rule) == ("deny", "rules[0]")  # current rules
    assert (decision.outcome, decision.approver) == ("approved", "ada")  # recorded


def test_a_live_call_during_a_replay_is_never_left_waiting_for_a_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flight_recorder.replay import engine as replay

    policy(tmp_path, '[gate]\napproval_timeout = "30s"\n')
    monkeypatch.setattr(replay, "replaying_now", lambda: True)
    started = time.monotonic()
    with (
        fr.run("triage"),
        pytest.raises(ToolDenied, match="approval needed during a replay"),
    ):
        refund("o1", 600.0)
    assert time.monotonic() - started < 5
    assert approvals.pending() == []


def test_env_defaults_to_dev(tmp_path: Path) -> None:
    policy(
        tmp_path, '[[gate.rules]]\ntool = "delete_*"\nenv = "prod"\naction = "deny"\n'
    )
    with fr.run("triage"):
        assert delete_note("n1") == "deleted"  # FR_ENV unset: dev


def test_when_fn_rules_get_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[gate.GateCall] = []

    def external(call: gate.GateCall) -> bool:
        seen.append(call)
        return call.args["order_id"] == "o2"

    import types

    module = types.ModuleType("fr_test_rules")
    module.external = external  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "fr_test_rules", module)
    policy(
        tmp_path,
        '[[gate.rules]]\ntool = "refund"\nwhen_fn = "fr_test_rules:external"\naction = "deny"\n'
        '[[gate.rules]]\ntool = "refund"\naction = "allow"\n',
    )
    with fr.run("triage"):
        assert refund("o1", 5.0)
        with pytest.raises(ToolDenied):
            refund("o2", 5.0)
    assert (
        seen[0].tool == "refund"
        and seen[0].effect == "irreversible"
        and seen[0].env == "dev"
    )


def test_an_undeclared_tool_is_irreversible_unless_the_policy_says_otherwise(
    tmp_path: Path, sink: Sink
) -> None:
    policy(
        tmp_path,
        '[gate]\napproval_timeout = "0.3s"\n[gate.effects]\nlookup = "irreversible"\n',
    )
    with fr.run("triage"):
        with pytest.raises(ToolDenied, match="not approved in time"):
            undeclared(1)
        with pytest.raises(ToolDenied):
            lookup("o1")  # the policy wins over the decorator
    first, second = sink.steps
    assert first.tool is not None and first.tool.effect == "irreversible"
    assert first.gate is not None and first.gate.outcome == "expired"
    assert second.tool is not None and second.tool.effect == "irreversible"
    policy(tmp_path, '[gate.effects]\nundeclared = "read"\n')
    with fr.run("triage"):
        assert undeclared(2) == 2


# Rows 4.4 and 4.5: the approval store and waiting


def test_an_approval_request_carries_every_field_and_is_decided_once(
    tmp_path: Path, sink: Sink
) -> None:
    policy(tmp_path, '[gate]\napproval_timeout = "5s"\n')
    fr.configure(
        dir=tmp_path / ".fr",
        policy=tmp_path / "fr.toml",
        redact=[fr.redact.path("$.tool.args.order_id")],
    )
    recorder.sinks.append(sink)
    captured: list[approvals.Request] = []

    def approve_and_race() -> None:
        until(approvals.pending)
        (request,) = approvals.pending()
        captured.append(request)
        assert approvals.resolve(request.id, "deny", "bob", "too much")
        assert not approvals.resolve(request.id, "approve", "eve")  # too late

    thread = threading.Thread(target=approve_and_race, daemon=True)
    thread.start()
    with fr.run("triage") as run, pytest.raises(ToolDenied, match="rejected.*too much"):
        lookup("o1")
        refund("o1", 600.0)
    thread.join(timeout=10)
    (request,) = captured
    assert (request.run_id, request.run_name, request.tool, request.effect) == (
        run.run_id,
        "triage",
        "refund",
        "irreversible",
    )
    assert request.args == {"order_id": "[REDACTED]", "amount": 600.0}
    assert request.preceding and request.preceding[0]["tool"]["name"] == "lookup"  # type: ignore[index]
    assert request.reason == "irreversible tools need approval"
    final = approvals.get(request.id)
    assert final is not None and (final.state, final.approver) == ("rejected", "bob")
    assert sink.steps[-1].gate is not None and sink.steps[-1].gate.outcome == "rejected"


def test_an_unanswered_request_expires_as_a_denial(tmp_path: Path) -> None:
    policy(tmp_path, '[gate]\napproval_timeout = "0.2s"\n')
    started = time.monotonic()
    with fr.run("triage"), pytest.raises(ToolDenied, match="not approved in time"):
        refund("o1", 600.0)
    assert time.monotonic() - started < 3
    (request,) = [r for r in _all(tmp_path)]
    assert request["state"] == "expired"
    assert not approvals.resolve(request["id"], "approve")


def _all(tmp_path: Path) -> list[Any]:
    import sqlite3
    from contextlib import closing

    # closing(): a connection's own "with" commits but doesn't close.
    with closing(sqlite3.connect(tmp_path / ".fr" / "state.sqlite")) as db:
        db.row_factory = sqlite3.Row
        return list(db.execute("SELECT * FROM approvals"))


def test_waiting_never_blocks_the_event_loop(tmp_path: Path, sink: Sink) -> None:
    policy(tmp_path, '[gate]\napproval_timeout = "5s"\n')

    @fr.tool(effect="irreversible")
    async def charge(amount: float) -> str:
        return "charged"

    ticks: list[int] = []

    async def heartbeat() -> None:
        for i in range(10):
            ticks.append(i)
            await asyncio.sleep(0.02)

    async def approver() -> None:
        deadline = time.monotonic() + 10
        while not approvals.pending():
            assert time.monotonic() < deadline, "no approval was asked"
            await asyncio.sleep(0.02)
        approvals.resolve(approvals.pending()[0].id, "approve", "ada")

    async def agent() -> str:
        async with fr.run("triage"):
            result, _, _ = await asyncio.gather(charge(9.0), heartbeat(), approver())
            return result

    assert asyncio.run(agent()) == "charged"
    assert len(ticks) == 10  # the heartbeat ran while charge waited
    assert sink.steps[0].gate is not None and sink.steps[0].gate.outcome == "approved"


# Row 4.6: what a denial does


def test_a_denial_can_be_returned_to_the_model(tmp_path: Path, sink: Sink) -> None:
    policy(
        tmp_path,
        '[gate]\non_denied = "return_to_model"\n[[gate.rules]]\ntool = "refund"\naction = "deny"\nreason = "refunds are manual"\n',
    )
    with fr.run("triage"):
        result = refund("o1", 5.0)
    assert result == "Denied: refund was denied (rule rules[0]): refunds are manual"
    (step,) = sink.steps
    assert step.status == "denied"
    assert step.tool is not None and step.tool.result == result
    assert step.tool.error is not None and step.tool.error.type.endswith("ToolDenied")


def test_langchain_tools_are_gated_and_can_return_the_denial(
    tmp_path: Path, sink: Sink
) -> None:
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def wire(amount: float) -> str:
        """Send money."""
        return "sent"

    @lc_tool
    def balance() -> str:
        """Read the balance."""
        return "100"

    tools = fr.langchain.wrap_tools([wire, balance], effects={"balance": "read"})
    policy(
        tmp_path,
        '[gate]\non_denied = "return_to_model"\n'
        '[[gate.rules]]\ntool = "wire"\nwhen = "args.amount > 10"\naction = "deny"\n'
        '[[gate.rules]]\ntool = "wire"\naction = "allow"\n',
    )
    with fr.run("triage"):
        assert tools[1].invoke({}) == "100"
        assert tools[0].invoke({"amount": 5}) == "sent"
        assert tools[0].invoke({"amount": 50}).startswith("Denied: wire was denied")
    policy(tmp_path, '[[gate.rules]]\ntool = "wire"\naction = "deny"\n')
    with fr.run("triage"), pytest.raises(ToolDenied):
        tools[0].invoke({"amount": 5})
    assert [s.status for s in sink.steps] == ["ok", "ok", "denied", "denied"]


# Row 4.7: the Python API, the CLI and the terminal prompt


def test_the_cli_approves_and_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    policy(tmp_path, '[gate]\napproval_timeout = "5s"\n')
    monkeypatch.setenv("FR_DIR", str(tmp_path / ".fr"))

    def by_cli(verb: str) -> None:
        until(approvals.pending)
        assert (
            cli.main(
                [verb, approvals.pending()[0].id, "--approver", "ops", "--reason", "ok"]
            )
            == 0
        )

    for verb, expected in (("approve", "refunded 600.0"), ("deny", None)):
        thread = threading.Thread(target=by_cli, args=(verb,), daemon=True)
        thread.start()
        with fr.run("triage"):
            if expected:
                assert refund("o1", 600.0) == expected
            else:
                with pytest.raises(ToolDenied, match="rejected"):
                    refund("o1", 600.0)
        thread.join(timeout=10)
    out = capsys.readouterr().out
    assert "approved refund" in out and "denied refund" in out
    assert cli.main(["approve", "no-such-id"]) == 1
    decided = _all(tmp_path)[0]["id"]
    assert cli.main(["deny", decided]) == 1
    assert "already approved" in capsys.readouterr().err


def test_the_terminal_prompt_asks_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy(tmp_path, '[gate]\nprompt = true\napproval_timeout = "5s"\n')

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty("y\n"))
    monkeypatch.setattr("sys.stderr", Tty())
    with fr.run("triage"):
        assert refund("o1", 600.0) == "refunded 600.0"
    monkeypatch.setattr("sys.stdin", Tty("\n"))
    with fr.run("triage"), pytest.raises(ToolDenied, match="rejected"):
        refund("o1", 600.0)


# Row 4.8: webhooks


SECRET = "whsec-test"


class Hook(BaseHTTPRequestHandler):
    received: ClassVar[list[tuple[dict[str, str], bytes]]] = []
    status: ClassVar[int] = 200

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Hook.received.append(({k.lower(): v for k, v in self.headers.items()}, body))
        self.send_response(Hook.status)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def hook_server() -> Iterator[str]:
    Hook.received, Hook.status = [], 200
    server = HTTPServer(("127.0.0.1", 0), Hook)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/approvals"
    server.shutdown()
    server.server_close()


def test_a_webhook_gets_a_signed_request_that_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook_server: str
) -> None:
    monkeypatch.setenv("FR_WEBHOOK_SECRET", SECRET)
    policy(
        tmp_path,
        f'[gate]\napproval_timeout = "5s"\n[gate.webhook]\nurl = "{hook_server}"\n',
    )

    def approve_from_hook() -> None:
        until(lambda: Hook.received)
        headers, body = Hook.received[0]
        request = json.loads(body)
        assert approvals.get(request["id"]) is not None  # durable before notified
        expected = webhook.sign(SECRET, headers["x-fr-timestamp"], body)
        assert hmac.compare_digest(headers["x-fr-signature"], expected)
        approvals.resolve(request["id"], "approve", "hook")

    thread = threading.Thread(target=approve_from_hook, daemon=True)
    thread.start()
    with fr.run("triage"):
        assert refund("o1", 600.0) == "refunded 600.0"
    thread.join(timeout=10)
    request = json.loads(Hook.received[0][1])
    assert (request["tool"], request["args"], request["state"]) == (
        "refund",
        {"order_id": "o1", "amount": 600.0},
        "pending",
    )


def test_a_failed_delivery_is_logged_and_the_call_expires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hook_server: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("FR_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(webhook, "_ATTEMPTS", 1)
    Hook.status = 500
    policy(
        tmp_path,
        f'[gate]\napproval_timeout = "1s"\n[gate.webhook]\nurl = "{hook_server}"\n',
    )
    with (
        caplog.at_level(logging.WARNING, "flight_recorder"),
        fr.run("triage"),
        pytest.raises(ToolDenied, match="not approved in time"),
    ):
        refund("o1", 600.0)
    assert any("webhook delivery" in r.getMessage() for r in caplog.records)


def call_app(
    app: Any, body: bytes, headers: dict[str, str], method: str = "POST"
) -> tuple[int, dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    scope = {
        "type": "http",
        "method": method,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent[0]["status"], json.loads(sent[1]["body"])


def test_the_callback_app_checks_signatures_and_decides_once(tmp_path: Path) -> None:
    policy(tmp_path, '[gate]\napproval_timeout = "5s"\n')
    app = fr.approvals.asgi_app(SECRET, dir=tmp_path / ".fr")
    request = approvals.create(
        run_id="r1",
        run_name="triage",
        tool="refund",
        effect="irreversible",
        args={},
        preceding=[],
        rule=None,
        reason=None,
        timeout=60,
    )
    body = json.dumps(
        {"id": request.id, "decision": "approve", "approver": "ada"}
    ).encode()
    good = webhook.headers(SECRET, body)
    assert call_app(app, body, {**good, "X-FR-Signature": "sha256=00"})[0] == 401
    stale = str(int(time.time()) - 600)
    old = {
        **good,
        "X-FR-Timestamp": stale,
        "X-FR-Signature": webhook.sign(SECRET, stale, body),
    }
    assert call_app(app, body, old)[0] == 401
    assert call_app(app, body, good, method="GET")[0] == 405
    assert call_app(app, body, good) == (200, {"id": request.id, "decision": "approve"})
    assert call_app(app, body, good)[0] == 409
    final = approvals.get(request.id)
    assert final is not None and (final.state, final.approver) == ("approved", "ada")
    unknown = json.dumps({"id": "nope", "decision": "deny"}).encode()
    assert call_app(app, unknown, webhook.headers(SECRET, unknown))[0] == 404
    bad = b'{"id": 1}'
    assert call_app(app, bad, webhook.headers(SECRET, bad))[0] == 400
    expected = hmac.new(
        SECRET.encode(), good["X-FR-Timestamp"].encode() + b"." + body, hashlib.sha256
    )
    assert good["X-FR-Signature"] == "sha256=" + expected.hexdigest()


# Row 4.9: the audit export


def test_audit_lists_every_irreversible_call_and_its_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    @fr.tool(effect="irreversible")
    def ungated(x: int) -> int:
        return x

    with fr.run("before"):  # no policy yet: recorded, not gated
        pass
    policy(
        tmp_path,
        '[[gate.rules]]\ntool = "refund"\nwhen = "args.amount > 100"\naction = "deny"\n[[gate.rules]]\ntool = "refund"\naction = "allow"\n',
    )
    with fr.run("triage"):
        refund("o1", 5.0)
        lookup("o1")
        with pytest.raises(ToolDenied):
            refund("o2", 600.0)
    fr.configure(dir=tmp_path / ".fr")  # flushes and closes the store
    with fr.run("after"):
        ungated(1)
    fr.configure(dir=tmp_path / ".fr")
    monkeypatch.setenv("FR_DIR", str(tmp_path / ".fr"))
    assert cli.main(["audit", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(r["tool"], r["action"], r["outcome"], r["status"]) for r in rows] == [
        ("refund", "allow", "allowed", "ok"),
        ("lookup", "allow", "allowed", "ok"),
        ("refund", "deny", "denied", "denied"),
        ("ungated", "", "ungated", "ok"),
    ]
    assert rows[2]["rule"] == "rules[0]"
    assert cli.main(["audit"]) == 0
    csv_out = capsys.readouterr().out.splitlines()
    assert csv_out[0].startswith(
        "run_id,run_name,seq,started_at,tool,effect,status,action,outcome"
    )
    assert len(csv_out) == 5
    assert cli.main(["audit", "--since", "2999-01-01"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1


# Row 4.11: the gate fails closed


def test_an_error_in_the_gate_denies_all_but_reads(tmp_path: Path, sink: Sink) -> None:
    policy(tmp_path, '[[gate.rules]]\nwhen = "args.missing > 1"\naction = "allow"\n')
    with fr.run("triage"):
        assert lookup("o1") == "o1 shipped"
        with pytest.raises(ToolDenied, match="the gate failed: KeyError"):
            refund("o1", 5.0)
        with pytest.raises(ToolDenied, match="the gate failed"):
            delete_note("n1")
    assert [s.gate.outcome for s in sink.steps if s.gate] == [
        "allowed",
        "denied",
        "denied",
    ]


def test_an_unusable_approval_store_denies(tmp_path: Path) -> None:
    file = tmp_path / "fr.toml"
    file.write_text('[gate]\napproval_timeout = "5s"\n', encoding="utf-8")
    blocker = tmp_path / ".fr"
    blocker.write_text("a file where the store directory should be")
    fr.configure(dir=blocker, policy=file)
    with fr.run("triage"):
        assert lookup("o1") == "o1 shipped"
        with pytest.raises(ToolDenied, match="the gate failed"):
            refund("o1", 600.0)
