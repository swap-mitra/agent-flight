"""The webhook approval demo (plan row 4.12, the v0.4 exit criterion).

The reference agent updates two customers. The policy makes CRM writes
irreversible and sends each one to a webhook; an approver service, running
locally like a team's own, approves the first and denies the second through
the signed callback. Both decisions end up in ``fr audit``.

CI runs this with FR_DIR set and puts the audit export in the job summary.
"""

import json
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from conftest import ScriptedClaude, chat_model, serve, text, tool_use
from reference_agent import answer, build, heptapod_tools

import flight_recorder as fr
from flight_recorder import cli, recorder
from flight_recorder.approvals import webhook
from flight_recorder.policy import loader
from flight_recorder.store import state
from flight_recorder.store.cassette import CassetteStore

SECRET = "demo-webhook-secret"  # a throwaway, like the mocks' credentials
TASK = "Update Ada's phone, and Grace's too."

POLICY = """
[gate]
approval_timeout = "30s"
on_denied = "return_to_model"   # the agent reads the denial and carries on

[gate.effects]
crm = "irreversible"            # the code says write; the policy is stricter

[gate.webhook]
url = "{url}/hook"

[[gate.rules]]
id = "crm-reads"
tool = "crm"
when = "args.operation in ['list_customers', 'get_customer']"
action = "allow"
"""


def approver(callback: list[str]) -> Any:
    """A team's approver service: it hears about each request on /hook and
    answers through Flight Recorder's callback app, mounted at /callback.
    Ada's change is approved, anyone else's denied."""
    decide = fr.approvals.asgi_app(SECRET)

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"] != "/hook":
            await decide(scope, receive, send)
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        request = json.loads(body)
        ada = request["args"].get("customer_id") == "C-1001"
        answer = {
            "id": request["id"],
            "decision": "approve" if ada else "deny",
            "approver": "support-lead",
            "reason": "verified with the customer"
            if ada
            else "no ticket for this change",
        }
        threading.Thread(target=post, args=(callback[0], answer), daemon=True).start()
        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    return app


def post(url: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers=webhook.headers(SECRET, body), method="POST"
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


def test_one_approved_and_one_denied_call_in_the_audit(
    registry: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(os.environ.get("FR_DEMO_DIR") or tmp_path / "demo")
    monkeypatch.setenv("FR_DIR", str(root))  # the callback app and fr audit use it
    monkeypatch.setenv("FR_WEBHOOK_SECRET", SECRET)
    callback: list[str] = []
    with serve(approver(callback)) as url:
        callback.append(f"{url}/callback")
        file = tmp_path / "fr.toml"
        file.write_text(POLICY.format(url=url), encoding="utf-8")
        store = CassetteStore(root)
        monkeypatch.setattr(recorder, "sinks", [store])
        monkeypatch.setattr(loader, "configured", loader.load(file))
        monkeypatch.setattr(state, "path", root / "state.sqlite")
        script = ScriptedClaude(
            [
                [tool_use("t1", "crm", {"operation": "list_customers"})],
                [
                    tool_use(
                        "t2",
                        "crm",
                        {
                            "operation": "update_customer",
                            "customer_id": "C-1001",
                            "phone": "+1-555-0142",
                        },
                    )
                ],
                [
                    tool_use(
                        "t3",
                        "crm",
                        {
                            "operation": "update_customer",
                            "customer_id": "C-1002",
                            "phone": "+1-555-0199",
                        },
                    )
                ],
                [text("Ada's phone is updated; Grace's change was denied.")],
            ]
        )
        graph = build(chat_model(script), heptapod_tools(registry), record=True)
        with fr.run("gate-demo", inputs={"task": TASK}):
            final = answer(graph, TASK)
        assert store.flush(timeout=10)
        store.close()
    assert final == "Ada's phone is updated; Grace's change was denied."
    results = [r["content"] for r in script.tool_results()]
    assert results[2].startswith("Denied: crm was rejected")
    assert "no ticket for this change" in results[2]

    assert cli.main(["audit", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    decisions = [(r["tool"], r["action"], r["outcome"], r["approver"]) for r in rows]
    assert decisions == [
        ("crm", "allow", "allowed", ""),
        ("crm", "approve", "approved", "support-lead"),
        ("crm", "approve", "rejected", "support-lead"),
    ]
    assert rows[0]["rule"] == "crm-reads"
