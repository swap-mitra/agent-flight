"""Approvals: a paused tool call waits here for a person (FR-G3 to G5, G8).

A request is written to ``state.sqlite`` before anyone is told about it, so a
notification never names a request that doesn't exist. It is decided once:
resolution is a single ``UPDATE ... WHERE state = 'pending'``, so two
approvers racing can't both win. A request not decided by its deadline
expires, which counts as a denial.

    fr.approvals.pending()                      # what is waiting
    fr.approvals.resolve(id, "approve", approver="ada", reason="known customer")
"""

import asyncio
import getpass
import json
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import JsonValue

from flight_recorder.store import state

State = Literal["pending", "approved", "rejected", "expired"]
Decision = Literal["approve", "deny"]

# Polling for a decision: from 50 ms, doubling to at most 1 s.
_FIRST_POLL, _MAX_POLL = 0.05, 1.0


@dataclass(frozen=True)
class Request:
    """An approval request, as stored and as sent to a webhook."""

    id: str
    state: State
    created_at: str
    expires_at: str
    run_id: str
    run_name: str
    tool: str
    effect: str
    args: dict[str, JsonValue]
    preceding: list[JsonValue]
    rule: str | None
    reason: str | None
    approver: str | None
    decided_at: str | None

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "state": self.state,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "tool": self.tool,
            "effect": self.effect,
            "args": self.args,
            "preceding": self.preceding,
            "rule": self.rule,
            "reason": self.reason,
        }


def create(
    *,
    run_id: str,
    run_name: str,
    tool: str,
    effect: str,
    args: dict[str, JsonValue],
    preceding: list[JsonValue],
    rule: str | None,
    reason: str | None,
    timeout: float,
) -> Request:
    """Write a pending request; it is durable when this returns."""
    now = datetime.now(UTC)
    values = (
        str(uuid.uuid4()),
        "pending",
        _iso(now),
        _iso(now + timedelta(seconds=timeout)),
        run_id,
        run_name,
        tool,
        effect,
        json.dumps(args),
        json.dumps(preceding),
        rule,
        reason,
    )
    with closing(state.connect()) as db, db:
        db.execute(
            "INSERT INTO approvals (id, state, created_at, expires_at, run_id,"
            " run_name, tool, effect, args, preceding, rule, reason)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
    found = get(values[0])
    assert found is not None
    return found


def get(request_id: str, *, at: Path | None = None) -> Request | None:
    with closing(state.connect(at)) as db:
        row = db.execute(
            "SELECT * FROM approvals WHERE id = ?", (request_id,)
        ).fetchone()
    return None if row is None else _request(row)


def pending(*, at: Path | None = None) -> list[Request]:
    """Requests still waiting for a decision, oldest first."""
    now = _iso(datetime.now(UTC))
    with closing(state.connect(at)) as db:
        rows = db.execute(
            "SELECT * FROM approvals WHERE state = 'pending' AND expires_at > ?"
            " ORDER BY created_at",
            (now,),
        ).fetchall()
    return [_request(r) for r in rows]


def resolve(
    request_id: str,
    decision: Decision,
    approver: str | None = None,
    reason: str | None = None,
    *,
    at: Path | None = None,
) -> bool:
    """Approve or deny a pending request. False if it was already decided,
    has expired, or doesn't exist; the first decision always stands."""
    if decision not in ("approve", "deny"):
        raise ValueError(f"decision must be 'approve' or 'deny', not {decision!r}")
    now = _iso(datetime.now(UTC))
    with closing(state.connect(at)) as db, db:
        changed = db.execute(
            "UPDATE approvals SET state = ?, approver = ?, reason = ?, decided_at = ?"
            " WHERE id = ? AND state = 'pending' AND expires_at > ?",
            (
                "approved" if decision == "approve" else "rejected",
                approver or getpass.getuser(),
                reason,
                now,
                request_id,
                now,
            ),
        ).rowcount
    return changed == 1


def wait(request: Request, ask: Callable[[Request], None] | None = None) -> Request:
    """Block until ``request`` is decided or expires (expiry is stored first)."""
    if ask is not None:
        ask(request)
    delay = _FIRST_POLL
    while True:
        current = _settled(request.id)
        if current is not None:
            return current
        time.sleep(min(delay, _left(request)))
        delay = min(delay * 2, _MAX_POLL)


async def await_decision(
    request: Request, ask: Callable[[Request], None] | None = None
) -> Request:
    """``wait`` for async agents: polls without blocking the event loop."""
    if ask is not None:
        await asyncio.to_thread(ask, request)
    delay = _FIRST_POLL
    while True:
        current = _settled(request.id)
        if current is not None:
            return current
        await asyncio.sleep(min(delay, _left(request)))
        delay = min(delay * 2, _MAX_POLL)


def prompt(request: Request) -> None:
    """Ask on the terminal; does nothing without one (another channel decides)."""
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return
    print(
        f"\nApprove {request.effect} call {request.tool}"
        f"({json.dumps(request.args)}) in run {request.run_name!r}?"
        f" {request.reason or ''}\n[approval {request.id}] y/N: ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    answer = sys.stdin.readline().strip().lower()
    resolve(
        request.id, "approve" if answer in ("y", "yes") else "deny", reason="terminal"
    )


def _settled(request_id: str) -> Request | None:
    """The request once decided; a pending one past its deadline is expired
    here, unless a decision lands first."""
    now = _iso(datetime.now(UTC))
    with closing(state.connect()) as db, db:
        db.execute(
            "UPDATE approvals SET state = 'expired', decided_at = ?"
            " WHERE id = ? AND state = 'pending' AND expires_at <= ?",
            (now, request_id, now),
        )
        row = db.execute(
            "SELECT * FROM approvals WHERE id = ?", (request_id,)
        ).fetchone()
    if row is None:
        raise LookupError(f"approval {request_id} is missing from the state store")
    return None if row["state"] == "pending" else _request(row)


def _iso(moment: datetime) -> str:
    # Fixed width, so stored times compare correctly as strings.
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _left(request: Request) -> float:
    deadline = datetime.strptime(request.expires_at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=UTC
    )
    return max((deadline - datetime.now(UTC)).total_seconds(), 0.0) + 0.001


def _request(row: Any) -> Request:
    return Request(
        id=row["id"],
        state=row["state"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        run_id=row["run_id"],
        run_name=row["run_name"],
        tool=row["tool"],
        effect=row["effect"],
        args=json.loads(row["args"]),
        preceding=json.loads(row["preceding"]),
        rule=row["rule"],
        reason=row["reason"],
        approver=row["approver"],
        decided_at=row["decided_at"],
    )


def asgi_app(secret: str | bytes, *, dir: str | Path | None = None) -> Any:
    """The webhook callback, as an ASGI app to mount in your own service.

    See ``flight_recorder.approvals.webhook``.
    """
    from flight_recorder.approvals.webhook import app

    return app(secret, dir=dir)


__all__ = ["Request", "asgi_app", "get", "pending", "resolve"]
