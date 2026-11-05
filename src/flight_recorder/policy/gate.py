"""The action gate: no irreversible tool call runs unchecked (FR-G1 to G9).

Before a tool runs, the policy's gate rules are evaluated in order; the first
match wins. With no match, read and write calls are allowed and irreversible
calls need approval. A call that needs approval waits for a person (CLI,
Python API, webhook); expiry denies it.

The gate fails closed: an error while evaluating it, or an unusable approval
store, denies the call unless the tool only reads. With no ``[gate]`` section
in the policy, tool calls are not gated at all.
"""

import fnmatch
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import JsonValue

from flight_recorder import approvals, redact
from flight_recorder.errors import ToolDenied
from flight_recorder.model import Effect, GateDecision
from flight_recorder.policy import conditions, loader
from flight_recorder.policy.loader import Gate
from flight_recorder.recorder import ActiveRun
from flight_recorder.replay import engine as replay

Action = Literal["allow", "approve", "deny"]


@dataclass(frozen=True)
class GateCall:
    """What a ``when_fn`` rule is called with."""

    tool: str
    effect: Effect
    args: dict[str, JsonValue]
    run_name: str
    run_id: str
    env: str


@dataclass
class Verdict:
    """The gate's answer for one call, for the interceptor to act on."""

    decision: GateDecision
    effect: Effect

    @property
    def allowed(self) -> bool:
        return self.decision.outcome in ("allowed", "approved")

    def denial(self, tool: str, on_denied: str) -> tuple[ToolDenied, str | None]:
        """The exception to raise, and the text to return to the model instead
        when the policy says so."""
        d = self.decision
        by = f"rule {d.rule}" if d.rule else "the gate's default"
        what = {"rejected": "rejected", "expired": "not approved in time"}.get(
            d.outcome, "denied"
        )
        message = f"{tool} was {what} ({by})" + (f": {d.reason}" if d.reason else "")
        returned = f"Denied: {message}" if on_denied == "return_to_model" else None
        return ToolDenied(message, d), returned


def gate() -> Gate | None:
    policy = loader.current()
    return policy.gate if policy is not None else None


def effect_of(tool: str, declared: Effect | None) -> Effect:
    """A tool's class: the policy's ``[gate.effects]``, else the code's, else
    the gate's default (irreversible), so an unclassified tool is gated."""
    config = gate()
    if config is not None and tool in config.effects:
        return config.effects[tool]
    if declared is not None:
        return declared
    return config.default_effect if config is not None else "irreversible"


def check(
    active: ActiveRun, tool: str, effect: Effect, args: dict[str, JsonValue]
) -> Verdict | None:
    """Gate a live call, waiting for approval if it needs one. None: no gate."""
    config = gate()
    if config is None:
        return None
    try:
        action, rule, reason = _rules(config, active, tool, effect, args)
        if action != "approve":
            return _final(action, rule, reason, effect)
        if replay.replaying_now():
            return _final(
                "approve", rule, "approval needed during a replay", effect, "denied"
            )
        request = _ask(config, active, tool, effect, args, rule, reason)
        ask = approvals.prompt if config.prompt else None
        return _decided(approvals.wait(request, ask), effect)
    except Exception as exc:  # noqa: BLE001  the gate fails closed (ADR-006)
        return _failed(effect, exc)


async def acheck(
    active: ActiveRun, tool: str, effect: Effect, args: dict[str, JsonValue]
) -> Verdict | None:
    """``check`` for async tools: waiting never blocks the event loop (FR-G8)."""
    config = gate()
    if config is None:
        return None
    try:
        action, rule, reason = _rules(config, active, tool, effect, args)
        if action != "approve":
            return _final(action, rule, reason, effect)
        if replay.replaying_now():
            return _final(
                "approve", rule, "approval needed during a replay", effect, "denied"
            )
        request = _ask(config, active, tool, effect, args, rule, reason)
        ask = approvals.prompt if config.prompt else None
        return _decided(await approvals.await_decision(request, ask), effect)
    except Exception as exc:  # noqa: BLE001  the gate fails closed (ADR-006)
        return _failed(effect, exc)


def replayed(
    active: ActiveRun,
    tool: str,
    effect: Effect,
    args: dict[str, JsonValue],
    recorded: GateDecision | None,
) -> GateDecision | None:
    """In replay, the recorded decision stands (no one approves in CI), but the
    action is what the current rules say, so a rule change shows in the diff
    (FR-G9)."""
    config = gate()
    if config is None:
        return None
    try:
        action, rule, reason = _rules(config, active, tool, effect, args)
    except Exception as exc:  # noqa: BLE001  the gate fails closed (ADR-006)
        return _failed(effect, exc).decision
    if recorded is not None:
        outcome = recorded.outcome
    else:
        outcome = "allowed" if action == "allow" else "denied"
    return GateDecision(
        action=action,
        outcome=outcome,
        rule=rule,
        reason=reason,
        approver=recorded.approver if recorded is not None else None,
        decided_at=_now(),
    )


def _rules(
    config: Gate,
    active: ActiveRun,
    tool: str,
    effect: Effect,
    args: dict[str, JsonValue],
) -> tuple[Action, str | None, str | None]:
    """(action, rule, reason) from the first matching rule, or the default."""
    env = os.environ.get("FR_ENV") or "dev"
    run = active.run
    names = {
        "args": args,
        "run": {"name": run.name, "id": run.run_id},
        "env": {"name": env},
    }
    for i, rule in enumerate(config.rules):
        if (
            fnmatch.fnmatchcase(tool, rule.tool)
            and (rule.effect is None or rule.effect == effect)
            and (rule.env is None or rule.env == env)
            and (rule.when is None or conditions.evaluate(rule.when, names))
            and (
                rule.when_fn is None
                or loader.resolve(rule.when_fn)(
                    GateCall(tool, effect, args, run.name, run.run_id, env)
                )
            )
        ):
            return rule.action, rule.id or f"rules[{i}]", rule.reason
    if effect == "irreversible":
        return "approve", None, "irreversible tools need approval"
    return "allow", None, None


def _ask(
    config: Gate,
    active: ActiveRun,
    tool: str,
    effect: Effect,
    args: dict[str, JsonValue],
    rule: str | None,
    reason: str | None,
) -> approvals.Request:
    # Redacted like everything persisted; written before anyone is notified.
    shown: Any = redact.apply({"kind": "tool", "tool": {"name": tool, "args": args}})
    request = approvals.create(
        run_id=active.run.run_id,
        run_name=active.run.name,
        tool=tool,
        effect=effect,
        args=shown["tool"]["args"],
        preceding=active.recent_steps(),
        rule=rule,
        reason=reason,
        timeout=config.approval_timeout,
    )
    if config.webhook is not None:
        from flight_recorder.approvals import webhook

        hook = config.webhook
        webhook.notify(request, hook.url, hook.secret_env, hook.timeout)
    return request


def _decided(request: approvals.Request, effect: Effect) -> Verdict:
    outcome = {"approved": "approved", "rejected": "rejected"}.get(
        request.state, "expired"
    )
    return Verdict(
        GateDecision(
            action="approve",
            outcome=outcome,  # type: ignore[arg-type]
            rule=request.rule,
            reason=request.reason,
            approver=request.approver,
            decided_at=_now(),
            approval_id=request.id,
        ),
        effect,
    )


def _final(
    action: Action,
    rule: str | None,
    reason: str | None,
    effect: Effect,
    outcome: str | None = None,
) -> Verdict:
    outcome = outcome or ("allowed" if action == "allow" else "denied")
    return Verdict(
        GateDecision(
            action=action,
            outcome=outcome,  # type: ignore[arg-type]
            rule=rule,
            reason=reason,
            decided_at=_now(),
        ),
        effect,
    )


def _failed(effect: Effect, exc: Exception) -> Verdict:
    """A failure inside the gate: reads go ahead, anything else is denied."""
    reason = f"the gate failed: {type(exc).__name__}: {exc}"
    if effect == "read":
        return _final("allow", None, reason, effect)
    return _final("deny", None, reason, effect)


def _now() -> datetime:
    return datetime.now(UTC)


def denial(verdict: Verdict, tool: str) -> tuple[ToolDenied, str | None]:
    """The ToolDenied for a denied call, and the text to return to the model
    instead of raising it when ``on_denied = "return_to_model"``."""
    config = gate()
    return verdict.denial(tool, config.on_denied if config else "raise")
