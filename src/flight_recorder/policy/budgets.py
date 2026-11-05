"""Budgets: judge a call before it is sent (FR-B1, FR-B2).

A call's projected cost is its input tokens (estimated, or counted by the
provider) plus the output it allows (``max_tokens``), priced from the pricing
table. A budget is judged on what was actually spent so far plus that
projection. Per-run spending is the run's own totals; per-day spending lives in
``state.sqlite`` so several processes share one budget. A budget in USD is
skipped for a call whose cost is unknown (an unpriced model), since cost is
never guessed.
"""

import logging
import math
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime

from flight_recorder.errors import BudgetExceeded
from flight_recorder.model import ModelCall, Usage
from flight_recorder.policy import pricing
from flight_recorder.policy.loader import Budgets, Limits, provider_of, zone
from flight_recorder.recorder import ActiveRun
from flight_recorder.store import state

logger = logging.getLogger("flight_recorder")


@dataclass(frozen=True)
class Projection:
    input_tokens: int
    counted: bool  # by the provider; else estimated from characters
    max_tokens: int  # 0 when the request doesn't cap its output

    def describe(self) -> str:
        how = "counted by the provider" if self.counted else "estimated"
        return (
            f"{self.input_tokens} input tokens ({how}) + {self.max_tokens} max output"
        )


def estimate(text: str) -> int:
    """Input tokens from characters: about four per token in English."""
    return math.ceil(len(text) / 4)


def check(
    budgets: Budgets,
    active: ActiveRun,
    provider: str,
    model: str,
    projection: Projection,
    daily: bool,
) -> str:
    """The model to send: ``model``, or the downgrade on a breach.

    Raises BudgetExceeded when the call must not be sent. ``daily`` is False
    while replaying, so a replay neither reads nor spends the daily budget.
    """
    breach = _breach(budgets, active, provider, model, projection, daily)
    if breach is None:
        return model
    if budgets.on_breach == "warn":
        # Once per budget, not per message: the amounts grow with each call.
        label = breach.partition(" would be exceeded")[0]
        active.warn_once(f"budget:{label}", "%s; sending anyway", breach)
        return model
    if budgets.on_breach == "downgrade":
        smaller = _downgrade(budgets, provider)
        if smaller is not None and smaller != model:
            again = _breach(budgets, active, provider, smaller, projection, daily)
            if again is None:
                active.warn_once(
                    f"downgrade:{model}", "%s; sent to %s instead", breach, smaller
                )
                return smaller
            breach = f"{breach}, and on {smaller}: {again}"
    raise BudgetExceeded(f"{breach}; the call to {provider}/{model} was not sent")


def spend(budgets: Budgets, run_name: str, call: ModelCall) -> None:
    """Add a sent call's actual usage to today's counters, atomically.

    ponytail: the check before a call reads the counters and this adds after
    it, so processes calling at once can each pass and overshoot a daily
    limit by one call apiece. A reservation per call would close that.
    """
    scopes = _daily_scopes(budgets, run_name)
    if not scopes:
        return
    usage = call.usage
    tokens = usage.input_tokens + usage.output_tokens if usage is not None else 0
    unknown = 1 if call.cost_usd is None else 0
    day = _today(budgets)
    with closing(state.connect()) as db, db:
        for scope in scopes:
            db.execute(
                "INSERT INTO budget_day (day, scope, usd, tokens, unpriced)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT (day, scope) DO UPDATE SET"
                " usd = usd + excluded.usd, tokens = tokens + excluded.tokens,"
                " unpriced = unpriced + excluded.unpriced",
                (day, scope, call.cost_usd or 0.0, tokens, unknown),
            )


def _breach(
    budgets: Budgets,
    active: ActiveRun,
    provider: str,
    model: str,
    projection: Projection,
    daily: bool,
) -> str | None:
    """Why sending this call breaks a budget, or None; warns near a limit."""
    proj_tokens = projection.input_tokens + projection.max_tokens
    usage = Usage(
        input_tokens=projection.input_tokens, output_tokens=projection.max_tokens
    )
    proj_usd = pricing.cost(provider, model, usage)
    totals = active.run.totals
    run_tokens = totals.input_tokens + totals.output_tokens
    name = active.run.name
    named = budgets.by_run_name.get(name)
    # (label, limit, spent, projected), for every limit that applies.
    checks: list[tuple[str, float | None, float | None, float | None]] = []
    for label, limits in (("", budgets), (f" for {name!r}", named)):
        if limits is None:
            continue
        checks += [
            (
                f"per-run USD budget{label}",
                limits.per_run_usd,
                totals.cost_usd,
                proj_usd,
            ),
            (
                f"per-run token budget{label}",
                limits.per_run_tokens,
                run_tokens,
                proj_tokens,
            ),
        ]
    if daily:
        for scope, limits in _daily_scopes(budgets, name).items():
            usd, tokens = _spent_today(budgets, scope)
            label = "" if scope == "*" else f" for {name!r}"
            checks += [
                (f"per-day USD budget{label}", limits.per_day_usd, usd, proj_usd),
                (
                    f"per-day token budget{label}",
                    limits.per_day_tokens,
                    tokens,
                    proj_tokens,
                ),
            ]
    for label, limit, spent, projected in checks:
        if limit is None or spent is None or projected is None:
            continue
        after = spent + projected
        if after > limit:
            return (
                f"{label} {_amount(label, limit)} would be exceeded:"
                f" {_amount(label, spent)} spent, this call up to"
                f" {_amount(label, projected)} ({projection.describe()})"
            )
        if after >= budgets.soft_warning * limit:
            active.warn_once(
                f"soft:{label}",
                "%s at %d%%: %s of %s, counting this call",
                label,
                round(after / limit * 100),
                _amount(label, after),
                _amount(label, limit),
            )
    return None


def _daily_scopes(budgets: Budgets, run_name: str) -> dict[str, Limits]:
    """The daily counters that apply to a run: all runs, and its own name's."""
    scopes: dict[str, Limits] = {}
    if budgets.per_day_usd is not None or budgets.per_day_tokens is not None:
        scopes["*"] = budgets
    named = budgets.by_run_name.get(run_name)
    if named is not None and (
        named.per_day_usd is not None or named.per_day_tokens is not None
    ):
        scopes[f"run:{run_name}"] = named
    return scopes


def _spent_today(budgets: Budgets, scope: str) -> tuple[float | None, int]:
    with closing(state.connect()) as db:
        row = db.execute(
            "SELECT usd, tokens, unpriced FROM budget_day WHERE day = ? AND scope = ?",
            (_today(budgets), scope),
        ).fetchone()
    if row is None:
        return 0.0, 0
    usd, tokens, unpriced = row
    return (None if unpriced else usd), tokens


def _today(budgets: Budgets) -> str:
    return datetime.now(zone(budgets.timezone)).date().isoformat()


def _downgrade(budgets: Budgets, provider: str) -> str | None:
    target = budgets.downgrade_to
    if isinstance(target, dict):
        return target.get(provider)
    if target is None:
        return None
    owner, model = provider_of(target)
    return model if owner == provider else None


def _amount(label: str, value: float) -> str:
    return f"{value:,.0f} tokens" if "token" in label else f"${value:.4f}"
