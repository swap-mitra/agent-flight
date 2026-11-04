"""Routing, budgets and fallbacks around one model call (FR-B2, B3, B5).

Routing runs first, then the budget check on the routed model, so a budget
judges the call that will actually be sent. Both fail open: an error inside
them sends the original request unchanged and warns once per run. Only a
budget's own denial (``BudgetExceeded``) stops a call.
"""

import json
from collections.abc import Sequence
from typing import Any

from flight_recorder.errors import BudgetExceeded
from flight_recorder.model import Fallback, ModelCall
from flight_recorder.policy import budgets, loader
from flight_recorder.policy.loader import Policy, provider_of
from flight_recorder.recorder import ActiveRun, error_info
from flight_recorder.replay import engine as replay

# Request fields that cap a call's output, by API.
MAX_TOKENS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


class Plan:
    """What the policy decided for one call, carried by its Request."""

    def __init__(self, policy: Policy, projection: budgets.Projection) -> None:
        self.policy = policy
        self.projection = projection
        self.chain: list[str] = []  # fallbacks not tried yet
        self.fallbacks: list[Fallback] = []  # attempts that failed


def wants_count() -> bool:
    """True when budgets ask the provider to count input tokens."""
    policy = loader.current()
    return (
        policy is not None
        and policy.budgets is not None
        and policy.budgets.token_count == "provider"
        and not replay.replaying_now()  # a replay may have no network
    )


def before(
    active: ActiveRun,
    request: Any,
    tags: Sequence[str],
    counted: int | None,
) -> None:
    """Route ``request`` and check its budget; may raise BudgetExceeded.

    ``counted`` is the provider's input token count, when it was asked.
    """
    policy = loader.current()
    if policy is None:
        return
    try:
        text = json.dumps([request.messages, request.params], default=str)
        projection = budgets.Projection(
            input_tokens=counted if counted is not None else budgets.estimate(text),
            counted=counted is not None,
            max_tokens=_max_tokens(request.params),
        )
        plan = Plan(policy, projection)
        model = route(policy, request.provider, request.model, tags, active, projection)
        model = _checked(active, plan, request.provider, model)
        plan.chain = _chain(policy, request.provider, model)
        if model != request.model:
            request.send_to(model)
        request.plan = plan
    except BudgetExceeded:
        active.denied = True
        raise
    except Exception as exc:  # noqa: BLE001  routing and budgets fail open
        _failed(active, exc)


def fall_back(active: ActiveRun, request: Any, exc: BaseException) -> bool:
    """After ``exc``, move ``request`` to its next fallback model; False if
    there is none and ``exc`` should reach the caller. May raise
    BudgetExceeded when the fallback model would break a budget."""
    plan: Plan | None = getattr(request, "plan", None)
    if plan is None or not plan.chain or not retryable(exc):
        return False
    try:
        failed = request.kwargs.get("model", request.model)
        model = plan.chain.pop(0)
        plan.fallbacks.append(Fallback(model=str(failed), error=error_info(exc)))
        model = _checked(active, plan, request.provider, model)
        request.send_to(model)
    except BudgetExceeded:
        active.denied = True
        raise
    except Exception as failure:  # noqa: BLE001  routing and budgets fail open
        _failed(active, failure)
        return False
    return True


def after(active: ActiveRun, call: ModelCall) -> None:
    """Count a sent call against today's budgets."""
    policy = loader.current()
    if policy is None or policy.budgets is None or replay.replaying_now():
        return
    if call.error is not None and call.usage is None and call.cost_usd == 0.0:
        return  # rejected or never sent: nothing was spent
    try:
        budgets.spend(policy.budgets, active.run.name, call)
    except Exception as exc:  # noqa: BLE001  routing and budgets fail open
        _failed(active, exc)


def route(
    policy: Policy,
    provider: str,
    model: str,
    tags: Sequence[str],
    active: ActiveRun,
    projection: budgets.Projection,
) -> str:
    """The first matching route's model, or ``model``.

    A route applies only to calls of its own model's provider (FR-B6).
    """
    for rule in policy.routes:
        owner, target = provider_of(rule.model)
        match = rule.match
        if (
            owner == provider
            and (match.tag is None or match.tag in tags)
            and (match.run_name is None or match.run_name == active.run.name)
            and (
                match.min_input_tokens is None
                or projection.input_tokens >= match.min_input_tokens
            )
        ):
            return target
    return model


def retryable(exc: BaseException) -> bool:
    """A rate limit, a server error or a timeout, after the SDK's own retries."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(exc).__name__ == "APITimeoutError"


def _checked(active: ActiveRun, plan: Plan, provider: str, model: str) -> str:
    if plan.policy.budgets is None:
        return model
    daily = not replay.replaying_now()
    return budgets.check(
        plan.policy.budgets, active, provider, model, plan.projection, daily
    )


def _chain(policy: Policy, provider: str, model: str) -> list[str]:
    for source, chain in policy.fallback.items():
        if provider_of(source) == (provider, model):
            return [provider_of(m)[1] for m in chain]
    return []


def _max_tokens(params: dict[str, Any]) -> int:
    for name in MAX_TOKENS:
        value = params.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _failed(active: ActiveRun, exc: Exception) -> None:
    active.warn_once(
        "policy",
        "budget or routing failed for run %s; the request was sent unchanged",
        active.run.run_id,
        exc_info=exc,
    )
