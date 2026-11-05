"""The policy file: budgets, routes, fallbacks and the gate (FR-B1 to B6, FR-G1, G2).

``fr.configure(policy="fr.toml")`` loads it once. Anything invalid raises
``PolicyError`` there, never mid-run. Every model the file names belongs to one
provider: written ``provider/model``, or bare when the pricing table knows it.
A rule only ever moves a call to a model of the call's own provider; routing
across providers needs request translation and is not supported (FR-B6).
"""

import hashlib
import importlib
import os
import re
import tomllib
from collections.abc import Callable
from datetime import UTC, tzinfo
from typing import Any, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from flight_recorder.errors import PolicyError
from flight_recorder.model import Effect
from flight_recorder.policy import conditions, pricing

PROVIDERS = ("openai", "anthropic")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Limits(_Strict):
    """Hard limits; a field left out is no limit."""

    per_run_usd: float | None = Field(default=None, gt=0)
    per_run_tokens: int | None = Field(default=None, gt=0)
    per_day_usd: float | None = Field(default=None, gt=0)
    per_day_tokens: int | None = Field(default=None, gt=0)


class Budgets(Limits):
    soft_warning: float = Field(default=0.8, gt=0, le=1)
    on_breach: Literal["warn", "downgrade", "deny"] = "deny"
    # One model, or one per provider; a call is downgraded to its provider's.
    downgrade_to: str | dict[str, str] | None = None
    by_run_name: dict[str, Limits] = {}
    timezone: str = "UTC"  # where a day starts, for per_day limits
    # estimate: characters / 4, no extra request. provider: the provider's
    # token-counting endpoint where it has one (exact, one more round trip).
    token_count: Literal["estimate", "provider"] = "estimate"

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.on_breach == "downgrade" and not self.downgrade_to:
            raise ValueError('on_breach = "downgrade" needs downgrade_to')
        zone(self.timezone)
        return self


def zone(name: str) -> tzinfo:
    """The time zone ``name``; UTC needs no zone database (Windows has none)."""
    if name == "UTC":
        return UTC
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"unknown timezone {name!r} (on Windows, pip install tzdata)"
        ) from exc


class Match(_Strict):
    """Every condition given must hold; none given matches every call."""

    tag: str | None = None
    run_name: str | None = None
    min_input_tokens: int | None = Field(default=None, ge=0)


class Route(_Strict):
    match: Match = Match()
    model: str


class GateRule(_Strict):
    """One gate rule; every condition given must hold. First match wins."""

    id: str | None = None  # how the rule is named in decisions; else rules[i]
    tool: str = "*"  # a tool name, or a glob such as delete_*
    effect: Effect | None = None
    env: str | None = None  # the deployment name, from FR_ENV (default dev)
    when: str | None = None  # a condition, e.g. "args.amount > 500"
    when_fn: str | None = None  # "module:function", called with a GateCall
    action: Literal["allow", "approve", "deny"]
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.when is not None:
            conditions.parse(self.when)
        if self.when_fn is not None:
            resolve(self.when_fn)
        return self


class Webhook(_Strict):
    url: str
    # The environment variable holding the HMAC secret; never the secret.
    secret_env: str = "FR_WEBHOOK_SECRET"
    timeout: float = Field(default=10, gt=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.url.startswith(("https://", "http://")):
            raise ValueError(f"webhook url must be http(s): {self.url!r}")
        if not os.environ.get(self.secret_env):
            raise ValueError(f"webhook secret: set the {self.secret_env} variable")
        return self


class Gate(_Strict):
    # The class of a tool whose code doesn't declare one.
    default_effect: Effect = "irreversible"
    effects: dict[str, Effect] = {}  # by tool name; wins over the code's
    approval_timeout: float = 900.0  # seconds; written "15m", "30s", "2h"
    on_denied: Literal["raise", "return_to_model"] = "raise"
    prompt: bool = False  # ask on the terminal, when one is attached
    webhook: Webhook | None = None
    rules: list[GateRule] = []

    @field_validator("approval_timeout", mode="before")
    @classmethod
    def _duration(cls, value: object) -> object:
        if isinstance(value, str):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", value)
            if match is None:
                raise ValueError(f"approval_timeout {value!r}: write 30s, 15m or 2h")
            return float(match[1]) * {"s": 1, "m": 60, "h": 3600}[match[2]]
        return value

    @field_validator("approval_timeout")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("approval_timeout must be positive")
        return value


def resolve(path: str) -> Callable[..., Any]:
    """The function at ``module:name``; ValueError if it can't be imported."""
    module, sep, name = path.partition(":")
    if not sep or not module or not name:
        raise ValueError(f"when_fn {path!r}: write it as module:function")
    try:
        found = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"when_fn {path!r} can't be imported: {exc}") from exc
    if not callable(found):
        raise ValueError(f"when_fn {path!r} is not callable")  # noqa: TRY004
    return found


class Policy(_Strict):
    budgets: Budgets | None = None
    gate: Gate | None = None  # no gate section: tool calls are not gated
    routes: list[Route] = []  # first match wins
    # A model's fallback chain, tried in order on rate limits, server errors
    # and timeouts.
    fallback: dict[str, list[str]] = {}
    # Not in the file: set by load().
    hash: str = ""


def provider_of(model: str, table: pricing.Table | None = None) -> tuple[str, str]:
    """(provider, model id) of a model the policy names; ValueError if unknown."""
    provider, sep, rest = model.partition("/")
    if sep and provider in PROVIDERS:
        return provider, rest
    prices = pricing.table if table is None else table
    known = [p for p in PROVIDERS if model in prices.get(p, {})]
    if len(known) != 1:
        raise ValueError(
            f"say which provider model {model!r} is from: write it as"
            f" {' or '.join(f'{p}/{model}' for p in PROVIDERS)}"
        )
    return known[0], model


def parse(text: str, source: str, table: pricing.Table) -> Policy:
    """Parse a policy file; PolicyError names ``source`` and what is wrong."""
    try:
        policy = Policy.model_validate(tomllib.loads(text))
        _check_models(policy, table)
    except (tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise PolicyError(f"invalid policy file {source}: {exc}") from exc
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return policy.model_copy(update={"hash": digest})


def load(
    path: str | os.PathLike[str] | None, table: pricing.Table | None = None
) -> Policy | None:
    """The policy at ``path`` (None for none); bare model ids resolve in ``table``."""
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise PolicyError(f"cannot read policy file {os.fspath(path)}: {exc}") from exc
    return parse(text, os.fspath(path), pricing.table if table is None else table)


def _check_models(policy: Policy, table: pricing.Table) -> None:
    def provider(model: str) -> str:
        return provider_of(model, table)[0]

    for route in policy.routes:
        provider(route.model)
    target = policy.budgets.downgrade_to if policy.budgets is not None else None
    if isinstance(target, str):
        provider(target)
    elif target is not None and (unknown := target.keys() - set(PROVIDERS)):
        raise ValueError(f"downgrade_to: unknown providers {sorted(unknown)}")
    for model, chain in policy.fallback.items():
        source = provider(model)
        for fallback in chain:
            if provider(fallback) != source:
                raise ValueError(
                    f"fallback {model!r} -> {fallback!r} crosses providers;"
                    " v0.3 routes within one provider only (FR-B6)"
                )


# Set by fr.configure(); the pytest plugin's --fr-policy overrides it.
configured: Policy | None = None
override: Policy | None = None


def current() -> Policy | None:
    return override if override is not None else configured
