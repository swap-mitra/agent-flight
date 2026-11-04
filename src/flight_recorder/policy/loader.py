"""The policy file: budgets, routes and fallbacks (FR-B1 to B3, B5, B6).

``fr.configure(policy="fr.toml")`` loads it once. Anything invalid raises
``PolicyError`` there, never mid-run. Every model the file names belongs to one
provider: written ``provider/model``, or bare when the pricing table knows it.
A rule only ever moves a call to a model of the call's own provider; routing
across providers needs request translation and is not supported (FR-B6).
"""

import hashlib
import os
import tomllib
from datetime import UTC, tzinfo
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from flight_recorder.errors import PolicyError
from flight_recorder.policy import pricing

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


class Policy(_Strict):
    budgets: Budgets | None = None
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
