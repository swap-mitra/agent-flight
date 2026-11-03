"""The pricing table and the cost of one model call (FR-R2, FR-B8).

Cost is never guessed. A model missing from the table, or a call that used a
rate or modifier its entry does not price, records cost null and warns once.
"""

import logging
import os
import re
import tomllib
from importlib import resources

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from flight_recorder.errors import PricingError
from flight_recorder.model import Usage

logger = logging.getLogger("flight_recorder")


class Rates(BaseModel):
    """USD per million tokens; None where the provider has no such rate."""

    model_config = ConfigDict(extra="forbid")

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    cache_write_1h: float | None = None


class LongContext(Rates):
    above: int  # prompt tokens, cached ones included


class Entry(Rates):
    long_context: LongContext | None = None


Table = dict[str, dict[str, Entry]]  # provider -> model id -> entry

_TABLE = TypeAdapter(Table)
# A dated snapshot id falls back to its undated key: -20250929, -2024-08-06.
_DATE = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")


def parse(text: str, source: str) -> Table:
    """Parse a pricing file; PricingError names ``source`` if it is invalid."""
    try:
        return _TABLE.validate_python(tomllib.loads(text))
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise PricingError(f"invalid pricing file {source}: {exc}") from exc


SHIPPED = parse(
    resources.files("flight_recorder.policy")
    .joinpath("pricing.toml")
    .read_text("utf-8"),
    "pricing.toml",
)


def load(path: str | os.PathLike[str] | None) -> Table:
    """The shipped table, with each entry in the file at ``path`` replacing its own."""
    if path is None:
        return SHIPPED
    with open(path, encoding="utf-8") as f:
        override = parse(f.read(), os.fspath(path))
    return {
        provider: {**SHIPPED.get(provider, {}), **override.get(provider, {})}
        for provider in SHIPPED.keys() | override.keys()
    }


# Set by fr.configure().
table: Table = SHIPPED

_warned: set[str] = set()


def cost(
    provider: str,
    model: str,
    usage: Usage,
    *,
    cache_write_1h: int = 0,
    unpriced: str | None = None,
) -> float | None:
    """USD cost of one call, or None (warned once) when it cannot be priced.

    ``cache_write_1h`` is the part of ``usage.cache_write_tokens`` written to a
    one-hour cache; ``unpriced`` names a modifier the call used that the table
    cannot express, such as a non-standard service tier.
    """
    models = table.get(provider, {})
    entry = models.get(model) or models.get(_DATE.sub("", model))
    if entry is None:
        return _unpriced(f"no price for {provider} model {model!r}")
    if unpriced is not None:
        return _unpriced(f"{provider} model {model!r} used {unpriced}")
    rates: Rates = entry
    if entry.long_context and usage.input_tokens > entry.long_context.above:
        rates = entry.long_context
    tokens = {
        "input": usage.input_tokens
        - usage.cache_read_tokens
        - usage.cache_write_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_write": usage.cache_write_tokens - cache_write_1h,
        "cache_write_1h": cache_write_1h,
        "output": usage.output_tokens,
    }
    total = 0.0
    for name, count in tokens.items():
        if count:
            rate: float | None = getattr(rates, name)
            if rate is None:
                return _unpriced(f"no {name} price for {provider} model {model!r}")
            total += count * rate
    return total / 1_000_000


def _unpriced(reason: str) -> None:
    # ponytail: once per process per reason, unlocked; a race logs it twice.
    if reason not in _warned:
        _warned.add(reason)
        logger.warning(
            "%s; cost recorded as null. Add it with fr.configure(pricing=path).",
            reason,
        )
