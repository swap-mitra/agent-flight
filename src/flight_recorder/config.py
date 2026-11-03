"""fr.configure(): the one place settings are made."""

import os
from collections.abc import Sequence
from pathlib import Path

from flight_recorder import recorder
from flight_recorder import redact as _redact
from flight_recorder.policy import pricing as _pricing
from flight_recorder.store.cassette import CassetteStore


def configure(
    *,
    dir: str | os.PathLike[str] | None = None,
    redact: Sequence[_redact.Rule] = (),
    pricing: str | os.PathLike[str] | None = None,
) -> None:
    """Configure recording. Each call replaces the previous configuration.

    dir: where runs are written; defaults to the FR_DIR environment variable,
    or ``.fr``. Nothing is written there until the first run.
    redact: rules applied after the default secret rules (fr.redact.pattern,
    fr.redact.path, or any function from record to record).
    pricing: a TOML file in the format of the shipped pricing.toml; each of its
    entries replaces the shipped one. Raises PricingError if it is invalid.
    """
    table = _pricing.load(pricing)  # first, so an invalid file changes nothing
    root = Path(dir) if dir is not None else Path(os.environ.get("FR_DIR") or ".fr")
    for sink in recorder.sinks:
        if isinstance(sink, CassetteStore):
            sink.close()
    recorder.sinks = [CassetteStore(root)]
    _redact.rules = [*_redact.DEFAULT_RULES, *redact]
    _pricing.table = table


# Recording works without a configure() call (SDD Write path).
configure()
