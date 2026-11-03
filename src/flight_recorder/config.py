"""fr.configure(): the one place settings are made."""

from collections.abc import Sequence

from flight_recorder import redact as _redact


def configure(*, redact: Sequence[_redact.Rule] = ()) -> None:
    """Configure recording. Each call replaces the previous configuration.

    redact: rules applied after the default secret rules (fr.redact.pattern,
    fr.redact.path, or any function from record to record).
    """
    _redact.rules = [*_redact.DEFAULT_RULES, *redact]
