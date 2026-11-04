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
    otel: bool = False,
) -> None:
    """Configure recording. Each call replaces the previous configuration.

    dir: where runs are written; defaults to the FR_DIR environment variable,
    or ``.fr``. Nothing is written there until the first run.
    redact: rules applied after the default secret rules (fr.redact.pattern,
    fr.redact.path, or any function from record to record).
    pricing: a TOML file in the format of the shipped pricing.toml; each of its
    entries replaces the shipped one. Raises PricingError if it is invalid.
    otel: also export each run to the application's OpenTelemetry
    TracerProvider; needs the ``otel`` extra. Message content is exported only
    when OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT is SPAN_ONLY or
    SPAN_AND_EVENT, and always after redaction.
    """
    table = _pricing.load(pricing)  # first, so an invalid file changes nothing
    sinks: list[recorder.Sink] = []
    if otel:
        try:
            from flight_recorder.export.otel import OtelSink
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "fr.configure(otel=True) needs the otel extra:"
                " pip install 'ai-flight-recorder[otel]'",
                name=exc.name,
            ) from exc
        sinks.append(OtelSink())
    root = Path(dir) if dir is not None else Path(os.environ.get("FR_DIR") or ".fr")
    for sink in recorder.sinks:
        close = getattr(sink, "close", None)
        if close is not None:
            close()
    recorder.sinks = [CassetteStore(root), *sinks]
    _redact.rules = [*_redact.DEFAULT_RULES, *redact]
    _pricing.table = table


# Recording works without a configure() call (SDD Write path).
configure()
