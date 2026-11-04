"""Every error Flight Recorder raises subclasses FlightRecorderError."""


class FlightRecorderError(Exception):
    """Base class for Flight Recorder errors."""


class CassetteError(FlightRecorderError):
    """A cassette is corrupt or has an unknown schema version."""


class PricingError(FlightRecorderError):
    """A pricing file is invalid; raised by fr.configure(), never mid-run."""


class PolicyError(FlightRecorderError):
    """A policy or replay rule is invalid; raised at startup, never mid-run."""


class ReplayDivergence(FlightRecorderError):
    """A replayed run made a call its recording has no match for (FR-P3, P4).

    ``divergence`` holds the step index, the expected and actual calls and the
    step before them.
    """

    def __init__(self, divergence: object) -> None:
        super().__init__(str(divergence))
        self.divergence = divergence
