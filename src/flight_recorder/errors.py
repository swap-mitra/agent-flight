"""Every error Flight Recorder raises subclasses FlightRecorderError."""


class FlightRecorderError(Exception):
    """Base class for Flight Recorder errors."""


class CassetteError(FlightRecorderError):
    """A cassette is corrupt or has an unknown schema version."""


class PricingError(FlightRecorderError):
    """A pricing file is invalid; raised by fr.configure(), never mid-run."""


class PolicyError(FlightRecorderError):
    """A policy or replay rule is invalid; raised at startup, never mid-run."""


class BudgetExceeded(FlightRecorderError):
    """A model call would break a hard budget, so it was not sent (FR-B2)."""


class ToolDenied(FlightRecorderError):
    """The action gate denied a tool call, so it did not run (FR-G6).

    ``decision`` is the step's GateDecision: the rule, the outcome (denied,
    rejected or expired), the approver and the reason.
    """

    def __init__(self, message: str, decision: object = None) -> None:
        super().__init__(message)
        self.decision = decision


class ReplayDivergence(FlightRecorderError):
    """A replayed run made a call its recording has no match for (FR-P3, P4).

    ``divergence`` holds the step index, the expected and actual calls and the
    step before them.
    """

    def __init__(self, divergence: object) -> None:
        super().__init__(str(divergence))
        self.divergence = divergence
