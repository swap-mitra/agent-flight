"""Every error Flight Recorder raises subclasses FlightRecorderError."""


class FlightRecorderError(Exception):
    """Base class for Flight Recorder errors."""


class CassetteError(FlightRecorderError):
    """A cassette is corrupt or has an unknown schema version."""
