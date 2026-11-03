from collections.abc import Iterator

import pytest

from flight_recorder import recorder, redact


@pytest.fixture(autouse=True)
def _isolate_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # No test writes to the default .fr store, and configure() calls are undone.
    monkeypatch.setattr(recorder, "sinks", [])
    monkeypatch.setattr(redact, "rules", list(redact.DEFAULT_RULES))
    yield
