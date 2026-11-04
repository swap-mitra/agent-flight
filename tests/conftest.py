pytest_plugins = ["pytester"]

from collections.abc import Iterator

import pytest

from flight_recorder import recorder, redact
from flight_recorder.policy import budgets, loader, pricing


@pytest.fixture(autouse=True)
def _isolate_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # No test writes to the default .fr store, and configure() calls are undone.
    monkeypatch.setattr(recorder, "sinks", [])
    monkeypatch.setattr(recorder, "tag_sources", [])
    monkeypatch.setattr(redact, "rules", list(redact.DEFAULT_RULES))
    monkeypatch.setattr(pricing, "table", pricing.SHIPPED)
    monkeypatch.setattr(loader, "configured", None)
    monkeypatch.setattr(loader, "override", None)
    monkeypatch.setattr(budgets, "state", budgets.state)
    yield
