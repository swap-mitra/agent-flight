"""The pytest plugin: record agent runs as test baselines, replay them in CI (FR-P7).

Loaded by pytest's entry point when the package is installed, and inert for
tests that don't use the ``fr_run`` fixture.

    @pytest.mark.fr(cassette="triage/refund_over_limit")
    def test_refund_over_limit(fr_run):
        fr_run.output = triage_agent(TICKET)

``--fr-mode=record`` runs the test live and writes its run to
``<store>/cassettes/<cassette>.jsonl``, with child runs beside it. Replay is
the default, so CI never calls a real API unless asked.
"""

import re
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.errors import CassetteError
from flight_recorder.model import Run
from flight_recorder.store import index
from flight_recorder.store.cassette import (
    CassetteStore,
    load,
    record_case,
    safe_name,
)

MODES = ("record", "replay", "off")
_CHILD = re.compile(r"\d+-.*\.jsonl")
# How long a test waits for its baseline to reach disk before failing.
WRITE_TIMEOUT = 30.0


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flight-recorder")
    group.addoption(
        "--fr-mode",
        choices=MODES,
        default="replay",
        help="record: run live and write baselines; replay (default): serve them;"
        " off: run live, record nothing",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "fr(cassette=None): the fr_run baseline's name, a path under"
        " <store>/cassettes; default <test file>/<test name>",
    )


@pytest.fixture
def fr_run(request: pytest.FixtureRequest) -> Iterator[Run]:
    """The test's run: set ``fr_run.output`` to the agent's result."""
    mode = str(request.config.getoption("fr_mode"))
    name = request.node.name
    if mode == "off":
        with fr.run(name) as run:
            yield run
        return
    store = _store()
    marker = request.node.get_closest_marker("fr")
    case = (marker.kwargs.get("cassette") if marker else None) or (
        f"{request.node.path.stem}/{safe_name(name)}"
    )
    path = store.root / "cassettes" / f"{case}.jsonl"
    if mode == "replay":
        if not path.is_file():
            pytest.fail(
                f"no cassette at {path}; record it with pytest --fr-mode=record",
                pytrace=False,
            )
        pytest.fail("replay mode is not available yet", pytrace=False)
    _forget_children(store.root, path)
    with record_case(path), fr.run(name) as run:
        yield run
    # Recording fails open for an agent, but a test that records a baseline
    # must not pass without one: check the file on disk holds this run.
    if not store.flush(timeout=WRITE_TIMEOUT):
        pytest.fail(f"the baseline {path} was not written in time", pytrace=False)
    try:
        written, _ = load(path)
    except (OSError, CassetteError) as exc:
        written, reason = None, str(exc)
    else:
        reason = "it holds another run"
    if written is None or written.run_id != run.run_id:
        pytest.fail(
            f"the baseline {path} was not written ({reason}); see the logged warning",
            pytrace=False,
        )


def _store() -> CassetteStore:
    stores = [s for s in recorder.sinks if isinstance(s, CassetteStore)]
    if not stores:
        raise pytest.UsageError(
            "fr_run needs fr.configure() to keep its cassette store"
        )
    return stores[0]


def _forget_children(root: Path, case: Path) -> None:
    """Delete a case's child runs from its last recording, and their index rows,
    so a re-recorded case never keeps a sub-agent run that no longer happens."""
    folder = case.with_suffix("")
    # Only <n>-<name>.jsonl: the folder may also hold other cases' baselines.
    children = sorted(p for p in folder.glob("*.jsonl") if _CHILD.match(p.name))
    for child in children:
        child.unlink()
    if children:
        with closing(index.connect(root / "index.sqlite")) as db, db:
            for child in children:
                index.forget(db, child.relative_to(root).as_posix())
