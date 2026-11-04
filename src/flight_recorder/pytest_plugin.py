"""The pytest plugin: record agent runs as baselines, replay them in CI (FR-P1 to P9).

Loaded by pytest's entry point when the package is installed, and inert for
tests that don't use the ``fr_run`` fixture.

    @pytest.mark.fr(cassette="triage/refund_over_limit", repeats=3, min_pass=2)
    def test_refund_over_limit(fr_run):
        fr_run.output = triage_agent(TICKET)

``--fr-mode=record`` runs the test live and writes its run to
``<store>/cassettes/<cassette>.jsonl``, child runs beside it. Replay, the
default, serves the recording and diffs the new run against it: ``failed``
fails the test, ``changed`` warns (or fails with ``--fr-on-change=fail``).
"""

import inspect
import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.errors import CassetteError, PolicyError, ReplayDivergence
from flight_recorder.model import Run
from flight_recorder.policy import loader
from flight_recorder.replay import cases
from flight_recorder.replay.diff import COMPARATORS, Compare, RunDiff, diff
from flight_recorder.replay.engine import Session, replaying
from flight_recorder.store.cassette import CassetteStore, load, record_case, safe_name

MODES = ("record", "replay", "off")
# How long a test waits for its run to reach disk before failing.
WRITE_TIMEOUT = 30.0
_RESULTS = pytest.StashKey[list["_Result"]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flight-recorder")
    group.addoption(
        "--fr-mode",
        choices=MODES,
        default="replay",
        help="record: run live and write baselines; replay (default): serve them"
        " and diff; off: run live, no baseline",
    )
    group.addoption(
        "--fr-replay",
        choices=("tools", "full"),
        default="tools",
        help="tools (default): serve tool results, models run live;"
        " full: serve model responses too, no network",
    )
    group.addoption(
        "--fr-on-change",
        choices=("warn", "fail"),
        default="warn",
        help="what a 'changed' verdict does (default warn)",
    )
    group.addoption(
        "--fr-update",
        action="store_true",
        help="make each replay that didn't fail its case's new baseline",
    )
    group.addoption(
        "--fr-report", metavar="DIR", help="write report.json and report.md to DIR"
    )
    group.addoption(
        "--fr-policy",
        metavar="FILE",
        help="replay under this policy instead of the configured one, with models"
        " live (tools replay); a change of model alone is not a change"
        " (fr eval-route)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "fr(cassette=None, repeats=1, min_pass=None, compare='json', replay=None,"
        " on_divergence='strict', lookahead=0): how fr_run records and replays",
    )
    config.stash[_RESULTS] = []
    if (path := config.getoption("fr_policy", None)) is not None:
        try:
            loader.override = loader.load(path)
        except PolicyError as exc:
            raise pytest.UsageError(str(exc)) from exc


def pytest_unconfigure(config: pytest.Config) -> None:
    if config.getoption("fr_policy", None) is not None:
        loader.override = None


@dataclass
class _Result:
    nodeid: str
    case: str
    verdict: str  # same, changed, failed; or updated
    passed: int
    runs: int
    diffs: list[RunDiff] = field(default_factory=list[RunDiff])


@dataclass
class _Case:
    """One test's case: drives each run of the test body and judges it."""

    nodeid: str
    name: str
    case: str
    mode: str
    replay: str
    store: CassetteStore
    repeats: int
    min_pass: int
    compare: Compare
    policy: Any
    lookahead: int
    rerouted: bool  # --fr-policy: models may change by design
    on_change: str
    update: bool
    results: list[_Result]
    diffs: list[RunDiff] = field(default_factory=list[RunDiff])
    errors: list[Exception] = field(default_factory=list[Exception])

    @contextmanager
    def iteration(self) -> Generator[Run]:
        root = self.store.root
        if self.mode == "off":
            with fr.run(self.name) as run:
                yield run
            return
        if self.mode == "record":
            path = cases.baseline_path(root, self.case)
            cases.forget(root, path)
            with record_case(path), fr.run(self.name) as run:
                yield run
            self._check_written(path, run)
            return
        baseline, children = cases.load_case(cases.baseline_path(root, self.case))
        session = Session(
            baseline,
            children,
            mode="full" if self.replay == "full" else "tools",
            policy=self.policy,
            lookahead=self.lookahead,
        )
        path = cases.replay_path(root, self.case)
        cases.forget(root, path)
        error: Exception | None = None
        run: Run | None = None
        try:
            with replaying(session), record_case(path), fr.run(self.name) as run:
                yield run
        except Exception as exc:  # noqa: BLE001  judged with the diff, re-raised in conclude
            error = exc
        assert run is not None
        self._check_written(path, run)
        candidate, candidate_children = cases.load_case(path)
        count = max(len(children), len(candidate_children))
        pairs = [
            (
                children[i] if i < len(children) else None,
                candidate_children[i] if i < len(candidate_children) else None,
            )
            for i in range(count)
        ]
        strict = [d for d in session.divergences if d.policy == "strict"]
        result = diff(
            baseline,
            candidate,
            self.compare,
            divergences=strict,
            children=pairs,
            models_may_differ=self.rerouted,
        )
        if error is not None:
            listed = isinstance(error, ReplayDivergence) and error.divergence in strict
            if not listed:  # a divergence is already the first reason
                result.reasons.append(f"raised {type(error).__name__}: {error}")
            result.verdict = "failed"
            self.errors.append(error)
        self.diffs.append(result)

    def conclude(self) -> None:
        """Judge the case once every run is done; fail the test if it failed."""
        if self.mode != "replay":
            return
        passed = sum(d.verdict != "failed" for d in self.diffs)
        # Enough runs passed: judge by them, so a tolerated failure doesn't count.
        passing = [d for d in self.diffs if d.verdict != "failed"]
        verdict = "failed" if passed < self.min_pass else _worst(passing)
        if verdict != "failed" and self.update:
            cases.accept(self.store.root, self.case)
            verdict = "updated" if verdict == "changed" else verdict
        self.results.append(
            _Result(
                self.nodeid, self.case, verdict, passed, len(self.diffs), self.diffs
            )
        )
        if verdict == "failed" and self.repeats == 1 and self.errors:
            raise self.errors[0]  # the test's own failure, with its traceback
        if verdict == "failed" or (verdict == "changed" and self.on_change == "fail"):
            pytest.fail(_explain(self.case, verdict, passed, self.diffs), pytrace=False)

    def _check_written(self, path: Path, run: Run) -> None:
        # Recording fails open for an agent, but a test that records must not
        # pass without its run: check the file holds this run.
        if not self.store.flush(timeout=WRITE_TIMEOUT):
            pytest.fail(f"the run {path} was not written in time", pytrace=False)
        try:
            written, _ = load(path)
        except (OSError, CassetteError) as exc:
            written, reason = None, str(exc)
        else:
            reason = "it holds another run"
        if written is None or written.run_id != run.run_id:
            pytest.fail(
                f"the run {path} was not written ({reason}); see the logged warning",
                pytrace=False,
            )


@pytest.fixture
def fr_run(request: pytest.FixtureRequest) -> _Case:
    """The test's run. The test receives a Run: set ``fr_run.output`` to the
    agent's result, which replay compares with the baseline's."""
    config = request.config
    marker = request.node.get_closest_marker("fr")
    options: dict[str, Any] = dict(marker.kwargs) if marker else {}
    name = request.node.name
    mode = str(config.getoption("fr_mode"))
    store = _store()
    case = options.get("cassette") or f"{request.node.path.stem}/{safe_name(name)}"
    if mode == "replay" and not cases.baseline_path(store.root, case).is_file():
        pytest.fail(
            f"no cassette at {cases.baseline_path(store.root, case)};"
            " record it with pytest --fr-mode=record",
            pytrace=False,
        )
    repeats = int(options.get("repeats", 1)) if mode == "replay" else 1
    # A rerouted call has no recorded response, so --fr-policy replays tools only.
    rerouted = config.getoption("fr_policy") is not None
    replay = (
        "tools" if rerouted else options.get("replay") or config.getoption("fr_replay")
    )
    compare = options.get("compare", "json")
    if isinstance(compare, str):
        if compare not in COMPARATORS:
            known = list(COMPARATORS)
            raise pytest.UsageError(f"compare must be a function or one of {known}")
        compare = COMPARATORS[compare]
    return _Case(
        nodeid=request.node.nodeid,
        name=name,
        case=case,
        mode=mode,
        replay=str(replay),
        store=store,
        repeats=repeats,
        min_pass=int(options.get("min_pass") or repeats),
        compare=compare,
        policy=options.get("on_divergence", "strict"),
        lookahead=int(options.get("lookahead", 0)),
        rerouted=rerouted,
        on_change=str(config.getoption("fr_on_change")),
        update=bool(config.getoption("fr_update")),
        results=config.stash[_RESULTS],
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run a test that uses fr_run once per repeat, each in its own run."""
    case = pyfuncitem.funcargs.get("fr_run")
    if not isinstance(case, _Case):
        return None
    test: Callable[..., Any] = pyfuncitem.obj
    params = inspect.signature(test).parameters
    args = {name: pyfuncitem.funcargs[name] for name in params}
    for _ in range(case.repeats):
        with case.iteration() as run:
            args["fr_run"] = run
            if test(**args) is not None:
                raise pytest.UsageError("fr_run supports sync tests only")
    case.conclude()
    return True


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    results = config.stash.get(_RESULTS, list[_Result]())
    if not results:
        return
    terminalreporter.section("flight recorder")
    for r in results:
        rate = f" ({r.passed}/{r.runs} passed)" if r.runs > 1 else ""
        terminalreporter.write_line(f"{r.verdict:8} {r.case}{rate}")
        for d in r.diffs[-1:]:
            for line in [*d.reasons, *map(str, d.changes)]:
                terminalreporter.write_line(f"         {line}")
    if (folder := config.getoption("fr_report")) is not None:
        report(Path(folder), results)
        terminalreporter.write_line(f"reports written to {folder}")


def report(folder: Path, results: list[_Result]) -> None:
    """report.json, and report.md for a PR comment (FR-P8)."""
    folder.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "test": r.nodeid,
            "case": r.case,
            "verdict": r.verdict,
            "passed": r.passed,
            "runs": r.runs,
            "diffs": [d.to_json() for d in r.diffs],
        }
        for r in results
    ]
    (folder / "report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    verdicts = ("failed", "changed", "updated", "same")
    counts = {v: sum(r.verdict == v for r in results) for v in verdicts}
    lines = [
        "## Flight Recorder replay",
        "",
        " · ".join(f"**{n}** {v}" for v, n in counts.items() if n),
        "",
        "| Case | Verdict | Passed | Cost delta | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        last = r.diffs[-1] if r.diffs else None
        cost = ""
        if last is not None and last.cost_delta_usd is not None:
            cost = f"${last.cost_delta_usd:+.4f}"
        # The first reason in full (a divergence is the cause; later reasons are
        # usually its effects), then a count; the job log has everything.
        found = [] if last is None else [*last.reasons, *map(str, last.changes)]
        notes = found[0] if found else ""
        if len(found) > 1:
            notes += f" (+{len(found) - 1} more)"
        notes = " ".join(notes.replace("|", "/").split())
        lines.append(
            f"| `{r.case}` | {r.verdict} | {r.passed}/{r.runs} | {cost} | {notes} |"
        )
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worst(diffs: list[RunDiff]) -> str:
    verdicts = {d.verdict for d in diffs}
    return next((v for v in ("failed", "changed") if v in verdicts), "same")


def _explain(case: str, verdict: str, passed: int, diffs: list[RunDiff]) -> str:
    rate = f", {passed}/{len(diffs)} runs passed" if len(diffs) > 1 else ""
    last = next((d for d in reversed(diffs) if d.verdict == verdict), diffs[-1])
    lines = [f"replay of {case}: {verdict}{rate}"]
    lines += [f"  {r}" for r in last.reasons]
    lines += [f"  {c}" for c in last.changes]
    if verdict == "changed":
        lines.append(f"  accept it with: fr accept {case}")
    return "\n".join(lines)


def _store() -> CassetteStore:
    stores = [s for s in recorder.sinks if isinstance(s, CassetteStore)]
    if not stores:
        raise pytest.UsageError(
            "fr_run needs fr.configure() to keep its cassette store"
        )
    return stores[0]
