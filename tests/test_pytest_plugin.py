"""The pytest plugin's record side (FR-P7, FR-P1; SDD pytest plugin).

Each test runs a real pytest session in a subprocess, so the plugin loads
from its entry point exactly as it does for users.
"""

from pathlib import Path

import pytest

from flight_recorder.store.cassette import load

AGENT = """
import pytest
import flight_recorder as fr


@fr.tool(effect="read")
def lookup(order_id: str) -> str:
    return "shipped"


def helper_agent(order_id: str) -> str:
    with fr.run("helper"):
        return lookup(order_id)


def test_triage(fr_run):
    fr_run.output = lookup("o1")


@pytest.mark.fr(cassette="desk/refund")
def test_named(fr_run):
    fr_run.output = helper_agent("o2") + lookup("o3")


@pytest.mark.parametrize("order", ["a/b", "c"])
def test_param(fr_run, order):
    fr_run.output = lookup(order)


def test_untouched():
    lookup("o9")  # no fr_run: the plugin stays out of the way
"""


@pytest.fixture
def store(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = pytester.path / "store"
    monkeypatch.setenv("FR_DIR", str(root))
    pytester.makepyfile(test_agent=AGENT)
    return root


def test_record_writes_each_case_to_its_baseline(
    pytester: pytest.Pytester, store: Path
) -> None:
    pytester.runpytest_subprocess("--fr-mode=record").assert_outcomes(passed=5)

    cassettes = store / "cassettes"
    run, steps = load(cassettes / "test_agent" / "test_triage.jsonl")
    assert (run.name, run.status, run.output) == ("test_triage", "ok", "shipped")
    assert [s.tool.args for s in steps if s.tool] == [{"order_id": "o1"}]
    # The marker names the cassette; a child run is recorded beside it.
    run, steps = load(cassettes / "desk" / "refund.jsonl")
    assert run.output == "shippedshipped"
    child, child_steps = load(cassettes / "desk" / "refund" / "1-helper.jsonl")
    assert (child.name, child.parent_run_id) == ("helper", run.run_id)
    assert [s.tool.args for s in child_steps if s.tool] == [{"order_id": "o2"}]
    # Each parameter gets its own baseline, its id made safe as a file name.
    assert (cassettes / "test_agent" / "test_param[a_b].jsonl").is_file()
    assert (cassettes / "test_agent" / "test_param[c].jsonl").is_file()
    # Every run went to a baseline, and the call outside a run to nothing.
    assert not list((store / "runs").rglob("*.jsonl"))


def test_re_recording_replaces_the_baseline_and_drops_stale_children(
    pytester: pytest.Pytester, store: Path
) -> None:
    pytester.runpytest_subprocess("--fr-mode=record").assert_outcomes(passed=5)
    folder = store / "cassettes" / "desk" / "refund"
    other = folder / "other-case.jsonl"  # another case's baseline in the folder
    other.write_text("kept")
    # The agent stops calling its helper.
    pytester.makepyfile(test_agent=AGENT.replace('helper_agent("o2")', '"x"'))
    pytester.runpytest_subprocess("--fr-mode=record").assert_outcomes(passed=5)

    run, _ = load(store / "cassettes" / "desk" / "refund.jsonl")
    assert run.output == "xshipped"
    assert not (folder / "1-helper.jsonl").exists()
    assert other.read_text() == "kept"
    # One index row per baseline: the old recordings were replaced.
    result = pytester.run("fr", "runs", "--limit", "0")
    assert result.stdout.str().count("test_named") == 1
    assert "helper" not in result.stdout.str()
    # A rebuilt index has the baselines too.
    pytester.run("fr", "reindex")
    result = pytester.run("fr", "runs", "--limit", "0")
    assert result.stdout.str().count("test_named") == 1


def test_replay_is_the_default_and_a_missing_cassette_says_how_to_record(
    pytester: pytest.Pytester, store: Path
) -> None:
    result = pytester.runpytest_subprocess("-k", "test_triage")

    result.assert_outcomes(errors=1)  # in setup: the test never ran
    result.stdout.fnmatch_lines(
        ["*no cassette at *test_triage.jsonl*--fr-mode=record*"]
    )
    assert not (store / "cassettes").exists()


def test_off_runs_live_and_writes_no_baseline(
    pytester: pytest.Pytester, store: Path
) -> None:
    pytester.runpytest_subprocess("--fr-mode=off").assert_outcomes(passed=5)

    assert not (store / "cassettes").exists()


def test_record_fails_the_test_when_its_baseline_cannot_be_written(
    pytester: pytest.Pytester, store: Path
) -> None:
    # A directory where the baseline file goes: the write fails on any OS.
    (store / "cassettes" / "test_agent" / "test_triage.jsonl").mkdir(parents=True)

    result = pytester.runpytest_subprocess("--fr-mode=record", "-k", "test_triage")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*run *test_triage.jsonl was not written*"])
