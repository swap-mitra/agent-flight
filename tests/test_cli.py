from contextlib import suppress
from importlib.metadata import entry_points
from pathlib import Path

import pytest

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.cli import main
from flight_recorder.context import active_run
from flight_recorder.model import ModelCall, Run, Usage
from flight_recorder.store.cassette import CassetteStore


def summarise() -> None:
    active = active_run()
    assert active is not None
    pending = active.open_step("model")
    active.close_step(
        pending,
        "ok",
        model=ModelCall(
            provider="openai",
            model_requested="m",
            model_used="m-2026",
            params={},
            messages=[],
            usage=Usage(input_tokens=12, output_tokens=3),
        ),
    )


@fr.tool(effect="read")
def search_orders(customer_id: str) -> list[str]:
    summarise()  # a model call inside a tool, shown nested
    return ["o1"]


@fr.tool(effect="write")
def refund(order_id: str) -> None:
    raise PermissionError("over limit")


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FR_DIR", str(tmp_path))
    recorder.sinks = [CassetteStore(tmp_path, background=False)]
    return tmp_path


def record(name: str, fail: bool = False) -> Run:
    with suppress(PermissionError), fr.run(name, inputs={"ticket": "t1"}) as run:
        search_orders("c1")
        if fail:
            refund("o1")
        run.output = "done"
    return run


def lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_runs_lists_newest_first_with_filters_and_limit(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second, third = (
        record("triage"),
        record("triage", fail=True),
        record("billing"),
    )

    assert main(["runs"]) == 0
    header, *rows = lines(capsys)
    assert header.split() == [
        "RUN",
        "STARTED",
        "(UTC)",
        "NAME",
        "STATUS",
        "STEPS",
        "TOKENS",
        "COST",
    ]
    assert [r.split()[0] for r in rows] == [third.run_id, second.run_id, first.run_id]
    assert rows[1].split()[3:] == ["triage", "error", "3", "12/3", "-"]

    main(["runs", "--name", "triage", "--status", "ok"])
    assert [r.split()[0] for r in lines(capsys)[1:]] == [first.run_id]
    main(["runs", "--limit", "1"])
    assert len(lines(capsys)) == 2


@pytest.mark.parametrize("since, count", [("1h", 3), ("2d", 3), ("2999-01-01", 0)])
def test_runs_since_takes_a_duration_or_an_iso_date(
    store: Path, capsys: pytest.CaptureFixture[str], since: str, count: int
) -> None:
    for _ in range(3):
        record("triage")
    main(["runs", "--since", since])
    assert len(lines(capsys)) == 1 + count


def test_bad_since_is_a_usage_error(store: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["runs", "--since", "yesterday"])
    assert caught.value.code == 2


def test_missing_index_asks_for_reindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FR_DIR", str(tmp_path))
    assert main(["runs"]) == 1
    assert "run `fr reindex`" in capsys.readouterr().err
    assert not (tmp_path / "index.sqlite").exists()


def test_reindex_reports_corrupt_cassettes_and_exits_1(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record("triage")
    assert main(["reindex"]) == 0
    assert lines(capsys) == ["indexed 1 runs"]

    bad = record("triage")
    (bad_file,) = store.rglob(f"{bad.run_id}.jsonl")
    bad_file.write_text("{not json\n", encoding="utf-8")
    assert main(["reindex"]) == 1
    out, err = capsys.readouterr()
    assert out == "indexed 1 runs\n"
    assert f"{bad_file.name}:1:" in err


def test_show_prints_the_run_with_nested_steps_and_errors(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = record("triage", fail=True)

    assert main(["show", run.run_id[:13]]) == 0
    out = lines(capsys)
    assert out[0] == f"triage  error  {run.run_id}"
    assert out[1:4] == [
        '  inputs  {"ticket": "t1"}',
        "  output  null",
        "  totals  3 steps, 12/3 tokens, -",
    ]
    steps = [line.split() for line in out[6:]]
    assert [s[:3] for s in steps] == [
        ["0", "tool", "search_orders"],
        ["1", "model", "m-2026"],
        ["2", "tool", "refund"],
    ]
    assert out[7].startswith("  1")  # the model call inside search_orders
    assert "error (PermissionError: over limit)" in out[8]


def test_show_reads_a_cassette_path_without_an_index(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = record("triage")
    (store / "index.sqlite").unlink()
    (file,) = store.rglob("*.jsonl")
    assert main(["show", str(file)]) == 0
    assert lines(capsys)[0] == f"triage  ok  {run.run_id}"


@pytest.mark.parametrize(
    "ref, message", [("zzz", "no run matches"), ("", "2 runs match")]
)
def test_show_rejects_unknown_or_ambiguous_ids(
    store: Path, capsys: pytest.CaptureFixture[str], ref: str, message: str
) -> None:
    record("triage")
    record("triage")
    assert main(["show", ref]) == 1
    assert message in capsys.readouterr().err


def test_fr_command_is_installed() -> None:
    (script,) = entry_points(group="console_scripts", name="fr")
    assert script.load() is main
