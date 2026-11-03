import logging
import sqlite3
import threading
from contextlib import closing, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.context import active_run, uuid7
from flight_recorder.errors import CassetteError
from flight_recorder.model import ModelCall, Run, Usage
from flight_recorder.store import index
from flight_recorder.store.cassette import CassetteStore


@fr.tool(effect="read")
def search_orders(customer_id: str) -> list[str]:
    return ["o1"]


def model_step() -> None:
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
            cost_usd=0.5,
        ),
    )


def record(store: CassetteStore, fail: bool = False) -> Run:
    recorder.sinks = [store]
    with suppress(ValueError), fr.run("triage") as run:
        model_step()
        search_orders("c1")
        if fail:
            raise ValueError("boom")
    store.flush()
    return run


def tables(root: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    with closing(sqlite3.connect(root / "index.sqlite")) as db:
        runs = db.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
        steps = db.execute("SELECT * FROM steps ORDER BY run_id, seq").fetchall()
    return runs, steps


def test_a_recorded_run_is_indexed_with_its_steps(tmp_path: Path) -> None:
    run = record(CassetteStore(tmp_path))
    runs, steps = tables(tmp_path)

    ((run_id, name, status, started, tokens_in, tokens_out, cost, path),) = runs
    assert (run_id, name, status) == (run.run_id, "triage", "ok")
    assert (tokens_in, tokens_out, cost) == (12, 3, 0.5)
    assert (tmp_path / str(path)).is_file()
    age = datetime.now(UTC) - datetime.fromisoformat(str(started))
    assert timedelta(0) <= age < timedelta(minutes=1)
    assert steps == [
        (run.run_id, 0, "model", None, "m-2026", 0.5, "ok"),
        (run.run_id, 1, "tool", "search_orders", None, None, "ok"),
    ]


def test_rebuild_reproduces_the_live_index(tmp_path: Path) -> None:
    """FR-R5: the index is rebuildable from the cassettes."""
    store = CassetteStore(tmp_path, background=False)
    record(store)
    record(store, fail=True)
    crashed = Run(run_id=uuid7(), name="crashed", status="incomplete", inputs=None)
    store.run_started(crashed)  # no footer, as if the process died
    live = tables(tmp_path)
    assert [r[2] for r in live[0]] == ["ok", "error", "incomplete"]

    (tmp_path / "index.sqlite").unlink()
    assert index.rebuild(tmp_path) == []
    assert tables(tmp_path) == live


def test_rebuild_skips_corrupt_cassettes_and_returns_their_errors(
    tmp_path: Path,
) -> None:
    store = CassetteStore(tmp_path, background=False)
    good, bad = record(store), record(store)
    (bad_file,) = (tmp_path / "runs").rglob(f"{bad.run_id}.jsonl")
    bad_file.write_text("{not json\n", encoding="utf-8")

    (error,) = index.rebuild(tmp_path)
    assert isinstance(error, CassetteError)
    assert f"{bad_file.name}:1:" in str(error)
    assert [r[0] for r in tables(tmp_path)[0]] == [good.run_id]


def test_index_failure_is_logged_once_per_run_and_cassette_is_kept(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "index.sqlite").mkdir()  # sqlite cannot open a directory
    store = CassetteStore(tmp_path)

    with caplog.at_level(logging.WARNING, logger="flight_recorder"):
        run = record(store)

    assert len(caplog.records) == 1
    (file,) = (tmp_path / "runs").rglob("*.jsonl")
    assert file.read_text(encoding="utf-8").count("\n") == 4  # header, 2 steps, footer
    assert run.status == "ok"


def test_concurrent_writers_share_one_index(tmp_path: Path) -> None:
    # Synchronous writes from several threads, each with its own connection.
    recorder.sinks = [CassetteStore(tmp_path, background=False)]

    def agent() -> None:
        for _ in range(5):
            with fr.run("triage"):
                model_step()
                search_orders("c1")

    threads = [threading.Thread(target=agent) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    runs, steps = tables(tmp_path)
    assert (len(runs), len(steps)) == (20, 40)
