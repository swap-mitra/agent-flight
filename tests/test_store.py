import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.context import active_run
from flight_recorder.errors import CassetteError, FlightRecorderError
from flight_recorder.model import ModelCall, Run, Step, Usage
from flight_recorder.store.cassette import BLOB_THRESHOLD, CassetteStore, load


@fr.tool(effect="read")
def search_orders(customer_id: str) -> list[dict[str, str]]:
    return [{"order_id": "o1", "status": "shipped"}]


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
            model_used="m",
            params={"temperature": 0},
            messages=[{"role": "user", "content": "where is my order?"}],
            response={"content": "shipped", "id": "resp_1"},
            usage=Usage(input_tokens=12, output_tokens=3),
        ),
    )


def record(store: CassetteStore, inputs: object = None) -> Run:
    recorder.sinks = [store]
    with fr.run("triage", inputs=inputs or {"ticket": "t1"}) as run:
        model_step()
        search_orders("c1")
        run.output = {"answer": "shipped"}
    store.flush()
    return run


def only_file(root: Path) -> Path:
    (file,) = (root / "runs").rglob("*.jsonl")
    return file


@pytest.mark.parametrize("background", [False, True])
def test_run_round_trips_through_its_cassette(tmp_path: Path, background: bool) -> None:
    run = record(CassetteStore(tmp_path, background=background))
    loaded, steps = load(only_file(tmp_path))

    assert loaded == run
    assert loaded.status == "ok"
    assert [s.kind for s in steps] == ["model", "tool"]
    assert steps[1].tool is not None
    assert steps[1].tool.result == [{"order_id": "o1", "status": "shipped"}]


def test_file_is_named_by_utc_day_and_run_id(tmp_path: Path) -> None:
    run = record(CassetteStore(tmp_path, background=False))
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    assert only_file(tmp_path) == tmp_path / "runs" / day / f"{run.run_id}.jsonl"


def test_lines_are_typed_with_volatile_fields_last_and_nested_keys_sorted(
    tmp_path: Path,
) -> None:
    record(CassetteStore(tmp_path, background=False))
    lines = only_file(tmp_path).read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]

    assert [p["type"] for p in parsed] == ["header", "step", "step", "footer"]
    assert list(parsed[0])[-1] == "parent_run_id"
    assert list(parsed[1])[-1] == "latency_ms"
    assert list(parsed[1]["model"]["response"]) == ["content", "id"]


def test_rerecording_an_unchanged_run_changes_only_volatile_fields(
    tmp_path: Path,
) -> None:
    """FR-R5 acceptance."""

    def stable_lines(root: Path) -> list[str]:
        record(CassetteStore(root, background=False))
        out: list[str] = []
        for line in only_file(root).read_text(encoding="utf-8").splitlines():
            fields = json.loads(line)
            for volatile in ("run_id", "parent_run_id", "latency_ms"):
                fields.pop(volatile, None)
            out.append(json.dumps(fields))
        return out

    assert stable_lines(tmp_path / "a") == stable_lines(tmp_path / "b")


def test_missing_footer_or_cut_off_last_line_loads_as_incomplete(
    tmp_path: Path,
) -> None:
    record(CassetteStore(tmp_path, background=False))
    file = only_file(tmp_path)
    lines = file.read_text(encoding="utf-8").splitlines(keepends=True)

    file.write_text("".join(lines[:-1]), encoding="utf-8")  # no footer
    run, steps = load(file)
    assert (run.status, len(steps)) == ("incomplete", 2)

    file.write_text("".join(lines[:-1]) + lines[-1][:10], encoding="utf-8")
    run, steps = load(file)
    assert (run.status, len(steps)) == ("incomplete", 2)


def bad_json_in_a_step(lines: list[str]) -> list[str]:
    return [lines[0], "{not json\n", *lines[2:]]


def unknown_schema_version(lines: list[str]) -> list[str]:
    header = lines[0].replace('"schema_version": 1', '"schema_version": 99')
    return [header, *lines[1:]]


def no_header(lines: list[str]) -> list[str]:
    return lines[1:]


@pytest.mark.parametrize(
    "corrupt, line",
    [(bad_json_in_a_step, 2), (unknown_schema_version, 1), (no_header, 1)],
)
def test_corrupt_cassette_names_the_file_and_line(
    tmp_path: Path, corrupt: Callable[[list[str]], list[str]], line: int
) -> None:
    record(CassetteStore(tmp_path, background=False))
    file = only_file(tmp_path)
    lines = file.read_text(encoding="utf-8").splitlines(keepends=True)
    file.write_text("".join(corrupt(lines)), encoding="utf-8")

    with pytest.raises(CassetteError, match=rf"{file.name}:{line}:") as caught:
        load(file)
    assert isinstance(caught.value, FlightRecorderError)


def test_large_strings_go_to_blobs_once_and_load_back(tmp_path: Path) -> None:
    image = "A" * (BLOB_THRESHOLD + 1)
    store = CassetteStore(tmp_path, background=False)
    run = record(store, inputs={"image": image, "same_image": image})

    text = only_file(tmp_path).read_text(encoding="utf-8")
    assert image not in text
    assert '"fr_blob"' in text
    assert len(list((tmp_path / "blobs").iterdir())) == 1
    loaded, _ = load(only_file(tmp_path))
    assert loaded.inputs == run.inputs


def test_full_queue_drops_steps_but_never_header_or_footer(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path, capacity=0)
    record(store)
    run, steps = load(only_file(tmp_path))
    assert (run.status, steps, store.dropped_steps) == ("ok", [], 2)


def test_write_failure_is_logged_once_per_run_and_agent_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    store = CassetteStore(blocker)  # every mkdir under a file fails

    with caplog.at_level(logging.WARNING, logger="flight_recorder"):
        first = record(store)
        second = record(store)

    assert first.output == second.output == {"answer": "shipped"}
    assert len(caplog.records) == 2


def test_configure_dir_and_fr_dir_set_the_root_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FR_DIR", str(tmp_path / "from_env"))
    fr.configure()
    (store,) = recorder.sinks
    assert isinstance(store, CassetteStore)
    assert store.root == tmp_path / "from_env"
    assert not store.root.exists()  # nothing written before the first run

    fr.configure(dir=tmp_path / "explicit")
    (store,) = recorder.sinks
    assert isinstance(store, CassetteStore)
    with fr.run("triage"):
        pass
    store.flush()
    assert len(list((tmp_path / "explicit" / "runs").rglob("*.jsonl"))) == 1


def test_step_from_a_store_validates_like_any_step(tmp_path: Path) -> None:
    record(CassetteStore(tmp_path, background=False))
    _, steps = load(only_file(tmp_path))
    assert all(isinstance(s, Step) for s in steps)
