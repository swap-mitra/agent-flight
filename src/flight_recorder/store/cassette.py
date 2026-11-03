"""JSONL cassettes: the source of truth for every recorded run (FR-R5, ADR-002).

One file per run: a header line, one line per step as it closes, a footer.
"""

import atexit
import hashlib
import json
import logging
import queue
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import JsonValue, ValidationError

from flight_recorder.errors import CassetteError
from flight_recorder.model import SCHEMA_VERSION, Run, Step

logger = logging.getLogger("flight_recorder")

BLOB_THRESHOLD = 64 * 1024
# Last on each line, so a word diff of a re-recorded run shows them apart.
_VOLATILE = ("run_id", "parent_run_id", "latency_ms")
_FOOTER_FIELDS = ("status", "output", "totals")

LineType = Literal["header", "step", "footer"]
# (run_id, file, line type, fields); None stops the writer.
_Item = tuple[str, Path, LineType, dict[str, Any]] | None


class CassetteStore:
    """A sink that writes each run to ``<root>/runs/<yyyy-mm-dd>/<run_id>.jsonl``.

    With ``background=True`` writes go through a queue drained by one thread,
    so the agent never waits on disk; when ``capacity`` steps are pending,
    further steps are dropped and counted in ``dropped_steps``. Headers and
    footers are never dropped. ``background=False`` writes synchronously.
    """

    def __init__(
        self, root: Path, *, background: bool = True, capacity: int = 10_000
    ) -> None:
        self.root = root
        self.background = background
        self.capacity = capacity
        self.dropped_steps = 0
        self._files: dict[str, Path] = {}
        self._queue: queue.Queue[_Item] = queue.Queue()
        self._lock = threading.Lock()
        self._pending_steps = 0
        self._thread: threading.Thread | None = None
        self._failed_runs: set[str] = set()

    def run_started(self, run: Run) -> None:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        file = self.root / "runs" / day / f"{run.run_id}.jsonl"
        self._files[run.run_id] = file
        header = run.model_dump(mode="json", exclude=set(_FOOTER_FIELDS))
        self._put((run.run_id, file, "header", header))

    def step_closed(self, run_id: str, step: Step) -> None:
        file = self._files[run_id]
        self._put((run_id, file, "step", step.model_dump(mode="json")))

    def run_finished(self, run: Run) -> None:
        file = self._files.pop(run.run_id)
        footer = run.model_dump(mode="json", include=set(_FOOTER_FIELDS))
        self._put((run.run_id, file, "footer", footer))

    def flush(self) -> None:
        """Block until every queued record is written."""
        self._queue.join()

    def close(self) -> None:
        """Flush and stop the writer thread."""
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None

    def _put(self, item: _Item) -> None:
        assert item is not None
        if not self.background:
            self._write(item)
            return
        if item[2] == "step":
            with self._lock:
                if self._pending_steps >= self.capacity:
                    self.dropped_steps += 1
                    return
                self._pending_steps += 1
        self._ensure_writer()
        self._queue.put(item)

    def _ensure_writer(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._drain, name="flight_recorder-writer", daemon=True
                )
                self._thread.start()
                atexit.register(self.flush)

    def _drain(self) -> None:
        while (item := self._queue.get()) is not None:
            try:
                self._write(item)
            except Exception as exc:  # noqa: BLE001  recording fails open (FR-R9)
                self._log_failure(item[0], exc)
            finally:
                if item[2] == "step":
                    with self._lock:
                        self._pending_steps -= 1
                self._queue.task_done()
        self._queue.task_done()

    def _write(self, item: _Item) -> None:
        assert item is not None
        _, file, line_type, fields = item
        fields = {k: self._store_blobs(v) for k, v in fields.items()}
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a", encoding="utf-8", newline="\n") as f:
            f.write(_line(line_type, fields))

    def _store_blobs(self, value: Any) -> Any:
        if isinstance(value, str):
            data = value.encode()
            if len(data) <= BLOB_THRESHOLD:
                return value
            digest = hashlib.sha256(data).hexdigest()
            blob = self.root / "blobs" / digest
            if not blob.exists():
                blob.parent.mkdir(parents=True, exist_ok=True)
                blob.write_bytes(data)
            return {"fr_blob": digest, "size": len(data)}
        if isinstance(value, list):
            return [self._store_blobs(v) for v in value]  # type: ignore[reportUnknownVariableType]
        if isinstance(value, dict):
            return {k: self._store_blobs(v) for k, v in value.items()}  # type: ignore[reportUnknownVariableType]
        return value

    def _log_failure(self, run_id: str, exc: Exception) -> None:
        if run_id in self._failed_runs:
            return
        self._failed_runs.add(run_id)
        logger.warning(
            "writing run %s failed; the agent continues unaffected",
            run_id,
            exc_info=exc,
        )


def _line(line_type: LineType, fields: dict[str, Any]) -> str:
    """One JSONL line: type first, volatile fields last, nested keys sorted."""
    keys = [k for k in fields if k not in _VOLATILE]
    keys += [k for k in _VOLATILE if k in fields]
    body = ", ".join(
        f"{json.dumps(k)}: {json.dumps(fields[k], sort_keys=True, ensure_ascii=False)}"
        for k in keys
    )
    return f'{{"type": "{line_type}", {body}}}\n'


def load(file: Path, blobs: Path | None = None) -> tuple[Run, list[Step]]:
    """Read a cassette back into its Run and Steps.

    A run with no footer loads as ``incomplete``. ``blobs`` defaults to the
    store's ``blobs/`` directory, two levels above the file.
    """
    blobs = blobs or file.parents[2] / "blobs"
    lines = file.read_text(encoding="utf-8").split("\n")
    # The last element is "" after a final newline; anything else there is a
    # line cut off by a crash, which is dropped so the run loads as incomplete.
    lines = lines[:-1]
    if not lines:
        raise CassetteError(f"{file}:1: empty cassette")

    run: dict[str, Any] = {}
    steps: list[Step] = []
    for lineno, raw in enumerate(lines, start=1):
        try:
            fields = _load_blobs(json.loads(raw), blobs)
            line_type = fields.pop("type")
            if lineno == 1:
                if line_type != "header":
                    raise ValueError(f"first line is {line_type!r}, not a header")
                if fields.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError(
                        f"unknown schema_version {fields.get('schema_version')!r}"
                    )
                run = {**fields, "status": "incomplete"}
            elif line_type == "step":
                steps.append(Step.model_validate(fields))
            elif line_type == "footer":
                run.update(fields)
            else:
                raise ValueError(f"unexpected {line_type!r} line")
        except (ValueError, KeyError, ValidationError) as exc:
            raise CassetteError(f"{file}:{lineno}: {exc}") from exc
    try:
        return Run.model_validate(run), steps
    except ValidationError as exc:
        raise CassetteError(f"{file}:1: {exc}") from exc


def _load_blobs(value: JsonValue, blobs: Path) -> Any:
    if isinstance(value, dict):
        if set(value) == {"fr_blob", "size"}:
            blob = blobs / str(value["fr_blob"])
            try:
                return blob.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"missing blob {blob}") from exc
        return {k: _load_blobs(v, blobs) for k, v in value.items()}
    if isinstance(value, list):
        return [_load_blobs(v, blobs) for v in value]
    return value
