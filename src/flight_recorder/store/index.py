"""SQLite index over the cassettes (FR-R5, ADR-002).

The index only answers queries; deleting it loses nothing, since ``rebuild``
recreates it from the cassette files.
"""

import sqlite3
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flight_recorder.errors import CassetteError
from flight_recorder.store import (
    cassette as _cassette,
)  # module import: cassette imports us

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL,
    path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT,
    model TEXT,
    cost_usd REAL,
    status TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS runs_started_at ON runs (started_at);
"""


def connect(file: Path) -> sqlite3.Connection:
    """Open the index, creating it if needed. WAL and a busy timeout let
    several agent processes write at once."""
    db = sqlite3.connect(file, timeout=5)
    # WAL persists in the file, so only a new index needs switching. The
    # switch ignores the busy timeout; if another writer holds the lock, it
    # either is switching too or this write runs in rollback mode, both safe.
    if db.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
        with suppress(sqlite3.OperationalError):
            db.execute("PRAGMA journal_mode=WAL")
    db.executescript(_SCHEMA)
    return db


def started_at(run_id: str) -> str:
    """The run's start time, read from its UUIDv7 (first 48 bits are ms)."""
    ms = int(run_id.replace("-", "")[:12], 16)
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="milliseconds")


def write(
    db: sqlite3.Connection,
    run_id: str,
    path: str,
    line_type: str,
    fields: dict[str, Any],
) -> None:
    """Index one cassette line (JSON-mode fields) of the run at ``path``."""
    if line_type == "header":
        db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, 'incomplete', ?, 0, 0, 0.0, ?)",
            (run_id, fields["name"], started_at(run_id), path),
        )
        db.execute("DELETE FROM steps WHERE run_id = ?", (run_id,))
    elif line_type == "step":
        model, tool = fields.get("model"), fields.get("tool")
        db.execute(
            "INSERT OR REPLACE INTO steps VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                fields["seq"],
                fields["kind"],
                tool["name"] if tool else None,
                model["model_used"] if model else None,
                model.get("cost_usd") if model else None,
                fields["status"],
            ),
        )
    else:
        totals = fields["totals"]
        db.execute(
            "UPDATE runs SET status = ?, input_tokens = ?, output_tokens = ?,"
            " cost_usd = ? WHERE run_id = ?",
            (
                fields["status"],
                totals["input_tokens"],
                totals["output_tokens"],
                totals["cost_usd"],
                run_id,
            ),
        )


def rebuild(root: Path) -> list[CassetteError]:
    """Recreate ``<root>/index.sqlite`` from ``<root>/runs``.

    Corrupt cassettes are skipped; their errors are returned.
    """
    root.mkdir(parents=True, exist_ok=True)
    errors: list[CassetteError] = []
    # One transaction: live writers wait on the busy timeout, readers never
    # see a half-built index.
    with closing(connect(root / "index.sqlite")) as db, db:
        db.execute("DELETE FROM runs")
        db.execute("DELETE FROM steps")
        for cassette in sorted((root / "runs").rglob("*.jsonl")):
            try:
                run, steps = _cassette.load(cassette)
            except CassetteError as exc:
                errors.append(exc)
                continue
            path = cassette.relative_to(root).as_posix()
            fields = run.model_dump(mode="json")
            write(db, run.run_id, path, "header", fields)
            for step in steps:
                write(db, run.run_id, path, "step", step.model_dump(mode="json"))
            if run.status != "incomplete":
                write(db, run.run_id, path, "footer", fields)
    return errors
