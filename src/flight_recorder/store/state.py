"""``state.sqlite``: what must survive a restart and be shared by processes.

Daily budget counters (v0.3) and approval requests (v0.4). Unlike the index it
is not rebuildable from cassettes, so it is never deleted by Flight Recorder.
"""

import sqlite3
from pathlib import Path

# Set by fr.configure(), beside the cassette store.
path: Path = Path(".fr") / "state.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS budget_day (
    day TEXT NOT NULL,
    scope TEXT NOT NULL,
    usd REAL NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    unpriced INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, scope)
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,          -- pending, approved, rejected, expired
    created_at TEXT NOT NULL,     -- ISO 8601 UTC
    expires_at TEXT NOT NULL,
    run_id TEXT NOT NULL,
    run_name TEXT NOT NULL,
    tool TEXT NOT NULL,
    effect TEXT NOT NULL,
    args TEXT NOT NULL,           -- JSON, redacted
    preceding TEXT NOT NULL,      -- JSON list of the run's last steps, redacted
    rule TEXT,
    reason TEXT,                  -- why approval was asked; then the decision's
    approver TEXT,
    decided_at TEXT
);
"""


def connect(at: Path | None = None) -> sqlite3.Connection:
    """A connection to the state store at ``at`` (default: the configured one)."""
    file = path if at is None else at
    file.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(file, timeout=5)
    db.row_factory = sqlite3.Row
    db.executescript(_SCHEMA)
    return db
