"""The ``fr`` command: inspect the local store (SDD CLI)."""

import argparse
import csv
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, get_args

from flight_recorder import approvals
from flight_recorder.errors import CassetteError
from flight_recorder.model import Run, Step
from flight_recorder.replay import cases
from flight_recorder.replay.diff import diff
from flight_recorder.store import index
from flight_recorder.store.cassette import load

_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``fr`` command; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="fr", description="Inspect recorded runs. The store is FR_DIR, or .fr."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("reindex", help="rebuild the index from the cassettes")
    runs = commands.add_parser("runs", help="list runs, newest first")
    runs.add_argument("--name")
    status = get_args(Run.model_fields["status"].annotation)
    runs.add_argument("--status", choices=status)
    runs.add_argument(
        "--since", type=_since, help="a duration (30m, 12h, 7d) or an ISO date"
    )
    runs.add_argument("--limit", type=int, default=20, help="0 for all (default 20)")
    show = commands.add_parser("show", help="print a run step by step")
    show.add_argument("run", help="run id, unique id prefix, or cassette path")
    compare = commands.add_parser("diff", help="diff a run against a baseline")
    compare.add_argument("baseline", help="run id, unique id prefix, or cassette path")
    compare.add_argument("candidate", help="run id, unique id prefix, or cassette path")
    accept = commands.add_parser(
        "accept", help="make a test case's latest replay its baseline"
    )
    accept.add_argument("case", help="the case name, as in cassettes/<case>.jsonl")
    route = commands.add_parser(
        "eval-route",
        help="replay the pytest suite under a candidate routing policy",
        description="Replay recorded test cases under a candidate policy, with"
        " models called live (tool results from the recordings), and report"
        " the same-verdict rate and cost delta for each run name. Arguments"
        " after -- go to pytest.",
    )
    route.add_argument("--policy", required=True, type=Path, help="the policy file")
    route.add_argument(
        "--report", type=Path, help="also write eval-route.md and .json here"
    )
    route.add_argument("pytest_args", nargs=argparse.REMAINDER)
    for verb in ("approve", "deny"):
        decide = commands.add_parser(verb, help=f"{verb} a pending tool call")
        decide.add_argument("id", help="the approval id")
        decide.add_argument("--approver", help="who decides (default: your login)")
        decide.add_argument("--reason")
    audit = commands.add_parser(
        "audit", help="export every gated or irreversible tool call and its decision"
    )
    audit.add_argument(
        "--since", type=_since, help="a duration (30m, 12h, 7d) or an ISO date"
    )
    audit.add_argument("--format", choices=("csv", "json"), default="csv")
    args = parser.parse_args(argv)

    root = Path(os.environ.get("FR_DIR") or ".fr")
    try:
        if args.command == "reindex":
            return _reindex(root)
        if args.command == "runs":
            return _runs(root, args.name, args.status, args.since, args.limit)
        if args.command == "diff":
            return _diff(root, args.baseline, args.candidate)
        if args.command == "accept":
            return _accept(root, args.case)
        if args.command == "eval-route":
            return _eval_route(args.policy, args.report, args.pytest_args)
        if args.command in ("approve", "deny"):
            return _decide(root, args.command, args.id, args.approver, args.reason)
        if args.command == "audit":
            return _audit(root, args.since, args.format)
        return _show(root, args.run)
    except _Fail as exc:
        print(f"fr: {exc}", file=sys.stderr)
        return 1


class _Fail(Exception):
    """A user-facing failure: printed without a traceback, exit code 1."""


def _since(text: str) -> str:
    if match := re.fullmatch(r"(\d+)([mhd])", text):
        start = datetime.now(UTC) - timedelta(**{_UNITS[match[2]]: int(match[1])})
    else:
        try:
            start = datetime.fromisoformat(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{text!r} is not a duration (30m, 12h, 7d) or an ISO date"
            ) from None
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
    # Same format as index.started_at, so the two compare as strings.
    return start.astimezone(UTC).isoformat(timespec="milliseconds")


def _index(root: Path) -> sqlite3.Connection:
    file = root / "index.sqlite"
    if not file.is_file():
        raise _Fail(f"no index at {file}; run `fr reindex`")
    return sqlite3.connect(file)


def _reindex(root: Path) -> int:
    errors = index.rebuild(root)
    with closing(_index(root)) as db:
        (count,) = db.execute("SELECT COUNT(*) FROM runs").fetchone()
    for error in errors:
        print(f"fr: skipped {error}", file=sys.stderr)
    print(f"indexed {count} runs")
    return 1 if errors else 0


def _runs(
    root: Path, name: str | None, status: str | None, since: str | None, limit: int
) -> int:
    where: list[str] = []
    params: list[object] = []
    for clause, value in (
        ("name = ?", name),
        ("status = ?", status),
        ("started_at >= ?", since),
    ):
        if value is not None:
            where.append(clause)
            params.append(value)
    sql = (
        "SELECT run_id, started_at, name, status,"
        " (SELECT COUNT(*) FROM steps WHERE steps.run_id = runs.run_id),"
        " input_tokens, output_tokens, cost_usd FROM runs"
        + (" WHERE " + " AND ".join(where) if where else "")
        + " ORDER BY started_at DESC LIMIT ?"
    )
    params.append(limit or -1)  # SQLite reads a negative limit as no limit
    with closing(_index(root)) as db:
        rows = db.execute(sql, params).fetchall()

    table = [["RUN", "STARTED (UTC)", "NAME", "STATUS", "STEPS", "TOKENS", "COST"]]
    for run_id, started, run_name, run_status, steps, tin, tout, cost in rows:
        table.append(
            [
                run_id,
                started[:19].replace("T", " "),
                run_name,
                run_status,
                str(steps),
                f"{tin}/{tout}",
                _cost(cost),
            ]
        )
    print(_table(table))
    return 0


def _show(root: Path, ref: str) -> int:
    run, steps = _load(_resolve(root, ref))
    totals = run.totals
    print(f"{run.name}  {run.status}  {run.run_id}")
    print(f"  inputs  {json.dumps(run.inputs, ensure_ascii=False)}")
    print(f"  output  {json.dumps(run.output, ensure_ascii=False)}")
    print(
        f"  totals  {totals.steps} steps, "
        f"{totals.input_tokens}/{totals.output_tokens} tokens, {_cost(totals.cost_usd)}"
    )
    print()
    print(
        _table(
            [["STEP", "KIND", "CALL", "STATUS", "LATENCY", "TOKENS", "COST"]]
            + _step_rows(steps)
        )
    )
    return 0


def _diff(root: Path, baseline: str, candidate: str) -> int:
    """Print the diff and verdict; exit 1 when the verdict is failed."""
    result = diff(_load(_resolve(root, baseline)), _load(_resolve(root, candidate)))
    print(f"{result.verdict}  {result.name}")
    for line in [*result.reasons, *map(str, result.changes)]:
        print(f"  {line}")
    if result.cost_delta_usd is not None:
        print(f"  cost {result.cost_delta_usd:+.4f} USD")
    print(f"  latency {result.latency_delta_ms:+.0f} ms")
    return 1 if result.verdict == "failed" else 0


def _accept(root: Path, case: str) -> int:
    try:
        path = cases.accept(root, case)
    except CassetteError as exc:
        raise _Fail(str(exc)) from None
    print(f"accepted {path}")
    return 0


def _eval_route(policy: Path, report: Path | None, pytest_args: list[str]) -> int:
    """Run pytest under ``policy`` and summarise its replay report (FR-B7).

    Exit 1 if any case failed, as pytest does.
    """
    if not policy.is_file():
        raise _Fail(f"no policy file at {policy}")
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    with tempfile.TemporaryDirectory() as tmp:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--fr-mode=replay",
            f"--fr-policy={policy.resolve()}",
            f"--fr-report={tmp}",
            *pytest_args,
        ]
        code = subprocess.run(command, check=False).returncode
        produced = Path(tmp) / "report.json"
        if not produced.is_file():
            raise _Fail(f"pytest exited {code} without a replay report")
        results: list[dict[str, Any]] = json.loads(produced.read_text("utf-8"))
    rows = _route_rows(results)
    text = _route_markdown(policy.name, rows)
    print(text, end="")
    if report is not None:
        report.mkdir(parents=True, exist_ok=True)
        (report / "eval-route.md").write_text(text, encoding="utf-8")
        data = json.dumps({"policy": policy.name, "runs": rows}, indent=2)
        (report / "eval-route.json").write_text(data, encoding="utf-8")
    return 1 if any(r["verdict"] == "failed" for r in results) else 0


def _route_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per run name: runs, how many judged same, and costs (None if any is unknown)."""
    rows: dict[str, dict[str, Any]] = {}
    for result in results:
        for d in result["diffs"]:
            row = rows.setdefault(
                d["name"],
                {
                    "run": d["name"],
                    "runs": 0,
                    "same": 0,
                    "baseline_usd": 0.0,
                    "candidate_usd": 0.0,
                },
            )
            row["runs"] += 1
            row["same"] += d["verdict"] == "same"
            for key, cost in (
                ("baseline_usd", d["baseline_cost_usd"]),
                ("candidate_usd", d["candidate_cost_usd"]),
            ):
                row[key] = None if row[key] is None or cost is None else row[key] + cost
    for row in rows.values():
        row["same_rate"] = row["same"] / row["runs"]
        base, cand = row["baseline_usd"], row["candidate_usd"]
        row["delta_usd"] = None if base is None or cand is None else cand - base
    return list(rows.values())


def _route_markdown(policy: str, rows: list[dict[str, Any]]) -> str:
    runs = sum(r["runs"] for r in rows)
    lines = [
        "## Flight Recorder route evaluation",
        "",
        (
            f"Policy `{policy}` replayed against {runs} recorded"
            f" run{'s' if runs != 1 else ''}: tool results came from the"
            " recordings, model calls ran live under the policy."
        ),
        "",
        "| Run | Same verdict | Baseline cost | Candidate cost | Cost delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        delta = r["delta_usd"]
        change = "-"
        if delta is not None:
            change = f"{'-' if delta < 0 else '+'}${abs(delta):.4f}"
            if r["baseline_usd"]:
                change += f" ({delta / r['baseline_usd']:+.1%})"
        lines.append(
            f"| `{r['run']}` | {r['same_rate']:.0%} ({r['same']}/{r['runs']})"
            f" | {_cost(r['baseline_usd'])} | {_cost(r['candidate_usd'])} | {change} |"
        )
    return "\n".join(lines) + "\n"


def _decide(
    root: Path, verb: str, request_id: str, approver: str | None, reason: str | None
) -> int:
    store = root / "state.sqlite"
    request = approvals.get(request_id, at=store) if store.is_file() else None
    if request is None:
        raise _Fail(f"no approval {request_id} in {store}")
    decision = "approve" if verb == "approve" else "deny"
    if not approvals.resolve(request_id, decision, approver, reason, at=store):
        current = approvals.get(request_id, at=store)
        state = current.state if current is not None else "missing"
        raise _Fail(f"approval {request_id} is already {state}")
    print(
        f"{'approved' if decision == 'approve' else 'denied'} {request.tool} ({request_id})"
    )
    return 0


AUDIT_FIELDS = (
    "run_id",
    "run_name",
    "seq",
    "started_at",
    "tool",
    "effect",
    "status",
    "action",
    "outcome",
    "rule",
    "approver",
    "decided_at",
    "reason",
    "approval_id",
)


def _audit(root: Path, since: str | None, fmt: str) -> int:
    """Every recorded tool call that was gated or is irreversible (FR-G7).

    Read from the cassettes of production runs, the source of truth; a corrupt
    one is reported on stderr and makes the exit code 1.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    for file in sorted((root / "runs").rglob("*.jsonl")):
        try:
            run, steps = load(file)
        except (OSError, CassetteError) as exc:
            print(f"fr: skipped {file}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        for step in steps:
            tool, gate = step.tool, step.gate
            if tool is None or (gate is None and tool.effect != "irreversible"):
                continue
            started = step.started_at.isoformat() if step.started_at else ""
            if since is not None and started and started < since:
                continue
            rows.append(
                {
                    "run_id": run.run_id,
                    "run_name": run.name,
                    "seq": step.seq,
                    "started_at": started,
                    "tool": tool.name,
                    "effect": tool.effect,
                    "status": step.status,
                    "action": gate.action if gate else "",
                    "outcome": gate.outcome if gate else "ungated",
                    "rule": (gate.rule or "default") if gate else "",
                    "approver": (gate.approver or "") if gate else "",
                    "decided_at": gate.decided_at.isoformat()
                    if gate and gate.decided_at
                    else "",
                    "reason": (gate.reason or "") if gate else "",
                    "approval_id": (gate.approval_id or "") if gate else "",
                }
            )
    rows.sort(key=lambda r: (r["started_at"], r["run_id"], r["seq"]))
    if fmt == "json":
        print(json.dumps(rows, indent=2))
    else:
        writer = csv.DictWriter(
            sys.stdout, fieldnames=list[str](AUDIT_FIELDS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return 1 if skipped else 0


def _resolve(root: Path, ref: str) -> Path:
    """The cassette a run id, unique id prefix or path names."""
    file = Path(ref)
    if file.suffix != ".jsonl":
        with closing(_index(root)) as db:
            # substr, not LIKE: an id prefix must never act as a wildcard.
            matches = db.execute(
                "SELECT path FROM runs WHERE substr(run_id, 1, ?) = ?",
                (len(ref), ref),
            ).fetchall()
        if not matches:
            raise _Fail(f"no run matches {ref!r}")
        if len(matches) > 1:
            raise _Fail(f"{len(matches)} runs match {ref!r}; give more of the id")
        file = root / matches[0][0]
    return file


def _load(file: Path) -> tuple[Run, list[Step]]:
    try:
        return load(file)
    except OSError as exc:
        raise _Fail(f"cannot read {file}: {exc.strerror}") from None
    except CassetteError as exc:
        raise _Fail(str(exc)) from None


def _step_rows(steps: list[Step]) -> list[list[str]]:
    depth: dict[int, int] = {}
    rows: list[list[str]] = []
    # Cassettes hold steps in close order; print in start order so a
    # nested step sits under its parent.
    for step in sorted(steps, key=lambda s: s.seq):
        parent = step.parent_seq
        depth[step.seq] = 0 if parent is None else depth.get(parent, -1) + 1
        tokens = cost = ""
        if step.model is not None:
            call, error = step.model.model_used, step.model.error
            if step.model.usage is not None:
                tokens = (
                    f"{step.model.usage.input_tokens}/{step.model.usage.output_tokens}"
                )
            cost = _cost(step.model.cost_usd)
        else:
            assert step.tool is not None
            call, error = step.tool.name, step.tool.error
        status = (
            f"{step.status} ({error.type}: {error.message})" if error else step.status
        )
        rows.append(
            [
                "  " * depth[step.seq] + str(step.seq),
                step.kind,
                call,
                status,
                f"{step.latency_ms:.0f} ms",
                tokens,
                cost,
            ]
        )
    return rows


def _cost(cost: float | None) -> str:
    return "-" if cost is None else f"${cost:.4f}"


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    )
