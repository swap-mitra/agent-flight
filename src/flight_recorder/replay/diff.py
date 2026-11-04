"""Compare a candidate run with its baseline: one verdict per run (FR-P5).

Steps are aligned with difflib on their signatures (tool name, or provider and
requested model), then aligned pairs are compared field by field. The final
outputs go through a comparator: exact, JSON-equal, or any function.
"""

import difflib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import JsonValue

from flight_recorder.model import Run, Step

Verdict = Literal["same", "changed", "failed"]
Recording = tuple[Run, list[Step]]
Compare = Callable[[Any, Any], bool]


def exact(baseline: Any, candidate: Any) -> bool:
    """Equal as written: same values, same key order."""
    return json.dumps(baseline, default=str) == json.dumps(candidate, default=str)


def json_equal(baseline: Any, candidate: Any) -> bool:
    """Equal as JSON: key order ignored, and strings holding JSON compared as JSON."""
    return _parsed(baseline) == _parsed(candidate)


COMPARATORS: dict[str, Compare] = {"exact": exact, "json": json_equal}


@dataclass
class Change:
    kind: Literal["added", "removed", "changed"]
    step: str  # "tool lookup" or "model openai/gpt-x"
    baseline: int | None  # step seq in the baseline
    candidate: int | None
    fields: list[str] = field(default_factory=list[str])

    def __str__(self) -> str:
        where = f"#{self.baseline}" if self.candidate is None else f"#{self.candidate}"
        detail = f" ({', '.join(self.fields)})" if self.fields else ""
        return f"{self.kind} {self.step} {where}{detail}"


@dataclass
class RunDiff:
    name: str
    verdict: Verdict
    reasons: list[str]
    changes: list[Change]
    output_equal: bool
    baseline_output: Any
    candidate_output: Any
    cost_delta_usd: float | None
    latency_delta_ms: float
    children: list["RunDiff"] = field(default_factory=list["RunDiff"])
    baseline_cost_usd: float | None = None
    candidate_cost_usd: float | None = None

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "changes": [str(c) for c in self.changes],
            "output_equal": self.output_equal,
            "cost_delta_usd": self.cost_delta_usd,
            "baseline_cost_usd": self.baseline_cost_usd,
            "candidate_cost_usd": self.candidate_cost_usd,
            "latency_delta_ms": round(self.latency_delta_ms, 3),
            "children": [c.to_json() for c in self.children],
        }


def diff(
    baseline: Recording,
    candidate: Recording,
    compare: Compare = json_equal,
    *,
    divergences: Sequence[object] = (),
    children: Sequence[tuple[Recording | None, Recording | None]] = (),
    models_may_differ: bool = False,
) -> RunDiff:
    """The diff and verdict of ``candidate`` against ``baseline``.

    failed: a strict divergence, a run that raised where the baseline didn't,
    an aligned step whose status changed, or outputs the comparator rejects.
    changed: steps added, removed or changed (tool arguments, model used),
    with everything above passing. same: nothing differs.

    ``models_may_differ`` ignores a change of the model used, for judging a
    routing policy that is meant to change it.
    """
    (base_run, base_steps), (cand_run, cand_steps) = baseline, candidate
    reasons = [f"diverged: {d}" for d in divergences]
    changes: list[Change] = []
    sigs = [_signature(s) for s in base_steps], [_signature(s) for s in cand_steps]
    matcher = difflib.SequenceMatcher(None, sigs[0], sigs[1], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for b, c in zip(base_steps[i1:i2], cand_steps[j1:j2], strict=True):
                fields = _changed_fields(b, c, models_may_differ)
                if fields:
                    changes.append(Change("changed", _label(b), b.seq, c.seq, fields))
                if "status" in fields:
                    reasons.append(
                        f"step {_label(c)} #{c.seq}: {b.status} -> {c.status}"
                    )
            continue
        changes += [
            Change("removed", _label(s), s.seq, None) for s in base_steps[i1:i2]
        ]
        changes += [Change("added", _label(s), None, s.seq) for s in cand_steps[j1:j2]]
    if base_run.status != cand_run.status:
        reasons.append(f"run status {base_run.status} -> {cand_run.status}")
    output_equal = compare(base_run.output, cand_run.output)
    if not output_equal:
        reasons.append("final output differs")
    child_diffs = [_child(b, c, compare, models_may_differ) for b, c in children]
    reasons += [f"child {c.name}: failed" for c in child_diffs if c.verdict == "failed"]
    verdict: Verdict = "failed" if reasons else "same"
    if verdict == "same" and (changes or any(c.verdict != "same" for c in child_diffs)):
        verdict = "changed"
    base_cost, cand_cost = base_run.totals.cost_usd, cand_run.totals.cost_usd
    return RunDiff(
        name=cand_run.name,
        verdict=verdict,
        reasons=reasons,
        changes=changes,
        output_equal=output_equal,
        baseline_output=base_run.output,
        candidate_output=cand_run.output,
        cost_delta_usd=None
        if base_cost is None or cand_cost is None
        else cand_cost - base_cost,
        latency_delta_ms=sum(s.latency_ms for s in cand_steps)
        - sum(s.latency_ms for s in base_steps),
        children=child_diffs,
        baseline_cost_usd=base_cost,
        candidate_cost_usd=cand_cost,
    )


def _child(
    base: Recording | None,
    cand: Recording | None,
    compare: Compare,
    models_may_differ: bool,
) -> RunDiff:
    if base is None or cand is None:
        present = base or cand
        assert present is not None
        kind = "removed" if cand is None else "added"
        return RunDiff(
            present[0].name,
            "changed",
            [],
            [Change(kind, f"run {present[0].name}", None, None)],
            True,
            None,
            None,
            None,
            0.0,
        )
    return diff(base, cand, compare, models_may_differ=models_may_differ)


def _signature(step: Step) -> tuple[str, str]:
    if step.tool is not None:
        return ("tool", step.tool.name)
    assert step.model is not None
    return ("model", f"{step.model.provider}/{step.model.model_requested}")


def _label(step: Step) -> str:
    return " ".join(_signature(step))


def _changed_fields(b: Step, c: Step, models_may_differ: bool) -> list[str]:
    fields: list[str] = []
    if b.status != c.status:
        fields.append("status")
    if b.tool is not None and c.tool is not None:
        if _dumps(b.tool.args) != _dumps(c.tool.args):
            fields.append("args")
        if (b.tool.error and b.tool.error.type) != (c.tool.error and c.tool.error.type):
            fields.append("error")
    if (
        not models_may_differ
        and b.model is not None
        and c.model is not None
        and b.model.model_used != c.model.model_used
    ):
        fields.append("model_used")
    return fields


def _parsed(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return _parsed(json.loads(value))
        except ValueError:
            return value
    if isinstance(value, dict):
        return {k: _parsed(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [_parsed(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
