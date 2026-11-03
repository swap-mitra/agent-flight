import pytest
from pydantic import ValidationError

from flight_recorder.model import ModelCall, Run, Step, ToolCall

TOOL = ToolCall(name="search_orders", effect="read", args={"customer_id": "c1"})
MODEL = ModelCall(
    provider="openai",
    model_requested="m",
    model_used="m",
    params={},
    messages=[{"role": "user", "content": "hi"}],
)


@pytest.mark.parametrize(
    "kind, payload",
    [
        ("tool", {}),
        ("tool", {"model": MODEL}),
        ("model", {"tool": TOOL}),
        ("tool", {"tool": TOOL, "model": MODEL}),
    ],
)
def test_step_rejects_payload_not_matching_kind(
    kind: str, payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        Step.model_validate(
            {"seq": 0, "kind": kind, "latency_ms": 1.0, "status": "ok", **payload}
        )


def test_step_round_trips_through_json() -> None:
    step = Step(seq=0, kind="tool", latency_ms=1.0, status="ok", tool=TOOL)
    assert Step.model_validate_json(step.model_dump_json()) == step


def test_run_round_trips_through_json() -> None:
    run = Run(run_id="r1", name="triage", status="ok", inputs={"ticket": "t"})
    assert Run.model_validate_json(run.model_dump_json()) == run
