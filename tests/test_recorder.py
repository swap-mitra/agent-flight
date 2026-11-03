import dataclasses
import logging
from collections.abc import Iterator

import pytest

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.context import active_run
from flight_recorder.model import ModelCall, Run, Step, ToolCall, Usage
from flight_recorder.recorder import ActiveRun


class ListSink:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.runs: list[Run] = []  # as received at start and finish
        self.steps: list[Step] = []

    def run_started(self, run: Run) -> None:
        self.events.append("start")
        self.runs.append(run)

    def step_closed(self, run_id: str, step: Step) -> None:
        self.events.append("step")
        self.steps.append(step)

    def run_finished(self, run: Run) -> None:
        self.events.append("finish")
        self.runs.append(run)


class BrokenSink:
    def run_started(self, run: Run) -> None:
        raise OSError("disk full")

    def step_closed(self, run_id: str, step: Step) -> None:
        raise OSError("disk full")

    def run_finished(self, run: Run) -> None:
        raise OSError("disk full")


@pytest.fixture
def sink(monkeypatch: pytest.MonkeyPatch) -> Iterator[ListSink]:
    sink = ListSink()
    monkeypatch.setattr(recorder, "sinks", [sink])
    yield sink


def current() -> ActiveRun:
    active = active_run()
    assert active is not None
    return active


def tool_step(name: str = "search_orders") -> None:
    active = current()
    pending = active.open_step("tool")
    active.close_step(
        pending, "ok", tool=ToolCall(name=name, effect="read", args={}, result=[])
    )


def model_step(cost_usd: float | None) -> None:
    active = current()
    pending = active.open_step("model")
    active.close_step(
        pending,
        "ok",
        model=ModelCall(
            provider="openai",
            model_requested="m",
            model_used="m",
            params={},
            messages=[],
            usage=Usage(input_tokens=10, output_tokens=5),
            cost_usd=cost_usd,
        ),
    )


def test_sinks_see_start_each_step_then_finish(sink: ListSink) -> None:
    with fr.run("triage", inputs={"ticket": "t"}) as run:
        tool_step()
        tool_step()
        run.output = "done"

    assert sink.events == ["start", "step", "step", "finish"]
    assert sink.runs[0].status == "incomplete"
    finished = sink.runs[-1]
    assert (finished.status, finished.output, finished.totals.steps) == (
        "ok",
        "done",
        2,
    )
    assert [s.seq for s in sink.steps] == [0, 1]
    assert all(s.latency_ms >= 0 for s in sink.steps)


def test_parallel_steps_get_unique_seqs(sink: ListSink) -> None:
    with fr.run("triage"), fr.thread_executor(max_workers=8) as pool:
        for future in [pool.submit(tool_step) for _ in range(50)]:
            future.result()
    assert sorted(s.seq for s in sink.steps) == list(range(50))
    assert sink.runs[-1].totals.steps == 50


def test_totals_sum_tokens_and_cost(sink: ListSink) -> None:
    with fr.run("triage") as run:
        model_step(0.01)
        model_step(0.02)
        tool_step()
    assert (run.totals.input_tokens, run.totals.output_tokens) == (20, 10)
    assert run.totals.cost_usd == pytest.approx(0.03)
    assert run.totals.steps == 3


def test_one_unpriced_model_step_makes_total_cost_unknown(sink: ListSink) -> None:
    with fr.run("triage") as run:
        model_step(0.01)
        model_step(None)
        model_step(0.02)
    assert run.totals.cost_usd is None


def test_failing_sink_never_reaches_the_agent_and_logs_once_per_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    healthy = ListSink()
    monkeypatch.setattr(recorder, "sinks", [BrokenSink(), healthy])

    with caplog.at_level(logging.WARNING, logger="flight_recorder"):
        for name in ("first", "second"):
            with fr.run(name) as run:
                tool_step()
                tool_step()
                run.output = "result"

    assert len(caplog.records) == 2  # one per run, not one per failure
    assert healthy.events.count("step") == 4


def test_invalid_step_is_dropped_and_the_run_continues(
    sink: ListSink, caplog: pytest.LogCaptureFixture
) -> None:
    with fr.run("triage") as run:
        active = current()
        pending = active.open_step("tool")
        active.close_step(pending, "ok")  # a tool step with no tool payload
        tool_step()
    assert len(sink.steps) == 1
    assert run.status == "ok"
    assert "recording failed" in caplog.text


def test_agent_exception_survives_a_failing_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recorder, "sinks", [BrokenSink()])
    err = KeyError("missing")
    with pytest.raises(KeyError) as caught, fr.run("triage") as run:
        raise err
    assert caught.value is err
    assert run.status == "error"


@dataclasses.dataclass
class Ticket:
    id: int
    tags: set[str]


class Opaque:
    def __repr__(self) -> str:
        return "<Opaque>"


def test_non_json_inputs_and_output_are_converted_not_raised(sink: ListSink) -> None:
    with fr.run("triage", inputs={"ticket": Ticket(1, {"a"}), "raw": b"\xff"}) as run:
        run.output = Opaque()
    assert run.inputs == {"ticket": {"id": 1, "tags": ["a"]}, "raw": "_w=="}
    assert run.output == "<Opaque>"


def test_step_opened_inside_another_links_to_it(sink: ListSink) -> None:
    with fr.run("triage"):
        active = current()
        outer = active.open_step("tool")
        tool_step("inner")
        with fr.thread_executor() as pool:
            pool.submit(tool_step, "in_thread").result()
        active.close_step(
            outer, "ok", tool=ToolCall(name="outer", effect="read", args={})
        )
        tool_step("after")

    parents = {s.tool.name: s.parent_seq for s in sink.steps if s.tool}
    assert parents == {"inner": 0, "in_thread": 0, "outer": None, "after": None}


def test_child_run_inside_a_step_does_not_link_to_parent_runs_step(
    sink: ListSink,
) -> None:
    with fr.run("parent"):
        active = current()
        outer = active.open_step("tool")
        with fr.run("child"):
            tool_step("in_child")
        active.close_step(
            outer, "ok", tool=ToolCall(name="outer", effect="read", args={})
        )

    child_step = next(s for s in sink.steps if s.tool and s.tool.name == "in_child")
    assert child_step.parent_seq is None
