import asyncio
import inspect
from collections.abc import Iterator

import pytest

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.model import Run, Step, ToolCall


class StepSink:
    def __init__(self) -> None:
        self.steps: list[Step] = []

    def run_started(self, run: Run) -> None:
        pass

    def step_closed(self, run_id: str, step: Step) -> None:
        self.steps.append(step)

    def run_finished(self, run: Run) -> None:
        pass

    def tool_calls(self) -> list[ToolCall]:
        return [s.tool for s in self.steps if s.tool]


@pytest.fixture
def sink(monkeypatch: pytest.MonkeyPatch) -> Iterator[StepSink]:
    sink = StepSink()
    monkeypatch.setattr(recorder, "sinks", [sink])
    yield sink


@fr.tool(effect="read")
def search_orders(customer_id: str, limit: int = 10) -> list[dict[str, str]]:
    """Find a customer's orders."""
    return [{"order_id": "o1", "customer_id": customer_id}]


@fr.tool(effect="irreversible")
def issue_refund(order_id: str, amount: float) -> str:
    raise PermissionError(f"refund of {amount} on {order_id} not allowed")


def test_records_name_effect_args_by_name_and_result(sink: StepSink) -> None:
    with fr.run("triage"):
        positional = search_orders("c1")
        keyword = search_orders(limit=10, customer_id="c1")

    assert positional == keyword == [{"order_id": "o1", "customer_id": "c1"}]
    first, second = sink.tool_calls()
    assert first == second
    assert (first.name, first.effect) == ("search_orders", "read")
    assert first.args == {"customer_id": "c1", "limit": 10}
    assert first.result == [{"order_id": "o1", "customer_id": "c1"}]
    assert sink.steps[0].status == "ok"


def test_exception_is_recorded_and_reraised_unchanged(sink: StepSink) -> None:
    with fr.run("triage"), pytest.raises(PermissionError) as caught:
        issue_refund("o1", 600.0)

    assert str(caught.value) == "refund of 600.0 on o1 not allowed"
    (call,) = sink.tool_calls()
    assert sink.steps[0].status == "error"
    assert call.result is None
    assert call.error is not None
    assert (call.error.type, call.error.message) == (
        "PermissionError",
        "refund of 600.0 on o1 not allowed",
    )


def test_async_tool_is_recorded(sink: StepSink) -> None:
    @fr.tool(effect="read")
    async def lookup(key: str) -> str:
        await asyncio.sleep(0)
        return key.upper()

    async def agent() -> str:
        async with fr.run("triage"):
            return await lookup("k")

    assert asyncio.run(agent()) == "K"
    (call,) = sink.tool_calls()
    assert (call.args, call.result) == ({"key": "k"}, "K")


def test_signature_and_docstring_are_preserved() -> None:
    def plain(customer_id: str, limit: int = 10) -> list[dict[str, str]]:
        """Find a customer's orders."""
        return []

    assert search_orders.__name__ == "search_orders"
    assert search_orders.__doc__ == "Find a customer's orders."
    assert inspect.signature(search_orders) == inspect.signature(plain)
    assert inspect.iscoroutinefunction(fr.tool()(asyncio.sleep))


def test_outside_a_run_the_tool_runs_unrecorded(sink: StepSink) -> None:
    assert search_orders("c1") == [{"order_id": "o1", "customer_id": "c1"}]
    assert sink.steps == []


def test_unclassified_tool_is_irreversible_and_existing_callables_wrap(
    sink: StepSink,
) -> None:
    send = fr.tool()(print)
    with fr.run("triage"):
        send("hi")
    assert sink.tool_calls()[0].effect == "irreversible"


def test_unknown_effect_is_rejected_at_decoration() -> None:
    with pytest.raises(ValueError):
        fr.tool(effect="harmless")  # type: ignore[arg-type]


def test_methods_drop_self_and_non_json_values_are_converted(sink: StepSink) -> None:
    class Repo:
        @fr.tool(effect="read")
        def fetch(self, ids: set[int]) -> object:
            return {"ids": ids, "raw": b"\x00"}

    with fr.run("triage"):
        Repo().fetch({1})

    (call,) = sink.tool_calls()
    assert call.args == {"ids": [1]}
    assert call.result == {"ids": [1], "raw": "AA=="}


def test_tool_called_inside_a_tool_links_to_it(sink: StepSink) -> None:
    @fr.tool(effect="read")
    def outer() -> list[dict[str, str]]:
        return search_orders("c1")

    with fr.run("triage"):
        outer()

    parents = {s.tool.name: s.parent_seq for s in sink.steps if s.tool}
    assert parents == {"outer": None, "search_orders": 0}
