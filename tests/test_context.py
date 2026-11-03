import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import flight_recorder as fr
from flight_recorder.context import current_run, uuid7


def test_run_is_current_inside_the_block_only() -> None:
    with fr.run("triage", inputs={"ticket": "t"}) as run:
        assert current_run() is run
    assert current_run() is None
    assert run.status == "ok"
    assert run.inputs == {"ticket": "t"}


def test_nested_run_links_to_parent_and_restores_it() -> None:
    with fr.run("parent") as parent:
        with fr.run("child") as child:
            assert current_run() is child
        assert current_run() is parent
    assert parent.parent_run_id is None
    assert child.parent_run_id == parent.run_id


def test_exception_propagates_unchanged_and_marks_run_error() -> None:
    err = ValueError("boom")
    with pytest.raises(ValueError) as caught, fr.run("triage") as run:
        raise err
    assert caught.value is err
    assert run.status == "error"
    assert current_run() is None


def test_async_with_and_concurrent_tasks_see_their_own_run() -> None:
    async def agent(name: str) -> tuple[str, str]:
        async with fr.run(name) as run:
            await asyncio.sleep(0.01)  # interleave with the other task
            seen = current_run()
            assert seen is not None
            return run.run_id, seen.run_id

    async def main() -> list[tuple[str, str]]:
        return list(await asyncio.gather(agent("a"), agent("b")))

    results = asyncio.run(main())
    assert all(opened == seen for opened, seen in results)
    assert results[0][0] != results[1][0]


def test_thread_executor_tasks_see_the_run_and_plain_threads_do_not() -> None:
    with fr.run("triage") as run:
        with fr.thread_executor() as pool:
            assert pool.submit(current_run).result() is run
        with ThreadPoolExecutor() as plain:
            assert plain.submit(current_run).result() is None


def test_copy_context_carries_the_run_into_a_thread() -> None:
    seen: list[object] = []
    with fr.run("triage") as run:
        t = threading.Thread(target=fr.copy_context(lambda: seen.append(current_run())))
        t.start()
        t.join()
    assert seen == [run]


def test_uuid7_has_version_7_and_sorts_by_time() -> None:
    first = uuid7()
    time.sleep(0.002)
    second = uuid7()
    parsed = uuid.UUID(first)
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122
    assert first < second
