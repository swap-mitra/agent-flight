"""Interception overhead benchmark (FR-R8, SDD Non-functional requirements).

For every interception path, the same call runs plain and recorded in
alternating pairs, and the median difference is the overhead per call. Model
calls carry a conversation of 1, 20 and 100 turns (a tool, a result of as many
records); the 20-turn median of every path must be under 2 ms. Calls go to the
SDK contract tests' mock transports and steps to a real cassette store, whose
disk writes happen on its own thread, so I/O is excluded as FR-R8 says.

    uv run python benchmarks/overhead.py

Exits 1 if any path misses the budget.
"""

import asyncio
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))

import anthropic
import httpx2
import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from test_anthropic import Server as AnthropicServer
from test_openai import Server as OpenAIServer

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.store.cassette import CassetteStore

BUDGET_MS = 2.0
GATED_TURNS = 20
SIZES = (1, GATED_TURNS, 100)
WARMUP, PAIRS = 30, 400
# The mock servers' models, priced so the cost path runs too.
PRICES = """
[openai]
"gpt-x" = { input = 1.0, cache_read = 0.5, output = 2.0 }
[anthropic]
"claude-x" = { input = 3.0, cache_read = 0.3, cache_write = 3.75, cache_write_1h = 6.0, output = 15.0 }
"""

Pair = tuple[Callable[[], Any], Callable[[], Any]]


def conversation(turns: int) -> list[dict[str, str]]:
    """A support conversation of ``turns`` messages, about 400 bytes each."""
    line = "My order still has not arrived and the tracking page shows no update. "
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": line * 6}
        for i in range(turns)
    ]


def lc_messages(turns: int) -> list[BaseMessage]:
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in conversation(turns)
    ]


# Any: the SDKs' overloads cannot match arguments unpacked from a dict.
def openai_client(wrapped: bool) -> Any:
    http = httpx2.Client(transport=httpx2.MockTransport(OpenAIServer()))
    client = openai.OpenAI(api_key="sk-test", http_client=http, max_retries=0)
    return fr.wrap(client) if wrapped else client


def anthropic_client(wrapped: bool) -> Any:
    http = httpx2.Client(transport=httpx2.MockTransport(AnthropicServer()))
    client = anthropic.Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    return fr.wrap(client) if wrapped else client


def chat_openai(wrapped: bool) -> ChatOpenAI:
    transport = httpx2.MockTransport(OpenAIServer())
    settings: dict[str, Any] = {
        "model": "gpt-x",
        "api_key": "sk-test",
        "max_retries": 0,
        "http_client": httpx2.Client(transport=transport),
    }
    model = ChatOpenAI(**settings)
    return fr.langchain.wrap_model(model) if wrapped else model


def orders(customer_id: str, count: int) -> list[dict[str, str]]:
    """Look up a customer's orders."""
    return [
        {"order_id": f"o{i}", "customer_id": customer_id, "status": "shipped"}
        for i in range(count)
    ]


async def aorders(customer_id: str, count: int) -> list[dict[str, str]]:
    """Look up a customer's orders."""
    return orders(customer_id, count)


def pairs(turns: int) -> dict[str, Pair]:
    """Each path's (plain, recorded) call, for a payload of ``turns``."""
    messages: Any = conversation(turns)  # the SDKs' message types, as dicts
    ask: dict[str, Any] = {"model": "claude-x", "max_tokens": 100, "messages": messages}
    plain_oa, rec_oa = openai_client(False), openai_client(True)
    plain_an, rec_an = anthropic_client(False), anthropic_client(True)
    plain_lc, rec_lc = chat_openai(False), chat_openai(True)
    lc_tool = tool(orders)
    (rec_lc_tool,) = fr.langchain.wrap_tools([lc_tool], effects={"orders": "read"})
    rec_orders = fr.tool(effect="read")(orders)
    rec_aorders = fr.tool(effect="read")(aorders)
    loop = asyncio.new_event_loop()
    lc_ask = lc_messages(turns)
    tool_args = {"customer_id": "c1", "count": turns}

    def chat(client: Any, **extra: Any) -> Callable[[], Any]:
        return lambda: client.chat.completions.create(
            model="gpt-x", messages=messages, **extra
        )

    return {
        "tool": (lambda: orders("c1", turns), lambda: rec_orders("c1", turns)),
        "tool, async": (
            lambda: loop.run_until_complete(aorders("c1", turns)),
            lambda: loop.run_until_complete(rec_aorders("c1", turns)),
        ),
        "openai chat": (chat(plain_oa), chat(rec_oa)),
        "openai chat, stream": (
            lambda: list(chat(plain_oa, stream=True)()),
            lambda: list(chat(rec_oa, stream=True)()),
        ),
        "openai responses": (
            lambda: plain_oa.responses.create(model="gpt-x", input=messages),
            lambda: rec_oa.responses.create(model="gpt-x", input=messages),
        ),
        "anthropic": (
            lambda: plain_an.messages.create(**ask),
            lambda: rec_an.messages.create(**ask),
        ),
        "anthropic, stream": (
            lambda: list(plain_an.messages.create(**ask, stream=True)),
            lambda: list(rec_an.messages.create(**ask, stream=True)),
        ),
        "langchain tool": (
            lambda: lc_tool.invoke(tool_args),
            lambda: rec_lc_tool.invoke(tool_args),
        ),
        "langchain model": (
            lambda: plain_lc.invoke(lc_ask),
            lambda: rec_lc.invoke(lc_ask),
        ),
    }


def overhead_ms(plain: Callable[[], Any], recorded: Callable[[], Any]) -> float:
    """Median of recorded minus plain time, over pairs run in random order so
    drift on the machine falls on both sides alike."""
    with fr.run("benchmark"):
        for _ in range(WARMUP):
            plain()
            recorded()
        deltas: list[float] = []
        for _ in range(PAIRS):
            first, second = (plain, recorded)
            if random.random() < 0.5:
                first, second = second, first
            start = time.perf_counter()
            first()
            middle = time.perf_counter()
            second()
            end = time.perf_counter()
            took = {first: middle - start, second: end - middle}
            deltas.append((took[recorded] - took[plain]) * 1000)
    return statistics.median(deltas)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="fr-bench-"))
    prices = root / "prices.toml"
    prices.write_text(PRICES, encoding="utf-8")
    fr.configure(dir=root, pricing=prices)
    (store,) = [s for s in recorder.sinks if isinstance(s, CassetteStore)]

    results: dict[str, dict[int, float]] = {}
    for turns in SIZES:
        for name, (plain, recorded) in pairs(turns).items():
            results.setdefault(name, {})[turns] = overhead_ms(plain, recorded)
            store.flush(timeout=60)  # keep one path's writes out of the next

    print(f"Median overhead per call, ms (budget {BUDGET_MS} at {GATED_TURNS} turns)")
    print(f"{'path':<22}" + "".join(f"{f'{t} turns':>12}" for t in SIZES))
    missed: list[str] = []
    for name, by_size in results.items():
        row = "".join(f"{by_size[t]:>12.3f}" for t in SIZES)
        over = by_size[GATED_TURNS] >= BUDGET_MS
        print(f"{name:<22}{row}{'  OVER' if over else ''}")
        if over:
            missed.append(name)
    if missed:
        print(f"over budget: {', '.join(missed)}")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
