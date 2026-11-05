"""Budgets, routing and fallbacks against the real SDKs (FR-B1 to B6, rows 3.1 to 3.9).

The Anthropic SDK runs over a mock transport that answers as the model each
request names, so a routed call shows up in the server's log and the response.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic

import flight_recorder as fr
from flight_recorder import recorder
from flight_recorder.errors import BudgetExceeded, PolicyError
from flight_recorder.model import Run, Step
from flight_recorder.policy import engine

# claude-opus-5 costs $5 / $25 per million tokens, claude-haiku-4-5 $1 / $5.
BIG, SMALL = "claude-opus-5", "claude-haiku-4-5"
USAGE = {"input_tokens": 1000, "output_tokens": 100}  # $0.0075 on BIG
ASK: dict[str, Any] = {
    "model": BIG,
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


class Server:
    """Answers like the Messages API, as the requested model. ``statuses``
    are the HTTP statuses of the first replies (then 200)."""

    def __init__(self, statuses: list[int] | None = None, counted: int = 0) -> None:
        self.statuses = list(statuses or [])
        self.models: list[str] = []
        self.counts = 0
        self.counted = counted

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/count_tokens"):
            self.counts += 1
            if self.counted < 0:  # the endpoint is down
                return httpx2.Response(503, json={"type": "error", "error": {}})
            return httpx2.Response(200, json={"input_tokens": self.counted})
        self.models.append(body["model"])
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            error = {"type": "error", "error": {"type": "api_error", "message": "no"}}
            return httpx2.Response(status, json=error)
        message = {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": USAGE,
        }
        if not body.get("stream"):
            return httpx2.Response(200, json=message)
        start = {**message, "content": [], "stop_reason": None}
        events = [
            {"type": "message_start", "message": start},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 100},
            },
            {"type": "message_stop"},
        ]
        sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse.encode()
        )


class Sink:
    def __init__(self) -> None:
        self.runs: list[Run] = []
        self.steps: list[Step] = []

    def run_started(self, run: Run) -> None:
        pass

    def step_closed(self, run_id: str, step: Step) -> None:
        self.steps.append(step)

    def run_finished(self, run: Run) -> None:
        self.runs.append(run)


@pytest.fixture
def sink() -> Sink:
    sink = Sink()
    recorder.sinks = [sink]
    return sink


def policy(tmp_path: Path, text: str) -> None:
    """Configure ``text`` as the policy, with the store in ``tmp_path``."""
    file = tmp_path / "fr.toml"
    file.write_text(text, encoding="utf-8")
    kept = [s for s in recorder.sinks if isinstance(s, Sink)]  # the test's own
    fr.configure(dir=tmp_path / ".fr", policy=file)
    recorder.sinks += kept


def client(server: Server) -> Any:  # Any: tests pass kwargs as dicts
    http = httpx2.Client(transport=httpx2.MockTransport(server))
    return fr.wrap(Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0))


def async_client(server: Server) -> Any:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(server))
    return fr.wrap(
        AsyncAnthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    )


def model_steps(sink: Sink) -> list[Any]:
    return [s.model for s in sink.steps if s.model is not None]


# Row 3.1: the policy file


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("[budgets]\nper_run_usd = 1\nbogus = 2\n", "bogus"),
        ('[budgets]\non_breach = "explode"\n', "on_breach"),
        ('[budgets]\non_breach = "downgrade"\n', "needs downgrade_to"),
        ("[budgets]\nper_run_usd = -1\n", "per_run_usd"),
        ('[budgets]\ntimezone = "Mars/Olympus"\n', "unknown timezone"),
        ('[[routes]]\nmodel = "made-up-model"\n', "say which provider"),
        ('[budgets]\ndowngrade_to = { google = "x" }\n', "unknown providers"),
        (f'[fallback]\n"{BIG}" = ["gpt-5-mini"]\n', "crosses providers"),
        ("[budgets\n", "invalid policy file"),
    ],
)
def test_an_invalid_policy_raises_at_configure_and_changes_nothing(
    tmp_path: Path, text: str, error: str
) -> None:
    good = tmp_path / "good.toml"
    good.write_text(f'[[routes]]\nmodel = "{SMALL}"\n', encoding="utf-8")
    fr.configure(dir=tmp_path / ".fr", policy=good)
    bad = tmp_path / "bad.toml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(PolicyError, match=error):
        fr.configure(dir=tmp_path / ".fr", policy=bad)
    with fr.run("triage") as run:
        pass
    assert run.policy_hash is not None  # the good policy is still in force


def test_a_missing_policy_file_raises_policy_error(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="cannot read"):
        fr.configure(dir=tmp_path / ".fr", policy=tmp_path / "nope.toml")


def test_runs_carry_the_policy_hash(tmp_path: Path) -> None:
    with fr.run("before") as before:
        pass
    policy(tmp_path, f'[[routes]]\nmodel = "anthropic/{SMALL}"\n')
    with fr.run("after") as after, fr.run("child") as child:
        pass
    assert before.policy_hash is None
    assert after.policy_hash is not None and len(after.policy_hash) == 16
    assert child.policy_hash == after.policy_hash


# Row 3.2: step tags


def test_step_tag_tags_steps_as_context_manager_and_decorator(sink: Sink) -> None:
    llm = client(Server())

    @fr.step_tag("extract")
    def extract() -> None:
        llm.messages.create(**ASK)

    @fr.step_tag("summarise")
    async def summarise() -> None:
        with fr.step_tag("inner"):
            llm.messages.create(**ASK)

    with fr.run("triage"):
        with fr.step_tag("outer"):
            extract()
        asyncio.run(summarise())
        llm.messages.create(**ASK)
    assert [s.tags for s in sink.steps] == [
        ["outer", "extract"],
        ["summarise", "inner"],
        [],
    ]


# Row 3.3: routing


def test_a_tagged_step_is_routed_to_the_configured_model(
    tmp_path: Path, sink: Sink
) -> None:
    policy(
        tmp_path,
        f'[[routes]]\nmatch = {{ tag = "extract" }}\nmodel = "{SMALL}"\n',
    )
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        with fr.step_tag("extract"):
            reply = llm.messages.create(**ASK)
        llm.messages.create(**ASK)
    assert server.models == [SMALL, BIG]
    assert reply.model == SMALL
    routed, unrouted = model_steps(sink)
    assert (routed.model_requested, routed.model_used) == (BIG, SMALL)
    assert routed.cost_usd == pytest.approx(0.0015)  # priced as the small model
    assert (unrouted.model_requested, unrouted.model_used) == (BIG, BIG)


def test_routes_match_run_name_and_input_size_and_the_first_match_wins(
    tmp_path: Path,
) -> None:
    policy(
        tmp_path,
        f"""
[[routes]]
match = {{ run_name = "report", min_input_tokens = 50 }}
model = "claude-opus-5-5"

[[routes]]
match = {{ run_name = "report" }}
model = "{SMALL}"

[[routes]]  # another provider's model never applies to an Anthropic call
model = "openai/gpt-5-mini"
""",
    )
    server = Server()
    llm = client(server)
    long = {**ASK, "messages": [{"role": "user", "content": "x" * 400}]}
    with fr.run("report"):
        llm.messages.create(**ASK)
        llm.messages.create(**long)
    with fr.run("triage"):
        llm.messages.create(**ASK)
    assert server.models == [SMALL, "claude-opus-5-5", BIG]


def test_stream_helper_is_routed_sync_and_async(tmp_path: Path, sink: Sink) -> None:
    policy(tmp_path, f'[[routes]]\nmodel = "{SMALL}"\n')
    server = Server()
    llm, allm = client(server), async_client(server)

    async def read_async() -> str:
        async with allm.messages.stream(**ASK) as stream:
            return (await stream.get_final_message()).model

    with fr.run("triage"):
        with llm.messages.stream(**ASK) as stream:
            assert stream.get_final_message().model == SMALL
        assert asyncio.run(read_async()) == SMALL
    assert server.models == [SMALL, SMALL]
    assert [m.model_used for m in model_steps(sink)] == [SMALL, SMALL]


# Rows 3.4 and 3.5: budgets per run and per run name


def test_a_call_over_the_run_budget_is_denied_before_it_is_sent(
    tmp_path: Path, sink: Sink
) -> None:
    policy(tmp_path, "[budgets]\nper_run_usd = 0.01\n")
    server = Server()
    llm = client(server)
    with (
        pytest.raises(BudgetExceeded, match=r"per-run USD budget \$0\.0100") as info,
        fr.run("triage"),
    ):
        llm.messages.create(**ASK)  # spends $0.0075
        llm.messages.create(**ASK)  # up to ~$0.0025 more: over
    assert "estimated" in str(info.value)
    assert len(server.models) == 1
    sent, denied = sink.steps
    assert (sent.status, denied.status) == ("ok", "denied")
    assert denied.model is not None and denied.model.cost_usd == 0.0
    assert denied.model.error is not None
    assert denied.model.error.type == "flight_recorder.errors.BudgetExceeded"
    assert sink.runs[0].status == "denied"
    assert sink.runs[0].totals.cost_usd == pytest.approx(0.0075)


def test_a_caught_denial_still_marks_the_run_denied(tmp_path: Path, sink: Sink) -> None:
    policy(tmp_path, "[budgets]\nper_run_tokens = 1200\n")
    llm = client(Server())
    with fr.run("triage"):
        llm.messages.create(**ASK)  # 1,100 tokens
        with pytest.raises(BudgetExceeded, match="per-run token budget 1,200"):
            llm.messages.create(**ASK)
    assert sink.runs[0].status == "denied"


def test_a_run_name_budget_applies_to_that_name_only(tmp_path: Path) -> None:
    policy(tmp_path, "[budgets.by_run_name.triage]\nper_run_usd = 0.001\n")
    llm = client(Server())
    with fr.run("report"):
        llm.messages.create(**ASK)
    with pytest.raises(BudgetExceeded, match="for 'triage'"), fr.run("triage"):
        llm.messages.create(**ASK)


def test_on_breach_warn_sends_and_warns_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    policy(tmp_path, '[budgets]\nper_run_usd = 0.001\non_breach = "warn"\n')
    server = Server()
    llm = client(server)
    with caplog.at_level(logging.WARNING, "flight_recorder"), fr.run("triage"):
        llm.messages.create(**ASK)
        llm.messages.create(**ASK)
    assert len(server.models) == 2
    assert sum("sending anyway" in r.getMessage() for r in caplog.records) == 1


def test_on_breach_downgrade_sends_the_smaller_model_or_denies(
    tmp_path: Path, sink: Sink
) -> None:
    policy(
        tmp_path,
        f'[budgets]\nper_run_usd = 0.009\non_breach = "downgrade"\n'
        f'downgrade_to = {{ anthropic = "{SMALL}" }}\n',
    )
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        llm.messages.create(**ASK)  # $0.0075 on BIG
        llm.messages.create(**ASK)  # up to $0.0025 on BIG, $0.0005 on SMALL
        with pytest.raises(BudgetExceeded, match=f"and on {SMALL}"):
            llm.messages.create(**ASK)  # over even on SMALL
    assert server.models == [BIG, SMALL]
    assert model_steps(sink)[1].model_used == SMALL


def test_a_soft_warning_fires_once_near_the_limit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    policy(tmp_path, "[budgets]\nper_run_tokens = 5000\nsoft_warning = 0.5\n")
    llm = client(Server())
    with caplog.at_level(logging.WARNING, "flight_recorder"), fr.run("triage"):
        for _ in range(4):  # 1,100 tokens each, ~120 projected: the fourth
            # call would bring the run past 2,500
            llm.messages.create(**ASK)
    soft = [
        r for r in caplog.records if "at " in r.getMessage() and "%" in r.getMessage()
    ]
    assert len(soft) == 1
    assert "per-run token budget" in soft[0].getMessage()


def test_provider_token_counts_replace_the_estimate_when_asked(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    policy(tmp_path, '[budgets]\nper_run_tokens = 900\ntoken_count = "provider"\n')
    server = Server(counted=850)
    llm = client(server)
    with (
        pytest.raises(
            BudgetExceeded, match=r"850 input tokens \(counted by the provider\)"
        ),
        fr.run("triage"),
    ):
        llm.messages.create(**ASK)
    assert (server.counts, server.models) == (1, [])


def test_async_clients_ask_the_provider_to_count_too(tmp_path: Path) -> None:
    policy(tmp_path, '[budgets]\nper_run_tokens = 900\ntoken_count = "provider"\n')
    server = Server(counted=850)
    llm = async_client(server)

    async def agent() -> None:
        async with fr.run("triage"):
            await llm.messages.create(**ASK)

    with pytest.raises(BudgetExceeded, match="counted by the provider"):
        asyncio.run(agent())
    assert (server.counts, server.models) == (1, [])


def test_a_failed_token_count_falls_back_to_the_estimate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    policy(tmp_path, '[budgets]\nper_run_tokens = 900\ntoken_count = "provider"\n')
    server = Server(counted=-1)
    llm = client(server)
    with caplog.at_level(logging.WARNING, "flight_recorder"), fr.run("triage"):
        llm.messages.create(**ASK)  # ~115 estimated: under 900
        with pytest.raises(BudgetExceeded, match=r"\(estimated\)"):
            llm.messages.create(**ASK)
    assert server.counts == 2
    assert server.models == [BIG]
    assert sum("token counting failed" in r.getMessage() for r in caplog.records) == 1


def test_a_usd_budget_skips_an_unpriced_model(tmp_path: Path) -> None:
    policy(tmp_path, "[budgets]\nper_run_usd = 0.000001\n")
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        llm.messages.create(**{**ASK, "model": "claude-unpriced"})
    assert server.models == ["claude-unpriced"]


# Row 3.6: daily budgets


def test_a_daily_budget_is_shared_across_runs_and_processes(
    tmp_path: Path, sink: Sink
) -> None:
    policy(tmp_path, "[budgets]\nper_day_tokens = 2300\n")
    server = Server()
    llm = client(server)
    for _ in range(2):
        with fr.run("triage"):
            llm.messages.create(**ASK)  # 1,100 tokens each
    # Another process sees the same counters: configure afresh, as it would.
    policy(tmp_path, "[budgets]\nper_day_tokens = 2300\n")
    with (
        pytest.raises(BudgetExceeded, match="per-day token budget 2,300"),
        fr.run("triage"),
    ):
        llm.messages.create(**ASK)
    assert len(server.models) == 2
    assert (tmp_path / ".fr" / "state.sqlite").is_file()


def test_an_unpriced_call_makes_the_daily_usd_unknown_so_it_is_not_judged(
    tmp_path: Path,
) -> None:
    policy(tmp_path, "[budgets]\nper_day_usd = 0.009\n")
    server = Server()
    llm = client(server)
    with fr.run("triage"):
        llm.messages.create(**ASK)  # $0.0075 of today's $0.009
        llm.messages.create(**{**ASK, "model": "claude-unpriced"})  # cost unknown
        # $0.0075 + up to $0.0025 would break the budget, but today's total
        # is unknown now, and cost is never guessed.
        llm.messages.create(**ASK)
    assert len(server.models) == 3


def test_a_run_name_daily_budget_counts_that_name(tmp_path: Path) -> None:
    policy(tmp_path, "[budgets.by_run_name.triage]\nper_day_tokens = 1200\n")
    llm = client(Server())
    for name in ("report", "triage", "report"):
        with fr.run(name):
            llm.messages.create(**ASK)
    with (
        pytest.raises(BudgetExceeded, match="per-day token budget for 'triage'"),
        fr.run("triage"),
    ):
        llm.messages.create(**ASK)


# Row 3.7: fallbacks


@pytest.mark.parametrize("status", [429, 500, 529])
def test_a_retryable_error_falls_back_and_is_recorded(
    tmp_path: Path, sink: Sink, status: int
) -> None:
    policy(tmp_path, f'[fallback]\n"{BIG}" = ["claude-opus-5-5", "{SMALL}"]\n')
    server = Server(statuses=[status, status])
    llm = client(server)
    with fr.run("triage"):
        reply = llm.messages.create(**ASK)
    assert reply.model == SMALL
    assert server.models == [BIG, "claude-opus-5-5", SMALL]
    (call,) = model_steps(sink)
    assert call.model_used == SMALL
    assert [f.model for f in call.fallbacks] == [BIG, "claude-opus-5-5"]
    expected = {
        429: "RateLimitError",
        500: "InternalServerError",
        529: "OverloadedError",
    }
    assert all(f.error.type.endswith(expected[status]) for f in call.fallbacks)


def test_a_client_error_or_an_exhausted_chain_reaches_the_agent(
    tmp_path: Path, sink: Sink
) -> None:
    policy(tmp_path, f'[fallback]\n"{BIG}" = ["{SMALL}"]\n')
    server = Server(statuses=[400, 429, 429])
    llm = client(server)
    with fr.run("triage"):
        with pytest.raises(anthropic.BadRequestError):
            llm.messages.create(**ASK)
        with pytest.raises(anthropic.RateLimitError):
            llm.messages.create(**ASK)
    assert server.models == [BIG, BIG, SMALL]
    bad, exhausted = model_steps(sink)
    assert bad.fallbacks == []
    assert [f.model for f in exhausted.fallbacks] == [BIG]
    assert exhausted.error is not None and "RateLimitError" in exhausted.error.type


def test_a_fallback_is_checked_against_the_budget(tmp_path: Path, sink: Sink) -> None:
    policy(
        tmp_path,
        f'[budgets]\nper_run_usd = 0.002\n[fallback]\n"{SMALL}" = ["{BIG}"]\n',
    )
    server = Server(statuses=[429])
    llm = client(server)
    with (
        pytest.raises(BudgetExceeded, match=f"anthropic/{BIG} was not sent"),
        fr.run("triage"),
    ):
        llm.messages.create(**{**ASK, "model": SMALL})
    assert server.models == [SMALL]
    assert sink.steps[0].status == "denied"


def test_async_calls_fall_back_too(tmp_path: Path, sink: Sink) -> None:
    policy(tmp_path, f'[fallback]\n"{BIG}" = ["{SMALL}"]\n')
    server = Server(statuses=[503])
    llm = async_client(server)

    async def agent() -> str:
        async with fr.run("triage"):
            return (await llm.messages.create(**ASK)).model

    assert asyncio.run(agent()) == SMALL
    assert [f.model for f in model_steps(sink)[0].fallbacks] == [BIG]


def test_a_timeout_falls_back() -> None:
    timeout = anthropic.APITimeoutError(request=httpx2.Request("POST", "http://x"))
    assert engine.retryable(timeout)
    assert not engine.retryable(ValueError("no"))


# Row 3.9: routing and budgets fail open


def test_an_error_in_routing_sends_the_request_unchanged_and_warns_once(
    tmp_path: Path,
    sink: Sink,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy(
        tmp_path, f'[[routes]]\nmodel = "{SMALL}"\n[budgets]\nper_run_usd = 0.000001\n'
    )

    def broken(*args: Any) -> str:
        raise RuntimeError("route bug")

    monkeypatch.setattr(engine, "route", broken)
    server = Server()
    llm = client(server)
    with caplog.at_level(logging.WARNING, "flight_recorder"), fr.run("triage"):
        first = llm.messages.create(**ASK)
        llm.messages.create(**ASK)
    assert server.models == [BIG, BIG]
    assert first.model == BIG
    warnings = [r for r in caplog.records if "sent unchanged" in r.getMessage()]
    assert len(warnings) == 1
    assert sink.runs[0].status == "ok"


def test_an_unusable_state_store_fails_open(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / ".fr"
    blocker.write_text("a file where the store directory should be")
    file = tmp_path / "fr.toml"
    file.write_text("[budgets]\nper_day_usd = 1\n", encoding="utf-8")
    fr.configure(dir=blocker, policy=file)
    server = Server()
    llm = client(server)
    with caplog.at_level(logging.WARNING, "flight_recorder"), fr.run("triage"):
        llm.messages.create(**ASK)
    assert server.models == [BIG]
    assert any("sent unchanged" in r.getMessage() for r in caplog.records)


# Row 3.8: LangChain chat models route and fall back through their SDK clients


def test_a_langchain_model_is_routed_and_falls_back(tmp_path: Path, sink: Sink) -> None:
    from langchain_anthropic import ChatAnthropic

    policy(
        tmp_path,
        f'[[routes]]\nmatch = {{ tag = "extract" }}\nmodel = "{SMALL}"\n'
        f'[fallback]\n"{BIG}" = ["claude-opus-5-5"]\n',
    )
    server = Server(statuses=[429])
    settings: dict[str, Any] = {
        "model": BIG,
        "api_key": "sk-ant-test",
        "max_tokens": 100,
        "max_retries": 0,
    }
    model = ChatAnthropic(**settings)
    http = httpx2.Client(transport=httpx2.MockTransport(server))
    sdk = anthropic.Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    object.__setattr__(model, "_client", sdk)
    model = fr.langchain.wrap_model(model)
    with fr.run("triage"):
        fell_back = model.invoke("hi")
        with fr.step_tag("extract"):
            routed = model.invoke("hi")
    assert server.models == [BIG, "claude-opus-5-5", SMALL]
    assert fell_back.response_metadata["model"] == "claude-opus-5-5"
    assert routed.response_metadata["model"] == SMALL
    first, second = model_steps(sink)
    assert [f.model for f in first.fallbacks] == [BIG]
    assert second.model_used == SMALL


def test_an_openai_call_is_routed_within_openai(tmp_path: Path) -> None:
    from openai import OpenAI
    from test_openai import Server as OpenAIServer

    policy(
        tmp_path,
        f'[[routes]]\nmodel = "{SMALL}"\n'  # Anthropic's: never an OpenAI call's
        '[[routes]]\nmatch = { tag = "extract" }\nmodel = "openai/gpt-5-mini"\n',
    )
    server = OpenAIServer()
    http = httpx2.Client(transport=httpx2.MockTransport(server))
    llm: Any = fr.wrap(OpenAI(api_key="sk-test", http_client=http, max_retries=0))
    ask = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    with fr.run("triage"):
        llm.chat.completions.create(**ask)
        with fr.step_tag("extract"):
            llm.chat.completions.create(**ask)
            llm.responses.create(model="gpt-5", input="hi")
    assert [b["model"] for b in server.bodies] == ["gpt-5", "gpt-5-mini", "gpt-5-mini"]


def test_tags_and_fallbacks_are_on_the_spans(tmp_path: Path) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from flight_recorder.export.otel import OtelSink

    policy(tmp_path, f'[fallback]\n"{BIG}" = ["{SMALL}"]\n')
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    otel = OtelSink(provider)
    recorder.sinks = [otel]
    llm = client(Server(statuses=[429]))
    with fr.run("triage"), fr.step_tag("extract"):
        llm.messages.create(**ASK)
    assert otel.flush(timeout=10)
    otel.close()
    (chat,) = [s for s in spans.get_finished_spans() if s.name.startswith("chat")]
    attributes = dict(chat.attributes or {})
    assert attributes["fr.step.tags"] == ("extract",)
    assert attributes["fr.model.fallbacks"] == (BIG,)
    assert attributes["gen_ai.response.model"] == SMALL


def test_a_replay_neither_reads_nor_spends_the_daily_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flight_recorder.replay import engine as replay

    policy(tmp_path, "[budgets]\nper_day_tokens = 1200\n")
    llm = client(Server())
    with fr.run("triage"):
        llm.messages.create(**ASK)  # 1,100 of today's 1,200
    monkeypatch.setattr(replay, "replaying_now", lambda: True)
    for _ in range(2):  # would be denied, and would spend, outside a replay
        with fr.run("triage"):
            llm.messages.create(**ASK)
    monkeypatch.setattr(replay, "replaying_now", lambda: False)
    with fr.run("triage"), pytest.raises(BudgetExceeded, match="1,100 tokens spent"):
        llm.messages.create(**ASK)
