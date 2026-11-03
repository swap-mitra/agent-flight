import logging
from collections.abc import Iterator

import pytest

import flight_recorder as fr
from flight_recorder import recorder, redact
from flight_recorder.model import Run, Step
from flight_recorder.redact import REDACTED, Record

OPENAI_KEY = "sk-proj-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
ANTHROPIC_KEY = "sk-ant-api03-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2"
BEARER = "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"


class JsonSink:
    """Keeps every record as the JSON a cassette or exporter would write."""

    def __init__(self) -> None:
        self.written: list[str] = []
        self.steps: list[Step] = []

    def run_started(self, run: Run) -> None:
        self.written.append(run.model_dump_json())

    def step_closed(self, run_id: str, step: Step) -> None:
        self.written.append(step.model_dump_json())
        self.steps.append(step)

    def run_finished(self, run: Run) -> None:
        self.written.append(run.model_dump_json())


@pytest.fixture
def sink(monkeypatch: pytest.MonkeyPatch) -> Iterator[JsonSink]:
    sink = JsonSink()
    monkeypatch.setattr(recorder, "sinks", [sink])
    monkeypatch.setattr(redact, "rules", list(redact.DEFAULT_RULES))
    yield sink


@fr.tool(effect="write")
def call_api(url: str, headers: dict[str, str], body: str) -> str:
    return "ok"


def test_seeded_api_keys_and_bearer_token_never_reach_a_sink(sink: JsonSink) -> None:
    with fr.run("triage", inputs={"note": f"key is {OPENAI_KEY}"}) as run:
        call_api(
            "https://api.example.com",
            {"Authorization": BEARER, "X-Request-Id": "r1"},
            f"anthropic {ANTHROPIC_KEY}",
        )
        run.output = f"done with {OPENAI_KEY}"

    written = "\n".join(sink.written)
    for secret in (OPENAI_KEY, ANTHROPIC_KEY, "eyJhbGciOiJIUzI1NiJ9"):
        assert secret not in written
    assert REDACTED in written
    tool = sink.steps[0].tool
    assert tool is not None
    assert tool.args["headers"] == {"Authorization": REDACTED, "X-Request-Id": "r1"}


def test_the_agents_own_run_is_not_redacted(sink: JsonSink) -> None:
    with fr.run("triage", inputs={"api_key": "k"}) as run:
        pass
    assert run.inputs == {"api_key": "k"}
    assert '"api_key":"k"' not in "".join(sink.written)


def test_secret_key_names_match_exactly_and_ignore_case() -> None:
    record: Record = {
        "Api-Key": "a",
        "PASSWORD": "b",
        "nested": [{"client_secret": "c"}],
        "input_tokens": 10,
        "password_hint": "d",
    }
    assert redact.apply(record) == {
        "Api-Key": REDACTED,
        "PASSWORD": REDACTED,
        "nested": [{"client_secret": REDACTED}],
        "input_tokens": 10,
        "password_hint": "d",
    }


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("$.tool.args.card", {"tool": {"args": {"card": REDACTED, "n": 1}}}),
        ("$.tool.args.*", {"tool": {"args": {"card": REDACTED, "n": REDACTED}}}),
        ("$..card", {"tool": {"args": {"card": REDACTED, "n": 1}}}),
    ],
)
def test_path_rules_on_objects(expr: str, expected: Record) -> None:
    record: Record = {"tool": {"args": {"card": "4111", "n": 1}}}
    assert redact.path(expr)(record) == expected


def test_path_rules_on_lists_and_nested_any_depth() -> None:
    record: Record = {
        "messages": [{"content": "a"}, {"content": "b", "meta": {"content": "c"}}]
    }
    assert redact.path("$.messages[1].content")(record) == {
        "messages": [{"content": "a"}, {"content": REDACTED, "meta": {"content": "c"}}]
    }
    assert redact.path("$.messages[*].content")(record)["messages"] == [
        {"content": REDACTED},
        {"content": REDACTED, "meta": {"content": "c"}},
    ]
    assert redact.path("$..content")(record) == {
        "messages": [
            {"content": REDACTED},
            {"content": REDACTED, "meta": {"content": REDACTED}},
        ]
    }


@pytest.mark.parametrize("expr", ["tool.args", "$", "$.a..", "$.a[x]"])
def test_invalid_paths_are_rejected_when_the_rule_is_made(expr: str) -> None:
    with pytest.raises(ValueError):
        redact.path(expr)


def test_configured_pattern_path_and_callable_rules_apply(sink: JsonSink) -> None:
    def drop_result(record: Record) -> Record:
        tool = record.get("tool")
        if isinstance(tool, dict):
            tool["result"] = None
        return record

    fr.configure(
        redact=[
            redact.pattern(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
            redact.path("$.tool.args.url"),
            drop_result,
        ]
    )
    with fr.run("triage"):
        call_api("https://internal", {}, "ssn 123-45-6789")

    tool = sink.steps[0].tool
    assert tool is not None
    assert tool.args["url"] == REDACTED
    assert tool.args["body"] == f"ssn {REDACTED}"
    assert tool.result is None


def test_configure_replaces_rules_and_keeps_defaults(sink: JsonSink) -> None:
    fr.configure(redact=[redact.path("$.tool.args.url")])
    fr.configure()
    assert redact.rules == redact.DEFAULT_RULES


def test_failing_rule_drops_the_record_and_the_agent_continues(
    sink: JsonSink, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(record: Record) -> Record:
        raise RuntimeError("rule bug")

    fr.configure(redact=[broken])
    with (
        caplog.at_level(logging.WARNING, logger="flight_recorder"),
        fr.run("triage") as run,
    ):
        result = call_api("u", {}, "b")

    assert result == "ok"
    assert sink.written == []
    assert run.status == "ok"
    assert len(caplog.records) == 1
