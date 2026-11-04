# Changelog

## 0.2.0 (2026-11-04)

Replay: a recorded run becomes a regression test.

- A pytest plugin loads when the package is installed. Mark a test with `@pytest.mark.fr(...)` and use the `fr_run` fixture: `pytest --fr-mode=record` records the case once against live services, and plain `pytest` replays it.
- In `tools` replay mode, tool results are served from the recording and the model stays live, to test prompt and model changes. In `full` mode, model responses are served too, streams included, so the run is free, offline and deterministic.
- Each case gets a verdict: `same`, `changed` (the agent took another path but reached the same output) or `failed` (an unmatched call, an exception, a status change or a different output). `--fr-on-change=fail` makes changes fail too.
- Replay never calls an irreversible tool for real. A call with no recorded match raises a divergence that names the step and the field that differs.
- `fr.matcher()` sets how a tool's arguments are compared, which paths to ignore and what to do on divergence. `lookahead` lets read calls happen in a different order.
- `repeats` and `min_pass` run a case several times for agents that aren't deterministic.
- Child runs are recorded next to their parent and replayed with it.
- `--fr-report DIR` writes `report.json` and `report.md`, ready to post on a pull request. `fr diff` compares two runs and `fr accept` (or `pytest --fr-update`) makes a replay the new baseline.

## 0.1.1 (2026-11-04)

- OpenAI Responses calls that use web search or file search are now priced, with each tool call's fee added to the token cost, instead of recording a null cost. Code interpreter is priced by its tokens only, since its container time is billed per session. Chat Completions web search still records null.
- With OpenTelemetry content capture on, model content is exported in the GenAI conventions' attributes (`gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages`) and message schema, with streamed output rebuilt from its chunks. This replaces the provider-native `fr.model.messages` and `fr.model.response` attributes.

## 0.1.0 (2026-11-04)

The first release: recording.

- `fr.wrap()` records OpenAI (Chat Completions and Responses) and Anthropic clients, sync, async and streaming, without changing what the agent sees.
- `@fr.tool(effect=...)` records tool calls: arguments, result or exception, side-effect class and latency. Exceptions reach the agent unchanged.
- `fr.run()` groups steps into a run with inputs, output, status, total tokens and cost. Nested runs link to their parent.
- Runs are stored as JSONL cassettes, with a SQLite index rebuilt from them by `fr reindex`. `fr runs` lists runs and `fr show` prints one step by step.
- Cost comes from a shipped, overridable pricing table. A model missing from it records a null cost and warns once.
- Secrets are redacted before anything is written or exported, and you can add regex, JSON path or callable rules.
- `fr.configure(otel=True)` exports each run to OpenTelemetry as one trace of GenAI spans, with message content off by default.
- `fr.langchain.wrap_model()` and `fr.langchain.wrap_tools()` record LangChain chat models and tools, and steps inside LangGraph are tagged with their node.
- Recording fails open: a storage or export failure logs one warning per run and the agent continues.
- Interception overhead is under 2 ms median per call, checked in CI.
