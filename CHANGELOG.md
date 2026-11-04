# Changelog

## 0.1.0 (unreleased)

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
