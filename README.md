# Flight Recorder

Record every step of an AI agent run: each model call (messages, response, tokens, cost, latency) and each tool call (arguments, result, effect), in local JSONL files you can list and read.

v0.1 records. Replay in CI, cost budgets and approval gates come in later releases.

## Quickstart

Python 3.11 or newer. No API key needed.

```bash
pip install flight-recorder-ai
```

Save this as `agent.py`:

```python
import flight_recorder as fr


@fr.tool(effect="read")
def lookup_order(order_id: str) -> dict[str, str]:
    return {"order_id": order_id, "status": "shipped"}


@fr.tool(effect="irreversible")
def issue_refund(order_id: str, amount: float) -> str:
    return f"refunded {amount} on {order_id}"


with fr.run("refund-desk", inputs={"order_id": "A-17"}) as run:
    order = lookup_order("A-17")
    run.output = issue_refund(order["order_id"], 19.99)

print(run.run_id)
```

Run it, then look at what was recorded:

```bash
python agent.py          # prints the run id
fr runs                  # list runs, newest first
fr show <run id>         # every step, in order (a unique prefix of the id works)
```

Runs are written to `.fr/` in the current directory (set `FR_DIR` or `fr.configure(dir=...)` to change it).

## Recording model calls

Wrap the client instance. Calls, streams and async calls behave exactly as before.

```bash
pip install 'flight-recorder-ai[openai]'      # or [anthropic]
```

```python
from openai import OpenAI

client = fr.wrap(OpenAI())

with fr.run("triage"):
    client.chat.completions.create(model="gpt-4.1-mini", messages=[...])
```

`fr.wrap` also takes `AsyncOpenAI`, `Anthropic` and `AsyncAnthropic`. Cost comes from the shipped pricing table; a model missing from it records a null cost and warns once. Add or override prices with `fr.configure(pricing="prices.toml")`.

## LangChain and LangGraph

```bash
pip install 'flight-recorder-ai[langchain]'
```

```python
model = fr.langchain.wrap_model(ChatOpenAI(model="gpt-4.1-mini"))
tools = fr.langchain.wrap_tools(tools, effects={"search": "read"})
```

Steps inside a LangGraph graph are tagged with their node (`node:agent`).

## What is recorded, and what is not

- Secrets are redacted before anything is written. Add your own rules with `fr.configure(redact=[fr.redact.pattern(...), fr.redact.path(...)])`.
- Recording fails open: if a write fails, the agent keeps running and one warning is logged.
- Nothing leaves your machine except your own model and tool calls. With `fr.configure(otel=True)` (the `otel` extra), each run is also exported to your OpenTelemetry tracer provider, with message content off by default.
- Overhead is under 2 ms per call, checked in CI.

## License

Apache-2.0
