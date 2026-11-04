# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/swap-mitra/ai-flight-recorder.git
cd ai-flight-recorder
uv sync
```

Before you push, run what CI runs:

```bash
uv run ruff format . && uv run ruff check .
uv run pyright          # strict mode, must be clean
uv run pytest
uv run python benchmarks/overhead.py
```

The dogfood suite needs a [Heptapod](https://github.com/swap-mitra/heptapod) checkout:

```bash
HEPTAPOD_DIR=/path/to/heptapod uv run --project dogfood pytest dogfood
```

## Conventions

- **Interception is invisible.** A wrapped call returns or raises exactly what the real call did. Every change to an interceptor needs a test against the real SDK on a mock HTTP transport.
- **Recording fails open.** A storage or export failure logs once and lets the agent continue. Add a row to the fault-injection matrix in `tests/test_faults.py` for any new failure path.
- **Never guess cost.** An unpriced model or modifier records `cost: null`.
- **No new runtime dependencies** without discussing it in an issue first. The core depends on pydantic only; SDK integrations are optional extras.
- **Tests at the public contract.** Test what a user sees (the recorded run, the replay verdict), not internals. A bug fix starts with a failing test.
- **Commits:** imperative subject under about 72 characters, and a body that explains why.

## Reporting a security issue

Please don't open a public issue. See [SECURITY.md](SECURITY.md).
