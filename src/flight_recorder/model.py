"""The Run record: every feature reads or writes this one structure.

A cassette holds a Run header, one Step per line, then a footer. These models
validate that file at the boundary, so everything behind it can trust the data.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, JsonValue, model_validator

SCHEMA_VERSION = 1

Effect = Literal["read", "write", "irreversible"]


class ErrorInfo(BaseModel):
    """An exception raised by a model or tool call, recorded by type and message."""

    type: str
    message: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ModelCall(BaseModel):
    provider: str
    model_requested: str
    # Differs from model_requested once routing or a fallback changes the model.
    model_used: str
    params: dict[str, JsonValue]
    messages: JsonValue
    response: JsonValue | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    # None when the model is not in the pricing table; cost is never guessed.
    cost_usd: float | None = None
    error: ErrorInfo | None = None


class ToolCall(BaseModel):
    name: str
    effect: Effect
    # Bound by parameter name, so matching ignores positional order.
    args: dict[str, JsonValue]
    result: JsonValue | None = None
    error: ErrorInfo | None = None
    call_id: str | None = None


class Step(BaseModel):
    seq: int
    kind: Literal["model", "tool"]
    parallel_group: str | None = None
    tags: list[str] = []
    source: Literal["live", "replayed", "stubbed"] = "live"
    latency_ms: float
    status: Literal["ok", "error", "denied"]
    model: ModelCall | None = None
    tool: ToolCall | None = None

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> Self:
        # Exactly one payload, and it must be the one named by kind.
        if (self.kind == "model") != (self.model is not None) or (
            self.kind == "tool"
        ) != (self.tool is not None):
            raise ValueError(f"a {self.kind} step needs exactly a {self.kind} payload")
        return self


class Totals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    # None when any model step was unpriced, so a partial sum is never reported.
    cost_usd: float | None = 0.0
    steps: int = 0


class Run(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    name: str
    parent_run_id: str | None = None
    status: Literal["ok", "error", "denied", "incomplete"]
    inputs: JsonValue
    # Agent code sets this to any value; it is converted to JSON when the run
    # closes. A cassette only ever holds JSON, so loading loses no validation.
    output: Any = None
    tags: dict[str, str] = {}
    totals: Totals = Totals()
