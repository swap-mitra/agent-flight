"""The Run record: every feature reads or writes this one structure.

A cassette holds a Run header, one Step per line, then a footer. These models
validate that file at the boundary, so everything behind it can trust the data.
"""

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, JsonValue, model_validator

SCHEMA_VERSION = 1

Effect = Literal["read", "write", "irreversible"]


class ErrorInfo(BaseModel):
    """An exception raised by a model or tool call, recorded by type and message."""

    type: str
    message: str


class Usage(BaseModel):
    # The whole prompt, cached or not, on every provider.
    input_tokens: int
    output_tokens: int
    # Parts of input_tokens read from and written to the provider's prompt cache.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Fallback(BaseModel):
    """A model attempt that failed before a fallback model took the call."""

    model: str
    error: ErrorInfo


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
    # Earlier attempts in the policy's fallback chain, in order (FR-B5).
    fallbacks: list[Fallback] = []
    error: ErrorInfo | None = None


class ToolCall(BaseModel):
    name: str
    effect: Effect
    # Bound by parameter name, so matching ignores positional order.
    args: dict[str, JsonValue]
    result: JsonValue | None = None
    error: ErrorInfo | None = None
    call_id: str | None = None


class GateDecision(BaseModel):
    """What the action gate decided for one tool call (FR-G7)."""

    # What the rules said: allow, approve (ask a person) or deny.
    action: Literal["allow", "approve", "deny"]
    # What happened: allowed, or approved, rejected or expired after asking,
    # or denied by a rule or by an error in the gate (which fails closed).
    outcome: Literal["allowed", "approved", "rejected", "expired", "denied"]
    rule: str | None = None  # the matching rule; None for the default
    reason: str | None = None
    approver: str | None = None
    decided_at: datetime | None = None
    approval_id: str | None = None


class Step(BaseModel):
    seq: int
    kind: Literal["model", "tool"]
    parallel_group: str | None = None
    # The step open when this one started (e.g. a model call inside a tool).
    parent_seq: int | None = None
    tags: list[str] = []
    source: Literal["live", "replayed", "stubbed"] = "live"
    # UTC, when the step opened; None in cassettes recorded before it existed.
    started_at: datetime | None = None
    latency_ms: float
    status: Literal["ok", "error", "denied"]
    model: ModelCall | None = None
    tool: ToolCall | None = None
    gate: GateDecision | None = None  # set when the policy has a gate

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
    # The policy file in force when the run started (its content's hash).
    policy_hash: str | None = None
    totals: Totals = Totals()
