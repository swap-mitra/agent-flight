"""Redaction rules, applied to every record before a sink sees it (FR-R6).

A rule is a function from a record (run header, step or run footer, as JSON)
to the redacted record. pattern() and path() build the common ones.
"""

import re
from collections.abc import Callable

from pydantic import JsonValue

REDACTED = "[REDACTED]"

Record = dict[str, JsonValue]
Rule = Callable[[Record], Record]


def pattern(regex: str) -> Rule:
    """Replace every match of regex inside any string value."""
    compiled = re.compile(regex)

    def walk(node: JsonValue) -> JsonValue:
        if isinstance(node, str):
            return compiled.sub(REDACTED, node)
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        return node

    def rule(record: Record) -> Record:
        return {key: walk(value) for key, value in record.items()}

    return rule


# .name / ..name / .* / [n] / [*]
_TOKEN = re.compile(r"\.\.([^.\[\]]+)|\.([^.\[\]]+)|\[(\d+|\*)\]")


def path(expr: str) -> Rule:
    """Replace the value at a JSONPath subset: $, .name, ..name, .*, [n], [*]."""
    if not expr.startswith("$"):
        raise ValueError(f"path must start with '$': {expr!r}")
    steps: list[tuple[bool, str]] = []  # (any depth, key)
    pos = 1
    while pos < len(expr):
        match = _TOKEN.match(expr, pos)
        if match is None:
            raise ValueError(f"invalid path {expr!r} at position {pos}")
        deep, child, index = match.groups()
        steps.append((deep is not None, deep or child or index))
        pos = match.end()
    if not steps:
        raise ValueError("path '$' would redact the whole record")

    def rule(record: Record) -> Record:
        result = _redact_at(record, steps)
        assert isinstance(result, dict)
        return result

    return rule


def _redact_at(node: JsonValue, steps: list[tuple[bool, str]]) -> JsonValue:
    if not steps:
        return REDACTED
    (deep, key), rest = steps[0], steps[1:]
    if isinstance(node, dict):
        node = {
            k: _redact_at(v, rest) if key in ("*", k) else v for k, v in node.items()
        }
    elif isinstance(node, list):
        node = [
            _redact_at(v, rest) if key in ("*", str(i)) else v
            for i, v in enumerate(node)
        ]
    if deep:
        # ..name also matches below this level; the steps go on unchanged.
        if isinstance(node, dict):
            node = {k: _redact_at(v, steps) for k, v in node.items()}
        elif isinstance(node, list):
            node = [_redact_at(v, steps) for v in node]
    return node


_SECRET_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "x_api_key",
    "password",
    "secret",
    "client_secret",
    "access_token",
    "refresh_token",
}


def _secret_keys(record: Record) -> Record:
    def walk(node: JsonValue) -> JsonValue:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            return {
                key: REDACTED
                if key.lower().replace("-", "_") in _SECRET_KEYS
                else walk(value)
                for key, value in node.items()
            }
        return node

    result = walk(record)
    assert isinstance(result, dict)
    return result


DEFAULT_RULES: list[Rule] = [
    _secret_keys,
    pattern(r"\bsk-[A-Za-z0-9_-]{20,}"),  # OpenAI, Anthropic (sk-ant-...)
    pattern(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key id
    pattern(r"\bgh[pousr]_[A-Za-z0-9]{36,}|\bgithub_pat_[A-Za-z0-9_]{22,}"),
    pattern(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),  # Slack
    pattern(r"\bAIza[0-9A-Za-z_-]{35}"),  # Google API key
    pattern(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    pattern(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    ),
]

# Set by fr.configure(): the defaults, then the configured rules.
rules: list[Rule] = list(DEFAULT_RULES)


def apply(record: Record) -> Record:
    for rule in rules:
        record = rule(record)
    return record
