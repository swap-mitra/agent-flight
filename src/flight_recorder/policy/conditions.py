"""Gate rule conditions: ``when = "args.amount > 500"`` (SDD Rule conditions).

Never ``eval()``. A condition is parsed with ``ast.parse`` and every node is
checked against a whitelist when the policy loads: comparisons, ``and``,
``or``, ``not``, unary minus, literals (and lists or tuples of them), and
attribute or constant-key access on ``args``, ``run`` and ``env``. Evaluation
walks the same tree; a missing argument is an error, which the gate treats
as a failure and so fails closed.
"""

import ast
import operator
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

NAMES = ("args", "run", "env")

_COMPARE: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_ALLOWED = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Constant,
    ast.List,
    ast.Tuple,
    *_COMPARE,
)


@lru_cache(maxsize=512)
def parse(source: str) -> ast.Expression:
    """The checked tree of a condition; ValueError names what is not allowed."""
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"when {source!r} is not an expression: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(  # noqa: TRY004  pydantic reports ValueErrors
                f"when {source!r}: {type(node).__name__} is not allowed"
                " (comparisons, and/or/not, literals, args/run/env fields)"
            )
        if isinstance(node, ast.Name) and node.id not in NAMES:
            raise ValueError(f"when {source!r}: unknown name {node.id!r}; use {NAMES}")
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Constant):
            raise ValueError(f"when {source!r}: a subscript must be a literal key")  # noqa: TRY004
    return tree


def evaluate(source: str, names: Mapping[str, Any]) -> bool:
    """Whether the condition holds for ``names`` (args, run and env)."""
    return bool(_eval(parse(source).body, names))


def _eval(node: ast.expr, names: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.Attribute):
        return _field(_eval(node.value, names), node.attr)
    if isinstance(node, ast.Subscript):
        key = node.slice
        assert isinstance(key, ast.Constant)  # checked by parse()
        return _field(_eval(node.value, names), key.value)
    if isinstance(node, ast.List | ast.Tuple):
        items: list[Any] = [_eval(e, names) for e in node.elts]
        return items
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, names)
        return not value if isinstance(node.op, ast.Not) else -value
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval(v, names) for v in node.values)
        return any(_eval(v, names) for v in node.values)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, names)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _eval(right_node, names)
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    raise ValueError(f"{type(node).__name__} is not allowed")  # unreachable


def _field(value: Any, key: Any) -> Any:
    kind = type(value).__name__
    if isinstance(value, Mapping):
        if key not in value:
            raise KeyError(f"no field {key!r}")
        return value[key]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list) and isinstance(key, int):
        return value[key]  # pyright: ignore[reportUnknownVariableType]
    raise KeyError(f"no field {key!r} on {kind}")
