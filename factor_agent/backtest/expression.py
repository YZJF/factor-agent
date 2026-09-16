"""Safe factor-expression parser and vectorized evaluator (no eval/exec)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

from factor_agent.spec.expr import ALLOWED_FUNCS, ALLOWED_VARS


class ExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class Number:
    value: float


@dataclass(frozen=True)
class Variable:
    name: str


@dataclass(frozen=True)
class Unary:
    operator: str
    operand: Any


@dataclass(frozen=True)
class Binary:
    operator: str
    left: Any
    right: Any


@dataclass(frozen=True)
class Call:
    name: str
    arguments: tuple[Any, ...]


Token = tuple[str, str]
_TOKENS = re.compile(
    r"\s*(?:(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<variable>\$[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<operator>[+\-*/(),]))"
)
_ARITY = {
    "RANK": {1},
    "TS_MEAN": {2},
    "TS_STD": {2},
    "TS_MIN": {2},
    "TS_MAX": {2},
    "TS_PCTCHANGE": {2},
    "TS_SUM": {2},
    "INDUSTRY_NEUTRALIZE": {2},
    "DELAY": {2},
    "DELTA": {2},
    "LOG": {1},
    "ABS": {1},
    "SIGN": {1},
}
_PRECEDENCE = {"+": 10, "-": 10, "*": 20, "/": 20}


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    position = 0
    while position < len(text):
        match = _TOKENS.match(text, position)
        if not match:
            raise ExpressionError(f"unexpected token at offset {position}: {text[position:position + 16]!r}")
        kind = next(name for name, value in match.groupdict().items() if value is not None)
        tokens.append((kind, match.group(kind)))
        position = match.end()
    tokens.append(("eof", ""))
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.tokens = _tokenize(text)
        self.position = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.position]

    def take(self, value: str | None = None) -> Token:
        token = self.current
        if value is not None and token[1] != value:
            raise ExpressionError(f"expected {value!r}, got {token[1]!r}")
        self.position += 1
        return token

    def parse(self) -> Any:
        node = self.expression()
        if self.current[0] != "eof":
            raise ExpressionError(f"unexpected trailing token: {self.current[1]!r}")
        return node

    def expression(self, minimum_precedence: int = 0) -> Any:
        left = self.prefix()
        while self.current[1] in _PRECEDENCE:
            operator = self.current[1]
            precedence = _PRECEDENCE[operator]
            if precedence < minimum_precedence:
                break
            self.take()
            right = self.expression(precedence + 1)
            left = Binary(operator, left, right)
        return left

    def prefix(self) -> Any:
        kind, value = self.current
        if value in {"+", "-"}:
            self.take()
            return Unary(value, self.expression(30))
        if value == "(":
            self.take("(")
            node = self.expression()
            self.take(")")
            return node
        if kind == "number":
            self.take()
            return Number(float(value))
        if kind == "variable":
            self.take()
            name = value[1:]
            if name not in ALLOWED_VARS:
                raise ExpressionError(f"unsupported field: ${name}")
            return Variable(name)
        if kind == "name":
            self.take()
            name = value.upper()
            if name not in ALLOWED_FUNCS or name not in _ARITY:
                raise ExpressionError(f"unsupported function: {value}")
            self.take("(")
            arguments = []
            if self.current[1] != ")":
                while True:
                    arguments.append(self.expression())
                    if self.current[1] != ",":
                        break
                    self.take(",")
            self.take(")")
            if len(arguments) not in _ARITY[name]:
                raise ExpressionError(f"{name} expects {sorted(_ARITY[name])} arguments")
            return Call(name, tuple(arguments))
        raise ExpressionError(f"expected expression, got {value!r}")


def parse_expression(text: str) -> Any:
    if len(text) > 4096:
        raise ExpressionError("expression is too long")
    return _Parser(text).parse()


def required_fields(node: Any) -> set[str]:
    if isinstance(node, Variable):
        return {node.name}
    if isinstance(node, (Number,)):
        return set()
    if isinstance(node, Unary):
        return required_fields(node.operand)
    if isinstance(node, Binary):
        return required_fields(node.left) | required_fields(node.right)
    if isinstance(node, Call):
        fields: set[str] = set()
        for argument in node.arguments:
            fields |= required_fields(argument)
        return fields
    raise TypeError(type(node))


def max_lookback(node: Any) -> int:
    own = 0
    children: Iterable[Any] = ()
    if isinstance(node, Unary):
        children = (node.operand,)
    elif isinstance(node, Binary):
        children = (node.left, node.right)
    elif isinstance(node, Call):
        children = node.arguments
        if node.name.startswith("TS_") or node.name in {"DELAY", "DELTA"}:
            if not isinstance(node.arguments[-1], Number) or not node.arguments[-1].value.is_integer():
                raise ExpressionError(f"{node.name} window must be an integer literal")
            own = int(node.arguments[-1].value)
            if own < 1 or own > 504:
                raise ExpressionError(f"{node.name} window must be in [1, 504]")
    return own + max((max_lookback(child) for child in children), default=0)


def _pandas():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError('expression evaluation requires: pip install -e ".[backtest]"') from exc
    return np, pd


def stack_all(frame: Any):
    """Stack without dropping NaNs across pandas 2.x and 3.x."""

    try:
        return frame.stack(future_stack=True)
    except TypeError:  # pandas 2.0
        return frame.stack(dropna=False)


def _window(argument: Any, function: str) -> int:
    if not isinstance(argument, Number) or not argument.value.is_integer():
        raise ExpressionError(f"{function} window must be an integer literal")
    value = int(argument.value)
    if value < 1 or value > 504:
        raise ExpressionError(f"{function} window must be in [1, 504]")
    return value


def _time_series(series: Any, operation: str, window: int):
    wide = series.unstack("instrument")
    if operation == "mean":
        result = wide.rolling(window, min_periods=window).mean()
    elif operation == "std":
        result = wide.rolling(window, min_periods=window).std(ddof=0)
    elif operation == "min":
        result = wide.rolling(window, min_periods=window).min()
    elif operation == "max":
        result = wide.rolling(window, min_periods=window).max()
    elif operation == "sum":
        result = wide.rolling(window, min_periods=window).sum()
    elif operation == "pctchange":
        result = wide / wide.shift(window) - 1.0
    elif operation == "delay":
        result = wide.shift(window)
    elif operation == "delta":
        result = wide - wide.shift(window)
    else:  # pragma: no cover - internal contract
        raise AssertionError(operation)
    return stack_all(result).reindex(series.index)


def evaluate_expression(node: Any, frame: Any):
    """Evaluate an AST against a date/instrument MultiIndex DataFrame."""

    np, pd = _pandas()
    if frame.index.names != ["datetime", "instrument"]:
        raise ValueError("market frame index must be ['datetime', 'instrument']")

    def evaluate(item: Any):
        if isinstance(item, Number):
            return item.value
        if isinstance(item, Variable):
            if item.name not in frame:
                raise ExpressionError(f"market data is missing field: ${item.name}")
            return frame[item.name]
        if isinstance(item, Unary):
            value = evaluate(item.operand)
            return value if item.operator == "+" else -value
        if isinstance(item, Binary):
            left, right = evaluate(item.left), evaluate(item.right)
            if item.operator == "+":
                return left + right
            if item.operator == "-":
                return left - right
            if item.operator == "*":
                return left * right
            denominator = right.replace(0, np.nan) if hasattr(right, "replace") else right
            if not hasattr(right, "replace") and right == 0:
                raise ExpressionError("division by zero")
            return left / denominator
        if isinstance(item, Call):
            name = item.name
            if name == "RANK":
                value = evaluate(item.arguments[0])
                return value.groupby(level="datetime").rank(pct=True, method="average")
            if name == "INDUSTRY_NEUTRALIZE":
                value = evaluate(item.arguments[0])
                groups = evaluate(item.arguments[1])
                table = pd.DataFrame({"value": value, "group": groups})
                keys = [table.index.get_level_values("datetime"), table["group"]]
                centered = table["value"] - table.groupby(keys)["value"].transform("mean")
                scale = centered.groupby(keys).transform(lambda values: values.std(ddof=0))
                return centered / scale.replace(0, np.nan)
            if name == "LOG":
                value = evaluate(item.arguments[0])
                return np.log(value.where(value > 0)) if hasattr(value, "where") else math.log(value)
            if name == "ABS":
                return abs(evaluate(item.arguments[0]))
            if name == "SIGN":
                return np.sign(evaluate(item.arguments[0]))
            window = _window(item.arguments[1], name)
            value = evaluate(item.arguments[0])
            operation = {
                "TS_MEAN": "mean",
                "TS_STD": "std",
                "TS_MIN": "min",
                "TS_MAX": "max",
                "TS_SUM": "sum",
                "TS_PCTCHANGE": "pctchange",
                "DELAY": "delay",
                "DELTA": "delta",
            }.get(name)
            if operation is None:
                raise ExpressionError(f"unsupported function: {name}")
            return _time_series(value, operation, window)
        raise TypeError(type(item))

    result = evaluate(node)
    if not isinstance(result, pd.Series):
        result = pd.Series(float(result), index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan)
