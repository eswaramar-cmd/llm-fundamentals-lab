"""Calculator tool — safe arithmetic evaluation using Python's ast module.

No arbitrary code execution. Only arithmetic operators are allowed:
+, -, *, /, //, %, **, parentheses, and numeric literals.
"""

from __future__ import annotations

import ast
import operator
from langchain_core.tools import ToolException
from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel, Field

_SUPPORTED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SUPPORTED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node restricted to arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant type")

    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_func = _SUPPORTED_BINOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        try:
            return op_func(left, right)
        except ZeroDivisionError:
            raise ToolException("Cannot divide by zero")

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_func = _SUPPORTED_UNARYOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Evaluate a safe arithmetic expression.

    Supports: integers, floats, +, -, *, /, //, %, **, parentheses,
    unary plus/minus. No variables, no function calls, no names.
    """
    if not expression or not expression.strip():
        raise ToolException("Expression is empty")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolException(f"Invalid expression: {exc}") from exc

    try:
        result = _safe_eval(tree)
    except ValueError as exc:
        raise ToolException(f"Invalid expression: {exc}") from exc

    # Convert to plain Python type
    if isinstance(result, complex):
        return result.real
    return float(result) if isinstance(result, float) else int(result)


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""
    expression: str = Field(..., description="Arithmetic expression to evaluate, e.g. '25 * 50 + 100'", min_length=1, max_length=512)


calculator = lc_tool(
    "calculator",
    description=(
        "Evaluate a safe arithmetic expression and return the numeric result. "
        "Supports addition, subtraction, multiplication, division, modulo, "
        "power, and parentheses. Input must be a string like '25 * 50 + 100'. "
        "Do not use this for non-arithmetic tasks."
    ),
    args_schema=CalculatorInput,
)(calculate)
