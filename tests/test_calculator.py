"""Tests for the calculator tool."""

from __future__ import annotations

import pytest
from langchain_core.tools import ToolException

from backend.agent.tools.calculator import calculate


class TestCalculator:
    """Test safe arithmetic evaluation."""

    def test_simple_addition(self):
        assert calculate("25 + 40") == 65

    def test_multiplication(self):
        assert calculate("25 * 50") == 1250

    def test_division(self):
        assert calculate("100 / 4") == 25

    def test_complex_expression(self):
        assert calculate("(100 + 50) * 3 - 200") == 250

    def test_power(self):
        assert calculate("2 ** 10") == 1024

    def test_modulo(self):
        assert calculate("17 % 5") == 2

    def test_floats(self):
        assert calculate("3.14 * 2") == pytest.approx(6.28)

    def test_unary_minus(self):
        assert calculate("-5 + 10") == 5

    def test_empty_expression(self):
        with pytest.raises(ToolException, match="Expression is empty"):
            calculate("")

    def test_whitespace_only(self):
        with pytest.raises(ToolException, match="Expression is empty"):
            calculate("   ")

    def test_invalid_syntax(self):
        with pytest.raises(ToolException, match="Invalid expression"):
            calculate("25 +")

    def test_no_variables(self):
        """The calculator should reject non-arithmetic input."""
        with pytest.raises(ToolException):
            calculate("print('hello')")

    def test_no_code_execution(self):
        """Ensure the calculator does not execute arbitrary code."""
        with pytest.raises(ToolException):
            calculate("__import__('os').system('echo hacked')")

    def test_no_division_by_zero(self):
        with pytest.raises(ToolException, match="Cannot divide by zero"):
            calculate("10 / 0")
