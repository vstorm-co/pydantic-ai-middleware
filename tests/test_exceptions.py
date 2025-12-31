"""Tests for exceptions module."""

from __future__ import annotations

import pytest

from pydantic_ai_middleware import (
    AggregationFailed,
    AggregationStrategy,
    GuardrailTimeout,
    InputBlocked,
    MiddlewareError,
    OutputBlocked,
    ParallelExecutionFailed,
    ToolBlocked,
)


class TestMiddlewareError:
    """Tests for MiddlewareError."""

    def test_middleware_error_is_exception(self) -> None:
        """Test that MiddlewareError is an Exception."""
        error = MiddlewareError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"


class TestInputBlocked:
    """Tests for InputBlocked exception."""

    def test_input_blocked_default_reason(self) -> None:
        """Test InputBlocked with default reason."""
        error = InputBlocked()
        assert error.reason == "Input blocked"
        assert str(error) == "Input blocked"
        assert isinstance(error, MiddlewareError)

    def test_input_blocked_custom_reason(self) -> None:
        """Test InputBlocked with custom reason."""
        error = InputBlocked("Custom reason")
        assert error.reason == "Custom reason"
        assert str(error) == "Custom reason"


class TestToolBlocked:
    """Tests for ToolBlocked exception."""

    def test_tool_blocked_default_reason(self) -> None:
        """Test ToolBlocked with default reason."""
        error = ToolBlocked("my_tool")
        assert error.tool_name == "my_tool"
        assert error.reason == "Tool blocked"
        assert str(error) == "Tool 'my_tool' blocked: Tool blocked"
        assert isinstance(error, MiddlewareError)

    def test_tool_blocked_custom_reason(self) -> None:
        """Test ToolBlocked with custom reason."""
        error = ToolBlocked("dangerous_tool", "Not authorized")
        assert error.tool_name == "dangerous_tool"
        assert error.reason == "Not authorized"
        assert str(error) == "Tool 'dangerous_tool' blocked: Not authorized"


class TestOutputBlocked:
    """Tests for OutputBlocked exception."""

    def test_output_blocked_default_reason(self) -> None:
        """Test OutputBlocked with default reason."""
        error = OutputBlocked()
        assert error.reason == "Output blocked"
        assert str(error) == "Output blocked"
        assert isinstance(error, MiddlewareError)

    def test_output_blocked_custom_reason(self) -> None:
        """Test OutputBlocked with custom reason."""
        error = OutputBlocked("Contains PII")
        assert error.reason == "Contains PII"
        assert str(error) == "Contains PII"


class TestParallelExecutionFailed:
    """Tests for ParallelExecutionFailed exception."""

    def test_parallel_execution_failed_basic(self) -> None:
        """Test ParallelExecutionFailed with basic parameters."""
        errors = [ValueError("error1"), ValueError("error2")]
        error = ParallelExecutionFailed(errors=errors)

        assert error.errors == errors
        assert error.failed_count == 2
        assert error.success_count == 0
        assert "2 failed" in str(error)
        assert isinstance(error, MiddlewareError)

    def test_parallel_execution_failed_with_results(self) -> None:
        """Test ParallelExecutionFailed with results."""
        errors = [ValueError("error1")]
        results = ["success1", "success2"]
        error = ParallelExecutionFailed(errors=errors, results=results)

        assert error.errors == errors
        assert error.results == results
        assert error.failed_count == 1
        assert error.success_count == 2
        assert "1 failed" in str(error)
        assert "2 succeeded" in str(error)

    def test_parallel_execution_failed_custom_message(self) -> None:
        """Test ParallelExecutionFailed with custom message."""
        errors = [ValueError("error1")]
        error = ParallelExecutionFailed(errors=errors, message="Custom failure")

        assert "Custom failure" in str(error)


class TestGuardrailTimeout:
    """Tests for GuardrailTimeout exception."""

    def test_guardrail_timeout_basic(self) -> None:
        """Test GuardrailTimeout with basic parameters."""
        error = GuardrailTimeout("my_guardrail", 5.0)

        assert error.guardrail_name == "my_guardrail"
        assert error.timeout == 5.0
        assert "my_guardrail" in str(error)
        assert "5.00s" in str(error)
        assert isinstance(error, MiddlewareError)


class TestAggregationFailed:
    """Tests for AggregationFailed exception."""

    def test_aggregation_failed_basic(self) -> None:
        """Test AggregationFailed with basic parameters."""
        error = AggregationFailed(
            strategy=AggregationStrategy.FIRST_SUCCESS,
            reason="All middleware failed",
        )

        assert error.strategy == AggregationStrategy.FIRST_SUCCESS
        assert error.reason == "All middleware failed"
        assert "first_success" in str(error)
        assert "All middleware failed" in str(error)
        assert isinstance(error, MiddlewareError)

    def test_aggregation_failed_with_errors(self) -> None:
        """Test AggregationFailed with underlying errors."""
        underlying = [ValueError("err1"), ValueError("err2")]
        error = AggregationFailed(
            strategy=AggregationStrategy.ALL_MUST_PASS,
            reason="Multiple failures",
            errors=underlying,
        )

        assert error.errors == underlying
        assert len(error.errors) == 2


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_exceptions_inherit_from_middleware_error(self) -> None:
        """Test that all custom exceptions inherit from MiddlewareError."""
        assert issubclass(InputBlocked, MiddlewareError)
        assert issubclass(ToolBlocked, MiddlewareError)
        assert issubclass(OutputBlocked, MiddlewareError)
        assert issubclass(ParallelExecutionFailed, MiddlewareError)
        assert issubclass(GuardrailTimeout, MiddlewareError)
        assert issubclass(AggregationFailed, MiddlewareError)

    def test_can_catch_all_with_middleware_error(self) -> None:
        """Test that all exceptions can be caught with MiddlewareError."""
        with pytest.raises(MiddlewareError):
            raise InputBlocked()

        with pytest.raises(MiddlewareError):
            raise ToolBlocked("tool")

        with pytest.raises(MiddlewareError):
            raise OutputBlocked()

        with pytest.raises(MiddlewareError):
            raise ParallelExecutionFailed(errors=[])

        with pytest.raises(MiddlewareError):
            raise GuardrailTimeout("test", 1.0)

        with pytest.raises(MiddlewareError):
            raise AggregationFailed(AggregationStrategy.ALL_MUST_PASS, "test")
