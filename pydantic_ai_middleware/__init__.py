"""pydantic-ai-middleware - Simple middleware library for Pydantic-AI.

This library provides clean middleware hooks for Pydantic-AI agents,
allowing you to intercept and modify agent behavior at various points
in the execution lifecycle.
"""

from __future__ import annotations

from .agent import MiddlewareAgent
from .async_guardrail import AsyncGuardrailMiddleware
from .base import AgentMiddleware
from .context import (
    ContextAccessError,
    HookType,
    MiddlewareContext,
    ScopedContext,
)
from .decorators import (
    after_run,
    after_tool_call,
    before_model_request,
    before_run,
    before_tool_call,
    on_error,
)
from .exceptions import (
    AggregationFailed,
    GuardrailTimeout,
    InputBlocked,
    MiddlewareError,
    OutputBlocked,
    ParallelExecutionFailed,
    ToolBlocked,
)
from .parallel import ParallelMiddleware
from .strategies import AggregationStrategy, GuardrailTiming
from .toolset import MiddlewareToolset

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "AgentMiddleware",
    "MiddlewareAgent",
    "MiddlewareToolset",
    # Context
    "MiddlewareContext",
    "ScopedContext",
    "HookType",
    "ContextAccessError",
    # Parallel execution
    "ParallelMiddleware",
    "AsyncGuardrailMiddleware",
    # Strategies
    "AggregationStrategy",
    "GuardrailTiming",
    # Decorators
    "before_run",
    "after_run",
    "before_model_request",
    "before_tool_call",
    "after_tool_call",
    "on_error",
    # Exceptions
    "MiddlewareError",
    "InputBlocked",
    "ToolBlocked",
    "OutputBlocked",
    "ParallelExecutionFailed",
    "GuardrailTimeout",
    "AggregationFailed",
]
