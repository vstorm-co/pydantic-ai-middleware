"""pydantic-ai-middleware - Simple middleware library for Pydantic-AI.

This library provides clean middleware hooks for Pydantic-AI agents,
allowing you to intercept and modify agent behavior at various points
in the execution lifecycle.
"""

from __future__ import annotations

from .agent import MiddlewareAgent
from .async_guardrail import AsyncGuardrailMiddleware
from .base import AgentMiddleware
from .builder import (
    MiddlewareFactory,
    MiddlewarePipelineCompiler,
    MiddlewareRegistry,
    PredicateFactory,
)
from .chain import MiddlewareChain
from .conditional import ConditionalMiddleware
from .config_loaders import (
    dump_middleware_config,
    load_middleware_config_path,
    load_middleware_config_text,
    register_middleware,
    register_predicate,
    save_middleware_config_path,
)
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
    MiddlewareConfigError,
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
    "MiddlewareChain",
    "MiddlewareToolset",
    # Config loading/saving
    "load_middleware_config_path",
    "load_middleware_config_text",
    "dump_middleware_config",
    "save_middleware_config_path",
    "register_middleware",
    "register_predicate",
    "MiddlewareFactory",
    "PredicateFactory",
    "MiddlewareRegistry",
    "MiddlewarePipelineCompiler",
    # Context
    "MiddlewareContext",
    "ScopedContext",
    "HookType",
    "ContextAccessError",
    # Parallel execution
    "ParallelMiddleware",
    "AsyncGuardrailMiddleware",
    # Composition helpers
    "ConditionalMiddleware",
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
    "MiddlewareConfigError",
    "InputBlocked",
    "ToolBlocked",
    "OutputBlocked",
    "ParallelExecutionFailed",
    "GuardrailTimeout",
    "AggregationFailed",
]
