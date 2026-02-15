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
    load_middleware_config_path,
    load_middleware_config_text,
    register_middleware,
    register_predicate,
)
from .context import (
    ContextAccessError,
    HookType,
    MiddlewareContext,
    ScopedContext,
)
from .cost_tracking import (
    CostCallback,
    CostInfo,
    CostTrackingMiddleware,
    create_cost_tracking_middleware,
)
from .decorators import (
    after_run,
    after_tool_call,
    before_model_request,
    before_run,
    before_tool_call,
    on_error,
    on_tool_error,
)
from .exceptions import (
    AggregationFailed,
    BudgetExceededError,
    GuardrailTimeout,
    InputBlocked,
    MiddlewareConfigError,
    MiddlewareError,
    MiddlewareTimeout,
    OutputBlocked,
    ParallelExecutionFailed,
    ToolBlocked,
)
from .parallel import ParallelMiddleware
from .permissions import PermissionHandler, ToolDecision, ToolPermissionResult
from .pipeline_spec import PipelineSpec
from .strategies import AggregationStrategy, GuardrailTiming
from .toolset import MiddlewareToolset

try:
    from importlib.metadata import version as _metadata_version

    __version__ = _metadata_version("pydantic-ai-middleware")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    # Core classes
    "AgentMiddleware",
    "MiddlewareAgent",
    "MiddlewareChain",
    "MiddlewareToolset",
    # Config loading/saving
    "load_middleware_config_path",
    "load_middleware_config_text",
    "register_middleware",
    "register_predicate",
    "MiddlewareFactory",
    "PredicateFactory",
    "MiddlewareRegistry",
    "MiddlewarePipelineCompiler",
    "PipelineSpec",
    # Context
    "MiddlewareContext",
    "ScopedContext",
    "HookType",
    "ContextAccessError",
    # Parallel execution
    "ParallelMiddleware",
    "AsyncGuardrailMiddleware",
    # Cost tracking
    "CostTrackingMiddleware",
    "CostInfo",
    "CostCallback",
    "create_cost_tracking_middleware",
    # Composition helpers
    "ConditionalMiddleware",
    "AggregationStrategy",
    "GuardrailTiming",
    # Permissions
    "ToolDecision",
    "ToolPermissionResult",
    "PermissionHandler",
    # Decorators
    "before_run",
    "after_run",
    "before_model_request",
    "before_tool_call",
    "after_tool_call",
    "on_tool_error",
    "on_error",
    # Exceptions
    "MiddlewareError",
    "MiddlewareConfigError",
    "BudgetExceededError",
    "MiddlewareTimeout",
    "InputBlocked",
    "ToolBlocked",
    "OutputBlocked",
    "ParallelExecutionFailed",
    "GuardrailTimeout",
    "AggregationFailed",
]
