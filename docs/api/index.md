# API Reference

Complete API documentation for pydantic-ai-middleware.

## Core Classes

- [`AgentMiddleware`](middleware.md) - Base middleware class
- [`MiddlewareAgent`](agent.md) - Agent wrapper with middleware
- [`MiddlewareChain`](chain.md) - Composable middleware sequences
- [`ConditionalMiddleware`](conditional.md) - Conditional middleware routing
- [`PipelineSpec`](pipeline_spec.md) - Pipeline spec builder/exporter
- [`MiddlewareToolset`](toolset.md) - Toolset wrapper for tool interception

## Decorators

- [`@before_run`](decorators.md#before_run)
- [`@after_run`](decorators.md#after_run)
- [`@before_model_request`](decorators.md#before_model_request)
- [`@before_tool_call`](decorators.md#before_tool_call)
- [`@after_tool_call`](decorators.md#after_tool_call)
- [`@on_error`](decorators.md#on_error)

## Exceptions

- [`MiddlewareError`](exceptions.md#middlewareerror)
- [`InputBlocked`](exceptions.md#inputblocked)
- [`ToolBlocked`](exceptions.md#toolblocked)
- [`OutputBlocked`](exceptions.md#outputblocked)

## Quick Import

```python
from pydantic_ai_middleware import (
    # Core
    AgentMiddleware,
    MiddlewareAgent,
    MiddlewareChain,
    ConditionalMiddleware,
    PipelineSpec,
    MiddlewareToolset,
    # Decorators
    before_run,
    after_run,
    before_model_request,
    before_tool_call,
    after_tool_call,
    on_error,
    # Exceptions
    MiddlewareError,
    InputBlocked,
    ToolBlocked,
    OutputBlocked,
)
```
