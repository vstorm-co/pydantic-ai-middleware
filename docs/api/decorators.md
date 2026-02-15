# Decorators

Decorator functions for creating middleware from simple async functions. Each decorator
wraps a single hook function into a full `AgentMiddleware` instance, so you do not need
to subclass `AgentMiddleware` for straightforward use cases.

The `before_tool_call`, `after_tool_call`, and `on_tool_error` decorators also accept
an optional `tools` parameter to restrict the middleware to specific tool names.

## before_run

::: pydantic_ai_middleware.before_run

## after_run

::: pydantic_ai_middleware.after_run

## before_model_request

::: pydantic_ai_middleware.before_model_request

## before_tool_call

::: pydantic_ai_middleware.before_tool_call

## after_tool_call

::: pydantic_ai_middleware.after_tool_call

## on_tool_error

::: pydantic_ai_middleware.on_tool_error

## on_error

::: pydantic_ai_middleware.on_error
