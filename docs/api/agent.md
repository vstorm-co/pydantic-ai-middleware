# MiddlewareAgent

`MiddlewareAgent` wraps a pydantic-ai `AbstractAgent` and applies middleware hooks
around its `run()` and `iter()` methods. It is the main entry point for using
middleware with any pydantic-ai agent.

Key responsibilities:

- Applies `before_run` hooks (in order) before the agent processes a prompt.
- Wraps all toolsets with `MiddlewareToolset` so `before_tool_call`, `after_tool_call`,
  and `on_tool_error` hooks intercept every tool invocation.
- Applies `after_run` hooks (in reverse order) after the agent completes.
- Applies `on_error` hooks when an exception occurs.
- Stores run metadata (usage, prompts, outputs) in the `MiddlewareContext` when provided.

::: pydantic_ai_middleware.MiddlewareAgent
    options:
      show_source: true
      members:
        - __init__
        - wrapped
        - middleware
        - run
        - iter
        - override
