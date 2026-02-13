# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Development Commands

### Core Development Tasks

- **Install dependencies**: `make install`
- **Run all checks**: `make all` (format, lint, typecheck, test)
- **Run tests**: `make test`
- **Build docs**: `make docs` or `make docs-serve`

### Single Test Commands

- **Run specific test**: `uv run pytest tests/test_agent.py::test_function_name -v`
- **Run test file**: `uv run pytest tests/test_agent.py -v`
- **Run with debug**: `uv run pytest tests/test_agent.py -v -s`

## Project Architecture

### Core Components

**Base Middleware (`src/pydantic_ai_middleware/base.py`)**
- `AgentMiddleware[DepsT]` - Abstract base class for all middleware
- Lifecycle hooks: `before_run`, `after_run`, `before_model_request`, `before_tool_call`, `on_tool_error`, `after_tool_call`, `on_error`
- `tool_names: set[str] | None` - Filter which tools a middleware handles
- `timeout: float | None` - Per-middleware timeout for all hooks

**Middleware Agent (`src/pydantic_ai_middleware/agent.py`)**
- `MiddlewareAgent` - Wraps an agent and applies middleware
- Delegates to wrapped agent while intercepting lifecycle events

**Middleware Toolset (`src/pydantic_ai_middleware/toolset.py`)**
- `MiddlewareToolset` - Wraps a toolset to intercept tool calls
- Applies `before_tool_call`, `on_tool_error`, and `after_tool_call` middleware hooks
- `permission_handler` - Callback for handling ASK permission decisions

**Decorators (`src/pydantic_ai_middleware/decorators.py`)**
- `@before_run`, `@after_run`, `@on_tool_error`, etc. - Create middleware from functions
- `@before_tool_call(tools={"send_email"})` - Decorator with tool name filtering
- `_FunctionMiddleware` - Internal class that wraps functions

**Exceptions (`src/pydantic_ai_middleware/exceptions.py`)**
- `MiddlewareError` - Base exception
- `InputBlocked` - Block input processing
- `ToolBlocked` - Block tool execution
- `OutputBlocked` - Block output
- `MiddlewareTimeout` - Hook exceeded timeout

**Permissions (`src/pydantic_ai_middleware/permissions.py`)**
- `ToolDecision` - Enum: ALLOW, DENY, ASK
- `ToolPermissionResult` - Structured result from before_tool_call
- `PermissionHandler` - Callback type for ASK decisions

### Key Design Patterns

**Middleware Chain**
```python
# Middleware executes in order for before_*, reverse for after_*
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[mw1, mw2, mw3],  # before: 1->2->3, after: 3->2->1
)
```

**Type-Safe Dependencies**
```python
class MyMiddleware(AgentMiddleware[MyDeps]):
    async def before_run(self, prompt, deps: MyDeps | None):
        # deps is properly typed
        ...
```

## Testing Strategy

- **Unit tests**: `tests/` directory
- **Test model**: Use `TestModel` from pydantic-ai for deterministic testing
- **Coverage**: 100% required
- **pytest-asyncio**: Auto mode enabled

## Key Configuration Files

- **`pyproject.toml`**: Project configuration
- **`Makefile`**: Development automation
- **`.pre-commit-config.yaml`**: Pre-commit hooks
- **`mkdocs.yml`**: Documentation configuration

## Coverage

Every pull request MUST have 100% coverage. Check with `make test`.
