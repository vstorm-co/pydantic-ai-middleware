# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-02-15

### Added

- **Cost Tracking Middleware** — `CostTrackingMiddleware` for automatic token usage and USD cost monitoring
  - Tracks cumulative input/output tokens and USD costs across agent runs
  - USD cost calculation via `genai-prices` library (supports `"provider:model"` format)
  - `budget_limit_usd` — enforces cost budget limits, raises `BudgetExceededError` when exceeded
  - `on_cost_update` callback with `CostInfo` dataclass (supports sync and async)
  - `reset()` method to clear accumulators
  - `create_cost_tracking_middleware()` factory function
- **`BudgetExceededError`** exception for budget limit enforcement
- `run_usage` metadata — `MiddlewareAgent` now stores `result.usage()` in context metadata after each run, enabling cost tracking and other usage-aware middleware

### Changed

- **`before_model_request` hook** — now invoked via pydantic-ai's history processor mechanism instead of manual message interception. This ensures the hook fires for every model request in both `run()` and `iter()` modes.
- **Toolset wrapping** — `MiddlewareAgent` now always wraps all toolsets (agent's own + explicitly passed) using `agent.override(toolsets=...)`, preventing duplicate tool registration.
- **`ToolBlocked` handling** — `MiddlewareToolset` now catches `ToolBlocked` exceptions in `before_tool_call` and returns a descriptive string instead of raising, avoiding message ordering issues with pydantic-ai retries.

### Dependencies

- Added `genai-prices>=0.0.49` as a required dependency (for cost calculation)

## [0.2.0] - 2025-02-12

### Added

- **Middleware Chains** - `MiddlewareChain` for grouping middleware into reusable, ordered units
  - Fluent API: `add()`, `insert()`, `remove()`, `replace()`, `pop()`, `clear()`, `copy()`
  - Chain composition: chains flatten when added to other chains
  - Operator support: `+` and `+=` for combining chains
  - Iteration support: `len()`, `[]`, `in`, iteration
- **Conditional Middleware** - `ConditionalMiddleware` for branching execution based on runtime conditions
  - Route to different middleware based on predicate functions
  - Supports single middleware or middleware pipelines for each branch
  - Predicate receives `ScopedContext` for context-aware decisions
  - `when(condition, then, otherwise)` helper for fluent syntax
- **Pipeline Spec Builder** - `PipelineSpec` for building middleware pipelines as portable config dictionaries
  - Fluent builder API: `add_type()`, `add_chain()`, `add_parallel()`, `add_when()`
  - Export to files: `dump(format="json"|"yaml")`, `save(path)` with auto-detection from extension
  - Direct compilation: `compile(compiler)` delegates to compiler
  - Helper functions: `type_node()`, `chain_node()`, `parallel_node()`, `when_node()`
- **Config Loaders** - Functions to load middleware pipelines from JSON/YAML
  - `load_middleware_config_text(text, registry, predicates)` - parse from string
  - `load_middleware_config_path(path, registry, predicates)` - parse from file
  - Auto-detection of format from file extension or content
  - Registration helpers: `register_middleware()`, `register_predicate()` decorators
- **Pipeline Compiler** - `MiddlewarePipelineCompiler` for compiling config dictionaries into middleware
  - `MiddlewareRegistry` for storing middleware factories and predicates
  - Built-in node handlers: `type`, `chain`, `parallel`, `when`
  - Extensible via `register_node_handler()` for custom node types
  - Convenience wrappers: `build_middleware()`, `build_middleware_list()`
- **Context Sharing System** - Share data between middleware hooks with access control
  - `MiddlewareContext` class for managing shared state across hooks
  - `ScopedContext` class for enforcing access control based on hook execution order
  - `HookType` enum defining hook execution order: `BEFORE_RUN(1)` → `BEFORE_MODEL_REQUEST(2)` → `BEFORE_TOOL_CALL(3)` → `ON_TOOL_ERROR(4)` → `AFTER_TOOL_CALL(5)` → `AFTER_RUN(6)` → `ON_ERROR(7)`
  - Immutable `config` for read-only global settings
  - Mutable `metadata` for shared state
  - Namespaced hook storage with strict access control
  - Hooks can only write to their own namespace
  - Hooks can only read from earlier or same-phase hooks
  - Enable context by passing a `MiddlewareContext` instance to `MiddlewareAgent`
- **Parallel Execution** - `ParallelMiddleware` for running multiple middleware concurrently
  - `AggregationStrategy` enum: `ALL_MUST_PASS`, `FIRST_SUCCESS`, `RACE`, `COLLECT_ALL`
  - Early cancellation: remaining tasks are cancelled when result is determined
  - Configurable timeout support
- **Async Guardrails** - `AsyncGuardrailMiddleware` for concurrent guardrail + LLM execution
  - `GuardrailTiming` enum: `BLOCKING`, `CONCURRENT`, `ASYNC_POST`
  - `cancel_on_failure` option to short-circuit LLM calls when guardrail fails
  - Background task management for async post-processing
- **Tool Name Filtering** - Scope middleware to specific tools
  - `tool_names: set[str] | None` class attribute on `AgentMiddleware` (default `None` = all tools)
  - `_should_handle_tool(tool_name)` helper method
  - Filtering in `MiddlewareToolset`, `MiddlewareChain`, and composite middleware
  - `tools` parameter on `@before_tool_call`, `@after_tool_call`, `@on_tool_error` decorators
- **`on_tool_error` Hook** - Handle tool execution failures with tool-specific context
  - `on_tool_error(tool_name, tool_args, error, deps, ctx)` method on `AgentMiddleware`
  - Returns `Exception | None` — return a different exception or `None` to re-raise original
  - New `HookType.ON_TOOL_ERROR` enum value
  - Integrated in `MiddlewareToolset`, `MiddlewareChain`, `ParallelMiddleware`, `ConditionalMiddleware`, `AsyncGuardrailMiddleware`
  - `@on_tool_error` decorator with optional `tools` parameter
- **Hook Timeout** - Per-middleware timeout enforcement
  - `timeout: float | None` class attribute on `AgentMiddleware` (default `None` = no timeout)
  - `call_with_timeout()` utility wrapping `asyncio.wait_for()`
  - Enforced in `MiddlewareToolset` and `MiddlewareAgent` for all hooks
- **Permission Decision Protocol** - Structured ALLOW/DENY/ASK decisions for tool authorization
  - `ToolDecision` enum: `ALLOW`, `DENY`, `ASK`
  - `ToolPermissionResult` dataclass with `decision`, `reason`, and optional `modified_args`
  - `before_tool_call` return type extended: `dict[str, Any] | ToolPermissionResult`
  - `PermissionHandler` callback type for `ASK` decisions: `async (tool_name, tool_args, reason) -> bool`
  - `permission_handler` parameter on `MiddlewareAgent` and `MiddlewareToolset`
- New exceptions:
  - `ParallelExecutionFailed` - When parallel execution fails
  - `GuardrailTimeout` - When guardrail times out
  - `AggregationFailed` - When result aggregation fails
  - `MiddlewareConfigError` - When config loading or compilation fails
  - `MiddlewareTimeout` - When a middleware hook exceeds its timeout
- Project restructured to `src/` layout

### Changed

- **BREAKING**: All hook signatures now include `ctx: ScopedContext | None = None` as the last keyword argument
  - `before_run(prompt, deps, ctx)`
  - `after_run(prompt, output, deps, ctx)`
  - `before_model_request(messages, deps, ctx)`
  - `before_tool_call(tool_name, tool_args, deps, ctx)` — return type extended to `dict | ToolPermissionResult`
  - `on_tool_error(tool_name, tool_args, error, deps, ctx)` — **NEW**
  - `after_tool_call(tool_name, tool_args, tool_result, deps, ctx)`
  - `on_error(error, deps, ctx)`
- `__version__` now uses `importlib.metadata` instead of a hardcoded string
- Bumped minimum `pydantic-ai-slim` dependency from `>=0.1.0` to `>=1.38`

### Fixed

- Replaced private pydantic-ai API usage with public alternatives ([#11](https://github.com/vstorm-co/pydantic-ai-middleware/pull/11) by [@pedroallenrevez](https://github.com/pedroallenrevez))
  - `from pydantic_ai._run_context import RunContext` → `from pydantic_ai import RunContext`
  - `AgentRunResult(output=..., _output_tool_name=..., ...)` → `dataclasses.replace(result, output=output)`
  - Removed dependency on `pydantic_ai._utils.UNSET/Unset` by simplifying `override()` to use `**kwargs` pass-through

## [0.1.0] - 2024-12-29

### Added

- Initial release
- `AgentMiddleware` base class with lifecycle hooks:
  - `before_run` - Called before agent runs
  - `after_run` - Called after agent finishes
  - `before_model_request` - Called before each model request
  - `before_tool_call` - Called before tool execution
  - `after_tool_call` - Called after tool execution
  - `on_error` - Called when errors occur
- `MiddlewareAgent` - Wrapper agent that applies middleware
- `MiddlewareToolset` - Toolset wrapper for tool call interception
- Decorator-based middleware creation:
  - `@before_run`
  - `@after_run`
  - `@before_model_request`
  - `@before_tool_call`
  - `@after_tool_call`
  - `@on_error`
- Custom exceptions:
  - `MiddlewareError` - Base exception
  - `InputBlocked` - Block input
  - `ToolBlocked` - Block tool calls
  - `OutputBlocked` - Block output
- Full type safety with generics
- 100% test coverage
- Documentation with MkDocs

[0.2.1]: https://github.com/vstorm-co/pydantic-ai-middleware/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/vstorm-co/pydantic-ai-middleware/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vstorm-co/pydantic-ai-middleware/releases/tag/v0.1.0
