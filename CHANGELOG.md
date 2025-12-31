# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Parallel Execution** - `ParallelMiddleware` for running multiple middleware concurrently
  - `AggregationStrategy` enum: `ALL_MUST_PASS`, `FIRST_SUCCESS`, `RACE`, `COLLECT_ALL`
  - Early cancellation: remaining tasks are cancelled when result is determined
  - Configurable timeout support
- **Async Guardrails** - `AsyncGuardrailMiddleware` for concurrent guardrail + LLM execution
  - `GuardrailTiming` enum: `BLOCKING`, `CONCURRENT`, `ASYNC_POST`
  - `cancel_on_failure` option to short-circuit LLM calls when guardrail fails
  - Background task management for async post-processing
- New exceptions:
  - `ParallelExecutionFailed` - When parallel execution fails
  - `GuardrailTimeout` - When guardrail times out
  - `AggregationFailed` - When result aggregation fails

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

[Unreleased]: https://github.com/vstorm-co/pydantic-ai-middleware/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vstorm-co/pydantic-ai-middleware/releases/tag/v0.1.0
