# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**pydantic-ai-shields** provides guardrail capabilities for Pydantic AI agents. Built on pydantic-ai's native capabilities API (v1.71+).

## Development Commands

- **Run tests**: `uv run pytest tests/ -v`
- **Run with coverage**: `uv run coverage run -m pytest tests/ && uv run coverage report --fail-under=100`
- **Lint**: `uv run ruff check src/ tests/`
- **Type check**: `uv run pyright src/`

## Project Structure

```
src/pydantic_ai_shields/
  __init__.py       — Package exports (all 5 capabilities + exceptions)
  guardrails.py     — All capability implementations
tests/
  test_guardrails.py — All tests
```

## Core Components

**Capabilities (`src/pydantic_ai_shields/guardrails.py`)**:
- `CostTracking` — token/USD tracking, budget enforcement (`before_run`, `after_run`)
- `ToolGuard` — block tools or require approval (`prepare_tools`, `before_tool_execute`)
- `InputGuard` — validate user input (`before_run`)
- `OutputGuard` — validate model output (`after_run`)
- `AsyncGuardrail` — concurrent guardrail + LLM execution (`wrap_run`)

**Exceptions**:
- `GuardrailError` — base
- `InputBlocked`, `OutputBlocked`, `ToolBlocked`, `BudgetExceededError`

**Data**:
- `CostInfo` — per-run and cumulative token/cost data

## Testing

- 100% coverage required
- pytest-asyncio with auto mode
- Use `TestModel` from pydantic-ai for deterministic tests
