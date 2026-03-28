# AGENTS.md

Instructions for AI coding assistants working on this repository.

## Project Overview

**pydantic-ai-shields** provides guardrail capabilities for [pydantic-ai](https://ai.pydantic.dev/) agents. Built on pydantic-ai's native capabilities API (v1.71+). No middleware wrappers — pure capabilities.

## Quick Reference

| Task | Command |
|------|---------|
| Test | `uv run pytest tests/ -v` |
| Test + Coverage | `uv run coverage run -m pytest tests/ && uv run coverage report --fail-under=100` |
| Lint | `uv run ruff check src/ tests/` |
| Typecheck | `uv run pyright src/` |

## Architecture

```
src/pydantic_ai_shields/
  __init__.py       — Package exports
  guardrails.py     — 5 capability implementations + exceptions + CostInfo
tests/
  test_guardrails.py — All tests
```

### Capabilities

| Capability | Hooks Used | Purpose |
|------------|-----------|---------|
| `CostTracking` | `before_run`, `after_run` | Token/USD tracking, budget enforcement |
| `ToolGuard` | `prepare_tools`, `before_tool_execute` | Block tools, require approval |
| `InputGuard` | `before_run` | Validate user input |
| `OutputGuard` | `after_run` | Validate model output |
| `AsyncGuardrail` | `wrap_run` | Concurrent guardrail + LLM |

## Code Standards

- **Coverage**: 100% required
- **Types**: Pyright strict on src/
- **Style**: ruff for formatting and linting

## Testing

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_shields import CostTracking, ToolGuard

agent = Agent(TestModel(), capabilities=[CostTracking()])
result = await agent.run("test")
```
