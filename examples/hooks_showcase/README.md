# Hooks Showcase

Simple, self-contained examples demonstrating every middleware hook in `pydantic-ai-middleware`.

Each file is a standalone script with a single concept, real OpenAI calls, and console logging.

## Examples

| # | File | Hook / Feature | Description |
|---|------|----------------|-------------|
| 01 | `01_before_run.py` | `before_run` | Log incoming prompts |
| 02 | `02_after_run.py` | `after_run` | Measure wall-clock time |
| 03 | `03_before_model_request.py` | `before_model_request` | Count LLM API calls |
| 04 | `04_before_tool_call.py` | `before_tool_call` | Log tool invocations |
| 05 | `05_after_tool_call.py` | `after_tool_call` | Log results + timing |
| 06 | `06_on_tool_error.py` | `on_tool_error` | Handle tool failures |
| 07 | `07_on_error.py` | `on_error` | Global error handler |
| 08 | `08_tool_blocking.py` | `ToolPermissionResult` | ALLOW / DENY decisions |
| 09 | `09_permission_handler.py` | `ToolDecision.ASK` | Interactive approval |
| 10 | `10_tool_name_filtering.py` | `tool_names` | Per-tool middleware |
| 11 | `11_middleware_chain.py` | `MiddlewareChain` | Compose middleware |
| 12 | `12_decorator_middleware.py` | `@before_run` etc. | Decorator syntax |
| 13 | `13_context_sharing.py` | `MiddlewareContext` | Share data between hooks |
| 14 | `14_cost_tracking.py` | `CostTrackingMiddleware` | Token & USD tracking |
| 15 | `15_multiple_middleware.py` | Execution order | FIFO before / LIFO after |
| 16 | `16_full_lifecycle.py` | All hooks | Complete lifecycle trace |

## Running

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run any example
uv run python examples/hooks_showcase/01_before_run.py
```

### Run all examples

```bash
export OPENAI_API_KEY=sk-...

uv run python examples/hooks_showcase/01_before_run.py
uv run python examples/hooks_showcase/02_after_run.py
uv run python examples/hooks_showcase/03_before_model_request.py
uv run python examples/hooks_showcase/04_before_tool_call.py
uv run python examples/hooks_showcase/05_after_tool_call.py
uv run python examples/hooks_showcase/06_on_tool_error.py
uv run python examples/hooks_showcase/07_on_error.py
uv run python examples/hooks_showcase/08_tool_blocking.py
uv run python examples/hooks_showcase/09_permission_handler.py
uv run python examples/hooks_showcase/10_tool_name_filtering.py
uv run python examples/hooks_showcase/11_middleware_chain.py
uv run python examples/hooks_showcase/12_decorator_middleware.py
uv run python examples/hooks_showcase/13_context_sharing.py
uv run python examples/hooks_showcase/14_cost_tracking.py
uv run python examples/hooks_showcase/15_multiple_middleware.py
uv run python examples/hooks_showcase/16_full_lifecycle.py
```
