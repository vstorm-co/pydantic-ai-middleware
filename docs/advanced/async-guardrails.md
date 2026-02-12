# Async Guardrails

`AsyncGuardrailMiddleware` runs guardrails with configurable timing relative to the LLM call. This lets you optimize latency by running safety checks in parallel with the model.

## Timing Modes

The `GuardrailTiming` enum controls when guardrails execute:

| Mode | Behavior | Latency Impact |
|------|----------|----------------|
| `BLOCKING` | Guardrail completes before LLM starts | Total = guardrail + LLM |
| `CONCURRENT` | Guardrail runs alongside LLM; can cancel on failure | Total = max(guardrail, LLM) |
| `ASYNC_POST` | Guardrail runs in background after response | Total = LLM only |

## Basic Usage

```python
from pydantic_ai_middleware import (
    AsyncGuardrailMiddleware,
    GuardrailTiming,
)

# BLOCKING mode - traditional, safe
blocking = AsyncGuardrailMiddleware(
    guardrail=SafetyChecker(),
    timing=GuardrailTiming.BLOCKING,
)

# CONCURRENT mode - optimized latency
concurrent = AsyncGuardrailMiddleware(
    guardrail=SafetyChecker(),
    timing=GuardrailTiming.CONCURRENT,
    cancel_on_failure=True,  # Cancel LLM if guardrail fails
)

# ASYNC_POST mode - fastest response
async_post = AsyncGuardrailMiddleware(
    guardrail=OutputValidator(),
    timing=GuardrailTiming.ASYNC_POST,
)
```

## Early Cancellation

CONCURRENT mode with `cancel_on_failure=True` can short-circuit LLM calls:

```python
# If guardrail fails at 0.5s while LLM is still running,
# the LLM call is cancelled immediately, saving costs!

guardrail = AsyncGuardrailMiddleware(
    guardrail=FastSafetyCheck(),
    timing=GuardrailTiming.CONCURRENT,
    cancel_on_failure=True,
)
```

## Combining with Parallel Execution

You can combine `ParallelMiddleware` and `AsyncGuardrailMiddleware` to run multiple guardrails in parallel while also running them concurrently with the LLM:

```python
from pydantic_ai_middleware import (
    AsyncGuardrailMiddleware,
    ParallelMiddleware,
    AggregationStrategy,
    GuardrailTiming,
    MiddlewareAgent,
)

# Multiple parallel input validators
input_validators = ParallelMiddleware(
    middleware=[
        ProfanityFilter(),
        PIIDetector(),
        InjectionGuard(),
    ],
    strategy=AggregationStrategy.ALL_MUST_PASS,
)

# Wrap in AsyncGuardrailMiddleware for concurrent execution with LLM
safety_check = AsyncGuardrailMiddleware(
    guardrail=input_validators,
    timing=GuardrailTiming.CONCURRENT,
    cancel_on_failure=True,
)

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[safety_check],
)
```

## Next Steps

- [Parallel Execution](parallel-execution.md) - Run multiple middleware concurrently
- [Config Loading](config-loading.md) - Load pipelines from config files
