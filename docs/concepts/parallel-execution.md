# Parallel Execution

The middleware library provides powerful parallel execution capabilities that help reduce latency when running multiple validations or guardrails.

## Overview

There are two main parallel execution features:

1. **ParallelMiddleware** - Execute multiple middleware simultaneously
2. **AsyncGuardrailMiddleware** - Run guardrails concurrently with LLM calls

## ParallelMiddleware

`ParallelMiddleware` executes multiple middleware instances in parallel, reducing total latency when you have multiple independent validators.

### Basic Usage

```python
from pydantic_ai_middleware import (
    ParallelMiddleware,
    AggregationStrategy,
    MiddlewareAgent,
)

# Create validators
profanity_filter = ProfanityFilter()
pii_detector = PIIDetector()
injection_guard = PromptInjectionGuard()

# Run them in parallel
parallel_validators = ParallelMiddleware(
    middleware=[profanity_filter, pii_detector, injection_guard],
    strategy=AggregationStrategy.ALL_MUST_PASS,
    timeout=5.0,
)

# Use with agent
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[parallel_validators],
)
```

### Aggregation Strategies

The `AggregationStrategy` enum controls how parallel results are combined:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `ALL_MUST_PASS` | All middleware must succeed, if any fails, raise the first exception and stop the others | Multiple validators that all must agree |
| `FIRST_SUCCESS` | Return first successful result, fails only when all middleware fail | Fallback patterns with alternatives |
| `RACE` | Return fastest successful result, even if it errors | Redundant validators for speed |
| `COLLECT_ALL` | Collect all results (errors included) | Comprehensive reporting |

### Example: Input Validation

```python
from pydantic_ai_middleware import ParallelMiddleware, AggregationStrategy

# Sequential: 0.5s + 0.5s + 0.5s + 0.5s = 2.0s total
# Parallel:  max(0.5s, 0.5s, 0.5s, 0.5s) = 0.5s total

parallel_validators = ParallelMiddleware(
    middleware=[
        ProfanityFilter(delay=0.5),
        PIIDetector(delay=0.5),
        PromptInjectionGuard(delay=0.5),
        ToxicityChecker(delay=0.5),
    ],
    strategy=AggregationStrategy.ALL_MUST_PASS,
)
```

### Timeout Handling

Set a timeout to prevent slow validators from blocking:

```python
parallel = ParallelMiddleware(
    middleware=[...],
    timeout=5.0,  # Max 5 seconds for all validators
)
```

## AsyncGuardrailMiddleware

`AsyncGuardrailMiddleware` runs guardrails with configurable timing relative to the LLM call.

### Timing Modes

The `GuardrailTiming` enum controls when guardrails execute:

| Mode | Behavior | Latency Impact |
|------|----------|----------------|
| `BLOCKING` | Guardrail completes before LLM starts | Total = guardrail + LLM |
| `CONCURRENT` | Guardrail runs alongside LLM, but can increase costs due to additional inference overhead | Total = max(guardrail, LLM) |
| `ASYNC_POST` | Guardrail runs in background after response (for logging purposes) | Total = LLM only |

### Basic Usage

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

### Example: Early Cancellation

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

## Combining Parallel Features

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
    guardrail=input_validators,  # Can wrap ParallelMiddleware!
    timing=GuardrailTiming.CONCURRENT,
    cancel_on_failure=True,
)

agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[safety_check],
)
```

## Performance Considerations

### When to Use Parallel Execution

✅ **Good candidates:**
- Multiple independent validators
- External API calls (rate limiting, content moderation)
- ML model inference calls
- Database lookups

❌ **Not recommended for:**
- Validators with dependencies
- Fast, CPU-bound checks
- Validators that share state

## Full Example

See the complete examples in:
- `examples/parallel_validation.py` - ParallelMiddleware demo
- `examples/async_guardrails.py` - AsyncGuardrailMiddleware demo

Run them with:
```bash
uv run examples/parallel_validation.py
uv run examples/async_guardrails.py
```
