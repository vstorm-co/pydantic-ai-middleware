# Parallel Execution

`ParallelMiddleware` executes multiple middleware instances in parallel, reducing total latency when you have multiple independent validators.

## Basic Usage

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

## Aggregation Strategies

The `AggregationStrategy` enum controls how parallel results are combined:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `ALL_MUST_PASS` | All middleware must succeed; cancels remaining on first failure | Multiple validators that all must agree |
| `FIRST_SUCCESS` | Return first successful result; fails only when all fail | Fallback patterns with alternatives |
| `RACE` | Return fastest successful result, even if others error | Redundant validators for speed |
| `COLLECT_ALL` | Collect all results (errors included) | Comprehensive reporting |

## Performance Example

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

## Timeout Handling

Set a timeout to prevent slow validators from blocking:

```python
parallel = ParallelMiddleware(
    middleware=[...],
    timeout=5.0,  # Max 5 seconds for all validators
)
```

## When to Use Parallel Execution

**Good candidates:**

- Multiple independent validators
- External API calls (rate limiting, content moderation)
- ML model inference calls
- Database lookups

**Not recommended for:**

- Validators with dependencies on each other
- Fast, CPU-bound checks
- Validators that share mutable state

## Next Steps

- [Async Guardrails](async-guardrails.md) - Run guardrails alongside LLM calls
- [Middleware Chains](middleware-chains.md) - Sequential composition
- [API Reference](../api/middleware.md) - ParallelMiddleware API
