# Async Guardrails

`AsyncGuardrail` runs a guard concurrently with the LLM call — if the guard fails
first, the LLM is cancelled to save cost.

## Timing Modes

| Mode | Behavior |
|------|----------|
| `"concurrent"` | Guard runs alongside LLM. If guard fails, LLM result is discarded. |
| `"blocking"` | Guard completes before LLM starts (traditional). |
| `"monitoring"` | LLM runs first, guard runs after (fire-and-forget for audit/logging). |

## Concurrent Mode (Default)

```python
from pydantic_ai import Agent
from pydantic_ai_shields import AsyncGuardrail, InputGuard

async def check_policy(prompt: str) -> bool:
    # Call your policy API
    return await policy_api.check(prompt)

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[AsyncGuardrail(
        guard=InputGuard(guard=check_policy),
        timing="concurrent",
        cancel_on_failure=True,
        timeout=5.0,
    )],
)
```

If `check_policy` returns `False` before the model finishes, the result is discarded
and `InputBlocked` is raised — no tokens wasted.

## Blocking Mode

Traditional sequential execution — guard completes before model starts:

```python
AsyncGuardrail(
    guard=InputGuard(guard=check_policy),
    timing="blocking",
)
```

## Monitoring Mode

Fire-and-forget — model runs first, guard runs after for logging/audit:

```python
AsyncGuardrail(
    guard=InputGuard(guard=log_for_compliance),
    timing="monitoring",
)
```

## Combining with Other Shields

```python
agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        PromptInjection(),                    # Fast regex check (before_run)
        AsyncGuardrail(                       # Slow API check (concurrent with LLM)
            guard=InputGuard(guard=external_moderation_api),
            timing="concurrent",
        ),
        SecretRedaction(),                    # Output check (after_run)
    ],
)
```
