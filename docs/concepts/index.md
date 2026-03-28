# Core Concepts

Pydantic AI Shields provides guardrail [capabilities](https://ai.pydantic.dev/capabilities/) that plug directly into any Pydantic AI agent. No wrappers, no middleware layers — just native capabilities.

## How Shields Work

Each shield is an `AbstractCapability` that hooks into specific points of the agent lifecycle:

| Hook | When | Used by |
|------|------|---------|
| `before_run` | Before agent starts | `InputGuard`, `PromptInjection`, `PiiDetector`, `BlockedKeywords`, `CostTracking` |
| `after_run` | After agent finishes | `OutputGuard`, `SecretRedaction`, `NoRefusals`, `CostTracking` |
| `prepare_tools` | Before model sees tools | `ToolGuard` (hides blocked tools) |
| `before_tool_execute` | Before tool runs | `ToolGuard` (approval check) |
| `wrap_run` | Wraps entire run | `AsyncGuardrail` (concurrent execution) |

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_shields import PromptInjection, SecretRedaction

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        PromptInjection(sensitivity="high"),
        SecretRedaction(),
    ],
)
```

All shields compose naturally — add as many as you need. Input shields fire in order, output shields fire in reverse order.

## Two Types of Shields

### Infrastructure Shields

General-purpose shields where **you provide the logic**:

- `InputGuard(guard=my_function)` — your function decides what's safe
- `OutputGuard(guard=my_function)` — your function validates output
- `ToolGuard(blocked=[...], require_approval=[...])` — configure tool access
- `CostTracking(budget_usd=5.0)` — track costs and enforce budgets
- `AsyncGuardrail(guard=..., timing="concurrent")` — run any guard concurrently with LLM

### Content Shields

**Ready-to-use** shields with built-in detection patterns:

- `PromptInjection()` — 6 injection categories, 3 sensitivity levels
- `PiiDetector()` — email, phone, SSN, credit card, IP detection
- `SecretRedaction()` — API keys, tokens, credentials in output
- `BlockedKeywords(keywords=[...])` — forbidden words/phrases
- `NoRefusals()` — prevent LLM from refusing to help

## Next Steps

- [Cost Tracking](../advanced/cost-tracking.md) — budget enforcement details
- [Async Guardrails](../advanced/async-guardrails.md) — concurrent guard + LLM
- [Examples](../examples/index.md) — real-world patterns
