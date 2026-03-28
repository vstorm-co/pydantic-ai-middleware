<h1 align="center">Pydantic AI Shields</h1>
<p align="center">
  <em>Guardrail Capabilities for Pydantic AI Agents</em>
</p>
<p align="center">
  <a href="https://pypi.org/project/pydantic-ai-shields/"><img src="https://img.shields.io/pypi/v/pydantic-ai-shields.svg" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

---

**Pydantic AI Shields** provides ready-to-use guardrail [capabilities](https://ai.pydantic.dev/capabilities/) for [Pydantic AI](https://ai.pydantic.dev/) agents. Drop them into any agent for cost control, tool permissions, content safety, and more.

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_shields import (
    CostTracking, PromptInjection, PiiDetector, SecretRedaction,
)

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        CostTracking(budget_usd=5.0),
        PromptInjection(sensitivity="high"),
        PiiDetector(),
        SecretRedaction(),
    ],
)
```

## Available Shields

### Infrastructure Shields

| Shield | Description |
|--------|-------------|
| [`CostTracking`](advanced/cost-tracking.md) | Token/USD tracking with budget enforcement |
| `ToolGuard` | Block tools or require human approval |
| `InputGuard` | Custom input validation (pluggable function) |
| `OutputGuard` | Custom output validation (pluggable function) |
| [`AsyncGuardrail`](advanced/async-guardrails.md) | Run guard concurrently with LLM call |

### Content Shields

| Shield | Description |
|--------|-------------|
| `PromptInjection` | Detect prompt injection / jailbreak (6 categories, 3 sensitivity levels) |
| `PiiDetector` | Detect PII — email, phone, SSN, credit card, IP |
| `SecretRedaction` | Block API keys, tokens, credentials in output |
| `BlockedKeywords` | Block forbidden keywords/phrases |
| `NoRefusals` | Block LLM refusals ("I cannot help with that") |

## Next Steps

- [Installation](installation.md) — install the package
- [Examples](examples/index.md) — real-world usage patterns
- [API Reference](api/index.md) — full API docs
