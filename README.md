<h1 align="center">Pydantic AI Shields</h1>

<p align="center">
  <em>Guardrail Capabilities for Pydantic AI Agents</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/pydantic-ai-shields/"><img src="https://img.shields.io/pypi/v/pydantic-ai-shields.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
</p>

<p align="center">
  <b>Cost Tracking</b>
  &nbsp;&bull;&nbsp;
  <b>Prompt Injection</b>
  &nbsp;&bull;&nbsp;
  <b>PII Detection</b>
  &nbsp;&bull;&nbsp;
  <b>Secret Redaction</b>
  &nbsp;&bull;&nbsp;
  <b>Tool Permissions</b>
  &nbsp;&bull;&nbsp;
  <b>Async Guardrails</b>
</p>

---

**Pydantic AI Shields** provides ready-to-use guardrail [capabilities](https://ai.pydantic.dev/capabilities/) for [Pydantic AI](https://ai.pydantic.dev/) agents. Drop them into any agent for cost control, tool permissions, and safety checks — no middleware wrappers needed.

> **Full framework?** Check out [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) — complete agent framework with planning, filesystem, subagents, and skills.

## Installation

```bash
pip install pydantic-ai-shields
```

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_shields import CostTracking, ToolGuard, InputGuard

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        CostTracking(budget_usd=5.0),
        ToolGuard(blocked=["execute"], require_approval=["write_file"]),
        InputGuard(guard=lambda prompt: "ignore all instructions" not in prompt.lower()),
    ],
)

result = await agent.run("Hello!")
```

## Available Shields

### CostTracking

Track token usage and API costs with optional budget enforcement:

```python
from pydantic_ai_shields import CostTracking

tracking = CostTracking(budget_usd=10.0)
agent = Agent("openai:gpt-4.1", capabilities=[tracking])

result = await agent.run("Hello")
print(f"Total cost: ${tracking.total_cost:.4f}")
print(f"Total tokens: {tracking.total_request_tokens + tracking.total_response_tokens}")
```

Raises `BudgetExceededError` when the cumulative cost exceeds the budget. Pricing auto-detected from model via [genai-prices](https://pypi.org/project/genai-prices/).

### ToolGuard

Control which tools the agent can use:

```python
from pydantic_ai_shields import ToolGuard

async def ask_user(tool_name: str, args: dict) -> bool:
    return input(f"Allow {tool_name}? (y/n) ") == "y"

guard = ToolGuard(
    blocked=["execute", "rm"],              # Hidden from model entirely
    require_approval=["write_file"],        # User must approve each call
    approval_callback=ask_user,
)
agent = Agent("openai:gpt-4.1", capabilities=[guard])
```

- **`blocked`** tools are removed via `prepare_tools` — the model never sees them
- **`require_approval`** tools trigger the callback before execution

### InputGuard

Block or validate user input before the agent runs:

```python
from pydantic_ai_shields import InputGuard

# Sync guard
agent = Agent("openai:gpt-4.1", capabilities=[
    InputGuard(guard=lambda prompt: "jailbreak" not in prompt.lower()),
])

# Async guard (e.g., call moderation API)
async def check_toxicity(prompt: str) -> bool:
    result = await moderation_api.check(prompt)
    return result.is_safe

agent = Agent("openai:gpt-4.1", capabilities=[InputGuard(guard=check_toxicity)])
```

Raises `InputBlocked` when the guard returns `False`.

### OutputGuard

Block or validate model output after the agent runs:

```python
from pydantic_ai_shields import OutputGuard

agent = Agent("openai:gpt-4.1", capabilities=[
    OutputGuard(guard=lambda output: "SSN" not in output),
])
```

Raises `OutputBlocked` when the guard returns `False`.

### AsyncGuardrail

Run a guardrail concurrently with the LLM call — if the guard fails first, the LLM is cancelled (saves cost):

```python
from pydantic_ai_shields import AsyncGuardrail, InputGuard

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[AsyncGuardrail(
        guard=InputGuard(guard=check_policy),
        timing="concurrent",       # "concurrent" | "blocking" | "monitoring"
        cancel_on_failure=True,     # Cancel LLM if guard fails
        timeout=5.0,                # Guard timeout in seconds
    )],
)
```

| Timing | Behavior |
|--------|----------|
| `"concurrent"` | Guard runs alongside LLM, fail-fast on violation |
| `"blocking"` | Guard completes before LLM starts (traditional) |
| `"monitoring"` | Guard runs after LLM, fire-and-forget (logging/audit) |

## Built-in Content Shields

### PromptInjection

Detect and block prompt injection / jailbreak attempts:

```python
from pydantic_ai_shields import PromptInjection

agent = Agent("openai:gpt-4.1", capabilities=[
    PromptInjection(sensitivity="high"),  # "low" | "medium" | "high"
])
```

6 detection categories: ignore_instructions, system_override, role_play, delimiter_injection, prompt_leaking, jailbreak. Add custom patterns with `custom_patterns=[r"my_pattern"]`.

### PiiDetector

Detect PII (email, phone, SSN, credit card, IP) in user input:

```python
from pydantic_ai_shields import PiiDetector

agent = Agent("openai:gpt-4.1", capabilities=[
    PiiDetector(detect=["email", "ssn", "credit_card"]),
])
```

Use `action="log"` to allow through while recording detections in `cap.last_detections`.

### SecretRedaction

Block API keys, tokens, and credentials from appearing in model output:

```python
from pydantic_ai_shields import SecretRedaction

agent = Agent("openai:gpt-4.1", capabilities=[SecretRedaction()])
```

Detects: OpenAI, Anthropic, AWS, GitHub, Slack keys, JWTs, private keys, generic API keys.

### BlockedKeywords

Block prompts containing forbidden words or phrases:

```python
from pydantic_ai_shields import BlockedKeywords

agent = Agent("openai:gpt-4.1", capabilities=[
    BlockedKeywords(
        keywords=["competitor_name", "internal_only"],
        whole_words=True,
    ),
])
```

Supports `case_sensitive`, `whole_words`, and `use_regex` modes.

### NoRefusals

Block LLM refusals — ensure the model attempts to answer:

```python
from pydantic_ai_shields import NoRefusals

agent = Agent("openai:gpt-4.1", capabilities=[NoRefusals()])
```

Use `allow_partial=True` to allow responses that contain refusal language but also have substance.

## Composing Shields

All shields compose naturally as pydantic-ai capabilities:

```python
agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        CostTracking(budget_usd=5.0),
        PromptInjection(sensitivity="high"),
        PiiDetector(),
        SecretRedaction(),
        BlockedKeywords(keywords=["classified"]),
        NoRefusals(),
    ],
)
```

## API Reference

### Infrastructure Shields

| Class | Description |
|-------|-------------|
| `CostTracking` | Token/USD tracking with budget enforcement |
| `ToolGuard` | Block tools or require approval |
| `InputGuard` | Custom input validation (pluggable function) |
| `OutputGuard` | Custom output validation (pluggable function) |
| `AsyncGuardrail` | Concurrent guardrail + LLM execution |

### Content Shields

| Class | Description |
|-------|-------------|
| `PromptInjection` | Detect prompt injection / jailbreak (6 categories, 3 sensitivity levels) |
| `PiiDetector` | Detect PII — email, phone, SSN, credit card, IP (regex-based) |
| `SecretRedaction` | Block API keys, tokens, credentials in output |
| `BlockedKeywords` | Block forbidden keywords/phrases (case, word boundary, regex modes) |
| `NoRefusals` | Block LLM refusals ("I cannot help with that") |

### Data

| Class | Description |
|-------|-------------|
| `CostInfo` | Per-run and cumulative token/cost data |

### Exceptions

| Exception | Raised by |
|-----------|-----------|
| `GuardrailError` | Base exception for all shields |
| `InputBlocked` | `InputGuard`, `PromptInjection`, `PiiDetector`, `BlockedKeywords`, `AsyncGuardrail` |
| `OutputBlocked` | `OutputGuard`, `SecretRedaction`, `NoRefusals` |
| `ToolBlocked` | `ToolGuard` |
| `BudgetExceededError` | `CostTracking` |

## Related Projects

| Package | Description |
|---------|-------------|
| [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) | Full agent framework |
| [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) | Task planning capability |
| [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai) | Multi-agent delegation |
| [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) | File storage and Docker sandbox |
| [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) | Context management |
| [pydantic-ai](https://github.com/pydantic/pydantic-ai) | The foundation — agent framework by Pydantic |

## License

MIT

---

<div align="center">

### Need help implementing this in your company?

<p>We're <a href="https://vstorm.co"><b>Vstorm</b></a> — an Applied Agentic AI Engineering Consultancy<br>with 30+ production AI agent implementations.</p>

<a href="https://vstorm.co/contact-us/">
  <img src="https://img.shields.io/badge/Talk%20to%20us%20%E2%86%92-0066FF?style=for-the-badge&logoColor=white" alt="Talk to us">
</a>

<br><br>

Made with ❤️ by <a href="https://vstorm.co"><b>Vstorm</b></a>

</div>
