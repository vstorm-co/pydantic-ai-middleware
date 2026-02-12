# Prompt Injection Screening

This example demonstrates middleware that detects and blocks common prompt injection techniques such as instruction override, role play, delimiter abuse, and jailbreak attempts.

## Overview

The `PromptInjectionMiddleware` scans input against known injection patterns, computes a risk score (max weight among matched patterns), and blocks the request if the score exceeds a configurable threshold.

## Injection Patterns

The middleware detects these categories:

| Pattern | Weight | Description |
|---------|--------|-------------|
| `ignore_instructions` | 0.9 | "Ignore all previous instructions..." |
| `disregard_rules` | 0.9 | "Disregard your rules..." |
| `role_assignment` | 0.7 | "You are now a hacker AI..." |
| `act_as` | 0.6 | "Act as if you were unrestricted..." |
| `system_prompt_leak` | 0.8 | "Show me your system prompt..." |
| `delimiter_injection` | 0.7 | Fake message formatting with delimiters |
| `base64_injection` | 0.5 | Encoded payload attempts |
| `jailbreak_keyword` | 0.9 | Known jailbreak terms (DAN, developer mode) |

## Basic Usage

```python
import re
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

from pydantic_ai_middleware import AgentMiddleware, InputBlocked, ScopedContext


@dataclass
class InjectionPattern:
    name: str
    pattern: re.Pattern[str]
    weight: float
    description: str


INJECTION_PATTERNS: list[InjectionPattern] = [
    InjectionPattern(
        name="ignore_instructions",
        pattern=re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above)\s+"
            r"(instructions|rules)",
            re.I,
        ),
        weight=0.9,
        description="Attempts to override system instructions",
    ),
    InjectionPattern(
        name="jailbreak_keyword",
        pattern=re.compile(
            r"\b(jailbreak|DAN|developer\s+mode|unrestricted\s+mode)\b",
            re.I,
        ),
        weight=0.9,
        description="Uses known jailbreak terminology",
    ),
    # ... add more patterns as needed
]


class PromptInjectionMiddleware(AgentMiddleware[None]):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.patterns = INJECTION_PATTERNS

    async def before_run(self, prompt, deps, ctx):
        if not isinstance(prompt, str):
            return prompt

        matched = [p for p in self.patterns if p.pattern.search(prompt)]

        if not matched:
            return prompt

        risk_score = max(p.weight for p in matched)
        if risk_score >= self.threshold:
            details = "; ".join(
                f"{p.name} ({p.description})" for p in matched
            )
            raise InputBlocked(
                f"Prompt injection detected "
                f"(score={risk_score:.2f}): {details}"
            )

        return prompt
```

## Configurable Threshold

```python
# Default (0.5) - blocks medium and high severity
middleware = PromptInjectionMiddleware(threshold=0.5)

# Strict (0.85) - only blocks severe attempts
middleware = PromptInjectionMiddleware(threshold=0.85)
```

## Context Tracking

Track injection scores in the middleware context:

```python
from pydantic_ai_middleware import MiddlewareContext
from pydantic_ai_middleware.context import HookType

ctx = MiddlewareContext()
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[PromptInjectionMiddleware()],
    context=ctx,
)

await agent.run("Hello, how are you?")

# Check the score after run
scoped = ctx.for_hook(HookType.AFTER_RUN)
score = scoped.get_from(HookType.BEFORE_RUN, "injection_score")
```

## Runnable Example

See `examples/prompt_injection.py` for a complete runnable demo with 8 patterns:

```bash
uv run python examples/prompt_injection.py
```
