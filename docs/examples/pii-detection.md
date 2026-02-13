# PII Detection & Redaction

This example demonstrates middleware that detects and redacts Personally Identifiable Information (PII) from prompts and agent outputs.

## Overview

The `PIIDetectionMiddleware` scans text for common PII patterns (emails, phone numbers, SSNs, credit cards, IP addresses) and either redacts them with placeholders or blocks the request entirely.

## Two Modes

| Mode | Behavior |
|------|----------|
| `redact` | Replace PII with `[CATEGORY_REDACTED]` placeholders |
| `block` | Raise `InputBlocked` / `OutputBlocked` when PII is found |

## Redact Mode

```python
import re
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

from pydantic_ai_middleware import AgentMiddleware, InputBlocked, OutputBlocked, ScopedContext

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b"),
    "phone_us": re.compile(
        r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}


@dataclass
class PIIConfig:
    mode: str = "redact"  # "redact" or "block"
    categories: set[str] = field(
        default_factory=lambda: set(PII_PATTERNS.keys())
    )
    scan_output: bool = True


class PIIDetectionMiddleware(AgentMiddleware[None]):
    def __init__(self, config: PIIConfig | None = None) -> None:
        self.config = config or PIIConfig()

    def _find_and_redact(self, text: str) -> tuple[str, list[str]]:
        found: list[str] = []
        for name, pattern in PII_PATTERNS.items():
            if name not in self.config.categories:
                continue
            for m in pattern.finditer(text):
                found.append(name)
                text = (
                    text[: m.start()]
                    + f"[{name.upper()}_REDACTED]"
                    + text[m.end() :]
                )
        return text, found

    async def before_run(
        self, prompt, deps, ctx
    ):
        if not isinstance(prompt, str):
            return prompt
        redacted, found = self._find_and_redact(prompt)
        if found and self.config.mode == "block":
            raise InputBlocked(f"PII detected: {', '.join(sorted(set(found)))}")
        return redacted

    async def after_run(self, prompt, output, deps, ctx):
        if not self.config.scan_output or not isinstance(output, str):
            return output
        redacted, found = self._find_and_redact(output)
        if found and self.config.mode == "block":
            raise OutputBlocked(f"PII in output: {', '.join(sorted(set(found)))}")
        return redacted
```

## Block Mode

```python
config = PIIConfig(mode="block")
middleware = PIIDetectionMiddleware(config)

# This will raise InputBlocked:
# await middleware.before_run("My email is john@example.com", None, None)
```

## Selective Categories

Scan only specific PII types:

```python
config = PIIConfig(mode="redact", categories={"email", "ssn"})
middleware = PIIDetectionMiddleware(config)

# Only emails and SSNs are redacted; phone numbers pass through
```

## Usage with Agent

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent

agent = MiddlewareAgent(
    agent=Agent('openai:gpt-4o', output_type=str),
    middleware=[PIIDetectionMiddleware(PIIConfig(mode="redact"))],
)

result = await agent.run("My email is john@example.com")
# PII is redacted before reaching the LLM
```

## Runnable Example

See `examples/pii_detection.py` for a complete runnable demo:

```bash
uv run python examples/pii_detection.py
```
