# Toxicity Check

This example demonstrates middleware that screens both prompts and agent outputs for toxic, offensive, or inappropriate content using a keyword-based scoring system.

## Overview

The `ToxicityMiddleware` checks text against 5 toxicity categories with weighted scoring. When the score exceeds a threshold, the request is blocked.

!!! tip
    In production, replace the keyword scorer with an ML-based classifier (e.g., Perspective API, OpenAI moderation endpoint, or a local model).

## Toxicity Categories

| Category | Weight | Example Keywords |
|----------|--------|-----------------|
| `profanity` | 0.3 | damn, crap, idiot, stupid |
| `hate_speech` | 0.9 | hate, racist, bigot, supremacy |
| `threat` | 0.9 | kill, destroy, attack, bomb |
| `harassment` | 0.5 | loser, worthless, pathetic |
| `self_harm` | 1.0 | suicide, self-harm |

## Basic Usage

```python
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

from pydantic_ai_middleware import (
    AgentMiddleware,
    InputBlocked,
    OutputBlocked,
    ScopedContext,
)

TOXICITY_KEYWORDS: dict[str, set[str]] = {
    "profanity": {"damn", "crap", "idiot", "stupid", "moron"},
    "hate_speech": {"hate", "racist", "bigot", "supremacy"},
    "threat": {"kill", "destroy", "attack", "bomb"},
    "harassment": {"loser", "worthless", "pathetic"},
    "self_harm": {"suicide", "self-harm"},
}

CATEGORY_WEIGHTS: dict[str, float] = {
    "profanity": 0.3,
    "hate_speech": 0.9,
    "threat": 0.9,
    "harassment": 0.5,
    "self_harm": 1.0,
}


@dataclass
class ToxicityConfig:
    threshold: float = 0.5
    check_input: bool = True
    check_output: bool = True


class ToxicityMiddleware(AgentMiddleware[None]):
    def __init__(self, config: ToxicityConfig | None = None) -> None:
        self.config = config or ToxicityConfig()

    def _score(self, text: str) -> float:
        words = set(text.lower().split())
        max_score = 0.0
        for category, keywords in TOXICITY_KEYWORDS.items():
            matched = words & keywords
            if matched:
                weight = CATEGORY_WEIGHTS.get(category, 0.5)
                cat_score = min(1.0, weight * len(matched))
                max_score = max(max_score, cat_score)
        return max_score

    async def before_run(self, prompt, deps, ctx):
        if not self.config.check_input or not isinstance(prompt, str):
            return prompt
        score = self._score(prompt)
        if score >= self.config.threshold:
            raise InputBlocked(
                f"Toxic content detected (score={score:.2f})"
            )
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        if not self.config.check_output or not isinstance(output, str):
            return output
        score = self._score(output)
        if score >= self.config.threshold:
            raise OutputBlocked(
                f"Toxic output detected (score={score:.2f})"
            )
        return output
```

## Configurable Thresholds

```python
# Low threshold (0.2) - catches mild profanity
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[ToxicityMiddleware(ToxicityConfig(threshold=0.2))],
)

# High threshold (0.8) - only blocks severe content
agent = MiddlewareAgent(
    agent=base_agent,
    middleware=[ToxicityMiddleware(ToxicityConfig(threshold=0.8))],
)
```

## Input + Output Screening

The middleware screens both directions by default:

- **Input screening**: Blocks toxic prompts before they reach the LLM
- **Output screening**: Blocks toxic model responses before they reach the user

## Runnable Example

See `examples/toxicity_check.py` for a complete runnable demo:

```bash
uv run python examples/toxicity_check.py
```
