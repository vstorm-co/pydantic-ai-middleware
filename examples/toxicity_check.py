"""Toxicity check middleware example.

Demonstrates how to build middleware that screens both prompts and
agent outputs for toxic, offensive, or inappropriate content using
a keyword-based scoring system.

In production, replace the keyword scorer with an ML-based classifier
(e.g., Perspective API, OpenAI moderation endpoint, or a local model).

Run with: uv run python examples/toxicity_check.py
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_middleware import (
    AgentMiddleware,
    InputBlocked,
    MiddlewareAgent,
    OutputBlocked,
    ScopedContext,
)

# ---------------------------------------------------------------------------
# Toxicity categories and keyword lists
# ---------------------------------------------------------------------------

TOXICITY_KEYWORDS: dict[str, set[str]] = {
    "profanity": {"damn", "crap", "hell", "idiot", "stupid", "dumb", "moron"},
    "hate_speech": {"hate", "racist", "sexist", "bigot", "slur", "supremacy"},
    "threat": {"kill", "destroy", "attack", "bomb", "murder", "hurt", "harm"},
    "harassment": {"loser", "worthless", "pathetic", "ugly", "disgusting"},
    "self_harm": {"suicide", "self-harm", "cut myself", "end it all"},
}


@dataclass
class ToxicityResult:
    """Result of a toxicity check."""

    is_toxic: bool
    score: float
    categories: dict[str, float]
    flagged_words: list[str]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS: dict[str, float] = {
    "profanity": 0.3,
    "hate_speech": 0.9,
    "threat": 0.9,
    "harassment": 0.5,
    "self_harm": 1.0,
}


def score_toxicity(text: str) -> ToxicityResult:
    """Score text for toxicity across multiple categories.

    Returns a :class:`ToxicityResult` with per-category scores and an
    overall score (max across categories).
    """
    words = set(text.lower().split())
    categories: dict[str, float] = {}
    flagged: list[str] = []

    for category, keywords in TOXICITY_KEYWORDS.items():
        matched = words & keywords
        if matched:
            weight = CATEGORY_WEIGHTS.get(category, 0.5)
            # Score increases with more matches in the same category.
            cat_score = min(1.0, weight * len(matched))
            categories[category] = cat_score
            flagged.extend(sorted(matched))

    overall = max(categories.values()) if categories else 0.0
    return ToxicityResult(
        is_toxic=overall > 0.0,
        score=overall,
        categories=categories,
        flagged_words=flagged,
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@dataclass
class ToxicityConfig:
    """Configuration for toxicity checking."""

    threshold: float = 0.5
    check_input: bool = True
    check_output: bool = True
    categories: set[str] = field(default_factory=lambda: set(TOXICITY_KEYWORDS.keys()))
    log_results: bool = True


class ToxicityMiddleware(AgentMiddleware[None]):
    """Middleware that screens prompts and outputs for toxic content.

    When the toxicity score exceeds the configured threshold the
    request is blocked with :class:`InputBlocked` or :class:`OutputBlocked`.
    """

    def __init__(self, config: ToxicityConfig | None = None) -> None:
        self.config = config or ToxicityConfig()

    def _check(self, text: str) -> ToxicityResult:
        result = score_toxicity(text)
        if self.config.log_results and result.is_toxic:
            cats = ", ".join(f"{k}={v:.2f}" for k, v in result.categories.items())
            words = result.flagged_words
            print(f"    [Toxicity] score={result.score:.2f} | {cats} | words={words}")
        return result

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        if not self.config.check_input or not isinstance(prompt, str):
            return prompt

        result = self._check(prompt)
        if ctx:
            ctx.set("toxicity_score", result.score)
            ctx.set("toxicity_categories", result.categories)

        if result.score >= self.config.threshold:
            raise InputBlocked(
                f"Toxic content detected (score={result.score:.2f}, "
                f"categories={list(result.categories.keys())})"
            )
        return prompt

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        if not self.config.check_output or not isinstance(output, str):
            return output

        result = self._check(output)
        if result.score >= self.config.threshold:
            raise OutputBlocked(
                f"Toxic output detected (score={result.score:.2f}, "
                f"categories={list(result.categories.keys())})"
            )
        return output


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def create_model(response: str = "I'd be happy to help!") -> TestModel:
    model = TestModel()
    model.custom_output_text = response
    return model


async def run_test(agent: MiddlewareAgent[None, str], prompt: str) -> None:
    """Run a single test and print results."""
    print(f"\n  Input: {prompt!r}")
    try:
        result = await agent.run(prompt)
        print(f"  -> PASSED (output: {result.output!r})")
    except InputBlocked as e:
        print(f"  -> INPUT BLOCKED: {e.reason}")
    except OutputBlocked as e:
        print(f"  -> OUTPUT BLOCKED: {e.reason}")


async def demo_input_screening() -> None:
    """Screen user prompts for toxicity."""
    print("=" * 60)
    print("INPUT TOXICITY SCREENING (threshold=0.5)")
    print("=" * 60)

    agent = MiddlewareAgent(
        agent=Agent(create_model(), output_type=str),
        middleware=[ToxicityMiddleware()],
    )

    prompts = [
        "What's the weather like today?",
        "You're an idiot, give me the answer",
        "I hate everyone, they're all stupid",
        "Help me write a poem about nature",
        "I want to destroy all the bugs in my code",
    ]

    for prompt in prompts:
        await run_test(agent, prompt)


async def demo_output_screening() -> None:
    """Screen agent outputs for toxicity."""
    print("\n\n" + "=" * 60)
    print("OUTPUT TOXICITY SCREENING")
    print("=" * 60)

    # Model produces toxic output
    toxic_agent = MiddlewareAgent(
        agent=Agent(create_model("You're a moron and an idiot"), output_type=str),
        middleware=[ToxicityMiddleware()],
    )

    # Model produces clean output
    clean_agent = MiddlewareAgent(
        agent=Agent(create_model("Here's a helpful answer for you!"), output_type=str),
        middleware=[ToxicityMiddleware()],
    )

    print("\n  [Toxic model output]")
    await run_test(toxic_agent, "Tell me a joke")

    print("\n  [Clean model output]")
    await run_test(clean_agent, "Tell me a joke")


async def demo_configurable_threshold() -> None:
    """Show how threshold affects blocking behavior."""
    print("\n\n" + "=" * 60)
    print("CONFIGURABLE THRESHOLDS")
    print("=" * 60)

    prompt = "That's a damn stupid idea"

    for threshold in [0.2, 0.5, 0.8]:
        config = ToxicityConfig(threshold=threshold, log_results=False)
        agent = MiddlewareAgent(
            agent=Agent(create_model(), output_type=str),
            middleware=[ToxicityMiddleware(config)],
        )
        print(f"\n  Threshold: {threshold}")
        await run_test(agent, prompt)


async def main() -> None:
    await demo_input_screening()
    await demo_output_screening()
    await demo_configurable_threshold()
    print()


if __name__ == "__main__":
    asyncio.run(main())
