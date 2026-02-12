"""PII detection and redaction middleware example.

Demonstrates how to build middleware that detects and redacts
Personally Identifiable Information (PII) from prompts and outputs.

Run with: uv run python examples/pii_detection.py
"""

import asyncio
import re
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
# PII patterns
# ---------------------------------------------------------------------------

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}


@dataclass
class PIIMatch:
    """A single PII match found in text."""

    category: str
    value: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Middleware: detect and redact
# ---------------------------------------------------------------------------


def find_pii(text: str, categories: set[str] | None = None) -> list[PIIMatch]:
    """Scan text for PII matches."""
    matches: list[PIIMatch] = []
    for name, pattern in PII_PATTERNS.items():
        if categories and name not in categories:
            continue
        for m in pattern.finditer(text):
            matches.append(PIIMatch(category=name, value=m.group(), start=m.start(), end=m.end()))
    # Sort by position so replacements work correctly from end to start.
    matches.sort(key=lambda m: m.start)
    return matches


def redact_text(text: str, matches: list[PIIMatch]) -> str:
    """Replace PII matches with redaction tokens."""
    # Replace from end to start so indices remain valid.
    for m in reversed(matches):
        placeholder = f"[{m.category.upper()}_REDACTED]"
        text = text[: m.start] + placeholder + text[m.end :]
    return text


@dataclass
class PIIConfig:
    """Configuration for the PII middleware."""

    mode: str = "redact"  # "redact" or "block"
    categories: set[str] = field(default_factory=lambda: set(PII_PATTERNS.keys()))
    scan_output: bool = True


class PIIDetectionMiddleware(AgentMiddleware[None]):
    """Middleware that detects and handles PII in prompts and outputs.

    Two modes:
      - **redact** (default): Replace detected PII with ``[CATEGORY_REDACTED]``.
      - **block**: Raise ``InputBlocked`` / ``OutputBlocked`` when PII is found.
    """

    def __init__(self, config: PIIConfig | None = None) -> None:
        self.config = config or PIIConfig()

    async def before_run(
        self,
        prompt: str | Sequence[Any],
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> str | Sequence[Any]:
        if not isinstance(prompt, str):
            return prompt

        matches = find_pii(prompt, self.config.categories)
        if not matches:
            return prompt

        if self.config.mode == "block":
            categories = {m.category for m in matches}
            raise InputBlocked(f"PII detected: {', '.join(sorted(categories))}")

        return redact_text(prompt, matches)

    async def after_run(
        self,
        prompt: str | Sequence[Any],
        output: Any,
        deps: None,
        ctx: ScopedContext | None = None,
    ) -> Any:
        if not self.config.scan_output or not isinstance(output, str):
            return output

        matches = find_pii(output, self.config.categories)
        if not matches:
            return output

        if self.config.mode == "block":
            categories = {m.category for m in matches}
            raise OutputBlocked(f"PII in output: {', '.join(sorted(categories))}")

        return redact_text(output, matches)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def create_model(response: str) -> TestModel:
    model = TestModel()
    model.custom_output_text = response
    return model


async def demo_redact_mode() -> None:
    """Demonstrate PII redaction (default mode)."""
    print("=" * 60)
    print("MODE: REDACT  (PII replaced with placeholders)")
    print("=" * 60)

    agent = MiddlewareAgent(
        agent=Agent(create_model("Sure, contact us at support@company.com"), output_type=str),
        middleware=[PIIDetectionMiddleware(PIIConfig(mode="redact"))],
    )

    test_cases = [
        "My email is john@example.com and phone is 555-123-4567",
        "SSN: 123-45-6789, please process my request",
        "Card number 4111-1111-1111-1111 for payment",
        "Server IP is 192.168.1.100, check the logs",
        "Hello, how are you today?",  # No PII
    ]

    for prompt in test_cases:
        result = await agent.run(prompt)
        print(f"\n  Input:  {prompt}")
        print(f"  Output: {result.output}")


async def demo_block_mode() -> None:
    """Demonstrate PII blocking mode."""
    print("\n" + "=" * 60)
    print("MODE: BLOCK  (reject messages containing PII)")
    print("=" * 60)

    agent = MiddlewareAgent(
        agent=Agent(create_model("OK"), output_type=str),
        middleware=[PIIDetectionMiddleware(PIIConfig(mode="block"))],
    )

    test_cases = [
        ("My email is test@example.com", True),
        ("Hello, no PII here", False),
        ("Call me at 555-987-6543", True),
    ]

    for prompt, should_block in test_cases:
        try:
            result = await agent.run(prompt)
            status = "PASSED" if not should_block else "UNEXPECTED PASS"
            print(f"\n  [{status}] {prompt}")
            print(f"           Output: {result.output}")
        except InputBlocked as e:
            status = "BLOCKED" if should_block else "UNEXPECTED BLOCK"
            print(f"\n  [{status}] {prompt}")
            print(f"           Reason: {e.reason}")


async def demo_selective_categories() -> None:
    """Demonstrate scanning only specific PII categories."""
    print("\n" + "=" * 60)
    print("MODE: SELECTIVE  (only scan for emails and SSNs)")
    print("=" * 60)

    config = PIIConfig(mode="redact", categories={"email", "ssn"})
    agent = MiddlewareAgent(
        agent=Agent(create_model("OK"), output_type=str),
        middleware=[PIIDetectionMiddleware(config)],
    )

    test_cases = [
        "Email: user@test.com, Phone: 555-111-2222, SSN: 999-88-7777",
        "Only a phone: 555-000-1111",  # Phone not in selected categories
    ]

    for prompt in test_cases:
        result = await agent.run(prompt)
        print(f"\n  Input:  {prompt}")
        print(f"  Output: {result.output}")


async def main() -> None:
    await demo_redact_mode()
    await demo_block_mode()
    await demo_selective_categories()
    print()


if __name__ == "__main__":
    asyncio.run(main())
