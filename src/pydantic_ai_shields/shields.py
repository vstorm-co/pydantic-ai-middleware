"""Built-in content and security shields for pydantic-ai agents.

Ready-to-use shields for prompt injection detection, PII filtering,
secret redaction, keyword blocking, and refusal detection.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_shields import PromptInjection, PiiDetector, SecretRedaction

    agent = Agent("openai:gpt-4.1", capabilities=[
        PromptInjection(sensitivity="high"),
        PiiDetector(detect=["email", "ssn", "credit_card"]),
        SecretRedaction(),
    ])
    ```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_shields.guardrails import InputBlocked, OutputBlocked

# ---------------------------------------------------------------------------
# Prompt Injection Shield
# ---------------------------------------------------------------------------

# Patterns organized by category and sensitivity
_INJECTION_PATTERNS: dict[str, list[tuple[str, str]]] = {
    # (pattern, sensitivity_level) — "low" = obvious, "medium" = balanced, "high" = aggressive
    "ignore_instructions": [
        (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)", "low"),
        (r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)", "low"),
        (r"forget\s+(everything|all)\s+(you|that)\s+(were|was)\s+told", "medium"),
        (r"do\s+not\s+follow\s+(your|the)\s+(previous|original)\s+instructions", "medium"),
    ],
    "system_override": [
        (r"you\s+are\s+now\s+(a|an)\s+", "medium"),
        (r"new\s+instructions?\s*:", "medium"),
        (r"system\s*:\s*", "high"),
        (r"<<\s*sys(tem)?\s*>>", "medium"),
        (r"\[system\]", "medium"),
    ],
    "role_play": [
        (r"pretend\s+(you\s+are|to\s+be)\s+", "medium"),
        (r"act\s+as\s+(if\s+you\s+are|a|an)\s+", "high"),
        (r"roleplay\s+as\s+", "medium"),
        (r"from\s+now\s+on\s+you\s+(are|will)\s+", "medium"),
    ],
    "delimiter_injection": [
        (r"```\s*(system|admin|root)", "low"),
        (r"---\s*(new|system)\s+(prompt|instructions?)", "low"),
        (r"<\|?(system|admin|root)\|?>", "low"),
    ],
    "prompt_leaking": [
        (
            r"(show|tell|reveal|repeat|print)\s+(me\s+)?(your|the)\s+(system\s+)?"
            r"(prompt|instructions)",
            "low",
        ),
        (r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)", "medium"),
        (r"output\s+(your|the)\s+(initial|original|system)\s+(prompt|instructions)", "low"),
    ],
    "jailbreak": [
        (r"do\s+anything\s+now", "low"),
        (r"DAN\s+mode", "low"),
        (r"developer\s+mode\s+(enabled|activated|on)", "low"),
        (
            r"(bypass|disable|override)\s+(your\s+)?(safety|content|ethical)\s+"
            r"(filter|guidelines|restrictions)",
            "low",
        ),
        (r"no\s+(ethical|moral|safety)\s+(guidelines|restrictions|filters)", "medium"),
    ],
}

_SENSITIVITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class PromptInjection(AbstractCapability[Any]):
    """Detect and block prompt injection attempts.

    Scans user input for common injection patterns across 6 categories:
    ignore_instructions, system_override, role_play, delimiter_injection,
    prompt_leaking, and jailbreak.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_shields import PromptInjection

        # Default: medium sensitivity
        agent = Agent("openai:gpt-4.1", capabilities=[PromptInjection()])

        # High sensitivity (more false positives, better detection)
        agent = Agent("openai:gpt-4.1", capabilities=[PromptInjection(sensitivity="high")])

        # Only check specific categories
        agent = Agent("openai:gpt-4.1", capabilities=[PromptInjection(
            categories=["jailbreak", "prompt_leaking"],
        )])

        # Add custom patterns
        agent = Agent("openai:gpt-4.1", capabilities=[PromptInjection(
            custom_patterns=[r"sudo\\s+mode", r"admin\\s+override"],
        )])
        ```

    Attributes:
        sensitivity: Detection sensitivity — "low" (obvious attacks only),
            "medium" (balanced), or "high" (aggressive, may have false positives).
        categories: Which injection categories to check. None = all.
        custom_patterns: Additional regex patterns to check.
    """

    sensitivity: Literal["low", "medium", "high"] = "medium"
    categories: list[str] | None = None
    custom_patterns: list[str] | None = None
    _compiled: list[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        threshold = _SENSITIVITY_ORDER[self.sensitivity]
        cats = self.categories or list(_INJECTION_PATTERNS.keys())

        patterns: list[str] = []
        for cat in cats:
            for pattern, level in _INJECTION_PATTERNS.get(cat, []):
                if _SENSITIVITY_ORDER.get(level, 0) <= threshold:
                    patterns.append(pattern)

        if self.custom_patterns:
            patterns.extend(self.custom_patterns)

        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    async def before_run(self, ctx: RunContext[Any]) -> None:
        prompt = ctx.prompt
        if prompt is None:
            return
        text = str(prompt) if not isinstance(prompt, str) else prompt

        for pattern in self._compiled:
            if pattern.search(text):
                raise InputBlocked(f"Prompt injection detected (pattern: {pattern.pattern})")


# ---------------------------------------------------------------------------
# PII Detector Shield
# ---------------------------------------------------------------------------

_PII_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


@dataclass
class PiiDetector(AbstractCapability[Any]):
    """Detect personally identifiable information (PII) in user input.

    Scans for email addresses, phone numbers, SSNs, credit card numbers,
    and IP addresses using regex patterns.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_shields import PiiDetector

        # Detect all PII types
        agent = Agent("openai:gpt-4.1", capabilities=[PiiDetector()])

        # Only specific types
        agent = Agent("openai:gpt-4.1", capabilities=[PiiDetector(
            detect=["email", "ssn", "credit_card"],
        )])

        # Add custom patterns
        agent = Agent("openai:gpt-4.1", capabilities=[PiiDetector(
            custom_patterns={"passport": r"[A-Z]{2}\\d{7}"},
        )])

        # Log instead of blocking
        agent = Agent("openai:gpt-4.1", capabilities=[PiiDetector(action="log")])
        ```

    Attributes:
        detect: Which PII types to detect. None = all built-in types.
            Built-in: "email", "phone", "ssn", "credit_card", "ip_address".
        custom_patterns: Dict of additional pattern name → regex string.
        action: "block" raises InputBlocked, "log" allows through (check `last_detections`).
    """

    detect: list[str] | None = None
    custom_patterns: dict[str, str] | None = None
    action: Literal["block", "log"] = "block"
    last_detections: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _compiled: dict[str, re.Pattern[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        types = self.detect or list(_PII_PATTERNS.keys())
        for pii_type in types:
            if pii_type in _PII_PATTERNS:
                self._compiled[pii_type] = re.compile(_PII_PATTERNS[pii_type])

        if self.custom_patterns:
            for name, pattern in self.custom_patterns.items():
                self._compiled[name] = re.compile(pattern)

    async def before_run(self, ctx: RunContext[Any]) -> None:
        prompt = ctx.prompt
        if prompt is None:
            return
        text = str(prompt) if not isinstance(prompt, str) else prompt

        self.last_detections = []
        for pii_type, pattern in self._compiled.items():
            matches = pattern.findall(text)
            if matches:
                self.last_detections.append(
                    {
                        "type": pii_type,
                        "count": len(matches),
                    }
                )

        if self.last_detections and self.action == "block":
            types = ", ".join(d["type"] for d in self.last_detections)
            raise InputBlocked(f"PII detected in input: {types}")


# ---------------------------------------------------------------------------
# Secret Redaction Shield
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: dict[str, str] = {
    "openai_key": r"sk-[a-zA-Z0-9]{20,}",
    "anthropic_key": r"sk-ant-[a-zA-Z0-9-]{20,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "aws_secret_key": (
        r"(?:aws)?_?(?:secret)?_?(?:access)?_?(?:key)?['\"]?"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}"
    ),
    "github_token": r"(?:ghp|gho|ghs|ghr)_[A-Za-z0-9_]{36,}",
    "slack_token": r"xox[bporas]-[A-Za-z0-9-]{10,}",
    "jwt": r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    "private_key": r"-----BEGIN\s+(?:RSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----",
    "generic_api_key": r"(?:api[_-]?key|apikey|token)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}",
}


@dataclass
class SecretRedaction(AbstractCapability[Any]):
    """Detect and block exposure of API keys, tokens, and credentials in model output.

    Scans model responses for common secret patterns including OpenAI, Anthropic,
    AWS, GitHub, Slack keys, JWTs, and private keys.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_shields import SecretRedaction

        # Block any secret in output
        agent = Agent("openai:gpt-4.1", capabilities=[SecretRedaction()])

        # Only check specific secret types
        agent = Agent("openai:gpt-4.1", capabilities=[SecretRedaction(
            detect=["openai_key", "aws_access_key", "private_key"],
        )])

        # Add custom patterns
        agent = Agent("openai:gpt-4.1", capabilities=[SecretRedaction(
            custom_patterns={"stripe_key": r"sk_live_[A-Za-z0-9]{24,}"},
        )])
        ```

    Attributes:
        detect: Which secret types to scan for. None = all built-in types.
            Built-in: "openai_key", "anthropic_key", "aws_access_key", "aws_secret_key",
            "github_token", "slack_token", "jwt", "private_key", "generic_api_key".
        custom_patterns: Dict of additional pattern name → regex string.
    """

    detect: list[str] | None = None
    custom_patterns: dict[str, str] | None = None
    _compiled: dict[str, re.Pattern[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        types = self.detect or list(_SECRET_PATTERNS.keys())
        for secret_type in types:
            if secret_type in _SECRET_PATTERNS:
                self._compiled[secret_type] = re.compile(_SECRET_PATTERNS[secret_type])

        if self.custom_patterns:
            for name, pattern in self.custom_patterns.items():
                self._compiled[name] = re.compile(pattern)

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        output = str(result.output) if hasattr(result, "output") else str(result)

        for secret_type, pattern in self._compiled.items():
            if pattern.search(output):
                raise OutputBlocked(f"Secret detected in output: {secret_type}")

        return result


# ---------------------------------------------------------------------------
# Blocked Keywords Shield
# ---------------------------------------------------------------------------


@dataclass
class BlockedKeywords(AbstractCapability[Any]):
    """Block prompts containing forbidden keywords or phrases.

    Configurable keyword blocking with support for case sensitivity,
    whole-word matching, and regex patterns.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_shields import BlockedKeywords

        # Simple keyword list
        agent = Agent("openai:gpt-4.1", capabilities=[BlockedKeywords(
            keywords=["competitor_name", "internal_only", "classified"],
        )])

        # Case-sensitive matching
        agent = Agent("openai:gpt-4.1", capabilities=[BlockedKeywords(
            keywords=["SECRET", "CLASSIFIED"],
            case_sensitive=True,
        )])

        # Whole-word only (won't match "classification")
        agent = Agent("openai:gpt-4.1", capabilities=[BlockedKeywords(
            keywords=["class", "secret"],
            whole_words=True,
        )])

        # Regex patterns
        agent = Agent("openai:gpt-4.1", capabilities=[BlockedKeywords(
            keywords=[r"password\\s*=\\s*\\S+"],
            use_regex=True,
        )])
        ```

    Attributes:
        keywords: List of keywords, phrases, or regex patterns to block.
        case_sensitive: Whether matching is case-sensitive (default: False).
        whole_words: Match whole words only — "class" won't match "classification" (default: False).
        use_regex: Treat keywords as regex patterns (default: False).
    """

    keywords: list[str] = field(default_factory=list)
    case_sensitive: bool = False
    whole_words: bool = False
    use_regex: bool = False
    _compiled: list[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        flags = 0 if self.case_sensitive else re.IGNORECASE

        for kw in self.keywords:
            if self.use_regex:
                pattern = kw
            elif self.whole_words:
                pattern = rf"\b{re.escape(kw)}\b"
            else:
                pattern = re.escape(kw)
            self._compiled.append(re.compile(pattern, flags))

    async def before_run(self, ctx: RunContext[Any]) -> None:
        prompt = ctx.prompt
        if prompt is None:
            return
        text = str(prompt) if not isinstance(prompt, str) else prompt

        for pattern in self._compiled:
            match = pattern.search(text)
            if match:
                raise InputBlocked(f"Blocked keyword detected: '{match.group()}'")


# ---------------------------------------------------------------------------
# No Refusals Shield
# ---------------------------------------------------------------------------

_DEFAULT_REFUSAL_PATTERNS = [
    r"I\s+cannot\s+help\s+with\s+that",
    r"I'?m\s+not\s+able\s+to",
    r"I\s+must\s+decline",
    r"I\s+can'?t\s+assist\s+with",
    r"I'?m\s+unable\s+to",
    r"I\s+don'?t\s+have\s+the\s+ability\s+to",
    r"as\s+an\s+AI,?\s+I\s+can(?:not|'t)",
    r"I'?m\s+sorry,?\s+(?:but\s+)?I\s+can(?:not|'t)",
    r"it'?s?\s+(?:not\s+)?(?:within|outside)\s+my\s+(?:capabilities|scope)",
    r"I\s+(?:need\s+to|have\s+to|must)\s+(?:refuse|decline)",
]


@dataclass
class NoRefusals(AbstractCapability[Any]):
    """Block LLM refusals — ensure the model attempts to answer.

    Detects common refusal phrases in model output and raises OutputBlocked.
    Useful for agents that should always attempt a task rather than refuse.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_shields import NoRefusals

        # Default refusal patterns
        agent = Agent("openai:gpt-4.1", capabilities=[NoRefusals()])

        # Custom patterns
        agent = Agent("openai:gpt-4.1", capabilities=[NoRefusals(
            patterns=[r"I cannot", r"I'm not able to", r"outside my scope"],
        )])

        # Allow partial refusals (refusal + substance)
        agent = Agent("openai:gpt-4.1", capabilities=[NoRefusals(
            allow_partial=True,
            min_response_length=50,
        )])
        ```

    Attributes:
        patterns: Refusal regex patterns to detect. None = built-in set of 10 patterns.
        allow_partial: If True, allow responses that contain refusal language but
            also have substantial content (above min_response_length).
        min_response_length: Minimum character count for a "substantial" response
            when allow_partial is True (default: 50).
    """

    patterns: list[str] | None = None
    allow_partial: bool = False
    min_response_length: int = 50
    _compiled: list[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        raw = self.patterns or _DEFAULT_REFUSAL_PATTERNS
        self._compiled = [re.compile(p, re.IGNORECASE) for p in raw]

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        output = str(result.output) if hasattr(result, "output") else str(result)

        for pattern in self._compiled:
            if pattern.search(output):
                if self.allow_partial and len(output) >= self.min_response_length:
                    continue
                raise OutputBlocked(f"Model refusal detected (pattern: {pattern.pattern})")

        return result


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PromptInjection",
    "PiiDetector",
    "SecretRedaction",
    "BlockedKeywords",
    "NoRefusals",
]
