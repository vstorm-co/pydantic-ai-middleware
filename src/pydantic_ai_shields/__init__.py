"""pydantic-ai-shields — Guardrail capabilities for Pydantic AI agents.

Ready-to-use capabilities for safety, cost control, and permission management.
Built on pydantic-ai's native capabilities API.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_shields import CostTracking, ToolGuard, InputGuard

    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[
            CostTracking(budget_usd=5.0),
            ToolGuard(blocked=["execute"]),
            InputGuard(guard=lambda prompt: "ignore" not in prompt.lower()),
        ],
    )
    ```
"""

from __future__ import annotations

from .guardrails import (
    AsyncGuardrail,
    BudgetExceededError,
    CostInfo,
    CostTracking,
    GuardrailError,
    InputBlocked,
    InputGuard,
    OutputBlocked,
    OutputGuard,
    ToolBlocked,
    ToolGuard,
)
from .shields import (
    BlockedKeywords,
    NoRefusals,
    PiiDetector,
    PromptInjection,
    SecretRedaction,
)

try:
    from importlib.metadata import version as _metadata_version

    __version__ = _metadata_version("pydantic-ai-shields")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    # Capabilities
    "CostTracking",
    "ToolGuard",
    "InputGuard",
    "OutputGuard",
    "AsyncGuardrail",
    # Data
    "CostInfo",
    # Built-in content shields
    "PromptInjection",
    "PiiDetector",
    "SecretRedaction",
    "BlockedKeywords",
    "NoRefusals",
    # Exceptions
    "GuardrailError",
    "InputBlocked",
    "OutputBlocked",
    "ToolBlocked",
    "BudgetExceededError",
]
