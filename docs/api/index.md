# API Reference

Complete API documentation for pydantic-ai-shields.

## Infrastructure Shields

- [`CostTracking`](guardrails.md#costtracking) — Token/USD tracking with budget enforcement
- [`ToolGuard`](guardrails.md#toolguard) — Block tools or require approval
- [`InputGuard`](guardrails.md#inputguard) — Custom input validation
- [`OutputGuard`](guardrails.md#outputguard) — Custom output validation
- [`AsyncGuardrail`](guardrails.md#asyncguardrail) — Concurrent guardrail + LLM execution

## Content Shields

- [`PromptInjection`](shields.md#promptinjection) — Prompt injection / jailbreak detection
- [`PiiDetector`](shields.md#piidetector) — PII detection
- [`SecretRedaction`](shields.md#secretredaction) — Secret/credential detection in output
- [`BlockedKeywords`](shields.md#blockedkeywords) — Keyword blocking
- [`NoRefusals`](shields.md#norefusals) — Refusal detection

## Exceptions

- [`GuardrailError`](exceptions.md) — Base exception
- [`InputBlocked`](exceptions.md) — Input validation failed
- [`OutputBlocked`](exceptions.md) — Output validation failed
- [`ToolBlocked`](exceptions.md) — Tool access denied
- [`BudgetExceededError`](exceptions.md) — Cost budget exceeded
