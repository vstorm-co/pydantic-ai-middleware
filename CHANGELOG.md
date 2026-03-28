# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-28

### Changed

- **Renamed to `pydantic-ai-shields`** — the package is now focused exclusively on guardrail capabilities built on pydantic-ai's native capabilities API. The old middleware layer (`MiddlewareAgent`, `AgentMiddleware`, etc.) has been removed — pydantic-ai v1.71+ provides this natively.

### Added

- **5 infrastructure capabilities**:
  - **`CostTracking`** — token usage tracking, USD cost calculation via genai-prices, budget enforcement
  - **`ToolGuard`** — block tools via `prepare_tools` or require approval via `before_tool_execute`
  - **`InputGuard`** — block/check user input with pluggable guard function (sync or async)
  - **`OutputGuard`** — block/check model output with pluggable guard function (sync or async)
  - **`AsyncGuardrail`** — run guardrail concurrently with LLM call, 3 timing modes
- **5 built-in content shields** (zero external dependencies):
  - **`PromptInjection`** — detect prompt injection / jailbreak across 6 categories with 3 sensitivity levels
  - **`PiiDetector`** — detect PII (email, phone, SSN, credit card, IP) with block or log action
  - **`SecretRedaction`** — block API keys, tokens, credentials in model output (OpenAI, Anthropic, AWS, GitHub, Slack, JWT, private keys)
  - **`BlockedKeywords`** — block forbidden keywords/phrases with case, whole-word, and regex modes
  - **`NoRefusals`** — block LLM refusals with 10 built-in patterns and partial refusal support

### Removed

- All legacy middleware modules: `MiddlewareAgent`, `AgentMiddleware`, `MiddlewareToolset`, `MiddlewareChain`, `MiddlewareContext`, `ConditionalMiddleware`, `ParallelMiddleware`, `AsyncGuardrailMiddleware`, `CostTrackingMiddleware`, decorators, config loaders, pipeline spec, builder
- All legacy examples and documentation
