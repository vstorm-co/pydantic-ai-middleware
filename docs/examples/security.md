# Security Example

Implement security checks with middleware.

## Input Validation

```python
from pydantic_ai_middleware import AgentMiddleware, InputBlocked

class InputValidator(AgentMiddleware[None]):
    blocked_patterns = [
        "ignore previous instructions",
        "system prompt",
        "jailbreak",
    ]

    async def before_run(self, prompt, deps, ctx):
        prompt_lower = str(prompt).lower()
        for pattern in self.blocked_patterns:
            if pattern in prompt_lower:
                raise InputBlocked(f"Blocked pattern: {pattern}")
        return prompt
```

## Tool Authorization

```python
from dataclasses import dataclass
from pydantic_ai_middleware import AgentMiddleware, ToolBlocked

@dataclass
class UserDeps:
    user_id: str
    roles: set[str]

class ToolAuthorization(AgentMiddleware[UserDeps]):
    tool_permissions = {
        "delete_file": {"admin"},
        "execute_code": {"admin", "developer"},
        "send_email": {"admin", "support"},
    }

    async def before_tool_call(
        self, tool_name, tool_args, deps, ctx
    ):
        required_roles = self.tool_permissions.get(tool_name, set())

        if required_roles and not (deps.roles & required_roles):
            raise ToolBlocked(
                tool_name,
                f"Requires roles: {required_roles}"
            )

        return tool_args
```

## Content Moderation

```python
class ContentModeration(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        if await self._is_inappropriate(prompt):
            raise InputBlocked("Content violates guidelines")
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        if await self._is_inappropriate(output):
            return "[Content removed due to policy violation]"
        return output

    async def _is_inappropriate(self, content):
        # Use content moderation API
        return False
```

## PII Redaction

```python
import re

class PIIRedaction(AgentMiddleware[None]):
    patterns = {
        "email": r'\b[\w.-]+@[\w.-]+\.\w+\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    }

    async def before_run(self, prompt, deps, ctx):
        return self._redact(str(prompt))

    def _redact(self, text):
        for name, pattern in self.patterns.items():
            text = re.sub(pattern, f"[REDACTED_{name.upper()}]", text)
        return text
```
