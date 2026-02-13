# Rate Limiting Example

Implement rate limiting with middleware.

## Simple Rate Limiter

```python
import time
from pydantic_ai_middleware import AgentMiddleware

class RateLimitExceeded(Exception):
    pass

class RateLimiter(AgentMiddleware[None]):
    def __init__(self, max_calls: int = 10, window: int = 60):
        self.max_calls = max_calls
        self.window = window
        self._calls: list[float] = []

    async def before_run(self, prompt, deps, ctx):
        now = time.time()

        # Remove old calls outside the window
        self._calls = [t for t in self._calls if now - t < self.window]

        if len(self._calls) >= self.max_calls:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {self.max_calls} calls per {self.window}s"
            )

        self._calls.append(now)
        return prompt
```

## Per-User Rate Limiter

```python
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class UserDeps:
    user_id: str

class PerUserRateLimiter(AgentMiddleware[UserDeps]):
    def __init__(self, max_calls: int = 10, window: int = 60):
        self.max_calls = max_calls
        self.window = window
        self._user_calls: dict[str, list[float]] = defaultdict(list)

    async def before_run(self, prompt, deps, ctx):
        now = time.time()
        user_id = deps.user_id

        # Clean old calls
        self._user_calls[user_id] = [
            t for t in self._user_calls[user_id]
            if now - t < self.window
        ]

        if len(self._user_calls[user_id]) >= self.max_calls:
            raise RateLimitExceeded(f"Rate limit exceeded for user {user_id}")

        self._user_calls[user_id].append(now)
        return prompt
```

## Token Bucket Rate Limiter

```python
import asyncio

class TokenBucket(AgentMiddleware[None]):
    def __init__(self, rate: float = 1.0, capacity: int = 10):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def before_run(self, prompt, deps, ctx):
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens = min(
                self.capacity,
                self._tokens + elapsed * self.rate
            )

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.rate
                raise RateLimitExceeded(
                    f"Rate limited. Try again in {wait_time:.1f}s"
                )

            self._tokens -= 1
            return prompt
```

## Tool-Specific Rate Limiting

```python
class ToolRateLimiter(AgentMiddleware[None]):
    def __init__(self):
        self._tool_calls: dict[str, list[float]] = defaultdict(list)
        self.limits = {
            "expensive_api": (5, 60),    # 5 calls per minute
            "database_query": (100, 60),  # 100 calls per minute
        }
        self.default_limit = (50, 60)

    async def before_tool_call(
        self, tool_name, tool_args, deps, ctx
    ):
        max_calls, window = self.limits.get(tool_name, self.default_limit)
        now = time.time()

        # Clean old calls
        self._tool_calls[tool_name] = [
            t for t in self._tool_calls[tool_name]
            if now - t < window
        ]

        if len(self._tool_calls[tool_name]) >= max_calls:
            raise ToolBlocked(
                tool_name,
                f"Rate limit: {max_calls} calls per {window}s"
            )

        self._tool_calls[tool_name].append(now)
        return tool_args
```
