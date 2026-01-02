# Logging Example

Implement comprehensive logging with middleware.

## Basic Logging

```python
import logging
from pydantic_ai_middleware import AgentMiddleware

logger = logging.getLogger(__name__)

class LoggingMiddleware(AgentMiddleware[None]):
    async def before_run(self, prompt, deps, ctx):
        logger.info(f"Agent started with: {prompt[:100]}...")
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        logger.info(f"Agent finished with: {output}")
        return output

    async def before_tool_call(
        self, tool_name, tool_args, deps, ctx
    ):
        logger.info(f"Tool call: {tool_name}({tool_args})")
        return tool_args

    async def on_error(self, error, deps, ctx):
        logger.error(f"Agent error: {error}")
        return None
```

## Structured Logging

```python
import json
from datetime import datetime

class StructuredLogger(AgentMiddleware[None]):
    def __init__(self):
        self.run_id = None

    async def before_run(self, prompt, deps, ctx):
        self.run_id = datetime.now().isoformat()
        self._log({
            "event": "run_start",
            "run_id": self.run_id,
            "prompt_length": len(str(prompt)),
        })
        return prompt

    async def after_run(self, prompt, output, deps, ctx):
        self._log({
            "event": "run_end",
            "run_id": self.run_id,
            "output_length": len(str(output)),
        })
        return output

    async def before_tool_call(
        self, tool_name, tool_args, deps, ctx
    ):
        self._log({
            "event": "tool_call",
            "run_id": self.run_id,
            "tool": tool_name,
            "args": tool_args,
        })
        return tool_args

    def _log(self, data):
        print(json.dumps(data))
```

## Audit Logging

```python
from dataclasses import dataclass

@dataclass
class AuditDeps:
    user_id: str
    session_id: str

class AuditMiddleware(AgentMiddleware[AuditDeps]):
    async def before_run(self, prompt, deps, ctx):
        await self._record({
            "action": "agent_start",
            "user_id": deps.user_id,
            "session_id": deps.session_id,
            "prompt": prompt,
        })
        return prompt

    async def before_tool_call(
        self, tool_name, tool_args, deps, ctx
    ):
        await self._record({
            "action": "tool_call",
            "user_id": deps.user_id,
            "tool": tool_name,
            "args": tool_args,
        })
        return tool_args

    async def _record(self, data):
        # Save to database, send to logging service, etc.
        pass
```
