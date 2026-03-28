# Examples

| Example | What it shows |
|---------|--------------|
| [basic_usage.py](basic_usage.py) | Combine multiple shields on one agent |
| [prompt_injection.py](prompt_injection.py) | Detect jailbreak attempts with sensitivity levels |
| [pii_detection.py](pii_detection.py) | Block or log PII (email, SSN, credit card) |
| [cost_tracking.py](cost_tracking.py) | Token tracking, budget enforcement, callbacks |
| [tool_guard.py](tool_guard.py) | Block tools or require human approval |
| [async_guardrail.py](async_guardrail.py) | Concurrent guard + LLM execution |

## Running

```bash
export OPENAI_API_KEY=your-key
uv run python examples/basic_usage.py
```
