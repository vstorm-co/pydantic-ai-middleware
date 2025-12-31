"""Simple chatbot with input filtering guardrail."""

import asyncio
import os
import re

from dotenv import load_dotenv
from pydantic_ai import Agent

from pydantic_ai_middleware import AgentMiddleware, InputBlocked, MiddlewareAgent

load_dotenv()


class InputFilter(AgentMiddleware[None]):
    """Blocks prompt injections and PII."""

    async def before_run(self, prompt, deps):
        if not isinstance(prompt, str):
            return prompt

        prompt_lower = prompt.lower()

        # Block prompt injection
        if any(
            phrase in prompt_lower
            for phrase in [
                "ignore previous",
                "disregard",
                "system prompt",
                "you are now",
                "act as",
            ]
        ):
            raise InputBlocked("Invalid instruction detected.")

        # Block PII
        if re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", prompt):
            raise InputBlocked("Please don't share email addresses.")

        if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", prompt):
            raise InputBlocked("Please don't share phone numbers.")

        return prompt


async def main():
    """Run the chatbot."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return

    # Create agent with input filter
    base_agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Be concise.",
        output_type=str,
    )

    chatbot = MiddlewareAgent(agent=base_agent, middleware=[InputFilter()])

    print("Chatbot ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        try:
            result = await chatbot.run(user_input)
            print(f"Bot: {result.output}\n")
        except InputBlocked as e:
            print(f"🚫 {e}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
