"""PII detection — block or log personally identifiable information."""

import asyncio

from pydantic_ai import Agent

from pydantic_ai_shields import InputBlocked, PiiDetector


async def main() -> None:
    # Block mode (default)
    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[PiiDetector(detect=["email", "ssn", "credit_card"])],
    )

    try:
        await agent.run("My email is john@example.com")
    except InputBlocked as e:
        print(f"Blocked: {e}")

    # Log mode — allow through but record detections
    detector = PiiDetector(action="log")
    agent_log = Agent("openai:gpt-4.1", capabilities=[detector])

    await agent_log.run("Contact me at test@example.com")
    print(f"Detections: {detector.last_detections}")

    # Custom patterns
    agent_custom = Agent(
        "openai:gpt-4.1",
        capabilities=[PiiDetector(custom_patterns={"passport": r"[A-Z]{2}\d{7}"})],
    )

    try:
        await agent_custom.run("Passport: AB1234567")
    except InputBlocked as e:
        print(f"Custom pattern blocked: {e}")


if __name__ == "__main__":
    asyncio.run(main())
