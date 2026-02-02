# app/agent/critic.py

import json
from app.llm.ollama_client import ollama_llm


CRITIC_PROMPT = """
You are a critic agent.

User goal:
{goal}

Execution observations:
{observations}

Evaluate the result.

Decide:
- "accept" → answer is complete, grounded, and useful
- "retry" → missing info, weak reasoning, or hallucination

Reply ONLY in valid JSON:
{
  "decision": "accept" | "retry",
  "reason": "short explanation"
}
"""


async def critic_node(state: dict) -> dict:
    """
    Input state:
    {
        "goal": str,
        "observations": list
    }
    """

    response = ollama_llm.generate(
        CRITIC_PROMPT.format(
            goal=state["goal"],
            observations=state["observations"]
        )
    )

    try:
        decision = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError(f"LLM did not return valid JSON:\n{response}")

    return {
        **state,
        "critic_decision": decision["decision"],
        "critic_reason": decision["reason"]
    }
