# app/agent/critic.py

import json
from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm


CRITIC_PROMPT = """
You are a critic agent in an Agentic RAG system.

User goal:
{goal}

Generated answer:
{answer}

Grounded:
{grounded}

Confidence score:
{confidence}

Citations:
{citations}

Execution observations:
{observations}

Evaluate the result.

Decide:
- "accept" → answer is complete, grounded, and useful
- "retry" → missing info, weak reasoning, hallucination, or low confidence

Reply ONLY in valid JSON:
{
  "decision": "accept" | "retry",
  "reason": "short explanation"
}
"""


def critic_node(state: AgentState) -> AgentState:
    """
    Critic evaluates final answer quality and grounding.
    NEVER crashes the graph.
    """

    response = ollama_llm.generate(
        CRITIC_PROMPT.format(
            goal=state.get("goal", ""),
            answer=state.get("answer", ""),
            grounded=state.get("grounded", False),
            confidence=state.get("confidence", 0.0),
            citations=state.get("citations", []),
            observations=state.get("observations", []),
        )
    )

    try:
        decision_json = json.loads(response)

        decision = decision_json.get("decision", "retry")
        reason = decision_json.get("reason", "No reason provided")

        # Enforce strict contract
        if decision not in ("accept", "retry"):
            decision = "retry"
            reason = "Invalid critic decision value"

        

    except Exception:
        # Absolute safety fallback
        state["critic_decision"] = "retry"
        state["critic_reason"] = "Critic returned invalid JSON"

    if decision == "retry":
        state["retry_count"] = state.get("retry_count", 0) + 1

        if state["retry_count"] >= state.get("max_retries", 3):
            decision = "accept"
            reason = "Max retries reached"
    
    state["critic_decision"] = decision
    state["critic_reason"] = reason

    # state.setdefault("steps", []).append(
    #     f"CriticNode → decision={state['critic_decision']}"
    # )
    state.setdefault("steps", []).append(
    f"CriticNode → decision={decision}"
)

    return state
