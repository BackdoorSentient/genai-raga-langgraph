import json
from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm


CRITIC_PROMPT = """
Evaluate the answer.

Return JSON only:
{{
  "decision": "accept" | "retry",
  "reason": "short reason"
}}
"""


def critic_node(state: AgentState) -> AgentState:
    response = ollama_llm.generate(
        CRITIC_PROMPT.format(
            answer=state.get("answer", ""),
            grounded=state.get("grounded", False),
            confidence=state.get("confidence", 0.0)
        )
    )

    try:
        parsed = json.loads(response)
        decision = parsed.get("decision", "retry")
        reason = parsed.get("reason", "")
        if decision not in ("accept", "retry"):
            decision = "retry"
    except Exception:
        decision = "retry"
        reason = "Invalid critic output"

    state["critic_decision"] = decision
    state["critic_reason"] = reason

    state.setdefault("steps", []).append(
        f"CriticNode → decision={decision}"
    )

    return state
