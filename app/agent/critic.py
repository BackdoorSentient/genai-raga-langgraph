import json
from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm


# CRITIC_PROMPT = """
# Evaluate the answer.

# Return JSON only:
# {{
#   "decision": "accept" | "retry",
#   "reason": "short reason"
# }}
# """


# def critic_node(state: AgentState) -> AgentState:
#     response = ollama_llm.generate(
#         CRITIC_PROMPT.format(
#             answer=state.get("answer", ""),
#             grounded=state.get("grounded", False),
#             confidence=state.get("confidence", 0.0)
#         )
#     )

#     try:
#         parsed = json.loads(response)
#         decision = parsed.get("decision", "retry")
#         reason = parsed.get("reason", "")
#         if decision not in ("accept", "retry"):
#             decision = "retry"
#     except Exception:
#         decision = "retry"
#         reason = "Invalid critic output"

#     state["critic_decision"] = decision
#     state["critic_reason"] = reason

#     # ✅ Grounding flag
#     state["grounded"] = (
#         decision == "accept"
#         and bool(state.get("documents"))
#     )

#     state.setdefault("steps", []).append(
#         f"CriticNode → decision={decision}, grounded={state['grounded']}"
#     )

#     return state
CRITIC_PROMPT = """
You are a strict JSON evaluator.

RULES:
- Output MUST be valid JSON
- NO markdown
- NO explanations
- Start with '{{' and end with '}}'

Return ONLY this structure:

{{
  "decision": "accept" or "retry",
  "reason": "short reason"
}}

Answer:
{answer}
"""

def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty critic output")

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end + 1])


def critic_node(state: AgentState) -> AgentState:
    response = ollama_llm.generate(
        CRITIC_PROMPT.format(
            answer=state.get("answer", "")
        )
    )

    try:
        parsed = _extract_json(response)
        decision = parsed.get("decision", "retry")
        reason = parsed.get("reason", "")

        if decision not in ("accept", "retry"):
            raise ValueError("Invalid decision value")

    except Exception as e:
        decision = "retry"
        reason = f"Critic parse failure: {str(e)}"

    state["critic_decision"] = decision
    state["critic_reason"] = reason

    # ✅ Grounded only if accepted AND documents exist
    state["grounded"] = (
        decision == "accept"
        and bool(state.get("documents"))
    )

    state.setdefault("steps", []).append(
        f"CriticNode → decision={decision}, grounded={state['grounded']}"
    )

    return state
