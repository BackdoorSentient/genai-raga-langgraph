# app/agent/planner.py
from app.llm.factory import get_llm          # ← Issue E: was: from app.llm.ollama_client import ollama_llm
from app.agent.state import AgentState
from app.utils.json_utils import extract_json


PLANNER_PROMPT = """
You are a planning agent.

User goal:
{goal}

Break the user goal into clear ordered steps.

STRICT RULES:
- Output MUST be valid JSON
- NO markdown
- NO explanations
- Start with '{{' and end with '}}'

Return ONLY this structure:

{{
  "steps": [
    "step one",
    "step two",
    "step three"
  ]
}}
"""

FALLBACK_STEPS = [
    "Identify the query intent",
    "Retrieve relevant information",
    "Summarize grounded answer"
]


def planner_node(state: AgentState) -> AgentState:
    goal = state.get("goal")
    if not goal:
        raise ValueError("Planner → goal missing")

    try:
        llm = get_llm()                      # ← Issue E: factory call, respects LLM_PROVIDER in .env
        response = llm.generate(
            PLANNER_PROMPT.format(goal=goal)
        )
    except Exception as exc:
        state["plan"] = FALLBACK_STEPS
        state["current_step"] = 0
        state["phase"] = "retrieve"
        state["used_vector"] = False
        state["used_web"] = False
        state.setdefault("observations", []).append({
            "node": "planner",
            "error": f"LLM unavailable: {exc}",
            "fallback": True
        })
        state.setdefault("steps", []).append(
            f"Planner → LLM failed, using {len(FALLBACK_STEPS)} fallback steps"
        )
        return state

    try:
        parsed = extract_json(response)
        steps = parsed.get("steps")

        if not isinstance(steps, list) or not all(
            isinstance(s, str) and s.strip() for s in steps
        ):
            raise ValueError("Invalid steps format")

    except Exception as e:
        steps = FALLBACK_STEPS
        state.setdefault("observations", []).append({
            "node": "planner",
            "error": str(e),
            "raw_output": response[:200]
        })

    state["plan"] = steps
    state["current_step"] = 0
    state["phase"] = "retrieve"
    state["used_vector"] = False
    state["used_web"] = False

    state.setdefault("steps", []).append(
        f"Planner → {len(steps)} steps created"
    )

    return state