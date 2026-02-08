import json
from app.llm.ollama_client import ollama_llm
from app.agent.state import AgentState


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


def _extract_json(text: str) -> dict:
    """
    Safely extract JSON from noisy LLM output.
    NEVER trusts the model.
    """
    if not text:
        raise ValueError("Empty planner output")

    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def planner_node(state: AgentState) -> AgentState:
    goal = state.get("goal")
    if not goal:
        raise ValueError("Planner → goal missing")

    response = ollama_llm.generate(
        PLANNER_PROMPT.format(goal=goal)
    )

    try:
        parsed = _extract_json(response)
        steps = parsed.get("steps")

        if not isinstance(steps, list) or not all(
            isinstance(s, str) and s.strip() for s in steps
        ):
            raise ValueError("Invalid steps format")

    except Exception as e:
        # 🔒 Absolute safety fallback
        steps = [
            "Identify the query intent",
            "Retrieve relevant information",
            "Summarize grounded answer"
        ]

        state.setdefault("observations", []).append({
            "node": "planner",
            "error": str(e),
            "raw_output": response[:200]
        })

    # -------- Initialize state deterministically --------
    state["plan"] = steps
    state["current_step"] = 0
    state["phase"] = "retrieve"
    state["used_vector"] = False
    state["used_web"] = False

    state.setdefault("steps", []).append(
        f"Planner → {len(steps)} steps created"
    )

    return state
