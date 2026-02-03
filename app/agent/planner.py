import json
from typing import Dict, Any, List
from app.llm.ollama_client import ollama_llm

# --------------------------------------------------
# Use a planner-specific model WITHOUT breaking global usage
# --------------------------------------------------
planner_llm = ollama_llm.with_model("qwen2.5:7b-instruct")

# --------------------------------------------------
# Planner prompt (escaped for .format)
# --------------------------------------------------
PLANNER_PROMPT = """
You are a planning agent in an Agentic AI system.

User goal:
"{goal}"

Break this goal into a list of clear, ordered steps.

Rules:
- Return ONLY valid JSON
- No explanations
- No markdown
- Output MUST strictly follow this format:

{{
  "steps": [
    "step 1",
    "step 2",
    "step 3"
  ]
}}
"""


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner Node

    Input state:
    {
        "goal": str
    }

    Output state additions:
    {
        "plan": list[str],
        "current_step": int,
        "observations": list
    }
    """

    goal = state.get("goal")

    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("PlannerNode: 'goal' must be a non-empty string")

    response = planner_llm.generate(
        PLANNER_PROMPT.format(goal=goal)
    )

    # ----------------------------
    # Parse LLM output safely
    # ----------------------------
    try:
        plan_json = json.loads(response)
        steps = plan_json.get("steps", [])

        if not isinstance(steps, list):
            raise ValueError("Invalid steps format")

    except Exception:
        # Hard fallback to keep agent alive
        steps = [f"Answer the user goal directly: {goal}"]

    # ----------------------------
    # Return updated state
    # ----------------------------
    return {
        **state,
        "plan": steps,
        "current_step": 0,
        "observations": [
            {
                "node": "planner",
                "raw_output": response[:300]
            }
        ]
    }
