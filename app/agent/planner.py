import json
from typing import Dict, Any
from app.llm.ollama_client import ollama_llm

planner_llm = ollama_llm.with_model("qwen2.5:7b-instruct")


PLANNER_PROMPT = """
You are a planning agent.

Your task:
Break the user goal into clear, ordered steps.

User goal:
{goal}

STRICT OUTPUT RULES:
- Output MUST be valid JSON
- Output MUST start with '{{' and end with '}}'
- NO markdown
- NO explanations
- NO extra text

Return JSON in this EXACT structure:

{{
  "steps": [
    "first step",
    "second step",
    "third step"
  ]
}}
"""


def extract_json(text: str) -> str:
    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON found in LLM output")

    return text[start : end + 1]


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    goal = state.get("goal")

    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("PlannerNode: goal must be a non-empty string")

    response = planner_llm.generate(
        PLANNER_PROMPT.format(goal=goal)
    )

    try:
        cleaned = extract_json(response)
        plan_json = json.loads(cleaned)
        steps = plan_json.get("steps", [])

        if not isinstance(steps, list) or not steps:
            raise ValueError("Invalid steps format")

    except Exception as e:
        steps = [f"Answer the user goal directly: {goal}"]
        state.setdefault("observations", []).append({
            "node": "planner",
            "error": str(e),
            "raw_output": response[:300]
        })

    return {
        **state,
        "plan": steps,
        "current_step": 0
    }
