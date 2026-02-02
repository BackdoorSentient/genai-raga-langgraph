# app/agent/planner.py

import json
from typing import List, Literal
from pydantic import BaseModel
from app.llm.ollama_client import ollama_llm
import re

class PlanStep(BaseModel):
    step: int
    tool: Literal["rag", "tool", "llm"]
    action: str

class Plan(BaseModel):
    steps: List[PlanStep]

PLANNER_PROMPT = """
You are a planning agent for an Agentic RAG system.

Available tools:
- rag : internal document retrieval system
- tool : external tools or web search
- llm : reasoning, summarization, final answer

Rules:
- Prefer using rag first if the answer may exist internally
- Use tool only if external info is needed
- Always end with llm
- Keep steps minimal and logical
- Output ONLY valid JSON, NOTHING ELSE
- Do NOT add greetings, explanations, or extra text. Only JSON.

User goal:
{goal}

Output JSON format:
{{
  "steps": [
    {{
      "step": 1,
      "tool": "rag|tool|llm",
      "action": "description of what to do"
    }}
  ]
}}
"""

def planner_node(state: dict) -> dict:
    """
    Input state:
    {
        "goal": str
    }
    """

    goal = state.get("goal")
    if not goal:
        raise ValueError("PlannerNode: 'goal' missing from state")

    response = ollama_llm.generate(
        PLANNER_PROMPT.format(goal=goal)
    )
    # if not response or "{" not in response:
    #     raise ValueError(f"LLM did not return any JSON-like content:\n{response}")
    # match = re.search(r"\{.*\}", response, re.DOTALL)
    match = re.search(r"\{(?:.|\n)*\}", response)
    if not match:
        raise ValueError(f"LLM did not return valid JSON:\n{response}")


    try:
        # plan_json = json.loads(response)
        plan_json = json.loads(match.group())
        plan = Plan.model_validate(plan_json)
    except json.JSONDecodeError:
        raise ValueError(f"LLM did not return valid JSON:\n{response}")

    return {
        **state,
        "plan": plan.steps,
        "current_step": 0,
        "observations": []
    }
