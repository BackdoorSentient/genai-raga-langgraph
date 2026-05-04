# app/agent/critic.py
from app.agent.state import AgentState
from app.llm.factory import get_llm
from app.schema import Document
from app.utils.json_utils import extract_json


CRITIC_PROMPT = """
You are a quality evaluator for research answers.

RULES:
- Output MUST be valid JSON
- NO markdown, NO explanations
- Start with '{{' and end with '}}'

Evaluate whether the answer provides useful information about the question asked.

Reasons to choose RETRY:
- Answer is completely empty or just says "no information found"
- Answer is about a completely unrelated subject
- Answer contains zero relevant facts

Reasons to choose ACCEPT:
- Answer contains ANY relevant information from search results
- Answer compiles information from multiple sources
- Answer clearly states what was and was not found
- Even partial information is a valid answer
- Answer addresses the question even if incompletely

Return ONLY:
{{
  "decision": "accept" or "retry",
  "reason": "short reason"
}}

Question: {query}
Sources used: {origins}
Answer: {answer}
"""


def critic_node(state: AgentState) -> AgentState:
    origins = list({
        d.origin
        for d in state.get("documents", [])
        if isinstance(d, Document)
    })

    try:
        llm = get_llm()
        response = llm.generate(
            CRITIC_PROMPT.format(
                query=state.get("query", ""),
                origins=", ".join(origins) if origins else "none",
                answer=state.get("answer", "")
            )
        )
    except Exception as exc:
        state["critic_decision"] = "retry"
        state["critic_reason"] = f"LLM unavailable: {exc}"
        state["grounded"] = False
        state.setdefault("steps", []).append(
            "CriticNode → LLM failed, defaulting to retry"
        )
        return state

    try:
        parsed = extract_json(response)
        decision = parsed.get("decision", "retry")
        reason = parsed.get("reason", "")

        if decision not in ("accept", "retry"):
            raise ValueError("Invalid decision value")

    except Exception as e:
        decision = "retry"
        reason = f"Critic parse failure: {str(e)}"

    state["critic_decision"] = decision
    state["critic_reason"] = reason
    state["grounded"] = (
        decision == "accept"
        and bool(state.get("documents"))
    )

    state.setdefault("steps", []).append(
        f"CriticNode → decision={decision}, grounded={state['grounded']}"
    )

    return state