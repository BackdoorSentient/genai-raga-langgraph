# app/agent/critic.py
from app.agent.state import AgentState
from app.llm.factory import get_llm          # ← Issue E: was: from app.llm.ollama_client import ollama_llm
from app.schema import Document
from app.utils.json_utils import extract_json


CRITIC_PROMPT = """
You are a relevance evaluator.

RULES:
- Output MUST be valid JSON
- NO markdown, NO explanations
- Start with '{{' and end with '}}'

Evaluate whether the answer addresses the question asked.

Reasons to choose RETRY:
- Answer is completely off-topic
- Answer is empty or says only "I don't know"
- Answer is about a completely different subject

Reasons to choose ACCEPT:
- Answer directly addresses the question using retrieved content
- Answer is grounded in web or vector sources
- Answer provides relevant information about the topic, even if partial
- For person queries — answer uses LinkedIn, news, company pages, or social profiles

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
        llm = get_llm()                      # ← Issue E: factory call, respects LLM_PROVIDER in .env
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