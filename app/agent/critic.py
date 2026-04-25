# app/agent/critic.py
from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm
from app.schema import Document
from app.utils.json_utils import extract_json      


CRITIC_PROMPT = """
You are a strict relevance evaluator.

RULES:
- Output MUST be valid JSON
- NO markdown, NO explanations
- Start with '{{' and end with '}}'

Evaluate whether the answer DIRECTLY and SPECIFICALLY answers the question asked.
Reasons to choose RETRY:
- Answer is vague or generic
- Answer is about a different topic than the question
- Answer pulls from unrelated documents
- Answer contains social media stats for a person query without authoritative source
- Answer says it does not have information

Reasons to choose ACCEPT:
- Answer directly addresses the question
- Answer is grounded in relevant retrieved content
- Answer is specific and factual

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
        response = ollama_llm.generate(
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