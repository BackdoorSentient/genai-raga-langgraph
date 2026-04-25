# app/agent/critic.py
import json
from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm
from app.schema import Document


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
    # Build origins list from Document objects
    origins = list({
        d.origin
        for d in state.get("documents", [])
        if isinstance(d, Document)
    })

    response = ollama_llm.generate(
        CRITIC_PROMPT.format(
            query=state.get("query", ""),
            origins=", ".join(origins) if origins else "none",
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

    state["grounded"] = (
        decision == "accept"
        and bool(state.get("documents"))
    )

    state.setdefault("steps", []).append(
        f"CriticNode → decision={decision}, grounded={state['grounded']}"
    )

    return state