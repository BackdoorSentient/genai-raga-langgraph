# app/utils/json_utils.py
import json


def extract_json(text: str) -> dict:
    """
    Safely extract a JSON object from noisy LLM output.

    Handles:
    - Leading/trailing whitespace
    - ```json ... ``` markdown fences
    - Preamble text before the first '{'
    - Postamble text after the last '}'
    """
    if not text:
        raise ValueError("Empty LLM output — cannot extract JSON")

    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in output: {text[:200]!r}")

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed JSON: {exc} — raw: {text[start:end+1][:200]!r}"
        ) from exc