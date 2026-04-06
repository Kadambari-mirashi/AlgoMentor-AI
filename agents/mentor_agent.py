"""
mentor_agent.py
Agent 3 — Mentor Agent

Provides progressive hints and pattern-level guidance for a retrieved
DSA problem.  Calls the pattern_tool to enrich responses with
well-known algorithmic strategies.

When an Ollama Cloud API key is provided, the agent uses the LLM to
generate a natural, conversational mentor response grounded in the
hint and pattern data.  Otherwise it falls back to a structured
template.
"""

from tools.pattern_tool import get_pattern_hint
from utils.llm import query_llm
from utils.prompts import MENTOR_ROLE


def get_mentor_response(
    problem: dict,
    hint_index: int = 0,
    api_key: str | None = None,
) -> dict:
    """Generate a mentor response with progressive hints.

    Parameters
    ----------
    problem : dict
        The current problem from the knowledge base.
    hint_index : int
        Which hint to reveal (0-based).  Earlier calls show fewer hints.
    api_key : str or None
        Ollama Cloud API key.  If provided, the LLM enriches the response.

    Returns
    -------
    dict  with keys ``hint_number``, ``hint_text``, ``pattern_guidance``,
    ``tools_called``, and ``formatted_response``.
    """
    hints = problem.get("hints", [])
    title = problem.get("title", "Unknown Problem")
    tags = problem.get("tags", [])
    topic = problem.get("topic", "")

    # Clamp hint_index to available range
    if not hints:
        hint_text = (
            "No pre-written hints are available for this problem. "
            "Try breaking it into smaller sub-problems and consider "
            "what data structure would give you efficient lookups."
        )
        current_index = 0
    else:
        current_index = min(hint_index, len(hints) - 1)
        hint_text = hints[current_index]

    # --- Tool call: pattern_tool ---
    pattern_result = get_pattern_hint(tags if tags else [topic])
    patterns = pattern_result.get("patterns", [])
    pattern_text = "\n".join(f"- {p}" for p in patterns)

    # --- LLM call (if API key available) ---
    llm_response = None
    if api_key:
        user_prompt = (
            f"Problem: \"{title}\" ({topic}, {problem.get('difficulty', 'N/A')})\n"
            f"Hint {current_index + 1}/{len(hints)}: \"{hint_text}\"\n"
            f"Patterns: {', '.join(patterns[:4])}\n\n"
            f"Expand on this hint in 3-5 sentences. Be encouraging. "
            f"Reference the patterns. Do not reveal the full solution."
        )
        messages = [
            {"role": "system", "content": MENTOR_ROLE},
            {"role": "user", "content": user_prompt},
        ]
        llm_response = query_llm(messages, api_key=api_key)

    # Build formatted response
    if llm_response:
        progress = ""
        if hints:
            progress = f"*(Hint {current_index + 1} of {len(hints)})*\n\n"

        formatted = (
            f"## Mentor Hint for *{title}*\n\n"
            f"{progress}"
            f"{llm_response}\n\n"
            f"---\n\n"
            f"**Relevant Patterns:**\n\n{pattern_text}"
        )
    else:
        # Fallback: template-based response
        progress = ""
        if hints:
            progress = f"*(Hint {current_index + 1} of {len(hints)})*\n\n"

        formatted = (
            f"## Hint for *{title}*\n\n"
            f"{progress}"
            f"{hint_text}\n\n"
            f"---\n\n"
            f"**Relevant Patterns:**\n\n{pattern_text}"
        )

    if current_index < len(hints) - 1:
        formatted += "\n\n> Need more help? Click **Get Hint** again for the next hint."
    else:
        formatted += "\n\n> You've seen all available hints for this problem."

    tools_called = ["get_pattern_hint"]
    if llm_response:
        tools_called.append("query_llm (Ollama Cloud — gemma3:12b)")

    return {
        "hint_number": current_index + 1,
        "total_hints": len(hints),
        "hint_text": hint_text,
        "pattern_guidance": patterns,
        "tools_called": tools_called,
        "formatted_response": formatted,
    }
