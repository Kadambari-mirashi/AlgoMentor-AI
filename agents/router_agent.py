"""
router_agent.py
Agent 1 — Router Agent

Classifies user intent into one of four categories using deterministic
keyword matching.  Designed to be fast and predictable (no LLM needed).

Categories
----------
- problem_request    → user wants a new DSA problem
- hint_request       → user wants a hint for the current problem
- evaluation_request → user wants their solution evaluated
- explanation_request→ user wants a concept or approach explained
"""

import re

_INTENT_RULES: list[tuple[str, list[str]]] = [
    (
        "problem_request",
        [
            "give me a problem",
            "new problem",
            "get problem",
            "show me a problem",
            "practice problem",
            "random problem",
            "find a problem",
            "suggest a problem",
            "pick a problem",
            "start a problem",
            "i want to practice",
            "challenge me",
        ],
    ),
    (
        "hint_request",
        [
            "hint",
            "give me a hint",
            "clue",
            "help me",
            "stuck",
            "not sure",
            "guide me",
            "tip",
            "pointer",
            "nudge",
            "i need help",
            "what should i do",
        ],
    ),
    (
        "evaluation_request",
        [
            "evaluate",
            "review",
            "check my",
            "grade",
            "assess",
            "how is my",
            "is this correct",
            "feedback",
            "score",
            "analyze my",
            "look at my",
            "rate my",
            "submit",
        ],
    ),
    (
        "explanation_request",
        [
            "explain",
            "what is",
            "how does",
            "why does",
            "describe",
            "tell me about",
            "what are",
            "how do",
            "teach me",
            "walk me through",
            "concept",
            "definition",
        ],
    ),
]


def classify_intent(user_text: str, action_override: str | None = None) -> dict:
    """Classify the user's intent.

    Parameters
    ----------
    user_text : str
        Free-form text entered by the user.
    action_override : str or None
        If set (e.g. from a UI selectbox), maps directly to an intent
        and bypasses keyword matching.

    Returns
    -------
    dict  with keys ``intent``, ``confidence``, and ``method``.
    """
    # Direct mapping from the sidebar action selector
    _ACTION_MAP = {
        "Get Problem": "problem_request",
        "Get Hint": "hint_request",
        "Evaluate Solution": "evaluation_request",
        "Explain Concept": "explanation_request",
    }

    if action_override and action_override in _ACTION_MAP:
        return {
            "intent": _ACTION_MAP[action_override],
            "confidence": "high",
            "method": "action_selector",
        }

    # Fall back to keyword matching on user text
    text_lower = user_text.lower().strip()

    for intent, keywords in _INTENT_RULES:
        for kw in keywords:
            if kw in text_lower:
                return {
                    "intent": intent,
                    "confidence": "high",
                    "method": "keyword_match",
                }

    # Default: treat as a problem request
    return {
        "intent": "problem_request",
        "confidence": "low",
        "method": "default_fallback",
    }
