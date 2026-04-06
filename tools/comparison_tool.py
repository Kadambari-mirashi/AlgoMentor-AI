"""
comparison_tool.py
Tool: compare_solution_keywords

Compares a user's solution text/code against a reference approach summary
using keyword overlap.  Returns a qualitative alignment report.
"""

import re

TOOL_METADATA = {
    "type": "function",
    "function": {
        "name": "compare_solution_keywords",
        "description": (
            "Compare a user's solution against a reference approach using "
            "keyword overlap and return a qualitative alignment summary."
        ),
        "parameters": {
            "type": "object",
            "required": ["user_solution", "reference_approach"],
            "properties": {
                "user_solution": {
                    "type": "string",
                    "description": "The user's submitted code or approach text.",
                },
                "reference_approach": {
                    "type": "string",
                    "description": "The reference approach summary from the knowledge base.",
                },
            },
        },
    },
}

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "and or but not no nor so yet both either neither each every all "
    "some any few more most other such than too very also just only "
    "if then else while until unless because since when where how what "
    "which who whom whose that this these those it its i we you they "
    "he she me him her us them my our your their his return def class "
    "self none true false import print range len append".split()
)

# Maps common code identifiers / patterns to the conceptual terms used in
# natural-language approach descriptions.  This bridges the gap between
# Python source code and the reference summaries in the knowledge base.
_CODE_TO_CONCEPT: dict[str, list[str]] = {
    "dict":        ["hash", "map", "hash_map"],
    "set":         ["hash", "set", "lookup"],
    "enumerate":   ["iterate", "index"],
    "sorted":      ["sort"],
    "sort":        ["sort"],
    "deque":       ["queue", "bfs"],
    "heapq":       ["heap", "priority"],
    "bisect":      ["binary_search"],
    "defaultdict": ["hash", "map", "hash_map"],
    "counter":     ["frequency", "count"],
    "stack":       ["stack"],
    "queue":       ["queue"],
    "visited":     ["visited", "mark"],
    "seen":        ["hash", "map", "store", "lookup"],
    "memo":        ["memoization", "memoize", "dp"],
    "cache":       ["memoization", "cache"],
    "complement":  ["complement"],
    "left":        ["pointer", "two_pointer", "left"],
    "right":       ["pointer", "two_pointer", "right"],
    "mid":         ["binary_search", "mid", "middle"],
    "dp":          ["dynamic", "programming", "dp"],
    "dfs":         ["dfs", "depth"],
    "bfs":         ["bfs", "breadth"],
    "node":        ["node", "linked", "tree"],
    "prev":        ["pointer", "previous"],
    "curr":        ["pointer", "current"],
    "head":        ["head", "linked"],
    "dummy":       ["dummy", "head"],
    "swap":        ["swap", "place"],
    "recursive":   ["recursion", "recursive"],
    "recurse":     ["recursion", "recursive"],
}


def _extract_keywords(text: str) -> set[str]:
    """Tokenize text, remove stop words, and expand code-to-concept aliases."""
    tokens = re.findall(r"[a-z_][a-z0-9_]*", text.lower())
    base = {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}

    expanded = set(base)
    for token in base:
        for code_token, concepts in _CODE_TO_CONCEPT.items():
            if code_token in token:
                expanded.update(concepts)

    return expanded


def compare_solution_keywords(
    user_solution: str,
    reference_approach: str,
) -> dict:
    """Compare user solution keywords against the reference approach.

    Parameters
    ----------
    user_solution : str
        Code or textual description submitted by the student.
    reference_approach : str
        The canonical approach summary from the knowledge base.

    Returns
    -------
    dict  with keys ``matched_keywords``, ``missing_keywords``,
    ``match_ratio``, ``alignment``, and ``suggestions``.
    """
    if not user_solution or not user_solution.strip():
        return {
            "matched_keywords": [],
            "missing_keywords": [],
            "match_ratio": 0.0,
            "alignment": "No solution provided",
            "suggestions": ["Submit your code or approach description to receive feedback."],
        }

    user_kw = _extract_keywords(user_solution)
    ref_kw = _extract_keywords(reference_approach)

    if not ref_kw:
        return {
            "matched_keywords": [],
            "missing_keywords": [],
            "match_ratio": 0.0,
            "alignment": "Reference approach unavailable",
            "suggestions": [],
        }

    matched = sorted(user_kw & ref_kw)
    missing = sorted(ref_kw - user_kw)
    ratio = len(matched) / len(ref_kw) if ref_kw else 0.0

    if ratio >= 0.55:
        alignment = "Strong alignment"
    elif ratio >= 0.3:
        alignment = "Partial alignment"
    else:
        alignment = "Low alignment"

    suggestions: list[str] = []
    if missing:
        top_missing = missing[:5]
        suggestions.append(
            f"Consider incorporating these concepts: {', '.join(top_missing)}."
        )
    if ratio < 0.3:
        suggestions.append(
            "Your approach may differ significantly from the expected solution. "
            "Review the problem constraints and consider alternative strategies."
        )

    return {
        "matched_keywords": matched,
        "missing_keywords": missing,
        "match_ratio": round(ratio, 2),
        "alignment": alignment,
        "suggestions": suggestions,
    }
