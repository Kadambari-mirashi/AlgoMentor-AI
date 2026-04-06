"""
complexity_tool.py
Tool: analyze_complexity

Uses simple heuristics to estimate the likely time and space complexity
of a user-submitted solution (code or plain-text approach description).
"""

import re

TOOL_METADATA = {
    "type": "function",
    "function": {
        "name": "analyze_complexity",
        "description": (
            "Estimate the likely time and space complexity of a submitted "
            "solution using keyword and structural heuristics."
        ),
        "parameters": {
            "type": "object",
            "required": ["solution_text"],
            "properties": {
                "solution_text": {
                    "type": "string",
                    "description": "The user's solution code or approach text.",
                },
            },
        },
    },
}

# Ordered from most to least specific so the first match wins for time.
_TIME_RULES: list[tuple[str, str]] = [
    (r"(\bfor\b.*\bfor\b.*\bfor\b)", "O(n^3)"),
    (r"(\bfor\b.*\bfor\b|\bnested\s+loop|\btwo\s+loops?\s+nested)", "O(n^2)"),
    (r"(\bsort\b|\bsorted\b|\b\.sort\(|\bmerge\s*sort|\bquick\s*sort|nlogn|n\s*log\s*n)", "O(n log n)"),
    (r"(\bbinary\s*search|\bbisect|\blog\s*n|\bmid\s*=)", "O(log n)"),
    (r"(\bfor\b|\bwhile\b|\biterat|\bsingle\s+pass|\bone\s+pass|\blinear)", "O(n)"),
    (r"(\bhash|\bdict|\bset\b|\blookup)", "O(n)"),
    (r"(\brecursi|\bdfs\b|\bbfs\b)", "O(n)"),
    (r"(\b2\^n|\bexponential|\bbacktrack)", "O(2^n)"),
    (r"(\bdp\b|\bdynamic\s+program|\btabul|\bmemoiz)", "O(n * m) or O(n^2)"),
]

_SPACE_RULES: list[tuple[str, str]] = [
    (r"(\bin[- ]?place|\bswap|\bconstant\s+space|\bO\(1\))", "O(1)"),
    (r"(\bdp\s*\[|\btable|\bmatrix|\b2d\s*array|\bgrid)", "O(n * m)"),
    (r"(\bstack|\bqueue|\brecursi|\bdfs\b)", "O(n)"),
    (r"(\bhash|\bdict|\bset\b|\bmap\b|\barray|\blist)", "O(n)"),
]


def analyze_complexity(solution_text: str) -> dict:
    """Estimate time and space complexity from solution text.

    Parameters
    ----------
    solution_text : str
        The user's code or textual approach description.

    Returns
    -------
    dict  with keys ``time_complexity``, ``space_complexity``,
    ``matched_time_signals``, and ``matched_space_signals``.
    """
    if not solution_text or not solution_text.strip():
        return {
            "time_complexity": "Unable to determine",
            "space_complexity": "Unable to determine",
            "matched_time_signals": [],
            "matched_space_signals": [],
        }

    text = solution_text.lower()

    time_cplx = "O(n)"
    time_signals: list[str] = []
    for pattern, cplx in _TIME_RULES:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            time_cplx = cplx
            time_signals.append(match.group(0).strip())
            break

    space_cplx = "O(n)"
    space_signals: list[str] = []
    for pattern, cplx in _SPACE_RULES:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            space_cplx = cplx
            space_signals.append(match.group(0).strip())
            break

    return {
        "time_complexity": time_cplx,
        "space_complexity": space_cplx,
        "matched_time_signals": time_signals,
        "matched_space_signals": space_signals,
    }
