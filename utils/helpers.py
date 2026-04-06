"""
helpers.py
Shared utility functions for AlgoMentor AI.
Handles knowledge-base loading, text formatting, and session-state helpers.
"""

import json
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KB_PATH = os.path.join(_BASE_DIR, "data", "dsa_knowledge_base.json")


def load_knowledge_base(path: str = _KB_PATH) -> list[dict]:
    """Load the DSA knowledge base from a JSON file.

    Returns a list of problem dictionaries. Returns an empty list
    if the file is missing or malformed.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.loads(fh.read())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[helpers] Failed to load knowledge base: {exc}")
        return []


def format_problem_card(problem: dict) -> str:
    """Render a problem dict as a human-readable markdown card."""
    return (
        f"### {problem['title']}\n"
        f"**Difficulty:** {problem['difficulty']}  \n"
        f"**Topic:** {problem['topic']}  \n"
        f"**Tags:** {', '.join(problem.get('tags', []))}\n\n"
        f"{problem['problem_statement']}\n\n"
        f"**Expected Time Complexity:** {problem['expected_time_complexity']}  \n"
        f"**Expected Space Complexity:** {problem['expected_space_complexity']}"
    )


def format_evaluation(evaluation: dict) -> str:
    """Render an evaluation result dict as readable markdown."""
    lines = [
        "### Evaluation Results\n",
        f"**Likely Correctness:** {evaluation.get('correctness', 'N/A')}",
        f"**Estimated Time Complexity:** {evaluation.get('time_complexity', 'N/A')}",
        f"**Estimated Space Complexity:** {evaluation.get('space_complexity', 'N/A')}",
    ]

    missing = evaluation.get("missing_concepts", [])
    if missing:
        lines.append(f"**Missing Concepts:** {', '.join(missing)}")

    suggestions = evaluation.get("suggestions", [])
    if suggestions:
        lines.append("\n**Suggestions:**")
        for s in suggestions:
            lines.append(f"- {s}")

    return "\n".join(lines)


def get_available_topics(knowledge_base: list[dict]) -> list[str]:
    """Return the sorted unique topics present in the knowledge base."""
    return sorted({p["topic"] for p in knowledge_base})


def get_available_difficulties(knowledge_base: list[dict]) -> list[str]:
    """Return unique difficulties present in the knowledge base, ordered logically."""
    order = {"Easy": 0, "Medium": 1, "Hard": 2}
    diffs = {p["difficulty"] for p in knowledge_base}
    return sorted(diffs, key=lambda d: order.get(d, 99))
