"""
rag_agent.py
Agent 2 — RAG Agent

Retrieves the most relevant DSA problem from the local JSON knowledge
base.  Supports filtering by topic and difficulty, then ranks candidates
using keyword overlap between the user query and each problem's tags,
title, and problem statement.
"""

import re
from utils.helpers import load_knowledge_base


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenization, keeping only alphanumeric tokens >= 2 chars."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2}


def _score_problem(problem: dict, query_tokens: set[str]) -> float:
    """Score a problem by how many query tokens appear in its fields."""
    searchable = " ".join(
        [
            problem.get("title", ""),
            problem.get("topic", ""),
            " ".join(problem.get("tags", [])),
            problem.get("problem_statement", ""),
        ]
    )
    problem_tokens = _tokenize(searchable)
    if not query_tokens:
        return 0.0
    return len(query_tokens & problem_tokens) / len(query_tokens)


def retrieve_problem(
    query: str,
    topic: str | None = None,
    difficulty: str | None = None,
) -> dict:
    """Search the knowledge base and return the best-matching problem.

    Parameters
    ----------
    query : str
        Free-text search query from the user.
    topic : str or None
        If provided, only problems with this topic are considered.
    difficulty : str or None
        If provided, only problems with this difficulty are considered.

    Returns
    -------
    dict  with keys ``problem`` (the matched entry or None),
    ``score``, ``candidates_considered``, and ``filters_applied``.
    """
    kb = load_knowledge_base()

    if not kb:
        return {
            "problem": None,
            "score": 0.0,
            "candidates_considered": 0,
            "filters_applied": {},
        }

    filters: dict[str, str] = {}

    # Filter by topic
    if topic and topic != "All":
        kb = [p for p in kb if p.get("topic", "").lower() == topic.lower()]
        filters["topic"] = topic

    # Filter by difficulty
    if difficulty and difficulty != "All":
        kb = [p for p in kb if p.get("difficulty", "").lower() == difficulty.lower()]
        filters["difficulty"] = difficulty

    if not kb:
        return {
            "problem": None,
            "score": 0.0,
            "candidates_considered": 0,
            "filters_applied": filters,
        }

    query_tokens = _tokenize(query)

    # If query is empty / no meaningful tokens, return the first candidate
    if not query_tokens:
        return {
            "problem": kb[0],
            "score": 1.0,
            "candidates_considered": len(kb),
            "filters_applied": filters,
        }

    scored = [(p, _score_problem(p, query_tokens)) for p in kb]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_problem, best_score = scored[0]

    return {
        "problem": best_problem,
        "score": round(best_score, 3),
        "candidates_considered": len(kb),
        "filters_applied": filters,
    }
