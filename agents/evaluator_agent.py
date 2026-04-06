"""
evaluator_agent.py
Agent 4 — Evaluator Agent

Evaluates a user-submitted solution against the current problem.
Calls the complexity_tool, comparison_tool, and testcase_tool to
produce structured feedback.

When an Ollama Cloud API key is provided, the agent uses the LLM to
synthesize the tool outputs into a natural, constructive review.
Otherwise it falls back to a structured template.
"""

from tools.complexity_tool import analyze_complexity
from tools.comparison_tool import compare_solution_keywords
from tools.testcase_tool import generate_test_cases
from utils.llm import query_llm
from utils.prompts import EVALUATOR_ROLE


def evaluate_solution(
    problem: dict,
    user_solution: str,
    api_key: str | None = None,
) -> dict:
    """Evaluate a user's solution against the reference problem.

    Parameters
    ----------
    problem : dict
        The current problem from the knowledge base.
    user_solution : str
        The code or approach text submitted by the student.
    api_key : str or None
        Ollama Cloud API key.  If provided, the LLM enriches the response.

    Returns
    -------
    dict  with keys ``correctness``, ``time_complexity``,
    ``space_complexity``, ``missing_concepts``, ``suggestions``,
    ``tools_called``, and ``formatted_response``.
    """
    title = problem.get("title", "Unknown Problem")
    reference = problem.get("approach_summary", "")
    expected_tc = problem.get("expected_time_complexity", "N/A")
    expected_sc = problem.get("expected_space_complexity", "N/A")

    # --- Tool call 1: complexity_tool ---
    complexity = analyze_complexity(user_solution)

    # --- Tool call 2: comparison_tool ---
    comparison = compare_solution_keywords(user_solution, reference)

    # --- Tool call 3: testcase_tool ---
    test_cases = generate_test_cases(problem)

    # Derive a correctness label from the comparison ratio
    ratio = comparison.get("match_ratio", 0.0)
    if ratio >= 0.55:
        correctness = "Likely Correct — strong alignment with expected approach"
    elif ratio >= 0.3:
        correctness = "Partially Correct — some key concepts present"
    else:
        correctness = "Needs Improvement — approach differs significantly from expected"

    # Complexity comparison
    est_tc = complexity.get("time_complexity", "N/A")
    est_sc = complexity.get("space_complexity", "N/A")

    tc_match = "Matches expected" if _complexity_matches(est_tc, expected_tc) else f"Expected {expected_tc}"
    sc_match = "Matches expected" if _complexity_matches(est_sc, expected_sc) else f"Expected {expected_sc}"

    # Aggregate suggestions
    suggestions = list(comparison.get("suggestions", []))
    if not _complexity_matches(est_tc, expected_tc):
        suggestions.append(
            f"Your estimated time complexity is {est_tc}, but the optimal is {expected_tc}. "
            "Consider whether a more efficient algorithm or data structure could help."
        )

    matched_kw = comparison.get("matched_keywords", [])
    missing_kw = comparison.get("missing_keywords", [])
    cases_text = "\n".join(f"- {c}" for c in test_cases.get("test_cases", []))

    # --- LLM call (if API key available) ---
    llm_response = None
    if api_key:
        solution_snippet = user_solution[:400]
        user_prompt = (
            f"Problem: \"{title}\" | Expected: {expected_tc} time, {expected_sc} space\n"
            f"Student submission:\n{solution_snippet}\n\n"
            f"Tool results: {correctness}. "
            f"Time: {est_tc} ({tc_match}). Space: {est_sc} ({sc_match}). "
            f"Match: {int(ratio * 100)}%. "
            f"Missing: {', '.join(missing_kw[:5]) if missing_kw else 'none'}.\n\n"
            f"Give a constructive 4-6 sentence review covering correctness, "
            f"complexity, and improvements."
        )
        messages = [
            {"role": "system", "content": EVALUATOR_ROLE},
            {"role": "user", "content": user_prompt},
        ]
        llm_response = query_llm(messages, api_key=api_key)

    # Build formatted response
    if llm_response:
        formatted = (
            f"## Evaluation for *{title}*\n\n"
            f"{llm_response}\n\n"
            f"---\n\n"
            f"**Tool Analysis Summary:**\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Correctness | {correctness} |\n"
            f"| Time Complexity | {est_tc} — {tc_match} |\n"
            f"| Space Complexity | {est_sc} — {sc_match} |\n"
            f"| Keyword Alignment | {comparison.get('alignment', 'N/A')} ({int(ratio * 100)}%) |\n\n"
            f"**Sample Test Cases to Verify:**\n\n{cases_text}"
        )
    else:
        # Fallback: template-based response
        formatted = (
            f"## Evaluation for *{title}*\n\n"
            f"**Correctness:** {correctness}\n\n"
            f"**Time Complexity:** {est_tc} — {tc_match}\n\n"
            f"**Space Complexity:** {est_sc} — {sc_match}\n\n"
            f"**Keyword Alignment:** {comparison.get('alignment', 'N/A')} "
            f"({int(ratio * 100)}% match)\n\n"
        )

        if matched_kw:
            formatted += f"**Key Concepts Found:** {', '.join(matched_kw)}\n\n"
        if missing_kw:
            formatted += f"**Missing Concepts:** {', '.join(missing_kw[:8])}\n\n"
        if suggestions:
            formatted += "**Suggestions:**\n\n"
            for s in suggestions:
                formatted += f"- {s}\n"
            formatted += "\n"

        formatted += f"---\n\n**Sample Test Cases to Verify:**\n\n{cases_text}"

    tools_called = [
        "analyze_complexity",
        "compare_solution_keywords",
        "generate_test_cases",
    ]
    if llm_response:
        tools_called.append("query_llm (Ollama Cloud — gemma3:12b)")

    return {
        "correctness": correctness,
        "time_complexity": est_tc,
        "space_complexity": est_sc,
        "expected_time_complexity": expected_tc,
        "expected_space_complexity": expected_sc,
        "missing_concepts": missing_kw[:8],
        "suggestions": suggestions,
        "match_ratio": ratio,
        "tools_called": tools_called,
        "formatted_response": formatted,
    }


def _complexity_matches(estimated: str, expected: str) -> bool:
    """Loose check whether two Big-O strings refer to the same class."""
    def _normalize(s: str) -> str:
        return s.lower().replace(" ", "").replace("o(", "").replace(")", "")

    return _normalize(estimated) == _normalize(expected)
