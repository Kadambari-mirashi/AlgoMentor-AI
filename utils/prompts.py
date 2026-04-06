"""
prompts.py
Centralized prompt templates and role descriptions for each agent.
"""

ROUTER_DESCRIPTION = (
    "The Router Agent classifies user intent into one of four categories: "
    "problem_request, hint_request, evaluation_request, or explanation_request. "
    "It uses deterministic keyword matching for fast, reliable classification."
)

RAG_DESCRIPTION = (
    "The RAG Agent searches the local DSA knowledge base for the most "
    "relevant problem. It filters by topic and difficulty, then scores "
    "candidates using keyword overlap between the user query and each "
    "problem's tags, title, and statement."
)

MENTOR_ROLE = (
    "You are AlgoMentor, a patient DSA tutor. Give progressive hints "
    "without revealing the full solution unless explicitly asked. Focus "
    "on helping the student discover the right approach on their own. "
    "Reference common algorithmic patterns when appropriate."
)

EVALUATOR_ROLE = (
    "You are a code reviewer evaluating a student's DSA solution. "
    "Assess correctness, time/space complexity, edge-case coverage, "
    "and code quality. Provide constructive, specific feedback."
)

MENTOR_TEMPLATE = (
    "## Hint {hint_number} for *{title}*\n\n"
    "{hint_text}\n\n"
    "---\n"
    "**Pattern guidance:** {pattern_text}"
)

EVALUATION_TEMPLATE = (
    "## Evaluation for *{title}*\n\n"
    "{evaluation_body}"
)
