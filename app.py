"""
app.py
AlgoMentor AI — Streamlit Application

An interactive DSA interview-prep app demonstrating:
  1. Multi-agent orchestration  (Router → RAG → Mentor / Evaluator)
  2. RAG integration            (local JSON knowledge-base retrieval)
  3. Function calling / tools   (pattern, complexity, comparison, testcase tools)

Run locally:
    streamlit run app.py
"""

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from agents.router_agent import classify_intent
from agents.rag_agent import retrieve_problem
from agents.mentor_agent import get_mentor_response
from agents.evaluator_agent import evaluate_solution
from utils.helpers import (
    load_knowledge_base,
    format_problem_card,
    get_available_topics,
    get_available_difficulties,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AlgoMentor AI",
    page_icon="🧠",
    layout="wide",
)

# ── Load knowledge base once ─────────────────────────────────────────────────

KB = load_knowledge_base()
TOPICS = ["All"] + get_available_topics(KB)
DIFFICULTIES = ["All"] + get_available_difficulties(KB)

# ── Session-state defaults ───────────────────────────────────────────────────


def _init_state():
    defaults = {
        "current_problem": None,
        "hint_index": 0,
        "agent_trace": [],
        "mentor_response": None,
        "evaluation_response": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ── Sidebar controls ────────────────────────────────────────────────────────

st.sidebar.title("🧠 AlgoMentor AI")
st.sidebar.markdown("*Your DSA interview prep companion*")
st.sidebar.divider()

topic = st.sidebar.selectbox("Topic", TOPICS, index=0)
difficulty = st.sidebar.selectbox("Difficulty", DIFFICULTIES, index=0)
action = st.sidebar.selectbox(
    "Action",
    ["Get Problem", "Get Hint", "Evaluate Solution"],
)

st.sidebar.divider()
api_key = os.environ.get("OLLAMA_API_KEY", "")
if api_key:
    st.sidebar.success("LLM enabled (Gemma 3 12B via Ollama Cloud)")
else:
    st.sidebar.info("No API key — using template-based responses.")

st.sidebar.divider()
st.sidebar.caption(
    "Built for SYSEN 5381 — demonstrates multi-agent orchestration, "
    "RAG integration, and function-calling tools."
)

# ── Main area ────────────────────────────────────────────────────────────────

st.title("🧠 AlgoMentor AI")
st.markdown(
    "An AI-powered DSA interview prep app.  "
    "Select a **topic** and **difficulty** in the sidebar, choose an action, "
    "and click **Run** to interact with the multi-agent system."
)

user_query = st.text_input(
    "Your request (optional — refine what you're looking for)",
    placeholder="e.g., 'Give me a problem about hash maps' or 'I'm stuck on the two-pointer approach'",
)

user_solution = st.text_area(
    "Your solution / code (for evaluation)",
    height=200,
    placeholder="Paste your Python code or describe your approach here…",
)

run_clicked = st.button("🚀 Run", type="primary", use_container_width=True)

st.divider()

# ── Orchestration logic ─────────────────────────────────────────────────────

if run_clicked:
    trace: list[str] = []

    # ── Step 1: Router Agent ─────────────────────────────────────────────
    router_result = classify_intent(user_query, action_override=action)
    intent = router_result["intent"]
    trace.append(
        f"**Router Agent** → intent = `{intent}` "
        f"(confidence: {router_result['confidence']}, method: {router_result['method']})"
    )

    # ── Step 2: RAG Agent (retrieve problem for any intent) ──────────────
    if intent == "problem_request" or st.session_state.current_problem is None:
        rag_result = retrieve_problem(
            query=user_query,
            topic=topic if topic != "All" else None,
            difficulty=difficulty if difficulty != "All" else None,
        )
        problem = rag_result.get("problem")
        if problem:
            st.session_state.current_problem = problem
            st.session_state.hint_index = 0
            st.session_state.mentor_response = None
            st.session_state.evaluation_response = None
        trace.append(
            f"**RAG Agent** → retrieved *{problem['title'] if problem else 'None'}* "
            f"(score: {rag_result['score']}, candidates: {rag_result['candidates_considered']}, "
            f"filters: {rag_result['filters_applied']})"
        )
    else:
        trace.append(
            f"**RAG Agent** → using cached problem: "
            f"*{st.session_state.current_problem['title']}*"
        )

    problem = st.session_state.current_problem

    # ── Step 3: Dispatch to Mentor or Evaluator ──────────────────────────
    if intent == "hint_request" or intent == "explanation_request":
        if problem:
            mentor_result = get_mentor_response(problem, st.session_state.hint_index, api_key=api_key or None)
            st.session_state.mentor_response = mentor_result
            st.session_state.hint_index += 1
            trace.append(
                f"**Mentor Agent** → delivered hint {mentor_result['hint_number']}"
                f"/{mentor_result['total_hints']}  |  tools called: "
                f"{', '.join(mentor_result['tools_called'])}"
            )
        else:
            st.warning("Retrieve a problem first (use **Get Problem**).")

    elif intent == "evaluation_request":
        if problem:
            if not user_solution.strip():
                st.warning("Paste your solution in the text area above before evaluating.")
            else:
                eval_result = evaluate_solution(problem, user_solution, api_key=api_key or None)
                st.session_state.evaluation_response = eval_result
                trace.append(
                    f"**Evaluator Agent** → correctness: {eval_result['correctness'][:30]}…  |  "
                    f"tools called: {', '.join(eval_result['tools_called'])}"
                )
        else:
            st.warning("Retrieve a problem first (use **Get Problem**).")

    st.session_state.agent_trace = trace

# ── Display: Current Problem ─────────────────────────────────────────────

st.subheader("📋 Current Problem")
if st.session_state.current_problem:
    st.markdown(format_problem_card(st.session_state.current_problem))
else:
    st.info("No problem loaded yet. Select a topic and click **Run** with *Get Problem*.")

# ── Display: Mentor Response ─────────────────────────────────────────────

st.subheader("🎯 Mentor Response")
if st.session_state.mentor_response:
    st.markdown(st.session_state.mentor_response["formatted_response"])
else:
    st.info("Select *Get Hint* and click **Run** to receive progressive hints.")

# ── Display: Evaluation Response ─────────────────────────────────────────

st.subheader("📊 Evaluation")
if st.session_state.evaluation_response:
    st.markdown(st.session_state.evaluation_response["formatted_response"])
else:
    st.info("Paste your solution, select *Evaluate Solution*, and click **Run**.")

# ── Display: Agent Trace ─────────────────────────────────────────────────

st.divider()
with st.expander("🔍 Agent Trace (click to expand)"):
    if st.session_state.agent_trace:
        for entry in st.session_state.agent_trace:
            st.markdown(f"- {entry}")
    else:
        st.caption("No trace yet — run an action to see the agent pipeline.")
