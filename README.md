# AlgoMentor AI

An interactive DSA (Data Structures & Algorithms) interview-prep application
built with **Streamlit**.  Demonstrates **multi-agent orchestration**,
**Retrieval-Augmented Generation (RAG)**, and **function-calling / tool usage**
as required by the SYSEN 5381 Homework 2 rubric.

---

## Project Overview

AlgoMentor AI lets a user:

1. **Request a DSA problem** filtered by topic and difficulty.
2. **Ask for progressive hints** grounded in retrieved problem data.
3. **Submit a solution** (code or plain text) and receive structured feedback
   including correctness estimate, complexity analysis, missing concepts,
   and improvement suggestions.

All intelligence is self-contained — no external LLM API is required at
runtime.  The app runs as a single Streamlit process and is deployable to
**Posit Connect** with no additional backend.

---

## Architecture Summary

```
User ──▶ Streamlit UI (app.py)
              │
              ▼
        ┌─────────────┐
        │ Router Agent │  classify intent (keyword matching)
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  RAG Agent   │  retrieve problem from JSON knowledge base
        └──────┬──────┘
               │
       ┌───────┴────────┐
       ▼                ▼
 ┌───────────┐   ┌──────────────┐
 │  Mentor   │   │  Evaluator   │
 │  Agent    │   │  Agent       │
 └─────┬─────┘   └──┬───┬───┬──┘
       │             │   │   │
  pattern_tool  complexity comparison testcase
                  _tool     _tool      _tool
```

---

## Agent Descriptions

| Agent | File | Role |
|-------|------|------|
| **Router Agent** | `agents/router_agent.py` | Classifies user intent into `problem_request`, `hint_request`, `evaluation_request`, or `explanation_request` using deterministic keyword matching. |
| **RAG Agent** | `agents/rag_agent.py` | Loads the local JSON knowledge base, filters by topic/difficulty, and scores candidates using keyword overlap to retrieve the most relevant problem. |
| **Mentor Agent** | `agents/mentor_agent.py` | Delivers progressive hints from the retrieved problem and calls the **pattern tool** to provide algorithmic-pattern guidance. |
| **Evaluator Agent** | `agents/evaluator_agent.py` | Evaluates a user's solution by calling the **complexity tool**, **comparison tool**, and **testcase tool**, then synthesizes structured feedback. |

---

## Tool Descriptions

| Tool | File | Function | Purpose |
|------|------|----------|---------|
| **Pattern Tool** | `tools/pattern_tool.py` | `get_pattern_hint(topic_or_tags)` | Returns common algorithmic patterns (e.g., Two Pointer, Sliding Window) for a given topic or tag list. |
| **Test Case Tool** | `tools/testcase_tool.py` | `generate_test_cases(problem)` | Returns 3–5 sample and edge test cases for a problem. |
| **Complexity Tool** | `tools/complexity_tool.py` | `analyze_complexity(solution_text)` | Uses regex heuristics to estimate time and space complexity from submitted code/text. |
| **Comparison Tool** | `tools/comparison_tool.py` | `compare_solution_keywords(user_solution, reference_approach)` | Compares user solution keywords against the reference approach and returns an alignment summary. |

---

## Setup Instructions

### Prerequisites

- Python 3.10 or later

### Install Dependencies

```bash
cd AlgoMentor-AI
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How to Use

1. **Select a topic** and **difficulty** in the sidebar.
2. Choose an **action**:
   - *Get Problem* — retrieves a matching DSA problem from the knowledge base.
   - *Get Hint* — shows the next progressive hint for the current problem.
   - *Evaluate Solution* — paste your code in the text area, then run.
3. Click **Run**.
4. View results in the main panel; expand **Agent Trace** at the bottom to
   see which agents ran and which tools were called.

---

## File Structure

```
AlgoMentor-AI/
├── app.py                          # Streamlit entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── agents/
│   ├── __init__.py
│   ├── router_agent.py             # Intent classification
│   ├── rag_agent.py                # Knowledge-base retrieval
│   ├── mentor_agent.py             # Progressive hints + pattern tool
│   └── evaluator_agent.py          # Solution evaluation + tools
├── tools/
│   ├── __init__.py
│   ├── pattern_tool.py             # Algorithmic pattern lookup
│   ├── testcase_tool.py            # Test case generation
│   ├── complexity_tool.py          # Complexity estimation
│   └── comparison_tool.py          # Keyword comparison
├── data/
│   └── dsa_knowledge_base.json     # 10 DSA problems (RAG source)
└── utils/
    ├── __init__.py
    ├── helpers.py                  # Loading, formatting utilities
    └── prompts.py                  # Agent role descriptions & templates
```

---

## Deployment on Posit Connect

1. Ensure `requirements.txt` is at the project root.
2. In Posit Connect, create a new **Streamlit** application.
3. Point it to this repository (or upload the project folder).
4. Set the main script to `app.py`.
5. Posit Connect will install dependencies from `requirements.txt` and
   serve the app.  No external APIs or servers are needed.

---

## Knowledge Base

The file `data/dsa_knowledge_base.json` contains 10 curated DSA problems
spanning Arrays, Hashing, Binary Search, Trees, Graphs, Dynamic Programming,
and Linked Lists.  Each entry includes a problem statement, hints, approach
summary, and expected complexities — providing the retrieval corpus for the
RAG agent.

---

*SYSEN 5381 — Data Science & AI · Cornell University*
