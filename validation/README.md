# AlgoMentor Hint Validation System

**SYSEN 5381 — Homework 3: AI Report Validation System**

This package implements a customized validation framework for the
AlgoMentor AI Mentor Agent and a statistical experiment comparing
three competing system prompts (A, B, C).

It builds directly on the AlgoMentor AI codebase from Homework 2 — the
`mentor_agent` is the AI under test, the JSON knowledge base is the
ground truth, and `utils/llm.py` is reused to call Ollama Cloud.

---

## 1. Validation Criteria Table

The rubric is defined formally in [`data/validation_rubric.json`](../data/validation_rubric.json).
It deliberately replaces the LAB's generic Likert axes with criteria
tailored to **DSA tutoring** as the use case.

| # | Dimension | Type | What it measures | Why it differs from the LAB |
|---|---|---|---|---|
| 1 | `pedagogical_progression` | Likert 1–5 | Does the hint guide discovery without revealing the full solution? | New tutoring-specific axis; LAB has no concept of "leakage". |
| 2 | `pattern_grounding` | Likert 1–5 | Does the hint correctly invoke an algorithmic pattern matching the KB tags? | Reframes LAB's `faithfulness` against a domain-specific ground truth (KB tags), not generic data interpretation. |
| 3 | `concept_specificity` | Likert 1–5 | Does the hint reference data structures specific to *this* problem? | More precise than LAB's `relevance`. |
| 4 | `actionability` | Likert 1–5 | Can a stuck student take a concrete next coding step from this hint? | Outcome-focused; absent from the LAB. |
| 5 | `tone_supportiveness` | Likert 1–5 | Is the language encouraging and mentor-like? | Replaces LAB's `formality` (irrelevant for tutoring) and `succinctness`. |
| 6 | `solution_leakage` | Boolean | Does the hint reveal pseudocode, the algorithm name + steps, or the optimal Big-O? | Inverts the LAB's `accurate` boolean to measure **teacher behavior** rather than data interpretation. |
| — | `overall_score` | Derived (1–5) | Mean of the five Likert axes | Single dependent variable for ANOVA. |

The judge model is asked to score conservatively (only award 5 when the
top-anchor description is clearly met) and to return a 20–40 word
justification in `details` for auditability.

---

## 2. Experimental Design

| Parameter | Value |
|---|---|
| Independent variable | `prompt_id` ∈ {A, B, C} |
| AI under test (generator) | `gemma3:12b` via Ollama Cloud, `temperature=0.7` |
| AI reviewer (validator) | `gpt-oss:120b` via Ollama Cloud, `temperature=0.1`, `think=false` (independent judge — different model from the generator) |
| Problems sampled | All 10 from [`data/dsa_knowledge_base.json`](../data/dsa_knowledge_base.json) |
| Repetitions per (prompt, problem) | 3 |
| **Sample size** | **n = 30 per prompt, N = 90 total** |
| Total LLM calls executed | 90 generations + 90 validations = **180** |

The three competing prompts are defined in
[`prompts_to_compare.py`](prompts_to_compare.py):

| Prompt | Hypothesis | Style |
|---|---|---|
| **A** (control) | Baseline — pedagogically sound but not maximally structured | Current production `MENTOR_ROLE` from `utils/prompts.py` |
| **B** | Best — Socratic + structured + explicit no-leak constraint | One guiding question + one named pattern + explicit ban on pseudocode |
| **C** | Worst — encourages full solution disclosure | Asks the model to walk through the complete solution with pseudocode |

---

## 3. Statistical Analysis

All tests run by [`analyze_results.py`](analyze_results.py) on
[`data/validation_scores.csv`](../data/validation_scores.csv); the full
log is saved to [`docs/anova_output.txt`](../docs/anova_output.txt).

### Tests performed

| # | Test | Purpose |
|---|---|---|
| 1 | Descriptive stats | Mean / SD per prompt; per-criterion means |
| 2 | Bartlett's test | Homogeneity of variance → chooses ANOVA flavor |
| 3 | One-way ANOVA on `overall_score` | Primary hypothesis: do prompts differ on aggregate quality? |
| 4 | Pairwise t-tests (Bonferroni) | Which pairs differ? |
| 5 | Per-criterion ANOVA | Which dimensions drive any difference? |
| 6 | Chi-square on `solution_leakage` | Categorical analog for the boolean criterion |
| 7 | OLS regression | `overall_score ~ C(prompt_id) + C(difficulty)` — controls for problem difficulty |

### Results (from this run)

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| Bartlett (variances) | χ² = 8.32 | 0.016 | Variances unequal → use Welch's ANOVA |
| Welch's ANOVA on `overall_score` | F(2, 55.4) = 2.18 | 0.122 | No significant overall effect (single-metric view) |
| Pairwise t (B vs C) | t(58) = 2.00, d = 0.52 | — | Medium-large effect favoring B |
| Per-criterion ANOVA on `pedagogical_progression` | F = 18.79 | < 0.0001 | **Highly significant** |
| Chi-square on `solution_leakage` | χ²(2) = 11.66 | 0.0029 | **Highly significant** |
| OLS — difficulty (Medium vs Easy) | β = -0.198 | 0.019 | Medium problems score lower regardless of prompt |

### Interpretation

The naive `overall_score` (an unweighted mean of all five Likert axes)
**hides** the true prompt effect because three of the five axes
(`pattern_grounding`, `concept_specificity`, `tone_supportiveness`)
remain near ceiling regardless of prompt — Gemma is consistently polite
and consistently names the right algorithmic pattern even when it
leaks the solution.

The customized rubric pays off when we look one level deeper:

* **`pedagogical_progression` collapses for Prompt C** (mean 2.67 vs.
  4.07/3.80 for A/B), F = 18.79, p < 0.0001 — a very large effect.
* **`solution_leakage` rate scales monotonically** with prompt
  aggressiveness: A 20% → B 40% → C 63% (χ² = 11.66, p = 0.003).
* The OLS regression confirms a real **difficulty covariate**: Medium
  problems score 0.20 points lower than Easy problems (p = 0.019).

So our headline finding is *not* "Prompt B is significantly better on
average" — it is the more nuanced (and more publishable) result that
**prompt design significantly affects which dimensions of hint quality
are preserved**, and a multi-dimensional rubric is required to detect
the effect at all.

---

## 4. System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                    validation/run_experiment.py                  │
│  (orchestrator — resumable, writes one row at a time)            │
└────────────┬─────────────────────────────────────────┬───────────┘
             │                                         │
             ▼                                         ▼
   ┌────────────────────┐                    ┌─────────────────────┐
   │  Generator         │   mentor hint      │  Validator (judge)  │
   │  agents.mentor_*   │ ─────────────────► │  validation/        │
   │  + PROMPT_{A|B|C}  │                    │  validator.py       │
   │  gemma3:12b        │                    │  gpt-oss:120b       │
   │  T=0.7             │                    │  T=0.1, think=False │
   └────────────────────┘                    └──────────┬──────────┘
                                                        │
                                                        ▼
                                          ┌─────────────────────────┐
                                          │ data/validation_scores  │
                                          │ .csv  (one row per hint)│
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────────────┐
                                          │  analyze_results.py     │
                                          │  Bartlett · ANOVA ·     │
                                          │  pairwise t · chi² ·    │
                                          │  OLS · boxplot          │
                                          └─────────────────────────┘
```

The validator receives the original problem (title, statement, expected
approach, expected complexity, KB tags) as **ground truth** alongside
the hint to evaluate. This is what makes the judge able to score
`pattern_grounding` and `solution_leakage` correctly — it knows what
the optimal answer looks like.

A different model is used for the validator (`gpt-oss:120b`) than the
generator (`gemma3:12b`) so the judge is methodologically independent
of the AI under test.

---

## 5. Technical Details

| Item | Value |
|---|---|
| Project root | `AlgoMentor-AI/` |
| Validation package | [`validation/`](.) |
| Rubric | [`data/validation_rubric.json`](../data/validation_rubric.json) |
| Raw mentor outputs (90 rows) | [`data/experiment_outputs.csv`](../data/experiment_outputs.csv) |
| Validator scores (90 rows) | [`data/validation_scores.csv`](../data/validation_scores.csv) |
| Statistical report | [`docs/anova_output.txt`](../docs/anova_output.txt) |
| Boxplot — overall score | [`docs/boxplot_overall_score.png`](../docs/boxplot_overall_score.png) |
| Detail plot — leakage rate + per-criterion means | [`docs/detail_per_criterion_and_leakage.png`](../docs/detail_per_criterion_and_leakage.png) |
| LLM provider | Ollama Cloud (`https://ollama.com/api/chat`) |
| Required env var | `OLLAMA_API_KEY` in `.env` |
| Python | 3.10+ (this run: 3.14.0) |
| Key packages | `pandas`, `scipy`, `pingouin`, `statsmodels`, `matplotlib`, `requests`, `python-dotenv` |

### File layout

```
AlgoMentor-AI/
├── validation/
│   ├── __init__.py
│   ├── prompts_to_compare.py      # PROMPT_A / PROMPT_B / PROMPT_C
│   ├── validator.py               # AI judge with custom rubric
│   ├── run_experiment.py          # resumable runner (--pilot / --full)
│   ├── analyze_results.py         # Bartlett, ANOVA, t-tests, regression, boxplot
│   └── README.md                  # this file
├── data/
│   ├── dsa_knowledge_base.json    # 10 DSA problems (RAG corpus from HW2)
│   ├── validation_rubric.json     # formal rubric definition
│   ├── experiment_outputs.csv     # 90 raw mentor hints
│   └── validation_scores.csv      # 90 validator scores
└── docs/
    ├── anova_output.txt                          # full statistical report
    ├── boxplot_overall_score.png                 # overall_score by prompt_id
    └── detail_per_criterion_and_leakage.png      # leakage rate + per-criterion means
```

---

## 6. Usage Instructions

### Install dependencies

```bash
cd AlgoMentor-AI
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add an `OLLAMA_API_KEY` to a `.env` file in the project root.

### Step 1 — Smoke-test the validator (1 LLM call)

Confirms the judge model is reachable on your account and returns
parseable JSON.

```bash
python -m validation.validator --smoke-test
```

### Step 2 — Pilot run (18 LLM calls, ~3–5 min)

1 problem × 3 prompts × 3 reps. Use this to verify prompt discrimination
before committing to the full run.

```bash
python -m validation.run_experiment --pilot
```

### Step 3 — Full experiment (180 LLM calls, ~18 min)

```bash
python -m validation.run_experiment --full --restart
```

* `--restart` wipes any prior CSVs so the dataset is balanced.
* Omit `--restart` to **resume** an interrupted run — the script writes
  one row at a time and re-running skips rows already present.
* Per-call delay defaults to 1.5 s for Free-tier rate-limit safety.

### Step 4 — Statistical analysis (no LLM calls)

```bash
python -m validation.analyze_results
```

Prints the full report to stdout, saves it to `docs/anova_output.txt`,
and writes the boxplot to `docs/boxplot_overall_score.png`.

### Optional flags

```
python -m validation.run_experiment --help
python -m validation.validator --help
```
