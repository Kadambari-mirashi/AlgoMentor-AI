"""
run_experiment.py
Generates AlgoMentor hint outputs under three competing prompts (A, B, C),
validates each hint via the AI judge, and writes results to disk.

Design
------
Independent variable:   prompt_id  ∈ {A, B, C}
Generator:              gemma3:12b at temperature 0.7
Validator:              gpt-oss:120b at temperature 0.1 (independent judge)
Problems:               all 10 from data/dsa_knowledge_base.json
Repetitions:            3 per (prompt, problem)  →  n=30 per prompt, N=90

The script is **resumable** — every result is appended to the CSV
immediately after generation+validation, and a re-run skips rows already
present in ``data/validation_scores.csv``.

Usage
-----

    python -m validation.run_experiment --pilot     # ~18 LLM calls, ~3-5 min
    python -m validation.run_experiment --full      # ~180 LLM calls

Flags
-----

    --pilot           Run only 1 problem × 3 prompts × 3 reps (= 9 + 9 calls).
    --full            Run the complete N=90 design (90 + 90 calls).
    --reps N          Override repetitions (default 3).
    --problems N      Override # of problems sampled (default: all 10).
    --generator MODEL Override generator model.
    --validator MODEL Override validator model.
    --resume          Skip rows already present in the scores CSV (default).
    --restart         Wipe the scores CSV and start over.
    --delay SECONDS   Sleep between calls (default 1.5s) for rate-limit safety.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────
# Path setup so this can be run as `python -m validation.run_experiment`
# from the project root.
# ──────────────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

from utils.helpers import load_knowledge_base  # noqa: E402
from utils.llm import query_llm  # noqa: E402
from utils import prompts as prompts_mod  # noqa: E402
from tools.pattern_tool import get_pattern_hint  # noqa: E402
from validation.prompts_to_compare import PROMPTS  # noqa: E402
from validation.validator import (  # noqa: E402
    DEFAULT_VALIDATOR_MODEL,
    LIKERT_CRITERIA,
    validate_hint,
)

DEFAULT_GENERATOR_MODEL = "gemma3:12b"
GENERATOR_TEMPERATURE = 0.7

OUTPUTS_CSV = _BASE_DIR / "data" / "experiment_outputs.csv"
SCORES_CSV = _BASE_DIR / "data" / "validation_scores.csv"

OUTPUTS_HEADER = [
    "row_id",
    "prompt_id",
    "problem_id",
    "problem_title",
    "difficulty",
    "topic",
    "repetition",
    "mentor_text",
    "generator_model",
]

SCORES_HEADER = [
    "row_id",
    "prompt_id",
    "problem_id",
    "problem_title",
    "difficulty",
    "topic",
    "repetition",
    "overall_score",
    *LIKERT_CRITERIA,
    "solution_leakage",
    "details",
    "validator_model",
]


# ──────────────────────────────────────────────────────────────────
# Mentor generation (replays the LLM path of mentor_agent.py but with
# a swappable system prompt so we can pin Prompt A / B / C).
# ──────────────────────────────────────────────────────────────────


def generate_mentor_hint(
    problem: dict,
    system_prompt: str,
    api_key: str,
    model: str = DEFAULT_GENERATOR_MODEL,
    hint_index: int = 0,
) -> str | None:
    """Run the Mentor Agent's LLM call with a custom system prompt.

    Mirrors ``agents.mentor_agent.get_mentor_response`` but lets us
    pin the system prompt to one of the experimental variants.
    """
    hints = problem.get("hints", [])
    title = problem.get("title", "Unknown Problem")
    topic = problem.get("topic", "")
    tags = problem.get("tags", [])

    current_index = min(hint_index, max(len(hints) - 1, 0))
    hint_text = hints[current_index] if hints else ""

    pattern_result = get_pattern_hint(tags if tags else [topic])
    patterns = pattern_result.get("patterns", [])

    user_prompt = (
        f"Problem: \"{title}\" ({topic}, {problem.get('difficulty', 'N/A')})\n"
        f"Hint {current_index + 1}/{len(hints)}: \"{hint_text}\"\n"
        f"Patterns: {', '.join(patterns[:4])}\n\n"
        "Expand on this hint in 3-5 sentences. Be encouraging. "
        "Reference the patterns. Do not reveal the full solution."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return query_llm(messages, model=model, api_key=api_key)


# ──────────────────────────────────────────────────────────────────
# CSV helpers (resumable)
# ──────────────────────────────────────────────────────────────────


def _ensure_csv(path: Path, header: list[str]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(header)


def _completed_row_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["row_id"] for row in reader if row.get("row_id")}


def _append_row(path: Path, header: list[str], row: dict[str, Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writerow({k: row.get(k, "") for k in header})


def _row_id(prompt_id: str, problem_id: str, rep: int) -> str:
    return f"{prompt_id}__{problem_id}__r{rep}"


# ──────────────────────────────────────────────────────────────────
# Experiment loop
# ──────────────────────────────────────────────────────────────────


def run_experiment(
    *,
    api_key: str,
    generator_model: str,
    validator_model: str,
    n_problems: int,
    n_reps: int,
    delay: float,
    restart: bool,
) -> dict[str, int]:
    kb = load_knowledge_base()
    if not kb:
        raise RuntimeError("Knowledge base is empty — nothing to generate.")

    problems = kb[:n_problems]
    prompt_ids = list(PROMPTS.keys())  # ['A', 'B', 'C']

    if restart:
        for path in (OUTPUTS_CSV, SCORES_CSV):
            if path.exists():
                path.unlink()
        print(f"[run] --restart: cleared {OUTPUTS_CSV.name} and {SCORES_CSV.name}")

    _ensure_csv(OUTPUTS_CSV, OUTPUTS_HEADER)
    _ensure_csv(SCORES_CSV, SCORES_HEADER)

    completed = _completed_row_ids(SCORES_CSV)
    total_planned = len(prompt_ids) * len(problems) * n_reps
    todo = total_planned - len(completed)

    print(f"[run] Generator:  {generator_model}")
    print(f"[run] Validator:  {validator_model}")
    print(f"[run] Prompts:    {prompt_ids}")
    print(f"[run] Problems:   {len(problems)}  Reps: {n_reps}")
    print(f"[run] Planned:    {total_planned} rows (gen + judge = "
          f"{2 * total_planned} LLM calls)")
    print(f"[run] Already complete: {len(completed)}  →  {todo} rows to do")
    print(f"[run] Per-call delay: {delay}s")
    print()

    stats = {"completed": 0, "skipped": 0, "gen_fail": 0, "val_fail": 0}
    start = time.time()

    for prompt_id in prompt_ids:
        system_prompt = PROMPTS[prompt_id]
        for problem in problems:
            problem_id = problem["id"]
            for rep in range(1, n_reps + 1):
                row_id = _row_id(prompt_id, problem_id, rep)
                if row_id in completed:
                    stats["skipped"] += 1
                    continue

                stats["completed"] += 1
                idx = stats["completed"] + stats["skipped"]
                print(
                    f"[{idx:3d}/{total_planned}] prompt={prompt_id} "
                    f"problem={problem_id:25s} rep={rep}"
                )

                # 1) Generate mentor hint
                mentor_text = generate_mentor_hint(
                    problem=problem,
                    system_prompt=system_prompt,
                    api_key=api_key,
                    model=generator_model,
                )
                if not mentor_text:
                    print("    [gen] FAILED — skipping row")
                    stats["gen_fail"] += 1
                    time.sleep(delay)
                    continue

                _append_row(
                    OUTPUTS_CSV,
                    OUTPUTS_HEADER,
                    {
                        "row_id": row_id,
                        "prompt_id": prompt_id,
                        "problem_id": problem_id,
                        "problem_title": problem.get("title", ""),
                        "difficulty": problem.get("difficulty", ""),
                        "topic": problem.get("topic", ""),
                        "repetition": rep,
                        "mentor_text": mentor_text,
                        "generator_model": generator_model,
                    },
                )

                time.sleep(delay)

                # 2) Validate hint
                score = validate_hint(
                    problem=problem,
                    mentor_output=mentor_text,
                    api_key=api_key,
                    model=validator_model,
                )
                if score is None or score.get("_parse_failed"):
                    print("    [val] FAILED — skipping row")
                    stats["val_fail"] += 1
                    time.sleep(delay)
                    continue

                row = {
                    "row_id": row_id,
                    "prompt_id": prompt_id,
                    "problem_id": problem_id,
                    "problem_title": problem.get("title", ""),
                    "difficulty": problem.get("difficulty", ""),
                    "topic": problem.get("topic", ""),
                    "repetition": rep,
                    "overall_score": score["overall_score"],
                    "solution_leakage": score["solution_leakage"],
                    "details": score.get("details", ""),
                    "validator_model": validator_model,
                }
                for k in LIKERT_CRITERIA:
                    row[k] = score[k]
                _append_row(SCORES_CSV, SCORES_HEADER, row)

                print(
                    f"    [val] overall={score['overall_score']:.2f}  "
                    f"leakage={score['solution_leakage']}"
                )
                time.sleep(delay)

    elapsed = time.time() - start
    print()
    print(f"[run] Done in {elapsed/60:.1f} min")
    print(
        f"[run] Completed: {stats['completed']}  Skipped (already done): "
        f"{stats['skipped']}  Gen-fail: {stats['gen_fail']}  Val-fail: "
        f"{stats['val_fail']}"
    )
    print(f"[run] Outputs CSV: {OUTPUTS_CSV}")
    print(f"[run] Scores  CSV: {SCORES_CSV}")
    return stats


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Generate + validate mentor hints across prompts A/B/C."
    )
    p.add_argument("--pilot", action="store_true",
                   help="Quick pilot: 1 problem × 3 prompts × 3 reps "
                        "(= 9 gen + 9 val = 18 LLM calls).")
    p.add_argument("--full", action="store_true",
                   help="Full design: 10 problems × 3 prompts × 3 reps "
                        "(= 90 gen + 90 val = 180 LLM calls).")
    p.add_argument("--problems", type=int, default=None,
                   help="Number of problems to sample (default: all 10).")
    p.add_argument("--reps", type=int, default=None,
                   help="Repetitions per (prompt, problem). Default: 3.")
    p.add_argument("--generator", default=DEFAULT_GENERATOR_MODEL)
    p.add_argument("--validator", default=DEFAULT_VALIDATOR_MODEL)
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--restart", action="store_true",
                   help="Wipe existing CSVs before running.")
    args = p.parse_args()

    if args.pilot and args.full:
        p.error("Choose --pilot OR --full, not both.")

    if args.pilot:
        n_problems = 1
        n_reps = 3
    elif args.full:
        n_problems = 10
        n_reps = 3
    else:
        n_problems = args.problems if args.problems else 10
        n_reps = args.reps if args.reps else 3

    if args.problems is not None:
        n_problems = args.problems
    if args.reps is not None:
        n_reps = args.reps

    load_dotenv()
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        p.error("OLLAMA_API_KEY not set in environment / .env file.")

    run_experiment(
        api_key=api_key,
        generator_model=args.generator,
        validator_model=args.validator,
        n_problems=n_problems,
        n_reps=n_reps,
        delay=args.delay,
        restart=args.restart,
    )


if __name__ == "__main__":
    _cli()
