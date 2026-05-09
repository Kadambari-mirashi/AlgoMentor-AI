"""
validator.py
AI-as-judge validator for AlgoMentor mentor hints.

Builds a strict-JSON quality-control prompt grounded in the
``data/validation_rubric.json`` rubric, sends it to an independent
Ollama Cloud model (default: ``gpt-oss:120b``), and parses the result
into a flat dict suitable for CSV writing.

Usage as a module
-----------------

    from validation.validator import validate_hint
    score = validate_hint(problem, mentor_text, api_key=...)

CLI smoke test
--------------

    python -m validation.validator --smoke-test

Costs 1 LLM call (validator only — no generator call).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

DEFAULT_VALIDATOR_MODEL = "gpt-oss:120b"
OLLAMA_CLOUD_URL = "https://ollama.com/api/chat"
REQUEST_TIMEOUT = 180
VALIDATOR_TEMPERATURE = 0.1

_BASE_DIR = Path(__file__).resolve().parent.parent
RUBRIC_PATH = _BASE_DIR / "data" / "validation_rubric.json"

# Likert criteria used to compute overall_score
LIKERT_CRITERIA = (
    "pedagogical_progression",
    "pattern_grounding",
    "concept_specificity",
    "actionability",
    "tone_supportiveness",
)
BOOLEAN_CRITERIA = ("solution_leakage",)


# ──────────────────────────────────────────────────────────────────
# Rubric loading & prompt construction
# ──────────────────────────────────────────────────────────────────


def load_rubric(path: Path = RUBRIC_PATH) -> dict[str, Any]:
    """Load the validation rubric JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_qc_prompt(problem: dict, mentor_output: str, rubric: dict | None = None) -> str:
    """Construct the strict-JSON QC prompt sent to the AI judge."""
    if rubric is None:
        rubric = load_rubric()

    # Assemble a compact, anchor-rich criteria block for the judge
    criteria_lines: list[str] = []
    for c in rubric["criteria"]:
        if c["type"] == "likert_1_5":
            anchors = c["anchors"]
            criteria_lines.append(
                f"- {c['name']} (1-5 Likert): {c['description']} "
                f"1 = {anchors['1']}; 3 = {anchors['3']}; 5 = {anchors['5']}."
            )
        else:  # boolean
            anchors = c["anchors"]
            criteria_lines.append(
                f"- {c['name']} (boolean true/false): {c['description']} "
                f"true = {anchors['true']}; false = {anchors['false']}."
            )
    criteria_block = "\n".join(criteria_lines)

    return f"""You are a strict validator of DSA tutoring hints. Score the hint
against the rubric below. Be conservative — only award 5/5 when the hint
clearly meets the top-anchor description. Return ONLY valid JSON. No prose
outside the JSON.

PROBLEM (ground truth from the AlgoMentor knowledge base):
- Title: {problem.get('title', 'N/A')}
- Difficulty: {problem.get('difficulty', 'N/A')}
- Topic: {problem.get('topic', 'N/A')}
- Tags: {", ".join(problem.get('tags', []))}
- Statement: {problem.get('problem_statement', 'N/A')}
- Expected approach: {problem.get('approach_summary', 'N/A')}
- Expected time complexity: {problem.get('expected_time_complexity', 'N/A')}
- Expected space complexity: {problem.get('expected_space_complexity', 'N/A')}

HINT TO EVALUATE:
\"\"\"
{mentor_output}
\"\"\"

RUBRIC:
{criteria_block}

Return ONLY this JSON object (no markdown fences, no commentary):
{{
  "pedagogical_progression": <1-5>,
  "pattern_grounding": <1-5>,
  "concept_specificity": <1-5>,
  "actionability": <1-5>,
  "tone_supportiveness": <1-5>,
  "solution_leakage": <true|false>,
  "details": "<20-40 word justification>"
}}
"""


# ──────────────────────────────────────────────────────────────────
# LLM call (Ollama Cloud)
# ──────────────────────────────────────────────────────────────────


def _call_ollama_cloud(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = VALIDATOR_TEMPERATURE,
    max_retries: int = 3,
    debug: bool = False,
) -> str | None:
    """Send a single chat request and return the raw assistant text.

    Retries on 429 / 5xx with exponential backoff. Returns ``None`` on
    permanent failure so the caller can decide what to do.

    Notes
    -----
    * ``think=False`` disables reasoning tokens for harmony-format models
      like ``gpt-oss``; without this, the model burns its ``num_predict``
      budget on internal reasoning and returns empty ``content``.
    * ``num_predict`` is set generously to avoid truncation; the JSON
      output is small (~150 tokens) but we leave headroom for safety.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",  # Ollama-side JSON enforcement
        "think": False,    # disable reasoning tokens for gpt-oss / o-series
        "options": {"temperature": temperature, "num_predict": 1500},
    }

    backoff = 4.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OLLAMA_CLOUD_URL,
                json=body,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                print(
                    f"    [validator] HTTP {resp.status_code} on attempt "
                    f"{attempt}/{max_retries}; sleeping {backoff:.1f}s"
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {}) or {}
            content = msg.get("content") or ""
            # Some harmony models put reasoning in `thinking`; if content
            # is empty but thinking has a JSON-shaped string, fall back to it.
            if not content.strip():
                thinking = msg.get("thinking") or ""
                if "{" in thinking and "}" in thinking:
                    content = thinking
            if debug:
                print("    [validator-debug] response keys:", list(data.keys()))
                print("    [validator-debug] message keys:", list(msg.keys()))
                print(
                    f"    [validator-debug] content length: {len(content)}, "
                    f"done_reason: {data.get('done_reason')!r}"
                )
            return content
        except requests.RequestException as exc:
            print(
                f"    [validator] request error on attempt "
                f"{attempt}/{max_retries}: {exc}"
            )
            if attempt == max_retries:
                return None
            time.sleep(backoff)
            backoff *= 2
    return None


# ──────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────


def parse_validator_response(raw: str) -> dict[str, Any] | None:
    """Extract and validate the JSON payload from the judge.

    Returns a dict containing all rubric fields plus ``overall_score``,
    or ``None`` if parsing fails or required fields are missing.
    """
    if not raw:
        return None

    # Some models still wrap JSON in fences or extra text; extract the
    # outermost {...} block defensively.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    parsed: dict[str, Any] = {}
    for key in LIKERT_CRITERIA:
        val = data.get(key)
        if not isinstance(val, (int, float)):
            return None
        # Clamp to the 1..5 range to defend against off-scale hallucinations
        parsed[key] = int(max(1, min(5, round(val))))

    leakage = data.get("solution_leakage")
    if not isinstance(leakage, bool):
        # Coerce common string variants
        if isinstance(leakage, str) and leakage.strip().lower() in ("true", "false"):
            leakage = leakage.strip().lower() == "true"
        else:
            return None
    parsed["solution_leakage"] = leakage

    parsed["details"] = str(data.get("details", ""))[:300]
    parsed["overall_score"] = round(
        sum(parsed[k] for k in LIKERT_CRITERIA) / len(LIKERT_CRITERIA), 3
    )
    return parsed


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────


def validate_hint(
    problem: dict,
    mentor_output: str,
    api_key: str | None = None,
    model: str = DEFAULT_VALIDATOR_MODEL,
    debug: bool = False,
) -> dict[str, Any] | None:
    """Run the AI judge on a single mentor hint.

    Returns a dict with all rubric fields + ``overall_score`` + raw
    judge response, or ``None`` on failure.
    """
    key = api_key or os.environ.get("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError(
            "OLLAMA_API_KEY not set. Add it to your .env file or pass "
            "api_key=... explicitly."
        )

    prompt = build_qc_prompt(problem, mentor_output)
    raw = _call_ollama_cloud(prompt, model=model, api_key=key, debug=debug)
    if raw is None:
        return None

    parsed = parse_validator_response(raw)
    if parsed is None:
        return {"_parse_failed": True, "_raw": raw}
    parsed["_raw"] = raw
    return parsed


# ──────────────────────────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────────────────────────


def _smoke_test(model: str) -> int:
    """Run one full validator call against a hand-written hint."""
    load_dotenv()
    fixture_problem = {
        "title": "Two Sum",
        "difficulty": "Easy",
        "topic": "Arrays",
        "tags": ["arrays", "hashing", "two pointer"],
        "problem_statement": (
            "Given an array of integers nums and an integer target, return "
            "the indices of the two numbers such that they add up to target."
        ),
        "approach_summary": (
            "Use a hash map storing each value -> index as you scan. For each "
            "element, check if (target - element) is already in the map."
        ),
        "expected_time_complexity": "O(n)",
        "expected_space_complexity": "O(n)",
    }
    fixture_hint = (
        "Have you considered what value you would need to find for each "
        "element so that the pair sums to the target? A hash map can make "
        "those lookups very efficient as you scan the array."
    )

    print(f"[smoke-test] Validator model: {model}")
    print("[smoke-test] Sending one validator call …")
    result = validate_hint(fixture_problem, fixture_hint, model=model, debug=True)
    if result is None:
        print("[smoke-test] FAIL — no response from validator.")
        return 1
    if result.get("_parse_failed"):
        raw = result.get("_raw") or ""
        print("[smoke-test] FAIL — could not parse JSON.")
        print(f"[smoke-test] Raw response length: {len(raw)} chars")
        print("[smoke-test] Raw response (first 1500 chars):")
        print(raw[:1500] if raw else "<EMPTY>")
        return 1

    print("[smoke-test] PASS — parsed validator output:")
    display = {k: v for k, v in result.items() if not k.startswith("_")}
    print(json.dumps(display, indent=2))
    return 0


def _cli() -> None:
    parser = argparse.ArgumentParser(description="AlgoMentor hint validator")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a single validator call against a fixed hint (1 LLM call).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_VALIDATOR_MODEL,
        help=f"Validator model (default: {DEFAULT_VALIDATOR_MODEL}).",
    )
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(_smoke_test(args.model))
    parser.print_help()


if __name__ == "__main__":
    _cli()
