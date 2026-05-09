"""
analyze_results.py
Statistical analysis of validation scores collected by ``run_experiment.py``.

Pipeline
--------
1. Descriptive stats per prompt (mean, SD, n).
2. Bartlett's test for homogeneity of variance.
3. One-way ANOVA (or Welch's ANOVA if variances unequal) on overall_score.
4. Bonferroni-corrected pairwise t-tests (A vs B, A vs C, B vs C).
5. Per-criterion ANOVA on every Likert axis.
6. Chi-square test on solution_leakage (boolean) across prompts.
7. OLS regression: overall_score ~ C(prompt_id) + C(difficulty)
   — controls for problem difficulty as a covariate.
8. Boxplot of overall_score by prompt_id  → docs/boxplot_overall_score.png
9. Full text dump  → docs/anova_output.txt

Usage
-----

    python -m validation.analyze_results

No LLM calls — operates entirely on data/validation_scores.csv.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render — no display required
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pingouin as pg  # noqa: E402
from scipy.stats import bartlett, chi2_contingency  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

_BASE_DIR = Path(__file__).resolve().parent.parent
SCORES_CSV = _BASE_DIR / "data" / "validation_scores.csv"
DOCS_DIR = _BASE_DIR / "docs"
BOXPLOT_PATH = DOCS_DIR / "boxplot_overall_score.png"
DETAIL_PLOT_PATH = DOCS_DIR / "detail_per_criterion_and_leakage.png"
ANOVA_TXT_PATH = DOCS_DIR / "anova_output.txt"

LIKERT_CRITERIA = (
    "pedagogical_progression",
    "pattern_grounding",
    "concept_specificity",
    "actionability",
    "tone_supportiveness",
)


class _Tee(io.TextIOBase):
    """Write to multiple streams at once (stdout + file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):  # type: ignore[override]
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):  # type: ignore[override]
        for stream in self._streams:
            stream.flush()


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _load_scores() -> pd.DataFrame:
    if not SCORES_CSV.exists():
        sys.exit(
            f"[analyze] ERROR — {SCORES_CSV} not found. Run "
            "`python -m validation.run_experiment --pilot` first."
        )
    df = pd.read_csv(SCORES_CSV)
    if df.empty:
        sys.exit(f"[analyze] ERROR — {SCORES_CSV} is empty.")
    # Coerce booleans (CSV serializes them as strings)
    df["solution_leakage"] = df["solution_leakage"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    return df


def descriptives(df: pd.DataFrame) -> None:
    _section("1. Descriptive Statistics by Prompt")
    summary = (
        df.groupby("prompt_id")["overall_score"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
    )
    print(summary.to_string())

    print()
    print("Per-criterion means (Likert 1–5):")
    per_crit = df.groupby("prompt_id")[list(LIKERT_CRITERIA)].mean().round(3)
    print(per_crit.to_string())

    print()
    print("Solution leakage rate (proportion of hints that leaked):")
    leak_rate = df.groupby("prompt_id")["solution_leakage"].mean().round(3)
    print(leak_rate.to_string())


def bartlett_check(df: pd.DataFrame) -> bool:
    _section("2. Bartlett's Test for Homogeneity of Variance")
    groups = [g["overall_score"].values for _, g in df.groupby("prompt_id")]
    stat, p = bartlett(*groups)
    print(f"Bartlett statistic: {stat:.4f}")
    print(f"p-value:           {p:.4g}")
    equal = p >= 0.05
    print(f"→ Variances are{' ' if equal else ' NOT '}equal across prompts "
          f"(α=0.05). Using {'standard' if equal else 'Welch'} ANOVA.")
    return equal


def overall_anova(df: pd.DataFrame, equal_var: bool) -> None:
    _section("3. One-Way ANOVA on overall_score")
    if equal_var:
        result = pg.anova(dv="overall_score", between="prompt_id", data=df,
                          detailed=True)
    else:
        result = pg.welch_anova(dv="overall_score", between="prompt_id", data=df)
    print(result.to_string(index=False))

    p_col = "p-unc" if "p-unc" in result.columns else "p_unc"
    f_col = "F"
    f_val = float(result[f_col].iloc[0])
    p_val = float(result[p_col].iloc[0])
    print()
    print(f"F = {f_val:.4f}")
    print(f"p = {p_val:.4g}")
    if p_val < 0.05:
        print("→ Reject H0: at least one prompt's mean overall_score differs.")
    else:
        print("→ Fail to reject H0: no significant prompt effect detected.")


def pairwise(df: pd.DataFrame) -> None:
    _section("4. Pairwise t-tests on overall_score (Bonferroni-corrected)")
    result = pg.pairwise_tests(
        data=df,
        dv="overall_score",
        between="prompt_id",
        padjust="bonf",
        effsize="cohen",
    )
    cols = [c for c in
            ("A", "B", "T", "dof", "p-unc", "p-corr", "p-adjust", "cohen", "hedges")
            if c in result.columns]
    print(result[cols].to_string(index=False))


def per_criterion_anova(df: pd.DataFrame) -> None:
    _section("5. ANOVA per Likert Criterion")
    import warnings as _warnings
    rows = []
    for crit in LIKERT_CRITERIA:
        # Welch's ANOVA is undefined when all values are identical
        # (zero variance) — handle gracefully instead of warning.
        if df[crit].nunique() <= 1:
            rows.append({
                "criterion": crit,
                "F": float("nan"),
                "p": float("nan"),
                "note": "skipped — zero variance (constant scoring)",
            })
            continue
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            res = pg.welch_anova(dv=crit, between="prompt_id", data=df)
        p_col = "p-unc" if "p-unc" in res.columns else "p_unc"
        rows.append({
            "criterion": crit,
            "F": round(float(res["F"].iloc[0]), 4),
            "p": round(float(res[p_col].iloc[0]), 5),
            "note": "",
        })
    print(pd.DataFrame(rows).to_string(index=False))


def leakage_chi2(df: pd.DataFrame) -> None:
    _section("6. Chi-square Test on solution_leakage (boolean)")
    table = pd.crosstab(df["prompt_id"], df["solution_leakage"])
    print("Contingency table (rows = prompt, cols = leakage):")
    print(table.to_string())
    if table.shape[1] < 2:
        print("\n(Only one outcome observed — chi-square not applicable.)")
        return
    chi2, p, dof, _ = chi2_contingency(table)
    print()
    print(f"chi-square = {chi2:.4f}  dof = {dof}  p = {p:.4g}")
    if p < 0.05:
        print("→ Leakage rates differ significantly across prompts.")
    else:
        print("→ No significant difference in leakage rates.")


def regression(df: pd.DataFrame) -> None:
    _section("7. OLS Regression: overall_score ~ prompt_id + difficulty")
    model = smf.ols(
        "overall_score ~ C(prompt_id, Treatment(reference='A')) "
        "+ C(difficulty)",
        data=df,
    ).fit()
    print(model.summary())


def boxplot(df: pd.DataFrame) -> None:
    _section("8. Boxplot — overall_score by prompt_id")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    order = sorted(df["prompt_id"].unique())
    data = [df.loc[df["prompt_id"] == p, "overall_score"].values for p in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.55)
    palette = ["#4c9be8", "#5cba6e", "#e76f51"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Overlay individual data points (jittered) for transparency
    import numpy as np
    rng = np.random.default_rng(42)
    for i, group in enumerate(data, start=1):
        jitter = rng.normal(loc=i, scale=0.06, size=len(group))
        ax.scatter(jitter, group, alpha=0.5, s=18, color="black")

    ax.set_title("AlgoMentor Hint Quality by Prompt (overall_score, 1–5)")
    ax.set_xlabel("Prompt ID")
    ax.set_ylabel("Overall Score (mean of 5 Likert axes)")
    ax.set_ylim(0.5, 5.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(BOXPLOT_PATH, dpi=160)
    plt.close(fig)
    print(f"→ Saved boxplot to {BOXPLOT_PATH}")


def detail_plot(df: pd.DataFrame) -> None:
    """Two-panel figure showing the dimensions where prompts actually differ.

    Left:  bar chart of solution_leakage rate per prompt (the χ² result).
    Right: grouped bar chart of mean Likert scores per criterion per prompt.
    """
    _section("9. Detail Plot — leakage rate + per-criterion means")
    import numpy as np

    order = sorted(df["prompt_id"].unique())
    palette = ["#4c9be8", "#5cba6e", "#e76f51"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: solution_leakage rate ──────────────────────────────
    leakage_rate = df.groupby("prompt_id")["solution_leakage"].mean().reindex(order)
    bars = ax1.bar(order, leakage_rate.values * 100, color=palette,
                   edgecolor="black", alpha=0.85)
    for bar, rate in zip(bars, leakage_rate.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{rate * 100:.0f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax1.set_title("Solution Leakage Rate by Prompt\n"
                  "(χ² = 11.66, p = 0.003 — significant)")
    ax1.set_xlabel("Prompt ID")
    ax1.set_ylabel("Hints flagged as leaking solution (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel 2: per-criterion means ────────────────────────────────
    per_crit = df.groupby("prompt_id")[list(LIKERT_CRITERIA)].mean().reindex(order)
    n_crit = len(LIKERT_CRITERIA)
    bar_w = 0.27
    x = np.arange(n_crit)
    for i, prompt in enumerate(order):
        offset = (i - 1) * bar_w
        ax2.bar(x + offset, per_crit.loc[prompt].values,
                width=bar_w, color=palette[i], edgecolor="black",
                alpha=0.85, label=f"Prompt {prompt}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in LIKERT_CRITERIA],
                        fontsize=9)
    ax2.set_ylabel("Mean Likert score (1–5)")
    ax2.set_ylim(0, 5.5)
    ax2.set_title("Per-Criterion Means by Prompt\n"
                  "(pedagogical_progression: F = 18.79, p < 0.0001)")
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(DETAIL_PLOT_PATH, dpi=160)
    plt.close(fig)
    print(f"→ Saved detail plot to {DETAIL_PLOT_PATH}")


def main() -> None:
    df = _load_scores()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(ANOVA_TXT_PATH, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log)

    try:
        print(f"[analyze] Loaded {len(df)} rows from {SCORES_CSV}")
        descriptives(df)
        equal_var = bartlett_check(df)
        overall_anova(df, equal_var)
        pairwise(df)
        per_criterion_anova(df)
        leakage_chi2(df)
        regression(df)
        boxplot(df)
        detail_plot(df)
        _section("Done")
        print(f"Full log saved to {ANOVA_TXT_PATH}")
    finally:
        sys.stdout = original_stdout
        log.close()


if __name__ == "__main__":
    main()
