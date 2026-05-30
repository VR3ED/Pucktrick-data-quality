# -*- coding: utf-8 -*-
"""Generate 6A-/6B-Experiment-evaluation.ipynb from a shared template.

Run:  python3 build_eval_notebooks.py
Both notebooks import the shared helpers in experiment_eval_utils.py.
"""
import os
import json

REPO = os.path.dirname(os.path.abspath(__file__))


def new_markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def new_code_cell(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def build(phase):
    assert phase in ("A", "B")
    is_A = phase == "A"
    ROOT = "complete_experiments/Experiment_A" if is_A else "complete_experiments/Experiment_B"
    cells = []
    md = lambda s: cells.append(new_markdown_cell(s))
    co = lambda s: cells.append(new_code_cell(s))

    # ------------------------------------------------------------------ #
    # 0. Title                                                            #
    # ------------------------------------------------------------------ #
    if is_A:
        md(r"""# 6A &mdash; Evaluation of *Experiment A* (reduced dataset, proof of concept)

This notebook evaluates **Experiment A**: the noise-injection runs stored under
`complete_experiments/Experiment_A`. Experiment A is the *proof-of-concept*
phase of the thesis (see `CLAUDE.md`): it uses a **feature-reduced** version of
the CAVAS dataset (redundant / zero-variance features removed) and a limited
grid of **5 random seeds**. Its purpose is to show that injecting controlled
noise with **PuckTrick** *can* affect &mdash; and in places *improve* &mdash; the
behaviour of the two IDS models (**TabNet** and **CNN-LSTM**).

> Experiment A is *not* the main contribution; it motivates the larger
> Experiment B (notebook `6B`). Conclusions drawn here should be read as
> indicative, not definitive, given the small seed count.

---

### What this notebook does (and why)

The evaluation is built to be **completely dynamic**: it discovers every
`*_artifacts.json` written under the experiment folder and recomputes all
tables, plots and confidence intervals from scratch. *Add more runs (new seeds,
new noise levels, new methods) and a single **Restart &amp; Run All** updates
every result &mdash; no code edits required.* This directly satisfies the
requirement that the confidence of the results refresh automatically as the
folder grows.

| Section | Question it answers | Why it matters for the thesis |
|---|---|---|
| 1. Dynamic loading | What runs exist on disk? | Reproducible, self-updating basis for everything below. |
| 2. Coverage | How many seeds back each cell? | A confidence interval over 2 seeds is not the same as over 20; this is reported honestly. |
| 3. Leaderboard (mean &plusmn; CI) | What are the headline numbers? | Thesis-ready point estimates *with* uncertainty. |
| 4. Metric vs. noise level | For each feature, how do **F1** and **MCC** evolve as noise grows, per PuckTrick method? | The central research question of the chapter. |
| 5. Heatmaps | Where does noise help vs. hurt? | Compact figures for the manuscript. |
| 6. Paired significance | Are observed changes *real* or within noise? | Lets us claim *improvement* rather than read it off overlapping error bars. |
| 7. Model comparison | Is TabNet or CNN-LSTM more noise-robust? | Architectural insight. |
| 8. Per-class analysis | Which attack classes benefit / suffer? | The macro metric can hide per-class effects on rare attacks. |
| 9. Feature-importance drift | Does corrupting a feature change what the model relies on? | Mechanistic explanation of the trends. |
| 10. Export | LaTeX/CSV artefacts | Drop-in tables/figures for `temp-latex`. |
| 11. Auto-summary | A written digest with the key numbers | Starting point for the chapter's conclusions. |

*Note on language:* the narrative is written in English so it can be lifted
straight into the (English) thesis in `temp-latex`.
""")
    else:
        md(r"""# 6B &mdash; Evaluation of *Experiment B* (complete dataset, main contribution)

This notebook evaluates **Experiment B**: the noise-injection runs stored under
`complete_experiments/Experiment_B`. Experiment B is the **main focus** of the
thesis (see `CLAUDE.md`): it uses the **complete** (non feature-reduced)
dataset and a much wider grid of **up to 20 random seeds**, so its conclusions
carry real statistical weight.

Experiment B is organised into two scenarios:

* **`Pucktrick_on_single_feature/`** &mdash; noise injected into *one* feature/column
  at a time (plus a `labels_experiment/` sub-folder that corrupts the target
  labels instead of features);
* **`Pucktrick_on_multiple_features/`** &mdash; noise injected into *several*
  features simultaneously (work in progress).

The loader records this as a `scenario` column (`single` / `multi`) so the same
analysis serves both.

> **Note.** This notebook is written to run *before* the runs are finished. If a
> section finds no data yet it says so and moves on. As the experiment folder
> fills up, **Restart &amp; Run All** refreshes every result, tightening the
> confidence intervals automatically as more seeds land.

---

### What this notebook does (and why)

The evaluation is **completely dynamic**: it discovers every
`*_artifacts.json` under the experiment folder and recomputes all tables,
plots and confidence intervals from scratch on every run. The structure mirrors
notebook `6A` but adds the single- vs. multiple-feature `scenario` dimension.

| Section | Question it answers | Why it matters for the thesis |
|---|---|---|
| 1. Dynamic loading | What runs exist on disk? | Reproducible, self-updating basis for everything below. |
| 2. Coverage | How many seeds back each cell? | A CI over few seeds &ne; a CI over 20; reported honestly. |
| 3. Leaderboard (mean &plusmn; CI) | Headline numbers per scenario | Thesis-ready estimates with uncertainty. |
| 4. Metric vs. noise level | For each feature, how do **F1** and **MCC** evolve as noise grows, per method? | Central research question. |
| 5. Heatmaps | Where does noise help vs. hurt? | Compact manuscript figures. |
| 6. Paired significance | Are observed changes real? | Supports causal *improvement* claims over many seeds. |
| 7. Model comparison | TabNet vs. CNN-LSTM robustness | Architectural insight. |
| 8. Per-class analysis | Which attack classes benefit / suffer? | Macro metrics hide rare-class effects. |
| 9. Feature-importance drift | Does corruption change feature reliance? | Mechanistic explanation. |
| 10. Export | LaTeX/CSV artefacts | Drop-in tables for `temp-latex`. |
| 11. Auto-summary | Written digest with key numbers | Seed for the chapter conclusions. |

*Note on language:* the narrative is written in English so it can be lifted
straight into the (English) thesis in `temp-latex`.
""")

    # ------------------------------------------------------------------ #
    # Config / imports                                                   #
    # ------------------------------------------------------------------ #
    co(f"""# --- imports & configuration -------------------------------------------------
%load_ext autoreload
%autoreload 2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import experiment_eval_utils as eu   # shared helpers (see experiment_eval_utils.py)

# ---- knobs you may want to change ------------------------------------------
PHASE        = {phase!r}                  # "A" or "B"
ROOT_DIR     = {ROOT!r}
CONF_LEVEL   = 0.95                       # confidence level for all CIs
RESULTS_DIR  = "evaluation_results/Experiment_{phase}"   # CSV/figure exports
SAVE_FIGURES = False                      # True -> also write PNGs under RESULTS_DIR/figures

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

os.makedirs(RESULTS_DIR, exist_ok=True)
print("Phase            : Experiment", PHASE)
print("Reading from     :", ROOT_DIR)
print("Confidence level :", CONF_LEVEL)
print("Exports ->       :", RESULTS_DIR)""")

    # ------------------------------------------------------------------ #
    # 1. Load                                                            #
    # ------------------------------------------------------------------ #
    md(r"""## 1. Dynamic data loading

`load_experiment_dataframe` walks the experiment tree and turns every artifact
into **two tidy rows** (one for the *binary* task, one for the *multiclass*
task). Every downstream table and figure is derived from this single
`DataFrame`, so the whole notebook stays consistent and refreshes together.

Each artifact stores the metrics (`accuracy`, `f1`, `mcc`, `auc`) per task; the
experimental coordinates (model / method / corrupted feature / noise %) are
parsed from the file name.""")
    co(r"""df = eu.load_experiment_dataframe(ROOT_DIR)
df.head()""")

    if not is_A:
        co(r"""# Experiment B only: quick look at the two scenarios (single vs multiple feature).
if df.empty:
    print("No Experiment B artifacts on disk yet - run the experiments, then re-run this notebook.")
else:
    print("Rows per scenario:")
    display(df.groupby("scenario").size().rename("rows").to_frame())
    print("\nLabels-variant runs (target corruption) per scenario:")
    display(df.groupby(["scenario", "is_labels_variant"]).size().rename("rows").to_frame())""")

    co(r"""# Guard used throughout: if nothing is on disk yet, the rest still runs cleanly.
HAS_DATA = not df.empty
if not HAS_DATA:
    print("[!] No artifacts found - downstream sections will be empty until runs exist.")
    MODELS, TASKS = [], ["binary", "multiclass"]
else:
    MODELS = sorted(df["model"].dropna().unique())
    TASKS  = ["binary", "multiclass"]
    print("Models  :", MODELS)
    print("Methods :", sorted(df["method"].dropna().unique()))
    print("Features:", sorted(df["feature"].dropna().unique()))
    print("Noise % :", sorted(df["noise_percentage"].dropna().unique()))
    print("Seeds   :", sorted(df["seed"].dropna().unique()),
          f"(n={df['seed'].nunique()})")""")

    # ------------------------------------------------------------------ #
    # 2. Coverage                                                        #
    # ------------------------------------------------------------------ #
    md(r"""## 2. Coverage &mdash; how trustworthy is each cell?

Before averaging anything we check **how many seeds** actually back each
`(method, feature, noise%)` combination. This is the empirical basis of every
confidence interval below:

* a cell with **1 seed** has *no* CI (reported as `NaN`) and must be read with
  caution;
* cells with uneven seed counts are expected while the grid is still filling in.

Surfacing this keeps the thesis honest: we never present a mean over two runs as
if it were as solid as a mean over twenty.""")
    co(r"""if HAS_DATA:
    for model in MODELS:
        print(f"\n=== Seed coverage - {model} (binary task) ===")
        cov = eu.coverage_table(df, model, task="binary")
        display(cov.style.background_gradient(cmap="Greens", axis=None))
    seeds_per_cell = (df.groupby(["model", "task", "method", "feature", "noise_percentage"])
                        ["seed"].nunique())
    print("\nSeeds per cell - min / median / max:",
          int(seeds_per_cell.min()), "/", int(seeds_per_cell.median()), "/",
          int(seeds_per_cell.max()))""")

    # ------------------------------------------------------------------ #
    # 3. Leaderboard                                                     #
    # ------------------------------------------------------------------ #
    md(r"""## 3. Aggregated results with confidence intervals

`aggregate_with_ci` groups the long table and reports, for every metric,
`mean`, `std`, the seed count `n`, and a **two-sided Student-t confidence
interval** whose width adapts to `n`. Student-t (rather than a normal
approximation) is the right choice for the small seed counts used here.

Below: the headline **MCC** and **F1** per `(model, task, method, feature,
noise%)`, formatted as `mean ± half-CI`.""")
    co(r"""def leaderboard(metric):
    if not HAS_DATA:
        return pd.DataFrame()
    agg = eu.aggregate_with_ci(
        df, ["model", "task", "method", "feature", "noise_percentage"],
        metric, conf=CONF_LEVEL)
    agg = agg.sort_values("mean", ascending=False)
    agg[f"{metric} (mean +/- CI)"] = [
        eu.format_mean_ci(m, h) for m, h in zip(agg["mean"], agg["ci_half"])]
    return agg

mcc_board = leaderboard("mcc")
if HAS_DATA:
    print("Top-15 configurations by mean MCC:")
    display(mcc_board[["model", "task", "method", "feature", "noise_percentage",
                       "n", "mcc (mean +/- CI)"]].head(15).reset_index(drop=True))""")
    co(r"""f1_board = leaderboard("f1")
if HAS_DATA:
    print("Top-15 configurations by mean F1:")
    display(f1_board[["model", "task", "method", "feature", "noise_percentage",
                      "n", "f1 (mean +/- CI)"]].head(15).reset_index(drop=True))""")

    # ------------------------------------------------------------------ #
    # 4. Core trends                                                     #
    # ------------------------------------------------------------------ #
    md(r"""## 4. Core trends &mdash; F1 / MCC vs. noise level, per feature & method

This is the central figure family of the chapter. For each **model** and
**task** we plot the metric (**MCC**, then **F1**) against the **noise level**,
with:

* **one subplot per corrupted feature**,
* **one line per PuckTrick method**,
* a **shaded band** = the Student-t confidence interval across seeds.

Reading guide: a line that *rises* with noise is evidence that the corresponding
corruption *helps* the model (the surprising effect the proof of concept is
after); a line that *falls* is the intuitive degradation; flat-within-band means
the model is *robust* to that corruption.""")
    co(r"""def trend_figs(metric):
    if not HAS_DATA:
        print("No data yet."); return
    for model in MODELS:
        for task in TASKS:
            save = (os.path.join(RESULTS_DIR, "figures",
                                 f"{metric}_{model}_{task}_by_feature.png")
                    if SAVE_FIGURES else None)
            fig = eu.plot_metric_vs_noise(df, model, task, metric,
                                          conf=CONF_LEVEL, facet_by="feature",
                                          hue="method", savepath=save)
            if fig is not None:
                plt.show()

trend_figs("mcc")""")
    co(r"""trend_figs("f1")""")

    md(r"""### Alternative view: one subplot per method, one line per feature

Swapping the facet and hue answers the dual question: *for a given corruption
method, which features are the most sensitive?* (shown for the binary task; edit
`task`/`metric` to explore).""")
    co(r"""if HAS_DATA:
    for model in MODELS:
        fig = eu.plot_metric_vs_noise(df, model, "binary", "mcc",
                                      conf=CONF_LEVEL, facet_by="method",
                                      hue="feature")
        if fig is not None:
            plt.show()""")

    # ------------------------------------------------------------------ #
    # 5. Heatmaps                                                        #
    # ------------------------------------------------------------------ #
    md(r"""## 5. Heatmaps &mdash; method &times; noise level

Heatmaps condense the same information into compact, manuscript-friendly
figures. We show the **delta vs. the reference noise level** (diverging colour
map centred at zero): **blue = better than reference, red = worse**. This makes
the "noise helped here" cells pop out at a glance.

> **About the reference.** These experiments do **not** include a clean (0%
> noise) run, so the *lowest available* noise level is used as a near-clean
> proxy and labelled as such. If you later add genuine 0% baselines to the
> folder, `reference_level()` will pick them up automatically.""")
    co(r"""if HAS_DATA:
    ref = eu.reference_level(df)
    print(f"Reference noise level (proxy baseline): {ref:g}%")
    for model in MODELS:
        for task in TASKS:
            fig = eu.plot_metric_heatmap(df, model, task, "mcc",
                                         delta_vs_ref=True, ref_pct=ref)
            if fig is not None:
                plt.show()""")

    # ------------------------------------------------------------------ #
    # 6. Significance                                                    #
    # ------------------------------------------------------------------ #
    md(r"""## 6. Is the change real? &mdash; paired significance vs. reference

Overlapping error bars are not a verdict. For every `(method, feature, noise%)`
we pair **each seed's** metric with the *same seed's* metric at the reference
level and run a **Wilcoxon signed-rank test** on the per-seed differences.
Pairing by seed cancels seed-to-seed variance, so this is a proper
repeated-measures test of *"does this noise level move the metric?"*

We then list the configurations with a **statistically significant
improvement** (`median_delta > 0` and `p < 0.05`) &mdash; the defensible "noise
helped" claims for the thesis.

> With few seeds the Wilcoxon test has limited power, so absence of significance
> here is *not* evidence of no effect; always read it next to `n_pairs`.""")
    co(r"""def significance(metric, task="binary"):
    if not HAS_DATA:
        return pd.DataFrame()
    ref = eu.reference_level(df)
    out = []
    for model in MODELS:
        s = eu.paired_delta_significance(df, model, task, metric, ref)
        if not s.empty:
            s.insert(0, "model", model)
            out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

sig_mcc = significance("mcc", task="binary")
if HAS_DATA and not sig_mcc.empty:
    improved = sig_mcc[(sig_mcc["median_delta"] > 0) & (sig_mcc["significant"])]
    print(f"Significant MCC improvements vs reference (binary): "
          f"{len(improved)} of {len(sig_mcc)} tested configs")
    display(improved.sort_values("median_delta", ascending=False).reset_index(drop=True))
    print("\nSignificant DEGRADATIONS (for completeness):")
    worsened = sig_mcc[(sig_mcc["median_delta"] < 0) & (sig_mcc["significant"])]
    display(worsened.sort_values("median_delta").reset_index(drop=True))
elif HAS_DATA:
    print("Not enough paired data to run the significance test yet.")""")

    # ------------------------------------------------------------------ #
    # 7. Model comparison                                                #
    # ------------------------------------------------------------------ #
    md(r"""## 7. Model comparison &mdash; TabNet vs. CNN-LSTM

Two complementary robustness summaries, averaged over all features/methods:

* **Mean metric across the whole noise grid** &mdash; overall quality under noise;
* **Degradation slope** &mdash; the linear trend of the metric vs. noise level
  (slope &approx; 0 &rarr; robust; strongly negative &rarr; fragile; positive
  &rarr; noise tends to help).""")
    co(r"""if HAS_DATA:
    rows = []
    for model in MODELS:
        for task in TASKS:
            sub = df[(df.model == model) & (df.task == task)]
            for metric in ["mcc", "f1"]:
                per_level = (sub.groupby("noise_percentage")[metric]
                               .mean().reset_index().dropna())
                slope = np.nan
                if len(per_level) >= 2:
                    slope = np.polyfit(per_level["noise_percentage"],
                                       per_level[metric], 1)[0]
                rows.append({"model": model, "task": task, "metric": metric,
                             "mean_over_grid": sub[metric].mean(),
                             "slope_per_pct": slope})
    comp = pd.DataFrame(rows)
    display(comp.pivot_table(index=["task", "metric"], columns="model",
                             values=["mean_over_grid", "slope_per_pct"]))""")

    # ------------------------------------------------------------------ #
    # 8. Per-class                                                       #
    # ------------------------------------------------------------------ #
    md(r"""## 8. Per-class behaviour (multiclass)

The macro MCC/F1 can mask what happens to individual **attack classes**, some
of which are rare. The artifacts do not store a per-class report, but they do
store the multiclass confusion matrix (`cm_mul`), from which per-class
precision/recall/F1 follow directly. Here we check which classes gain or lose F1
as noise increases &mdash; important because an IDS that improves overall while
collapsing on a rare-but-critical attack is *not* an improvement in practice.
(Classes are identified by integer index, since names are not stored.)""")
    co(r"""pcd = eu.per_class_dataframe(ROOT_DIR)
if pcd.empty:
    print("No per-class data available yet.")
else:
    ref = eu.reference_level(df)
    hi = sorted(pcd["noise_percentage"].dropna().unique())[-1]
    piv = (pcd[pcd["noise_percentage"].isin([ref, hi])]
           .groupby(["model", "class", "noise_percentage"])["f1"].mean()
           .unstack("noise_percentage"))
    if ref in piv.columns and hi in piv.columns:
        piv["delta(F1)"] = piv[hi] - piv[ref]
        piv = piv.rename(columns={ref: f"F1@{ref:g}%", hi: f"F1@{hi:g}%"})
        print(f"Per-class F1 (mean over methods/features): reference {ref:g}% -> highest {hi:g}% noise")
        display(piv.sort_values("delta(F1)").round(4))""")

    # ------------------------------------------------------------------ #
    # 9. Feature importance                                              #
    # ------------------------------------------------------------------ #
    md(r"""## 9. Feature-importance drift

A mechanistic check: when PuckTrick corrupts a feature, does the model's
**permutation importance** for that feature drop (it learns to ignore the
unreliable signal) or stay high (it keeps relying on noise)? We track the mean
importance of the *corrupted* feature as the noise level grows. (`is_corrupted`
is name-normalised, so the file's `Down_Up Ratio` matches the importance dict's
`Down/Up Ratio`.)""")
    co(r"""fid = eu.feature_importance_dataframe(ROOT_DIR)
if fid.empty:
    print("No feature-importance data available yet.")
else:
    self_imp = fid[fid["is_corrupted"]]
    if self_imp.empty:
        print("Corrupted feature not found among reported importances "
              "(e.g. row-level 'duplicated'/'labels' runs); showing top features instead.")
        top = (fid.groupby(["model", "fi_feature"])["importance"].mean()
                  .reset_index().sort_values("importance", ascending=False))
        display(top.groupby("model").head(8).reset_index(drop=True))
    else:
        tab = (self_imp.groupby(["model", "method", "corrupted_feature",
                                 "noise_percentage"])["importance"]
                       .mean().reset_index())
        models_here = sorted(tab["model"].dropna().unique())
        fig, axes = plt.subplots(1, len(models_here),
                                 figsize=(6 * len(models_here), 4.2), squeeze=False)
        for ax, model in zip(axes[0], models_here):
            for (mth, feat), g in tab[tab.model == model].groupby(["method", "corrupted_feature"]):
                g = g.sort_values("noise_percentage")
                ax.plot(g["noise_percentage"], g["importance"], marker="o",
                        label=f"{mth} / {feat}")
            ax.set_title(f"{model} - importance of corrupted feature")
            ax.set_xlabel("Noise level (%)"); ax.set_ylabel("Permutation importance")
            ax.legend(fontsize=8)
        plt.tight_layout(); plt.show()""")

    # ------------------------------------------------------------------ #
    # 10. Export                                                         #
    # ------------------------------------------------------------------ #
    md(r"""## 10. Export for the thesis

Aggregated tables are written to CSV under `RESULTS_DIR` so they can be pulled
straight into the LaTeX sources in `temp-latex` (e.g. with `pandas.to_latex` or
`pgfplots`). Re-running regenerates them from whatever is currently on disk.""")
    co(r"""if HAS_DATA:
    for metric in ["mcc", "f1", "accuracy"]:
        eu.export_aggregate(df, metric, RESULTS_DIR, conf=CONF_LEVEL)
    if not sig_mcc.empty:
        sig_mcc.to_csv(os.path.join(RESULTS_DIR, "significance_mcc_binary.csv"),
                       index=False)
        print("[export] significance_mcc_binary.csv")
    print("\nExports written to:", RESULTS_DIR)""")

    # ------------------------------------------------------------------ #
    # 11. Auto-summary                                                   #
    # ------------------------------------------------------------------ #
    md(r"""## 11. Auto-generated summary &amp; conclusions

A programmatic digest of the key numbers &mdash; a *starting point* for the
chapter's prose, recomputed every run so it never goes stale. Treat it as
scaffolding to be edited into academic English, not as final text.""")
    co(r"""def auto_summary():
    if not HAS_DATA:
        print("No data yet - nothing to summarise.")
        return
    ref = eu.reference_level(df)
    lines = []
    lines.append(f"Experiment {PHASE} - automatic evaluation summary")
    lines.append("=" * 60)
    lines.append(f"Runs parsed        : {len(df)//2} artifacts "
                 f"({df['seed'].nunique()} seeds, models={sorted(df['model'].dropna().unique())})")
    lines.append(f"Reference level    : {ref:g}% (near-clean proxy; no 0% baseline on disk)")

    bb = eu.aggregate_with_ci(df[df.task == 'binary'],
                              ['model','method','feature','noise_percentage'], 'mcc')
    if not bb.empty:
        top = bb.sort_values('mean', ascending=False).iloc[0]
        lines.append("")
        lines.append(f"Best binary MCC    : {top['mean']:.3f} "
                     f"({top['model']}, {top['method']} on {top['feature']} "
                     f"@ {top['noise_percentage']:g}%, n={int(top['n'])})")

    n_imp = 0
    for model in df['model'].dropna().unique():
        s = eu.paired_delta_significance(df, model, 'binary', 'mcc', ref)
        if not s.empty:
            n_imp += int(((s['median_delta'] > 0) & s['significant']).sum())
    lines.append(f"Sig. MCC gains     : {n_imp} (model x method x feature x level) "
                 f"configurations beat the reference (Wilcoxon p<0.05)")
    print("\n".join(lines))

auto_summary()""")

    if is_A:
        md(r"""---

### Caveats to carry into the write-up

* **Small sample (5 seeds).** Confidence intervals are wide and the Wilcoxon
  test has little power; phrase Experiment A findings as *indicative*. The
  larger Experiment B (`6B`) is where firm conclusions belong.
* **No clean baseline on disk.** Improvement is measured against the lowest
  noise level as a proxy. To make absolute "noise helps vs. clean" claims, add
  0%-noise runs to the folder &mdash; the notebook will use them automatically.
* **Feature-reduced dataset.** Results here are not directly comparable to
  Experiment B, which keeps the full feature set.""")
    else:
        md(r"""---

### Caveats to carry into the write-up

* **No clean baseline on disk.** Improvement is measured against the lowest
  noise level as a proxy. Add 0%-noise runs to the folder to enable absolute
  "noise helps vs. clean" claims &mdash; the notebook will pick them up
  automatically.
* **Single- vs. multiple-feature scenarios** are aggregated together by default.
  Filter `df[df.scenario == "single"]` (or `"multi"`) before any section to
  analyse them separately; the multiple-feature scenario is still in progress.
* As seeds accumulate toward the full 20, re-run the notebook periodically &mdash;
  the confidence intervals will tighten and previously non-significant effects
  may cross into significance.""")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(REPO, f"6{phase}-Experiment-evaluation.ipynb")
    with open(out, "w") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", out, "cells=", len(cells))


if __name__ == "__main__":
    build("A")
    build("B")