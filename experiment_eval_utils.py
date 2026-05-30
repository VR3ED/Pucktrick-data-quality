"""
experiment_eval_utils.py
========================

Shared, *dynamic* evaluation utilities for the CAVAS noise-injection experiments
(see CLAUDE.md). Both ``6A-Experiment-evaluation.ipynb`` (reduced dataset,
Experiment A) and ``6B-Experiment-evaluation.ipynb`` (complete dataset,
Experiment B) import these helpers.

Design goals
------------
1. **Fully dynamic.** Nothing is hard-coded about *which* seeds, methods,
   features or noise levels exist. The loader walks the experiment directory
   tree, parses every ``*_artifacts.json`` it finds and builds a tidy
   long-format :class:`pandas.DataFrame`. Drop new runs into the folder and a
   single *Restart & Run All* refreshes every table, plot and confidence
   interval.
2. **Honest uncertainty.** Because results are averaged over random seeds, every
   aggregate is reported with a Student-t confidence interval whose width adapts
   to the number of seeds actually present for that cell. Sparse cells are never
   hidden -- they are flagged in the coverage table.
3. **Reusable across both phases.** Experiment B adds a ``scenario`` dimension
   (single- vs. multiple-feature corruption) and a ``labels_experiment``
   sub-folder; the loader records these so the same code serves both notebooks.

Artifact schema (one ``*_artifacts.json`` per trained model)
-----------------------------------------------------------
Top-level keys::

    model               # checkpoint path (string)
    model_type          # "CNN-LSTM" | "TabNet"
    label               # e.g. "Experiment_missing_Protocol_10.0"
    mcc_bin, mcc_mul, mean_mcc, val_loss
    cm_bin              # 2x2 confusion matrix (binary task)
    cm_mul              # KxK confusion matrix (multiclass task)
    feature_importance  # {feature_name: importance}  (dict!)
    params              # best hyper-parameters
    metrics_binary      # {task, accuracy, f1, mcc, auc}
    metrics_multiclass  # {task, accuracy, f1, mcc, auc}

The experimental coordinates (model / method / corrupted feature / noise %) are
**not** stored as fields; they are encoded in the file name / ``label`` and are
recovered by :func:`parse_label`.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Presentation constants                                                       #
# --------------------------------------------------------------------------- #

#: Known PuckTrick noise types (used to disambiguate the label and to order plots).
KNOWN_METHODS = ["missing", "outliers", "noise", "duplicated", "labels"]
METHOD_ORDER = KNOWN_METHODS

#: Human-readable labels for the metrics we track.
METRIC_LABELS = {
    "mcc": "MCC",
    "f1": "F1-score",
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
}

#: Short prose describing each noise type (used in the notebook narrative).
METHOD_DESCRIPTIONS = {
    "missing": "values blanked out (missingness) on the target feature",
    "outliers": "extreme out-of-distribution values injected into the feature",
    "noise": "additive random perturbation of the feature values",
    "duplicated": "whole rows duplicated (row-level corruption)",
    "labels": "corruption of the target label itself",
}

#: Metrics available inside each ``metrics_<task>`` block.
_BASE_METRICS = ["accuracy", "f1", "mcc", "auc"]


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #

def _to_float(x) -> float:
    """Best-effort float conversion (returns NaN on failure)."""
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def _norm_feature(s) -> str:
    """Normalise a feature name for matching ('Down/Up Ratio' == 'Down_Up Ratio')."""
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_label(text: str) -> dict:
    """Recover (method, feature, noise %) from a file name or ``label`` string.

    Accepts either the bare label (``Experiment_missing_Protocol_10.0``) or a
    full artifact filename. Feature names may contain underscores *and* spaces
    (e.g. ``Down_Up Ratio``), so we anchor on the leading method token and the
    trailing numeric percentage and treat everything in between as the feature.
    """
    core = os.path.basename(text)
    core = re.sub(r"_artifacts\.json$", "", core)
    core = re.sub(r"^(cnn_lstm|tabnet)_trial_", "", core)
    core = re.sub(r"^Experiment_", "", core)
    m = re.match(r"^(?P<method>[A-Za-z]+)_(?P<feature>.+)_(?P<pct>\d+(?:\.\d+)?)$", core)
    if not m:
        return {"method": None, "feature": None, "noise_percentage": float("nan")}
    return {
        "method": m.group("method"),
        "feature": m.group("feature"),
        "noise_percentage": _to_float(m.group("pct")),
    }


def _model_from_path(path: str) -> Optional[str]:
    """Canonical model id from the filename prefix ('cnn_lstm' | 'tabnet')."""
    base = os.path.basename(path)
    if base.startswith("cnn_lstm"):
        return "cnn_lstm"
    if base.startswith("tabnet"):
        return "tabnet"
    return None


def _model_from_type(mtype: Optional[str]) -> Optional[str]:
    if not mtype:
        return None
    t = mtype.lower()
    if "cnn" in t or "lstm" in t:
        return "cnn_lstm"
    if "tabnet" in t:
        return "tabnet"
    return t


def _scenario_from_path(path: str) -> str:
    """Derive the Experiment-B scenario from the path (``single``/``multi``/``na``)."""
    low = path.replace("\\", "/").lower()
    if "multiple_feature" in low or "pucktrick_on_multiple" in low:
        return "multi"
    if "single_feature" in low or "pucktrick_on_single" in low:
        return "single"
    return "na"


def _seed_from_path(path: str) -> Optional[int]:
    """Recover the seed from an ``experiment_rs{seed}`` path component."""
    m = re.search(r"experiment_rs(\d+)", path.replace("\\", "/"))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def load_experiment_dataframe(root: str, verbose: bool = True) -> pd.DataFrame:
    """Walk ``root`` recursively and return a tidy long-format DataFrame.

    One row per (artifact file x task). Columns:

    ``model, method, feature, noise_percentage, seed, scenario,
    is_labels_variant, task, accuracy, f1, mcc, auc, n_classes, path``.

    Corrupt or partially-written JSON files are skipped (and counted) rather than
    aborting the whole load, so a re-run never breaks because a single run is
    mid-write.
    """
    pattern = os.path.join(root, "**", "*_artifacts.json")
    files = sorted(glob.glob(pattern, recursive=True))

    rows = []
    n_bad = 0
    for f in files:
        try:
            with open(f, "r") as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            n_bad += 1
            continue

        coords = parse_label(d.get("label") or os.path.basename(f))
        model = _model_from_path(f) or _model_from_type(d.get("model_type"))
        scenario = _scenario_from_path(f)
        is_labels = (coords["method"] == "labels") or (
            "labels_experiment" in f.replace("\\", "/").lower())
        seed = _seed_from_path(f)

        for task, mkey, cmkey in (("binary", "metrics_binary", "cm_bin"),
                                  ("multiclass", "metrics_multiclass", "cm_mul")):
            block = d.get(mkey, {}) or {}
            cm = d.get(cmkey)
            n_classes = len(cm) if isinstance(cm, (list, tuple)) else np.nan
            row = {
                "model": model,
                "method": coords["method"],
                "feature": coords["feature"],
                "noise_percentage": coords["noise_percentage"],
                "seed": seed,
                "scenario": scenario,
                "is_labels_variant": is_labels,
                "task": task,
                "n_classes": n_classes,
                "path": f,
            }
            for met in _BASE_METRICS:
                row[met] = _to_float(block.get(met))
            # Fallback to the top-level mcc fields if the block is missing one.
            if not np.isfinite(row["mcc"]):
                row["mcc"] = _to_float(d.get("mcc_bin" if task == "binary" else "mcc_mul"))
            rows.append(row)

    df = pd.DataFrame(rows)
    if verbose:
        n_files = len(files)
        n_ok = n_files - n_bad
        if df.empty:
            print(f"[load] No artifacts found under '{root}'.")
            print("       The notebook will run but every section stays empty "
                  "until experiments are written here.")
        else:
            print(f"[load] {n_ok}/{n_files} artifact files parsed "
                  f"({n_bad} skipped) -> {len(df)} task-rows.")
            print(f"       models   : {sorted(df['model'].dropna().unique())}")
            print(f"       methods  : {sorted(df['method'].dropna().unique())}")
            print(f"       features : {sorted(df['feature'].dropna().unique())}")
            print(f"       noise %  : {sorted(df['noise_percentage'].dropna().unique())}")
            print(f"       seeds    : {sorted(df['seed'].dropna().unique())}")
            if (df['scenario'] != 'na').any():
                print(f"       scenarios: {sorted(df['scenario'].dropna().unique())}")
    return df


# --------------------------------------------------------------------------- #
# Coverage / data-quality of the *evaluation* itself                          #
# --------------------------------------------------------------------------- #

def coverage_table(df: pd.DataFrame, model: str, task: str = "binary",
                   index: Sequence[str] = ("method", "feature"),
                   columns: str = "noise_percentage") -> pd.DataFrame:
    """How many seeds back each (method, feature, noise%) cell?

    This is the empirical basis of every confidence interval below. Cells with a
    single seed cannot have a CI and should be read with caution in the thesis.
    """
    sub = df[(df["model"] == model) & (df["task"] == task)]
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index=list(index), columns=columns, values="seed",
        aggfunc=pd.Series.nunique, fill_value=0,
    )
    return pivot.astype(int)


# --------------------------------------------------------------------------- #
# Aggregation with Student-t confidence intervals                             #
# --------------------------------------------------------------------------- #

def _ci_half_width(std: float, n: int, conf: float) -> float:
    """Half-width of a two-sided Student-t CI for the mean."""
    if n is None or n < 2 or not np.isfinite(std):
        return np.nan
    sem = std / math.sqrt(n)
    t = stats.t.ppf(0.5 + conf / 2.0, df=n - 1)
    return t * sem


def aggregate_with_ci(df: pd.DataFrame, group_cols: Sequence[str],
                      value_col: str, conf: float = 0.95) -> pd.DataFrame:
    """Group, then return mean / std / n / sem / CI for ``value_col``.

    The CI half-width adapts to the number of seeds in each group, so adding
    seeds automatically tightens the interval on the next run.
    """
    group_cols = list(group_cols)
    sub = df.dropna(subset=[value_col]).copy()
    if sub.empty:
        return pd.DataFrame(
            columns=group_cols + ["mean", "std", "n", "sem", "ci_half",
                                  "ci_low", "ci_high"])

    g = sub.groupby(group_cols)[value_col]
    out = g.agg(mean="mean", std=lambda s: s.std(ddof=1), n="count").reset_index()
    out["sem"] = out["std"] / np.sqrt(out["n"])
    out["ci_half"] = [
        _ci_half_width(s, int(n), conf) for s, n in zip(out["std"], out["n"])
    ]
    out["ci_low"] = out["mean"] - out["ci_half"]
    out["ci_high"] = out["mean"] + out["ci_half"]
    return out


# --------------------------------------------------------------------------- #
# Reference level & paired significance vs. the cleanest available run        #
# --------------------------------------------------------------------------- #

def reference_level(df: pd.DataFrame) -> Optional[float]:
    """Pick the reference noise level.

    Prefers a genuine clean baseline (0%) if present; otherwise falls back to
    the *lowest* available noise percentage, which acts as a near-clean proxy.
    Callers should label it accordingly in the narrative.
    """
    pcts = sorted(df["noise_percentage"].dropna().unique())
    if not pcts:
        return None
    return 0.0 if 0.0 in pcts else pcts[0]


def paired_delta_significance(df: pd.DataFrame, model: str, task: str,
                              metric: str, ref_pct: float) -> pd.DataFrame:
    """Per (method, feature, noise%): paired test vs. the reference level.

    For every (method, feature) we pair each seed's metric at a given noise
    level with the *same seed's* metric at ``ref_pct`` and run a Wilcoxon
    signed-rank test on the per-seed differences. Reported columns:
    ``n_pairs, median_delta, mean_delta, p_value, significant``.

    Pairing by seed removes seed-to-seed variance and makes the "does this
    noise level move the metric?" question a proper repeated-measures test --
    exactly what is needed to claim an *improvement* (or degradation) in the
    thesis rather than reading it off overlapping error bars.
    """
    sub = df[(df["model"] == model) & (df["task"] == task)].dropna(subset=[metric])
    if sub.empty:
        return pd.DataFrame()

    records = []
    for (method, feature), grp in sub.groupby(["method", "feature"]):
        ref = grp[grp["noise_percentage"] == ref_pct][["seed", metric]]
        ref = ref.dropna(subset=["seed"]).rename(columns={metric: "ref"})
        ref = ref.set_index("seed")["ref"]
        if ref.empty:
            continue
        for pct, pg in grp.groupby("noise_percentage"):
            if pct == ref_pct:
                continue
            cur = pg.dropna(subset=["seed"]).set_index("seed")[metric]
            common = ref.index.intersection(cur.index)
            if len(common) == 0:
                continue
            deltas = (cur.loc[common] - ref.loc[common]).values
            n_pairs = len(deltas)
            median_delta = float(np.median(deltas))
            mean_delta = float(np.mean(deltas))
            p_value = np.nan
            if n_pairs >= 2 and np.any(deltas != 0):
                try:
                    p_value = float(stats.wilcoxon(deltas).pvalue)
                except ValueError:
                    p_value = np.nan
            records.append({
                "method": method, "feature": feature, "noise_percentage": pct,
                "n_pairs": n_pairs, "median_delta": median_delta,
                "mean_delta": mean_delta, "p_value": p_value,
                "significant": (p_value < 0.05) if np.isfinite(p_value) else False,
            })
    res = pd.DataFrame(records)
    if not res.empty:
        res = res.sort_values(["method", "feature", "noise_percentage"]).reset_index(drop=True)
    return res


# --------------------------------------------------------------------------- #
# Per-class behaviour, derived from the multiclass confusion matrix           #
# --------------------------------------------------------------------------- #

def per_class_dataframe(root: str) -> pd.DataFrame:
    """Long per-class precision/recall/F1, computed from ``cm_mul``.

    The artifacts do not store a per-class report, but they do store the
    multiclass confusion matrix, from which per-class metrics follow directly:
    ``recall_i = cm[i,i]/row_sum_i``, ``precision_i = cm[i,i]/col_sum_i``.
    Classes are identified by their integer index (labels are not stored).
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*_artifacts.json"),
                             recursive=True))
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        cm = d.get("cm_mul")
        if not isinstance(cm, list) or not cm:
            continue
        cm = np.asarray(cm, dtype=float)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            continue
        coords = parse_label(d.get("label") or os.path.basename(f))
        model = _model_from_path(f) or _model_from_type(d.get("model_type"))
        seed = _seed_from_path(f)
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        diag = np.diag(cm)
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.where(row_sums > 0, diag / row_sums, np.nan)
            precision = np.where(col_sums > 0, diag / col_sums, np.nan)
            f1 = np.where((precision + recall) > 0,
                          2 * precision * recall / (precision + recall), np.nan)
        for i in range(cm.shape[0]):
            rows.append({
                "model": model, "method": coords["method"],
                "feature": coords["feature"],
                "noise_percentage": coords["noise_percentage"], "seed": seed,
                "class": int(i), "precision": float(precision[i]),
                "recall": float(recall[i]), "f1": float(f1[i]),
                "support": float(row_sums[i]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Feature-importance long table                                               #
# --------------------------------------------------------------------------- #

def feature_importance_dataframe(root: str) -> pd.DataFrame:
    """Long table of permutation feature importances across all runs.

    Columns: ``model, method, corrupted_feature, noise_percentage, seed,
    fi_feature, importance, is_corrupted``. ``is_corrupted`` flags the row whose
    reported feature is the one PuckTrick dirtied (name-normalised, so
    'Down/Up Ratio' matches the file's 'Down_Up Ratio').
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*_artifacts.json"),
                             recursive=True))
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        fi = d.get("feature_importance", {}) or {}
        if not isinstance(fi, dict):
            continue
        coords = parse_label(d.get("label") or os.path.basename(f))
        model = _model_from_path(f) or _model_from_type(d.get("model_type"))
        seed = _seed_from_path(f)
        corrupted_norm = _norm_feature(coords["feature"])
        for feat, imp in fi.items():
            rows.append({
                "model": model, "method": coords["method"],
                "corrupted_feature": coords["feature"],
                "noise_percentage": coords["noise_percentage"], "seed": seed,
                "fi_feature": feat, "importance": _to_float(imp),
                "is_corrupted": _norm_feature(feat) == corrupted_norm,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def _ordered_hue_values(values: Iterable[str]) -> list:
    """Order hue categories: known methods first (METHOD_ORDER), then the rest."""
    values = list(dict.fromkeys(values))
    known = [m for m in METHOD_ORDER if m in values]
    rest = sorted([v for v in values if v not in METHOD_ORDER], key=str)
    return known + rest


def plot_metric_vs_noise(df: pd.DataFrame, model: str, task: str, metric: str,
                         conf: float = 0.95, facet_by: str = "feature",
                         hue: str = "method", extra_filter: Optional[dict] = None,
                         figsize_per=(5.2, 4.0), ncols: int = 3,
                         title_suffix: str = "", savepath: Optional[str] = None):
    """Line plots of ``metric`` vs. noise %, with shaded t-CI bands.

    One subplot per ``facet_by`` value; one coloured line per ``hue`` value.
    This is the central figure requested for the thesis: *for each feature, how
    does F1 / MCC evolve as the noise threshold grows, and how does that depend
    on the PuckTrick method used?* (Swap ``facet_by``/``hue`` to pivot the view.)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = df[(df["model"] == model) & (df["task"] == task)].copy()
    if extra_filter:
        for k, v in extra_filter.items():
            sub = sub[sub[k] == v]
    if sub.empty:
        print(f"[plot] No data for model={model}, task={task}"
              f"{', ' + str(extra_filter) if extra_filter else ''}.")
        return None

    agg = aggregate_with_ci(sub, [facet_by, hue, "noise_percentage"], metric, conf)
    if agg.empty:
        print(f"[plot] Nothing to aggregate for metric '{metric}'.")
        return None

    facets = sorted(agg[facet_by].dropna().unique(), key=str)
    hue_vals = _ordered_hue_values(agg[hue].dropna().unique())
    palette = dict(zip(hue_vals, sns.color_palette("tab10", max(len(hue_vals), 3))))

    ncols = max(1, min(ncols, len(facets)))
    nrows = math.ceil(len(facets) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)

    for i, fac in enumerate(facets):
        ax = axes[i // ncols][i % ncols]
        for h in hue_vals:
            g = agg[(agg[facet_by] == fac) & (agg[hue] == h)].sort_values("noise_percentage")
            if g.empty:
                continue
            ax.plot(g["noise_percentage"], g["mean"], marker="o", ms=5,
                    label=str(h), color=palette[h])
            band = g.dropna(subset=["ci_low", "ci_high"])
            if not band.empty:
                ax.fill_between(band["noise_percentage"], band["ci_low"],
                                band["ci_high"], alpha=0.18, color=palette[h])
        ax.set_title(f"{facet_by} = {fac}", fontsize=11)
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.3)

    for j in range(len(facets), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title=hue.capitalize(),
                   loc="upper center", ncol=min(len(labels), 6),
                   bbox_to_anchor=(0.5, 1.02))
    metric_name = METRIC_LABELS.get(metric, metric)
    fig.suptitle(f"{metric_name} vs. noise level - {model.upper()} / {task}"
                 f"{(' - ' + title_suffix) if title_suffix else ''}",
                 y=1.06, fontsize=14)
    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"[plot] saved -> {savepath}")
    return fig


def plot_metric_heatmap(df: pd.DataFrame, model: str, task: str, metric: str,
                        feature: Optional[str] = None, delta_vs_ref: bool = False,
                        ref_pct: Optional[float] = None,
                        figsize=(9, 4.5), savepath: Optional[str] = None):
    """Heatmap of mean ``metric`` over (method x noise%).

    If ``delta_vs_ref`` is True, cells show the change relative to ``ref_pct``
    (defaults to the reference level), which makes "noise helped / hurt"
    immediately legible (diverging colormap centred at zero).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = df[(df["model"] == model) & (df["task"] == task)].copy()
    if feature is not None:
        sub = sub[sub["feature"] == feature]
    if sub.empty:
        print(f"[heatmap] No data for model={model}, task={task}, feature={feature}.")
        return None

    agg = aggregate_with_ci(sub, ["method", "noise_percentage"], metric)
    if agg.empty:
        print("[heatmap] Nothing to aggregate.")
        return None
    pivot = agg.pivot(index="method", columns="noise_percentage", values="mean")
    pivot = pivot.reindex([m for m in METHOD_ORDER if m in pivot.index]
                          + [m for m in pivot.index if m not in METHOD_ORDER])

    cmap, center, fmt = "viridis", None, ".3f"
    title = f"{METRIC_LABELS.get(metric, metric)} - {model.upper()} / {task}"
    if feature is not None:
        title += f" - feature '{feature}'"

    if delta_vs_ref:
        rp = ref_pct if ref_pct is not None else reference_level(sub)
        if rp is not None and rp in pivot.columns:
            pivot = pivot.sub(pivot[rp], axis=0)
            cmap, center = "RdBu_r", 0.0
            title += f"\n(delta vs. {rp:g}% reference)"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, center=center,
                linewidths=0.5, cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
                ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("PuckTrick method")
    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"[heatmap] saved -> {savepath}")
    return fig


# --------------------------------------------------------------------------- #
# Export helpers (LaTeX / CSV for the thesis)                                  #
# --------------------------------------------------------------------------- #

def export_aggregate(df: pd.DataFrame, metric: str, outdir: str, conf: float = 0.95,
                     group_cols=("model", "task", "method", "feature", "noise_percentage")):
    """Write a tidy aggregated CSV (mean +/- CI) for one metric, return the path."""
    os.makedirs(outdir, exist_ok=True)
    agg = aggregate_with_ci(df, list(group_cols), metric, conf)
    path = os.path.join(outdir, f"aggregate_{metric}.csv")
    agg.to_csv(path, index=False)
    print(f"[export] {metric}: {len(agg)} rows -> {path}")
    return agg, path


def format_mean_ci(mean: float, ci_half: float, decimals: int = 3) -> str:
    """Render ``mean +/- ci`` for tables; falls back to ``mean`` when no CI."""
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(ci_half):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {ci_half:.{decimals}f}"