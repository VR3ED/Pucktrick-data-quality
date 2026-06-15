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


def _is_baseline_path(path: str) -> bool:
    """True for the clean-baseline artifacts under a ``Baselines/`` sub-folder.

    Baseline runs (one per model, ``*_trial_Baseline_artifacts.json``) carry no
    noise/method/feature coordinates, so they must be kept out of the noisy-run
    loaders and handled separately via :func:`load_baselines`.
    """
    low = path.replace("\\", "/").lower()
    return "/baselines/" in low or "trial_baseline" in os.path.basename(low)


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
    n_baseline = 0
    for f in files:
        if _is_baseline_path(f):
            n_baseline += 1   # handled by load_baselines(), not here
            continue
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
        n_ok = n_files - n_bad - n_baseline
        if df.empty:
            print(f"[load] No noisy-run artifacts found under '{root}'.")
            print("       The notebook will run but every section stays empty "
                  "until experiments are written here.")
        else:
            print(f"[load] {n_ok}/{n_files} artifact files parsed "
                  f"({n_bad} skipped, {n_baseline} baselines handled separately) "
                  f"-> {len(df)} task-rows.")
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
# Clean baseline (0% noise) -- loaded from the Baselines/ sub-folder           #
# --------------------------------------------------------------------------- #

def load_baselines(root: str, verbose: bool = True) -> pd.DataFrame:
    """Load the clean-baseline artifacts under ``<root>/Baselines/``.

    These are the models trained on the *uncorrupted* dataset (one file per
    model, ``*_trial_Baseline_artifacts.json``). Returns one row per
    (model x task) with columns ``model, task, accuracy, f1, mcc, auc, path``.

    The baseline is the genuine clean reference the noise experiments are
    measured against; before it existed the notebooks fell back to the lowest
    available noise level as a proxy.
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*_artifacts.json"),
                             recursive=True))
    files = [f for f in files if _is_baseline_path(f)]
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        model = _model_from_path(f) or _model_from_type(d.get("model_type"))
        for task, mkey in (("binary", "metrics_binary"),
                           ("multiclass", "metrics_multiclass")):
            block = d.get(mkey, {}) or {}
            row = {"model": model, "task": task, "path": f}
            for met in _BASE_METRICS:
                row[met] = _to_float(block.get(met))
            if not np.isfinite(row["mcc"]):
                row["mcc"] = _to_float(d.get("mcc_bin" if task == "binary" else "mcc_mul"))
            rows.append(row)
    bl = pd.DataFrame(rows)
    if verbose:
        if bl.empty:
            print(f"[baseline] No clean baseline found under '{root}/Baselines'.")
        else:
            print(f"[baseline] loaded {len(files)} baseline file(s) -> "
                  f"models={sorted(bl['model'].dropna().unique())}")
    return bl


def baseline_value(baselines: pd.DataFrame, model: str, task: str,
                   metric: str) -> float:
    """Scalar clean-baseline value for (model, task, metric), or NaN if absent."""
    if baselines is None or baselines.empty:
        return float("nan")
    sel = baselines[(baselines["model"] == model) & (baselines["task"] == task)]
    if sel.empty or metric not in sel.columns:
        return float("nan")
    return float(sel[metric].iloc[0])


def baseline_delta_significance(df: pd.DataFrame, baselines: pd.DataFrame,
                                model: str, task: str, metric: str) -> pd.DataFrame:
    """Per (method, feature, noise%): one-sample test vs. the *clean* baseline.

    The baseline is a single deterministic value (no seeds), so we cannot pair
    by seed as in :func:`paired_delta_significance`. Instead we take the
    per-seed metric values at each noise cell and run a one-sample Wilcoxon
    signed-rank test of ``(value - baseline)`` against zero. This answers the
    headline question of the thesis directly: *does injecting this noise beat
    (or hurt) the model trained on clean data?*

    Columns: ``model, method, feature, noise_percentage, n, baseline,
    median_value, median_delta, mean_delta, p_value, significant``.
    """
    base = baseline_value(baselines, model, task, metric)
    sub = df[(df["model"] == model) & (df["task"] == task)].dropna(subset=[metric])
    if sub.empty or not np.isfinite(base):
        return pd.DataFrame()

    records = []
    for (method, feature, pct), g in sub.groupby(["method", "feature", "noise_percentage"]):
        vals = g[metric].values
        deltas = vals - base
        n = len(deltas)
        p_value = np.nan
        if n >= 2 and np.any(deltas != 0):
            try:
                p_value = float(stats.wilcoxon(deltas).pvalue)
            except ValueError:
                p_value = np.nan
        records.append({
            "model": model, "method": method, "feature": feature,
            "noise_percentage": pct, "n": n, "baseline": base,
            "median_value": float(np.median(vals)),
            "median_delta": float(np.median(deltas)),
            "mean_delta": float(np.mean(deltas)), "p_value": p_value,
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
        if _is_baseline_path(f):
            continue
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
        scenario = _scenario_from_path(f)
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
                "scenario": scenario,
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
    scenario, fi_feature, importance, is_corrupted``. ``is_corrupted`` flags the
    row whose reported feature is the one PuckTrick dirtied (name-normalised, so
    'Down/Up Ratio' matches the file's 'Down_Up Ratio').
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*_artifacts.json"),
                             recursive=True))
    rows = []
    for f in files:
        if _is_baseline_path(f):
            continue
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
        scenario = _scenario_from_path(f)
        # In the multiple-feature scenario the corrupted "feature" is a combo of
        # several names joined by '+', e.g. "FwdActDataPkts+DstPort+IdleMean".
        # Match each importance feature against ANY token of that combo.
        corrupted_tokens = {_norm_feature(t)
                            for t in str(coords["feature"]).split("+")}
        for feat, imp in fi.items():
            rows.append({
                "model": model, "method": coords["method"],
                "corrupted_feature": coords["feature"],
                "noise_percentage": coords["noise_percentage"], "seed": seed,
                "scenario": scenario,
                "fi_feature": feat, "importance": _to_float(imp),
                "is_corrupted": _norm_feature(feat) in corrupted_tokens,
            })
    return pd.DataFrame(rows)


def feature_rank_dataframe(fid: pd.DataFrame) -> pd.DataFrame:
    """Add a per-run importance ``rank`` column to a feature-importance table.

    Within each run -- identified by
    ``(model, method, corrupted_feature, noise_percentage, seed)`` -- the
    features are ranked by permutation importance, **rank 1 = most important**.
    Also adds ``n_features`` (how many features were ranked in that run, i.e. the
    worst possible rank). Returns a copy; the input is unchanged.

    This is the basis of the rank-drift plot: a feature's *absolute* importance
    is hard to compare across noise levels, but its *rank* among the other
    features is a stable, interpretable measure of how much the model relies on
    it relative to everything else.
    """
    if fid is None or fid.empty:
        return fid
    out = fid.copy()
    grp = ["model", "method", "corrupted_feature", "noise_percentage", "seed"]
    out["rank"] = (out.groupby(grp)["importance"]
                      .rank(ascending=False, method="min"))
    out["n_features"] = out.groupby(grp)["importance"].transform("size")
    return out


def baseline_feature_importance(root: str) -> pd.DataFrame:
    """Per-model feature importances of the clean baseline, with importance rank.

    Columns: ``model, fi_feature, importance, rank, n_features``. Used to draw a
    "clean" reference rank in the rank-drift plot.
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*_artifacts.json"),
                             recursive=True))
    files = [f for f in files if _is_baseline_path(f)]
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        fi = d.get("feature_importance", {}) or {}
        if not isinstance(fi, dict) or not fi:
            continue
        model = _model_from_path(f) or _model_from_type(d.get("model_type"))
        for feat, imp in fi.items():
            rows.append({"model": model, "fi_feature": feat,
                         "importance": _to_float(imp)})
    bfi = pd.DataFrame(rows)
    if bfi.empty:
        return bfi
    bfi["rank"] = (bfi.groupby("model")["importance"]
                      .rank(ascending=False, method="min"))
    bfi["n_features"] = bfi.groupby("model")["importance"].transform("size")
    return bfi


def baseline_feature_rank(baseline_fi: pd.DataFrame, model: str,
                          feature: str) -> float:
    """Clean-baseline importance rank of ``feature`` for ``model`` (NaN if absent)."""
    if baseline_fi is None or baseline_fi.empty:
        return float("nan")
    target = _norm_feature(feature)
    sel = baseline_fi[(baseline_fi["model"] == model)
                      & (baseline_fi["fi_feature"].map(_norm_feature) == target)]
    if sel.empty:
        return float("nan")
    return float(sel["rank"].iloc[0])


# --------------------------------------------------------------------------- #
# Binary confusion-matrix counts (TN / FP / FN / TP)                           #
# --------------------------------------------------------------------------- #

def _cm_bin_counts(cm) -> Optional[dict]:
    """Unpack a 2x2 ``cm_bin`` into TN/FP/FN/TP.

    The artifacts store ``cm_bin = [[TN, FP], [FN, TP]]`` (row 0 = actual
    benign/negative, row 1 = actual malicious/positive). Returns None if the
    matrix is not a well-formed 2x2.
    """
    if not isinstance(cm, list) or len(cm) != 2:
        return None
    try:
        (tn, fp), (fn, tp) = cm
    except (ValueError, TypeError):
        return None
    return {"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)}


def confusion_bin_dataframe(root: str) -> pd.DataFrame:
    """Long table of binary confusion-matrix counts across all noisy runs.

    Columns: ``model, method, feature, noise_percentage, seed, scenario,
    tn, fp, fn, tp, mcc``. Baseline runs are excluded (loaded separately by
    :func:`baseline_confusion_bin`).
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*_artifacts.json"),
                             recursive=True))
    rows = []
    for f in files:
        if _is_baseline_path(f):
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        counts = _cm_bin_counts(d.get("cm_bin"))
        if counts is None:
            continue
        coords = parse_label(d.get("label") or os.path.basename(f))
        model = _model_from_path(f) or _model_from_type(d.get("model_type"))
        row = {
            "model": model, "method": coords["method"],
            "feature": coords["feature"],
            "noise_percentage": coords["noise_percentage"],
            "seed": _seed_from_path(f), "scenario": _scenario_from_path(f),
            "mcc": _to_float((d.get("metrics_binary", {}) or {}).get("mcc",
                                                                     d.get("mcc_bin"))),
        }
        row.update(counts)
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_confusion_bin(root: str) -> pd.DataFrame:
    """Per-model binary confusion counts of the clean baseline.

    Columns: ``model, tn, fp, fn, tp, mcc``. One row per model.
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*_artifacts.json"),
                             recursive=True))
    files = [f for f in files if _is_baseline_path(f)]
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        counts = _cm_bin_counts(d.get("cm_bin"))
        if counts is None:
            continue
        model = _model_from_path(f) or _model_from_type(d.get("model_type"))
        row = {"model": model,
               "mcc": _to_float((d.get("metrics_binary", {}) or {}).get(
                   "mcc", d.get("mcc_bin")))}
        row.update(counts)
        rows.append(row)
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
                         baselines: Optional[pd.DataFrame] = None,
                         figsize_per=(5.2, 4.0), ncols: int = 3,
                         title_suffix: str = "", savepath: Optional[str] = None):
    """Line plots of ``metric`` vs. noise %, with shaded t-CI bands.

    One subplot per ``facet_by`` value; one coloured line per ``hue`` value.
    This is the central figure requested for the thesis: *for each feature, how
    does F1 / MCC evolve as the noise threshold grows, and how does that depend
    on the PuckTrick method used?* (Swap ``facet_by``/``hue`` to pivot the view.)

    If ``baselines`` is provided, the clean-baseline value for this
    (model, task, metric) is drawn as a horizontal dashed reference line in
    every subplot, so one can see at a glance whether each noise curve sits
    above or below the model trained on clean data.

    The legend is built from proxy handles covering *all* ``hue`` values across
    the whole figure (not just those present in the first subplot), so no curve
    is ever missing from it.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

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

    base_val = baseline_value(baselines, model, task, metric)
    has_base = np.isfinite(base_val)

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
                    color=palette[h])
            band = g.dropna(subset=["ci_low", "ci_high"])
            if not band.empty:
                ax.fill_between(band["noise_percentage"], band["ci_low"],
                                band["ci_high"], alpha=0.18, color=palette[h])
        if has_base:
            ax.axhline(base_val, ls="--", lw=1.4, color="0.35", zorder=0)
        ax.set_title(f"{facet_by} = {fac}", fontsize=11)
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.3)

    for j in range(len(facets), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Build the legend from proxy handles spanning EVERY hue value present
    # anywhere in the figure (reading handles off the first subplot alone would
    # drop any method/feature missing from that particular facet).
    handles = [Line2D([0], [0], marker="o", color=palette[h], label=str(h))
               for h in hue_vals]
    if has_base:
        handles.append(Line2D([0], [0], ls="--", color="0.35",
                              label=f"clean baseline ({base_val:.3f})"))
    if handles:
        fig.legend(handles=handles, title=hue.capitalize(),
                   loc="upper center", ncol=min(len(handles), 6),
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
                        delta_vs_baseline: bool = False,
                        baselines: Optional[pd.DataFrame] = None,
                        figsize=(9, 4.5), savepath: Optional[str] = None):
    """Heatmap of mean ``metric`` over (method x noise%).

    Three display modes:

    * default -> absolute mean metric per cell;
    * ``delta_vs_baseline=True`` (with ``baselines``) -> change vs. the genuine
      clean baseline scalar (diverging map, blue = beats clean, red = worse);
    * ``delta_vs_ref=True`` -> change vs. ``ref_pct`` / the lowest noise level
      (legacy proxy reference, used when no clean baseline is available).

    ``delta_vs_baseline`` takes precedence when a baseline value exists.
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

    base_val = baseline_value(baselines, model, task, metric)
    if delta_vs_baseline and np.isfinite(base_val):
        pivot = pivot - base_val
        cmap, center = "RdBu_r", 0.0
        title += f"\n(delta vs. clean baseline = {base_val:.3f})"
    elif delta_vs_ref:
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


def plot_feature_rank_drift(fid: pd.DataFrame, model: str,
                            baseline_fi: Optional[pd.DataFrame] = None,
                            ncols: int = 3, figsize_per=(5.2, 4.0),
                            annotate: bool = True, y_max: Optional[int] = None,
                            savepath: Optional[str] = None):
    """Rank-drift plot: how the corrupted feature's *importance rank* moves.

    One subplot **per corrupted feature**. In each subplot the y-axis is the
    permutation-importance **rank (position)** of *that same feature* (rank 1 =
    most important, at the top via an inverted axis), the x-axis is the noise
    level, and there is **one line per PuckTrick method**. Each point is
    annotated with the rank it sits on, so the dot and its label always agree
    (e.g. "missingness pushes Protocol from position 5 down to position 8").

    **Rank = rank-of-mean-importance.** For each (method, noise level) we first
    average the permutation importance over seeds for *every* feature, then rank
    those averages. This yields a genuine integer *position* in the importance
    ordering -- unlike averaging the per-seed ranks, which produces a misleading
    fractional number that no longer corresponds to any real position.

    Subplots are ordered by the feature's **clean-baseline importance** (the most
    important feature in the baseline first), so the grid reads top-left to
    bottom-right in decreasing baseline relevance. A dashed line marks each
    feature's baseline rank.

    ``y_max`` fixes the worst rank shown on the axis (default: ``max(10, observed
    max)`` -- i.e. 1..10 for the ~8-9 features of the single-feature scenario,
    auto-expanded if a run ranks more features). Only features that appear among
    the reported importances are shown (row-level ``duplicated``/``labels`` runs
    corrupt a column that is not a model input, so they are skipped).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    if fid is None or fid.empty:
        print("[rank-drift] No feature-importance data.")
        return None

    model_fi = fid[fid["model"] == model].copy()
    if model_fi.empty or not model_fi["is_corrupted"].any():
        print(f"[rank-drift] No corrupted-feature importances for model={model} "
              "(e.g. only row-level 'duplicated'/'labels' runs).")
        return None

    # 1) Average importance over seeds for EVERY feature within each run cell,
    #    keyed by the full corrupted-feature label (a single name in the single
    #    scenario, a '+'-joined combo in the multi scenario).
    imp = (model_fi.groupby(["method", "corrupted_feature", "noise_percentage",
                             "fi_feature"])
                   .agg(importance=("importance", "mean"),
                        is_corrupted=("is_corrupted", "first"))
                   .reset_index())
    # 2) Rank the averaged importances -> a true integer position per cell.
    imp["rank"] = (imp.groupby(["method", "corrupted_feature", "noise_percentage"])
                      ["importance"].rank(ascending=False, method="min"))

    # 3) Keep only the dirtied features; facet by the actual feature name. In the
    #    multi scenario a feature may be dirtied inside several combos at the same
    #    (method, noise), so average its position across them (then it can be
    #    fractional -- the annotation matches whatever is plotted).
    plot_df = imp[imp["is_corrupted"]]
    agg = (plot_df.groupby(["fi_feature", "method", "noise_percentage"])
                  ["rank"].mean().reset_index())

    # Order facets by clean-baseline importance (rank 1 first); unknowns last.
    def _facet_key(feat):
        br = baseline_feature_rank(baseline_fi, model, feat)
        return (br if np.isfinite(br) else float("inf"), str(feat))
    feats = sorted(agg["fi_feature"].dropna().unique(), key=_facet_key)

    methods = _ordered_hue_values(agg["method"].dropna().unique())
    palette = dict(zip(methods, sns.color_palette("tab10", max(len(methods), 3))))

    observed_max = int(np.ceil(np.nanmax(agg["rank"]))) if len(agg) else 10
    if y_max is None:
        y_max = max(10, observed_max)

    def _fmt(v):  # keep the dot and its label identical
        return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    ncols = max(1, min(ncols, len(feats)))
    nrows = math.ceil(len(feats) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)

    for i, feat in enumerate(feats):
        ax = axes[i // ncols][i % ncols]
        for m in methods:
            g = agg[(agg["fi_feature"] == feat) & (agg["method"] == m)]
            g = g.sort_values("noise_percentage")
            if g.empty:
                continue
            ax.plot(g["noise_percentage"], g["rank"], marker="o", ms=6,
                    color=palette[m])
            if annotate:
                for x, y in zip(g["noise_percentage"], g["rank"]):
                    ax.annotate(_fmt(y), (x, y), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=8,
                                color=palette[m])
        base_rank = baseline_feature_rank(baseline_fi, model, feat)
        title = f"feature = {feat}"
        if np.isfinite(base_rank):
            ax.axhline(base_rank, ls="--", lw=1.3, color="0.35", zorder=0)
            title += f"  (baseline #{base_rank:.0f})"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel("Importance rank (1 = most important)")
        ax.set_ylim(y_max + 0.5, 0.5)   # inverted: rank 1 on top, y_max at bottom
        step = 1 if y_max <= 12 else 2
        ax.set_yticks(range(1, y_max + 1, step))
        ax.grid(True, alpha=0.3)

    for j in range(len(feats), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles = [Line2D([0], [0], marker="o", color=palette[m], label=str(m))
               for m in methods]
    if baseline_fi is not None and not baseline_fi.empty:
        handles.append(Line2D([0], [0], ls="--", color="0.35",
                              label="clean-baseline rank"))
    fig.legend(handles=handles, title="Method", loc="upper center",
               ncol=min(len(handles), 6), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Importance-rank drift of the corrupted feature - {model.upper()}",
                 y=1.06, fontsize=14)
    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"[rank-drift] saved -> {savepath}")
    return fig


# --------------------------------------------------------------------------- #
# Baseline-vs-experiment confusion breakdown (TN/FP/TP/FN), per noise level    #
# --------------------------------------------------------------------------- #

def _fmt_signed(v, decimals=0):
    """Signed delta string, e.g. '+57' / '-0.0074'."""
    if not np.isfinite(v):
        return ""
    if decimals == 0:
        return f"{v:+,.0f}"
    return f"{v:+.{decimals}f}"


def plot_confusion_breakdown(cbin: pd.DataFrame, baseline_cbin: pd.DataFrame,
                             model: str, method: str, feature: str,
                             noise_percentage: float, figsize=(13, 5.5),
                             savepath: Optional[str] = None):
    """Baseline-vs-experiment stacked confusion bars for one (model, method,
    feature, noise%) cell.

    Reproduces the requested figure: two side-by-side panels (Baseline |
    Experiment), each with two stacked bars -- *Benign (actual)* split into
    correct=**TN** and wrong=**FP**, and *Malicious (actual)* split into
    correct=**TP** and wrong=**FN**. Every count is the **mean over the random
    seeds** for that cell. The footer reports the change vs. the clean baseline
    for **MCC** (delta-MCC, as requested instead of delta-F1) and for the false
    positives (delta-FP).

    Bars are annotated with the mean counts, and the wrong/correct segment labels
    additionally carry the signed delta vs. the baseline for that same segment
    (TN/FP/TP/FN), so "how far each mean is from the baseline" is read directly
    off the figure.

    Returns the Figure, or None if the cell or the baseline is missing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cell = cbin[(cbin["model"] == model) & (cbin["method"] == method)
                & (cbin["feature"] == feature)
                & (cbin["noise_percentage"] == noise_percentage)]
    if cell.empty:
        print(f"[confusion] No runs for {model}/{method}/{feature}@{noise_percentage:g}%.")
        return None
    bsel = baseline_cbin[baseline_cbin["model"] == model] if baseline_cbin is not None else None
    if bsel is None or bsel.empty:
        print(f"[confusion] No clean baseline for model={model}.")
        return None

    n_seeds = cell["seed"].nunique()
    exp = {k: float(cell[k].mean()) for k in ("tn", "fp", "fn", "tp")}
    exp_mcc = float(cell["mcc"].mean())
    base = {k: float(bsel[k].iloc[0]) for k in ("tn", "fp", "fn", "tp")}
    base_mcc = float(bsel["mcc"].iloc[0])

    c_correct, c_wrong = "#2a8fbd", "#a8c83a"   # blue = correct, green = wrong
    x = [0, 1]
    xlabels = ["Benign\n(actual)", "Malicious\n(actual)"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    def _panel(ax, counts, title):
        # Benign bar: correct=TN (bottom), wrong=FP (top).
        ax.bar(x[0], counts["tn"], color=c_correct)
        ax.bar(x[0], counts["fp"], bottom=counts["tn"], color=c_wrong)
        # Malicious bar: correct=TP (bottom), wrong=FN (top).
        ax.bar(x[1], counts["tp"], color=c_correct)
        ax.bar(x[1], counts["fn"], bottom=counts["tp"], color=c_wrong)
        ax.set_xticks(x); ax.set_xticklabels(xlabels)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        return counts

    seg_names = {"tn": "TN · True Negatives", "fp": "FP · False Positives",
                 "tp": "TP · True Positives", "fn": "FN · False Negatives"}

    def _annot(ax, counts, deltas):
        # counts: dict tn/fp/fn/tp; deltas: same keys or None for baseline panel.
        def lab(seg, xpos, y_bottom, val):
            txt = f"{seg_names[seg]}\n{val:,.0f}"
            if deltas is not None and np.isfinite(deltas[seg]):
                txt += f" ({_fmt_signed(deltas[seg])})"
            ax.text(xpos, y_bottom + val / 2, txt, ha="center", va="center",
                    fontsize=8.5, color="white" if seg in ("tn", "tp") else "black")
        lab("tn", x[0], 0, counts["tn"])
        lab("fp", x[0], counts["tn"], counts["fp"])
        lab("tp", x[1], 0, counts["tp"])
        lab("fn", x[1], counts["tp"], counts["fn"])

    _panel(axes[0], base, f"Baseline  (MCC={base_mcc:.4f})")
    _annot(axes[0], base, None)
    _panel(axes[1], exp, f"Experiment  (MCC={exp_mcc:.4f})")
    _annot(axes[1], exp, {k: exp[k] - base[k] for k in exp})

    axes[0].set_ylabel("Number of samples")
    legend = [Patch(facecolor=c_correct, label="Correct prediction"),
              Patch(facecolor=c_wrong, label="Wrong prediction")]
    axes[1].legend(handles=legend, loc="upper right", fontsize=9)

    fig.suptitle(f"{model.upper()} — {method} on '{feature}' @ {noise_percentage:g}%",
                 fontsize=14, fontweight="bold")
    d_mcc = exp_mcc - base_mcc
    d_fp = exp["fp"] - base["fp"]
    dcol = "green" if d_mcc >= 0 else "firebrick"
    fig.text(0.5, -0.02,
             f"$\\Delta$MCC = {_fmt_signed(d_mcc, 4)}   |   "
             f"$\\Delta$FP = {_fmt_signed(d_fp)}   "
             f"(means over {n_seeds} seeds)",
             ha="center", fontsize=11, style="italic", color=dcol)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"[confusion] saved -> {savepath}")
    return fig


def confusion_breakdown_table(cbin: pd.DataFrame, baseline_cbin: pd.DataFrame,
                              model: str, method: str,
                              conf: float = 0.95) -> pd.DataFrame:
    """Tidy mean(+/-CI) TN/FP/FN/TP and their baseline deltas, per noise level.

    For one (model, method) this returns one row per (feature, noise%) with the
    seed-averaged confusion counts, their Student-t CI half-widths, and the
    signed delta of each mean vs. the clean baseline. Handy as the numeric
    companion of :func:`plot_confusion_breakdown` and for export to the thesis.
    """
    sub = cbin[(cbin["model"] == model) & (cbin["method"] == method)].copy()
    if sub.empty:
        return pd.DataFrame()
    base = baseline_cbin[baseline_cbin["model"] == model] if baseline_cbin is not None else None
    base_row = None if base is None or base.empty else base.iloc[0]

    out = []
    for (feature, pct), g in sub.groupby(["feature", "noise_percentage"]):
        rec = {"model": model, "method": method, "feature": feature,
               "noise_percentage": pct, "n": g["seed"].nunique()}
        for seg in ("tn", "fp", "fn", "tp", "mcc"):
            mean = float(g[seg].mean())
            std = float(g[seg].std(ddof=1)) if len(g) > 1 else float("nan")
            rec[f"{seg}_mean"] = mean
            rec[f"{seg}_ci"] = _ci_half_width(std, len(g), conf)
            if base_row is not None:
                rec[f"{seg}_delta"] = mean - float(base_row[seg])
        out.append(rec)
    res = pd.DataFrame(out)
    if not res.empty:
        res = res.sort_values(["feature", "noise_percentage"]).reset_index(drop=True)
    return res


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