# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CAVAS** — a thesis project studying the impact of data quality degradation on IDS (Intrusion Detection System) models. The pipeline injects controlled noise into the CAVAS network traffic dataset using [PuckTrick](https://github.com/andreamaurino/pucktrick), then evaluates how two complementary models handle it:

- **TabNet** (`pytorch_tabnet`) — tabular classifier, input shape `(n_samples, n_features)` (2D)
- **CNN-LSTM** (`CNNLSTMMultiTask`) — time-series classifier, input shape `(n_samples, window_size, n_features)` (3D)

Both models are **multi-task**: they simultaneously predict a binary label (`label_generic`: benign/malicious) and a multiclass label (`Label`: attack type).

The primary evaluation metric is **MCC** (Matthews Correlation Coefficient), reported separately for binary/multiclass tasks.

## Thesis Helper

All the thesis chapeters written so far are insede the `temp-latex` folder. Those require to be:
1. Be writtent in latex
2. Use english as language to write down any further chapters
3. Use academic language

our main goal for this thesis will be developing following experiment:

- **Experiment A**: **Experiments performed using a reduced dataset**: These experiments serve as a proof of concept, providing evidence that the introduction of controlled noise into the dataset may positively influence the effectiveness of deep learning-based Intrusion Detection Systems (IDSs). The main objective of this phase was to investigate whether specific noise injection techniques could systematically improve model performance; consequently this set of experiments was carried out on a preprocessed version of the dataset in which redundant features (i.e., highly correlated features) as well as zero-variance features had been removed. Note that this is not main focus of the thesis, yet, just a way to prove one of the many applications of the newly developed Pucktrick functionality. For this reason those experiment are more limited and they're experiment were conducted only on 5 random seeds. This is just to arrive to a proof of concept. 

- **Experiment B**: **Experiments performed using the complete dataset**: unlike the previous scenario, this experimental setting relied on the original dataset without applying feature reduction procedures (this means that no feature from the original dataset was discarded before starting HPO, yet all experiments were conducted on dataset composed only of 10 features). The purpose of this phase is primarily analytical, focusing on the theoretical aspects of the study. More specifically, it aims to provide a comprehensive understanding of the behavior of the selected models when exposed to noise generated through the `Pucktrick` library. Those are the main focus of the thesis and have been developed using a wider range of 20 random seeds. 

Please note that all results from those experiments in following folders:
- Experiment A: are allocated `complete_experiments\Experiment_A`
- Experiment B: are allocated `complete_experiments\Experiment_B`

## Running the Experiments

### Local execution (localhost)
```bash
# Main experiment: all noise types (missing, outliers, noise, duplicated, labels)
python3 4-4c-cavas_model_Experiment_localhost.py

# Labels-only experiment (corrupts y_train targets across multiple seeds)
python3 5-cavas_model_experiment_targets.py
```

### HPC cluster (SLURM)
```bash
sbatch submit_dirtify.sh     # main experiment
sbatch submit_dirtifyB.sh    # variant B
```

### Watchdog (auto-restart on OOM crash)
```bash
# Run from within a tmux session with the venv active
python3 watchdog.py
python3 watchdog.py --script 5-cavas_model_experiment_targets.py --max-restarts 50 --wait 30
```

The watchdog launches each run in an isolated tmux session (`exp_run_N`), monitors `run_N.log` for the completion string, and auto-restarts on crash. 

### Install dependencies
```bash
pip install -r requirements.txt
# pucktrick is installed from GitHub (see requirements.txt line 65)
```

## Key Configuration Flags

In each experiment script, the top CONFIG block controls all behavior:

| Flag | Default | Effect |
|---|---|---|
| `LOCAL_RUN` | `True` | `True` = local paths; `False` = HPC paths |
| `JUST_COMPILE_DATASETS` | `False` | `True` = generate and save datasets to disk, skip training |
| `RANDOM_SEEDS` | list | Seeds to iterate over; local uses 20 seeds, HPC uses 10 |
| `PERCENTAGE_TO_USE` | `0.1` | Fraction of dataset to sample (stratified by `label_generic`) |
| `WINDOW_SIZE` | `50` | CNN-LSTM temporal window |
| `STEP_SIZE` | `10` | Stride between windows |

## Architecture and Data Flow

### Dataset (`DATASETS/all_elaborated.parquet`)
Loaded via PySpark. Key columns: `Timestamp`, `Label` (string multiclass), `label_generic` (int binary). Feature columns are split into continuous, categorical (`Fwd Seg Size Min`, `Protocol`), and binary (TCP flags).

### Pipeline per experiment (`run_single_experiment` / `run_single_experiment_load_or_create`)
1. **Spark** reads the parquet, applies temporal ordering, and splits 66%/17%/17% (train/val/test) by row index mod 3.
2. **PuckTrick** injects noise into `train_clean` only — validation and test remain clean. Noise types: `missing`, `outliers`, `noise`, `duplicated` (row-level), `labels` (target corruption).
3. **Preprocessing**: Spark → Pandas → `StandardScaler` on continuous features, `LabelEncoder` on categoricals. Rare classes (< 5 samples) are dropped and class indices are remapped to be contiguous.
4. **TabNet** receives 2D arrays; **CNN-LSTM** receives 3D sliding-window arrays built by `build_windows_from_arrays`.
5. Both models skip training if their `.zip` / `.pt` checkpoint already exists (idempotent reruns).
6. Metrics (`accuracy`, `F1`, `MCC`, `AUC`, confusion matrices) and feature importance are saved as `*_artifacts.json` alongside each model file.

### Labels-only variant (`5-cavas_model_experiment_targets.py`)
Same pipeline but noise is applied to the encoded **target columns** (`Label_enc`, `label_generic_enc`), not to features. PuckTrick `labels` requires an integer column, so encoding must happen before noise injection.

### Experiment output structure

Results are filed under `complete_experiments/` by thesis experiment (see **Thesis Helper** above). The experiment scripts write per-seed folders named `experiment_rs{seed}/`; the two top-level buckets correspond to the two thesis phases:

```
complete_experiments/
  Experiment_A/                          # reduced (feature-pruned) dataset — proof of concept, 5 seeds
    Baselines/                           # clean-data reference run (no noise)
      cnn_lstm_trial_Baseline_artifacts.json
      tabnet_trial_Baseline_artifacts.json
    experiment_rs{seed}/                 # rs1, rs2, rs42, rs51, rs84
      cnn_lstm_trial_Experiment_{method}_{col}_{pct}.pt   (+ _artifacts.json)
      tabnet_trial_Experiment_{method}_{col}_{pct}.zip    (+ _artifacts.json)

  Experiment_B/                          # complete dataset — main thesis focus, 20 seeds
    Baselines/                           # clean-data reference run (no noise)
      cnn_lstm_trial_Baseline_artifacts.json
      tabnet_trial_Baseline_artifacts.json
    Pucktrick_on_single_feature/         # noise injected into one feature/column at a time
      experiment_rs{seed}/               # 20 seeds (1, 11, 21, 31, 41, 42, 51, 61, 86, 101, ...)
        cnn_lstm_trial_Experiment_{method}_{col}_{pct}.pt  (+ _artifacts.json)
        tabnet_trial_Experiment_{method}_{col}_{pct}.zip   (+ _artifacts.json)
        images/                          # confusion matrices + validation-loss curves (verbose runs)
        labels_experiment/               # labels-only variant; corrupts the Label / label_generic targets
          cnn_lstm_trial_Experiment_labels_{Label|label_generic}_{pct}.pt  (+ _artifacts.json)
          tabnet_trial_Experiment_labels_{Label|label_generic}_{pct}.zip   (+ _artifacts.json)
    Pucktrick_on_multiple_features/      # noise injected into several features at once (work in progress)

models/
  important_features.csv        # ranked feature list; drives dataset column selection
  cnn_lstm_trial_{N}.pt         # hyperparameter tuning trial checkpoints
  tabnet_trial_{N}.zip
  *_artifacts.json
```

`{method}` ∈ `missing | outliers | noise | duplicated | labels`; `{col}` is the corrupted column (`Timestamp` for `duplicated`, since it acts row-wide); `{pct}` is the noise percentage (e.g. `10.0`). Note the percentage grids differ between phases — Experiment A uses `1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0`, Experiment B uses `5.0, 10.0, 20.0, 35.0, 50.0, 75.0`. Each phase also has a `Baselines/` folder holding the clean-data reference models that the noisy runs are compared against.

Best hyperparameters from tuning are hard-coded: TabNet trial 7, CNN-LSTM trial 1. These are read from `models/` via `reload_all_trial_metadata()` at the start of each experiment run.

## Evaluation (notebooks 6A / 6B)

Result analysis lives in a pair of generated notebooks plus a shared helper module:

| File | Purpose |
|---|---|
| `experiment_eval_utils.py` | Shared, **fully dynamic** evaluation helpers. Walks an experiment tree, parses every `*_artifacts.json`, and builds a tidy long-format DataFrame — nothing about which seeds/methods/features/noise levels exist is hard-coded. Experimental coordinates (model / method / corrupted feature / noise %) are recovered from each file's `label` via `parse_label`, not read from JSON fields. Aggregates are reported with Student-t confidence intervals whose width adapts to the seed count per cell. |
| `6A-Experiment-evaluation.ipynb` | Evaluates **Experiment A** (`complete_experiments/Experiment_A`, reduced dataset, 5 seeds). |
| `6B-single-Experiment-evaluation.ipynb` | Evaluates **Experiment B**, **single-feature** scenario only (`scenario == "single"`: `Pucktrick_on_single_feature/` + its `labels_experiment/`). |
| `6B-multi-Experiment-evaluation.ipynb` | Evaluates **Experiment B**, **multiple-feature** scenario only (`scenario == "multi"`: `Pucktrick_on_multiple_features/`). |
| `build_eval_notebooks.py` | Generator: `python3 build_eval_notebooks.py` regenerates **all three** notebooks (6A, 6B-single, 6B-multi) from one shared template. Edit the template here, not the notebooks directly. |

Each notebook also includes a **rank-drift** figure (§9b): one subplot per dirtied feature showing how that feature's permutation-importance *rank* (1 = most important) moves with the noise level, one line per PuckTrick method, each point annotated with its rank and a dashed clean-baseline reference.

Aggregated tables are written to `evaluation_results/Experiment_{A,B-single,B-multi}/`: `aggregate_{mcc,f1,accuracy}.csv`, `baseline_metrics.csv`, and significance tests (`significance_mcc_binary_vs_baseline.csv`, `significance_mcc_binary_vs_lownoise.csv`) comparing noisy runs against the clean `Baselines/` and against the lowest noise level.

## Notebook Workflow (numbered sequence)

| File | Purpose |
|---|---|
| `1-cavas_dataset_analysis.ipynb` | EDA, feature selection, correlation analysis |
| `2-cavas_model_HyperParameters_Tuning.ipynb` | Optuna HPO for both models |
| `3-cavas_model_Baselines.ipynb` | Baseline comparisons |
| `4-1/4-2/4-3-cavas_model_Experiment_*.ipynb` | Interactive experiment notebooks (Linux/Windows/GPU) |
| `4-4/4-4b/4-4c-cavas_model_Experiment_*.py` | Converted scripts for cluster/local |
| `5-cavas_model_experiment_targets.py` | Labels-only experiment, multi-seed |
| `5-Experiment-evaluation.ipynb` | Older result aggregation/visualization (superseded by 6A/6B) |
| `6A-/6B-single-/6B-multi-Experiment-evaluation.ipynb` | Current evaluation notebooks (see **Evaluation** above); generated by `build_eval_notebooks.py` |

## Memory Management

The `clear_memory()` function runs after every experiment: Python GC, CUDA cache flush, Spark catalog clear, and `malloc_trim`. This is critical — OOM crashes are a known issue, hence the watchdog. The CNN-LSTM feature importance computation (permutation-based) can be particularly memory-intensive.