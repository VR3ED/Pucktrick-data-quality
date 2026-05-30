# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CAVAS** — a thesis project studying the impact of data quality degradation on IDS (Intrusion Detection System) models. The pipeline injects controlled noise into the CAVAS network traffic dataset using [PuckTrick](https://github.com/andreamaurino/pucktrick), then evaluates how two complementary models handle it:

- **TabNet** (`pytorch_tabnet`) — tabular classifier, input shape `(n_samples, n_features)` (2D)
- **CNN-LSTM** (`CNNLSTMMultiTask`) — time-series classifier, input shape `(n_samples, window_size, n_features)` (3D)

Both models are **multi-task**: they simultaneously predict a binary label (`label_generic`: benign/malicious) and a multiclass label (`Label`: attack type).

The primary evaluation metric is **MCC** (Matthews Correlation Coefficient), reported separately for binary/multiclass tasks.

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
```
complete_experiments/
  experiment_rs{seed}/          # one folder per random seed
    labels_experiment/          # only for the labels-only variant
      cnn_lstm_trial_Experiment_labels_{col}_{pct}.pt
      cnn_lstm_trial_Experiment_labels_{col}_{pct}_artifacts.json
      tabnet_trial_Experiment_labels_{col}_{pct}.zip
      tabnet_trial_Experiment_labels_{col}_{pct}_artifacts.json
models/
  important_features.csv        # ranked feature list; drives dataset column selection
  cnn_lstm_trial_{N}.pt         # hyperparameter tuning trial checkpoints
  tabnet_trial_{N}.zip
  *_artifacts.json
```

Best hyperparameters from tuning are hard-coded: TabNet trial 7, CNN-LSTM trial 1. These are read from `models/` via `reload_all_trial_metadata()` at the start of each experiment run.

## Notebook Workflow (numbered sequence)

| File | Purpose |
|---|---|
| `1-cavas_dataset_analysis.ipynb` | EDA, feature selection, correlation analysis |
| `2-cavas_model_HyperParameters_Tuning.ipynb` | Optuna HPO for both models |
| `3-cavas_model_Baselines.ipynb` | Baseline comparisons |
| `4-1/4-2/4-3-cavas_model_Experiment_*.ipynb` | Interactive experiment notebooks (Linux/Windows/GPU) |
| `4-4/4-4b/4-4c-cavas_model_Experiment_*.py` | Converted scripts for cluster/local |
| `5-cavas_model_experiment_targets.py` | Labels-only experiment, multi-seed |
| `5-Experiment-evaluation.ipynb` | Result aggregation and visualization |

## Memory Management

The `clear_memory()` function runs after every experiment: Python GC, CUDA cache flush, Spark catalog clear, and `malloc_trim`. This is critical — OOM crashes are a known issue, hence the watchdog. The CNN-LSTM feature importance computation (permutation-based) can be particularly memory-intensive.