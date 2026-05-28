#!/usr/bin/env python3
# =============================================================================
# CAVAS — Deep Learning Model Evaluation (LABELS-ONLY VARIANT)
# IDS Intrusion Detection: TabNet (Tabular) + CNN-LSTM (Time Series)
#
# Variante derivata da "4-4b-cavas_model_Experiment_cluster" con queste
# differenze:
#   • Applica esclusivamente il metodo PuckTrick "labels"
#     (gli altri 4 metodi — missing/outliers/noise/duplicated — sono esclusi)
#   • Il noise viene iniettato SOLO sulle y_train. Le 2 colonne target del
#     dataset sono 'Label' (multiclass) e 'label_generic' (binary): queste
#     vengono sporcate SINGOLARMENTE (un esperimento per colonna), e poi
#     formano gli y_target dati in pasto ai modelli.
#   • JUST_COMPILE_DATASETS = True  → crea e salva i dataset su disco
#     JUST_COMPILE_DATASETS = False → carica i dataset da disco se presenti;
#     se NON sono presenti, li crea al volo (in RAM) SENZA salvarli su disco
#   • Tutti i risultati (modelli, artifacts, immagini, dataset) sono
#     salvati in:  complete_experiments/experiment_rs{seed}/labels_experiment/
# =============================================================================

# ── Matplotlib: headless backend (no display on HPC) ─────────────────────────
import matplotlib
matplotlib.use('Agg')

import os   # necessario per BASE_DIR prima del blocco CONFIG

# =============================================================================
# CONFIG — modifica solo qui
# =============================================================================
JUST_COMPILE_DATASETS  = False   # True  → crea e salva i dataset su disco (senza addestrare i modelli)
                                 # False → carica dataset da disco se esistono, altrimenti li crea al volo
                                 #         (SENZA salvarli) e addestra i due modelli
LOCAL_RUN              = True    # True → esecuzione locale | False → HPC cluster
RUNNING_ON_HPC         = not LOCAL_RUN
RANDOM_SEEDS           = []
TRIALS_ALREADY_EXECUTED= True
MIN_SAMPLES_PER_CLASS  = 5
PROF_DIR               = True

PERCENTAGE_TO_USE = 0.1          # 10 % del dataset
WINDOW_SIZE       = 50           # finestra temporale per CNN-LSTM
STEP_SIZE         = 10           # overlap tra finestre

# Percorso base: cartella dello script in locale, path cluster su HPC
if LOCAL_RUN:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RANDOM_SEEDS = [1,11,21,31,41,42,51,61,86,101,202,303,404,505,606,707,808,909,1010,1111]
else:
    RANDOM_SEEDS = [303, 404, 505, 606, 707, 808, 909, 1010, 1111, 42]
    if PROF_DIR:
        BASE_DIR = "/scratch_share/datai/maurinoa/dirtify/hpc"
    else:
        BASE_DIR = "/scratch_share/datai/fcavallini/dirtify"

DATASETS_DIR        = f"{BASE_DIR}/DATASETS"
CURRENT_RANDOM_SEED = RANDOM_SEEDS[0]   # aggiornato dinamicamente nel loop

# Nome della sotto-cartella dedicata a questo esperimento (LABELS-ONLY)
LABELS_EXPERIMENT_SUBDIR = "labels_experiment"

# =============================================================================
# 0. IMPORTS
# =============================================================================
import pucktrick
from pucktrick import Engine
from pucktrick import PuckTrick

import os, subprocess, warnings, json, gc, ctypes
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob, math

# Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer

# Sklearn
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.metrics          import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)

# TabNet
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

# PyTorch
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Optuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')
pl.seed_everything(CURRENT_RANDOM_SEED)
print("All imports OK")

# =============================================================================
# 0.1. PATH & EPOCH CONFIGURATION
# =============================================================================
PATH_IMG = f"{BASE_DIR}/images"

if RUNNING_ON_HPC:
    PATH = f"file://{DATASETS_DIR}"
else:
    PATH = "DATASETS"

# Epoche invariate rispetto alle run locali (no discrepanze)
TABNET_MAX_EPOCHS   = 20
CNN_LSTM_MAX_EPOCHS = 50

# Directory per-seed: aggiornata all'inizio di ogni iterazione nel main loop
# NOTA: include il sotto-percorso "labels_experiment"
CURRENT_RS_DIR = (
    f"{BASE_DIR}/complete_experiments/experiment_rs{CURRENT_RANDOM_SEED}/"
    f"{LABELS_EXPERIMENT_SUBDIR}"
)

# Crea le directory fisse (non dipendenti dal seed)
for d in [f"{BASE_DIR}/models",
          f"{BASE_DIR}/job_logs",
          f"{BASE_DIR}/complete_experiments"]:
    os.makedirs(d, exist_ok=True)

# Cambia working directory → tutti i path relativi puntano qui
os.chdir(BASE_DIR)
print(f"✅  Working dir: {os.getcwd()}")

# =============================================================================
# 0.2. SPARK SESSION (single-node HPC, usa tutti i core SLURM)
# =============================================================================
# JAVA_HOME
java_home = os.environ.get('JAVA_HOME', '')
if not java_home:
    try:
        java_path = subprocess.check_output(['which', 'java'], text=True).strip()
        os.environ['JAVA_HOME'] = os.path.dirname(os.path.dirname(
            os.path.realpath(java_path)))
    except subprocess.CalledProcessError:
        print("⚠️  Java not found — assicurati di avere java nel PATH")

os.environ['PYSPARK_PYTHON']        = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

spark = SparkSession.builder \
    .appName("CAVAS_Models_LabelsOnly") \
    .master("local[*]") \
    .config("spark.driver.memory",          "30g") \
    .config("spark.driver.maxResultSize",   "20g") \
    .config("spark.driver.host",            "localhost") \
    .config("spark.sql.shuffle.partitions", "60") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"✅  Spark {spark.version} ready")

# =============================================================================
# 0.3. GPU / DEVICE SETUP
# =============================================================================
if torch.cuda.is_available():
    GPU_DEVICE = torch.device('cuda:0')
    print(f"✅  GPU disponibile: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.set_per_process_memory_fraction(0.80, device=0)
    torch.backends.cudnn.benchmark = True
else:
    GPU_DEVICE = torch.device('cpu')
    print("⚠️  GPU non disponibile — training su CPU")

# =============================================================================
# 0.4. FEATURE TYPES
# =============================================================================
CATEGORICAL_FEATURES = ['Fwd Seg Size Min', 'Protocol']
BINARY_FEATURES      = [
    'FIN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
    'ACK Flag Cnt', 'URG Flag Cnt', 'Fwd URG Flag', 'Fwd PSH Flag'
]

# Globals popolati da prepare_whole_dataset_from_scratch()
CAT_IDXS    = []
CAT_DIMS    = []
label_classes = []
tabnet_trial_artifacts   = {}
cnn_lstm_trial_artifacts = {}

# =============================================================================
# STEP 0.5 — UTILITY FUNCTIONS
# =============================================================================

def label_encoding_spark(sdf):
    sdf = sdf.withColumn('label_generic_enc', col('label_generic').cast('int'))
    indexer = StringIndexer(inputCol='Label', outputCol='Label_enc', handleInvalid='keep')
    model = indexer.fit(sdf)
    sdf = model.transform(sdf)
    sdf = sdf.withColumn('Label_enc', col('Label_enc').cast('int'))
    label_classes = list(model.labels)
    n_binary = sdf.select('label_generic_enc').distinct().count()
    print(f"✅  label_generic_enc: {n_binary} classes | Label_enc: {len(label_classes)} classes")
    print(f"    Label mapping: { {i: l for i, l in enumerate(label_classes)} }")
    return sdf, label_classes


def preprocess_to_pandas(sdf, continuous_features, categorical_features, binary_features):
    print("⏳  Converting Spark → Pandas ...")
    pdf = sdf.toPandas()
    print(f"📊  Shape: {pdf.shape}")

    available  = set(pdf.columns)
    cont_cols  = [c for c in continuous_features  if c in available]
    cat_cols   = [c for c in categorical_features if c in available]
    bin_cols   = [c for c in binary_features      if c in available]

    for c in cont_cols:
        pdf[c] = pd.to_numeric(pdf[c], errors='coerce')
    pdf[cont_cols] = pdf[cont_cols].fillna(0.0)

    cat_encoders = {}
    cat_dims     = {}
    for c in cat_cols:
        le = LabelEncoder()
        pdf[c] = le.fit_transform(pdf[c].astype(str))
        cat_encoders[c] = le
        cat_dims[c]     = len(le.classes_)

    for c in bin_cols:
        pdf[c] = pd.to_numeric(pdf[c], errors='coerce').fillna(0).astype(int)

    print(f"✅  Preprocessed: {len(cont_cols)} continuous | {len(cat_cols)} categorical | {len(bin_cols)} binary")
    return pdf, cat_encoders, cat_dims


def print_metrics(y_true, y_pred, y_proba, task_name, class_names=None, verbose=1):
    is_binary = (len(np.unique(y_true)) == 2)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='binary' if is_binary else 'macro')
    try:
        auc = roc_auc_score(
            y_true,
            y_proba[:, 1] if is_binary else y_proba,
            multi_class='ovr' if not is_binary else 'raise'
        )
    except Exception:
        auc = float('nan')

    if verbose != 0:
        print(f"\n{'='*55}")
        print(f"  {task_name}")
        print(f"{'='*55}")
        print(f"  Accuracy : {acc:.4f}  |  F1: {f1:.4f}  |  MCC: {mcc:.4f}  |  AUC: {auc:.4f}")

    present_labels = sorted(np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
    target_names_filtered = (
        [class_names[i] for i in present_labels if i < len(class_names)]
        if class_names is not None else None
    )
    if verbose != 0:
        print(classification_report(y_true, y_pred, labels=present_labels,
                                    target_names=target_names_filtered))
    return dict(task=task_name, accuracy=acc, f1=f1, mcc=mcc, auc=auc)


def json_serializer(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return None if np.isnan(x) else x.item()
    if isinstance(x, float) and np.isnan(x):
        return None
    return str(x)


def clear_memory():
    """Libera RAM: Python heap + GPU + Spark cache + forza glibc malloc_trim."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        spark.catalog.clearCache()
    except Exception:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def experiment_already_exists(tag):
    tabnet_path   = f"{CURRENT_RS_DIR}/tabnet_trial_{tag}_artifacts.json"
    print(f"Checking if experiment exists: {tabnet_path}")
    print("esiste: ", os.path.exists(tabnet_path))
    cnn_lstm_path = f"{CURRENT_RS_DIR}/cnn_lstm_trial_{tag}_artifacts.json"
    print(f"Checking if experiment exists: {cnn_lstm_path}")
    print("esiste: ", os.path.exists(cnn_lstm_path))
    return os.path.exists(tabnet_path) and os.path.exists(cnn_lstm_path)


def _dataset_dir(label):
    """Restituisce la cartella su disco per il dataset identificato da *label*."""
    return os.path.join(CURRENT_RS_DIR, "datasets", label)


def dataset_already_saved(label):
    """True se il dataset per *label* e' gia' stato serializzato su disco."""
    d = _dataset_dir(label)
    return os.path.exists(os.path.join(d, "metadata.json"))


def save_dataset_to_disk(label, dataset_tuple):
    """
    Salva su disco il risultato di prepare_whole_dataset_from_scratch().
    """
    (FEATURE_COLS,
     X_train_2d, X_val_2d,
     y_tr_bin_2d, y_tr_mul_2d,
     y_val_bin_2d, y_val_mul_2d,
     X_train_3d, X_val_3d,
     y_tr_bin_3d, y_tr_mul_3d,
     y_val_bin_3d, y_val_mul_3d,
     corr_matrix) = dataset_tuple

    d = _dataset_dir(label)
    os.makedirs(d, exist_ok=True)

    arrays = {
        "X_train_2d":   X_train_2d,
        "X_val_2d":     X_val_2d,
        "y_tr_bin_2d":  y_tr_bin_2d,
        "y_tr_mul_2d":  y_tr_mul_2d,
        "y_val_bin_2d": y_val_bin_2d,
        "y_val_mul_2d": y_val_mul_2d,
        "X_train_3d":   X_train_3d,
        "X_val_3d":     X_val_3d,
        "y_tr_bin_3d":  y_tr_bin_3d,
        "y_tr_mul_3d":  y_tr_mul_3d,
        "y_val_bin_3d": y_val_bin_3d,
        "y_val_mul_3d": y_val_mul_3d,
    }
    for name, arr in arrays.items():
        np.save(os.path.join(d, f"{name}.npy"), arr)

    metadata = {
        "FEATURE_COLS":  FEATURE_COLS,
        "CAT_IDXS":      CAT_IDXS,
        "CAT_DIMS":      CAT_DIMS,
        "label_classes": label_classes,
        "corr_matrix":   corr_matrix,
    }
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, default=json_serializer)

    print(f"[DATASET SAVED] '{label}' -> {d}")


def load_dataset_from_disk(label):
    """
    Carica da disco il dataset salvato da save_dataset_to_disk().
    Ripristina anche i globali CAT_IDXS, CAT_DIMS e label_classes.
    Restituisce la stessa tupla di prepare_whole_dataset_from_scratch().
    """
    global CAT_IDXS, CAT_DIMS, label_classes

    d = _dataset_dir(label)
    if not os.path.exists(os.path.join(d, "metadata.json")):
        raise FileNotFoundError(
            f"Dataset '{label}' non trovato in {d}. "
            "Esegui prima con JUST_COMPILE_DATASETS=True oppure lascialo creare al volo."
        )

    with open(os.path.join(d, "metadata.json")) as f:
        metadata = json.load(f)

    FEATURE_COLS  = metadata["FEATURE_COLS"]
    CAT_IDXS      = metadata["CAT_IDXS"]
    CAT_DIMS      = metadata["CAT_DIMS"]
    label_classes = metadata["label_classes"]
    corr_matrix   = metadata["corr_matrix"]

    def _load(name):
        return np.load(os.path.join(d, f"{name}.npy"), allow_pickle=False)

    X_train_2d   = _load("X_train_2d")
    X_val_2d     = _load("X_val_2d")
    y_tr_bin_2d  = _load("y_tr_bin_2d")
    y_tr_mul_2d  = _load("y_tr_mul_2d")
    y_val_bin_2d = _load("y_val_bin_2d")
    y_val_mul_2d = _load("y_val_mul_2d")
    X_train_3d   = _load("X_train_3d")
    X_val_3d     = _load("X_val_3d")
    y_tr_bin_3d  = _load("y_tr_bin_3d")
    y_tr_mul_3d  = _load("y_tr_mul_3d")
    y_val_bin_3d = _load("y_val_bin_3d")
    y_val_mul_3d = _load("y_val_mul_3d")

    print(f"[DATASET LOADED] '{label}' <- {d}")
    return (
        FEATURE_COLS,
        X_train_2d, X_val_2d,
        y_tr_bin_2d, y_tr_mul_2d,
        y_val_bin_2d, y_val_mul_2d,
        X_train_3d, X_val_3d,
        y_tr_bin_3d, y_tr_mul_3d,
        y_val_bin_3d, y_val_mul_3d,
        corr_matrix,
    )

# =============================================================================
# STEP 1a — TabNet Hyperparameter Objective
# =============================================================================

def tabnet_multitask_objective(X_train, X_val,
                                y_tr_bin, y_tr_mul,
                                y_val_bin, y_val_mul,
                                N_a, N_steps, gamma, lambda_s, lr, batch_sz, mask_type,
                                verbose=0, trial=None, label_model=None,
                                feature_names=None, corr_matrix=None):
    FINAL_LABEL    = label_model if label_model is not None else trial.number
    model_path     = f'{CURRENT_RS_DIR}/tabnet_trial_{FINAL_LABEL}'
    artifacts_path = f'{CURRENT_RS_DIR}/tabnet_trial_{FINAL_LABEL}_artifacts.json'

    reloaded = False
    old_feature_importance = None

    if os.path.exists(model_path + '.zip'):
        print(f"♻️  Found existing TabNet model for {FINAL_LABEL}, reloading...")
        reloaded = True
        if os.path.exists(artifacts_path):
            with open(artifacts_path) as f:
                old_art = json.load(f)
            old_feature_importance = old_art.get('feature_importance')
        clf = TabNetMultiTaskClassifier()
        clf.load_model(model_path + '.zip')
    else:
        clf = TabNetMultiTaskClassifier(
            n_d=N_a, n_a=N_a,
            n_steps=N_steps,
            gamma=gamma,
            lambda_sparse=lambda_s,
            cat_idxs=CAT_IDXS if CAT_IDXS else [],
            cat_dims=CAT_DIMS if CAT_DIMS else [],
            cat_emb_dim=1,
            optimizer_params=dict(lr=lr),
            mask_type=mask_type,
            verbose=verbose,
            seed=CURRENT_RANDOM_SEED,
        )

        y_train_mt = np.column_stack([y_tr_bin, y_tr_mul])
        y_val_mt   = np.column_stack([y_val_bin, y_val_mul])

        clf.fit(
            X_train, y_train_mt,
            eval_set           = [(X_val, y_val_mt)],
            eval_metric        = ['accuracy'],
            max_epochs         = TABNET_MAX_EPOCHS,
            patience           = 4,
            batch_size         = batch_sz,
            virtual_batch_size = max(batch_sz // 4, 64),
            drop_last          = False,
        )
        clf.save_model(model_path)
        print(f"✅  Saved TabNet model for trial {FINAL_LABEL}")

    raw_preds     = clf.predict(X_val)
    pred_bin      = np.asarray(raw_preds[0]).astype(int)
    pred_mul      = np.asarray(raw_preds[1]).astype(int)
    y_val_bin_int = np.asarray(y_val_bin).astype(int)
    y_val_mul_int = np.asarray(y_val_mul).astype(int)

    mcc_bin = matthews_corrcoef(y_val_bin_int, pred_bin)
    mcc_mul = matthews_corrcoef(y_val_mul_int, pred_mul)
    cm_bin  = confusion_matrix(y_val_bin_int, pred_bin)
    cm_mul  = confusion_matrix(y_val_mul_int, pred_mul)

    if verbose != 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'])
        axes[0].set_title(f'Trial {FINAL_LABEL} — Binary CM')
        sns.heatmap(cm_mul, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                    xticklabels=label_classes[:cm_mul.shape[1]],
                    yticklabels=label_classes[:cm_mul.shape[0]])
        axes[1].set_title(f'Trial {FINAL_LABEL} — Multiclass CM')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{PATH_IMG}/tabnet_trial{FINAL_LABEL}_cm.png', dpi=150, bbox_inches='tight')
        plt.close()

    proba = clf.predict_proba(X_val)
    metrics_first_output  = print_metrics(y_val_bin_int, pred_bin, proba[0],
                                           f'Trial {FINAL_LABEL} - Binary Task',
                                           class_names=['Benign', 'Malicious'], verbose=verbose)
    metrics_second_output = print_metrics(y_val_mul_int, pred_mul, proba[1],
                                           f'Trial {FINAL_LABEL} - Multiclass Task',
                                           class_names=label_classes, verbose=verbose)

    if reloaded and old_feature_importance is not None:
        feature_importance = old_feature_importance
    else:
        feat_names = (list(feature_names) if feature_names is not None
                      else [f'f{i}' for i in range(X_train.shape[1])])
        importance_scores  = clf.feature_importances_
        feature_importance = dict(zip(feat_names, importance_scores.tolist()))

    mean_mcc = (mcc_bin + mcc_mul) / 2
    object_to_store = {
        'model':              f'tabnet_trial_{FINAL_LABEL}',
        'model_type':         'TabNet',
        'label':              str(FINAL_LABEL),
        'mcc_bin':            mcc_bin,
        'mcc_mul':            mcc_mul,
        'mean_mcc':           mean_mcc,
        'cm_bin':             cm_bin,
        'cm_mul':             cm_mul,
        'feature_importance': feature_importance,
        'params': trial.params if trial is not None else {
            'N_a': N_a, 'N_steps': N_steps, 'gamma': gamma,
            'lambda_sparse': lambda_s, 'lr': lr,
            'batch_size': batch_sz, 'mask_type': mask_type
        },
        'metrics_binary':     metrics_first_output,
        'metrics_multiclass': metrics_second_output,
        'correlation_matrix': corr_matrix,
    }

    with open(artifacts_path, 'w') as f:
        json.dump(object_to_store, f, indent=4, default=json_serializer)

    print(f"Trial {FINAL_LABEL}: MCC_bin={mcc_bin:.4f}, MCC_mul={mcc_mul:.4f}, mean={mean_mcc:.4f}")
    return mean_mcc

# =============================================================================
# STEP 1b — CNN-LSTM Model Definition
# =============================================================================

class CNNLSTMMultiTask(nn.Module):
    def __init__(self, n_features, n_timesteps, n_classes_bin, n_classes_mul,
                 nb_filters=64, kernel_size=3, lstm_units_1=64,
                 lstm_units_2=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=nb_filters,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(nb_filters),
        )
        self.lstm1   = nn.LSTM(input_size=nb_filters, hidden_size=lstm_units_1, batch_first=True)
        self.lstm2   = nn.LSTM(input_size=lstm_units_1, hidden_size=lstm_units_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Sequential(
            nn.Linear(lstm_units_2, lstm_units_2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_bin = nn.Linear(lstm_units_2, n_classes_bin)
        self.head_mul = nn.Linear(lstm_units_2, n_classes_mul)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return self.head_bin(x), self.head_mul(x)


# =============================================================================
# STEP 1b — CNN-LSTM Objective
# =============================================================================

def cnn_lstm_multitask_objective(X_train, X_val,
                                  y_tr_bin, y_tr_mul,
                                  y_val_bin, y_val_mul,
                                  nb_filters, kernel_size,
                                  lstm_units_1, lstm_units_2,
                                  dropout, lr, batch_size,
                                  verbose=0, trial=None, label_model=None,
                                  feature_names=None, corr_matrix=None):
    FINAL_LABEL    = label_model if label_model is not None else trial.number
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_features  = X_train.shape[2]
    n_timesteps = X_train.shape[1]

    def make_loader(X, yb, ym, shuffle):
        ds = TensorDataset(
            torch.tensor(X,  dtype=torch.float32),
            torch.tensor(yb, dtype=torch.long),
            torch.tensor(ym, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=torch.cuda.is_available())

    train_dl = make_loader(X_train, y_tr_bin, y_tr_mul, shuffle=False)
    val_dl   = make_loader(X_val,   y_val_bin, y_val_mul, shuffle=False)

    loss_bin = nn.CrossEntropyLoss()
    loss_mul = nn.CrossEntropyLoss()

    model_path     = f'{CURRENT_RS_DIR}/cnn_lstm_trial_{FINAL_LABEL}.pt'
    artifacts_path = f'{CURRENT_RS_DIR}/cnn_lstm_trial_{FINAL_LABEL}_artifacts.json'
    val_losses     = None

    reloaded = False
    old_feature_importance = None

    if os.path.exists(model_path):
        print(f"♻️  Found existing CNN-LSTM model for {FINAL_LABEL}, reloading...")
        reloaded = True
        if os.path.exists(artifacts_path):
            with open(artifacts_path) as f:
                old_art = json.load(f)
            old_feature_importance = old_art.get('feature_importance')
        model = CNNLSTMMultiTask(
            n_features=n_features, n_timesteps=n_timesteps,
            n_classes_bin=2,
            n_classes_mul=int(max(np.max(y_tr_mul), np.max(y_val_mul))) + 1,
            nb_filters=nb_filters, kernel_size=kernel_size,
            lstm_units_1=lstm_units_1, lstm_units_2=lstm_units_2, dropout=dropout,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = CNNLSTMMultiTask(
            n_features=n_features, n_timesteps=n_timesteps,
            n_classes_bin=2,
            n_classes_mul=int(max(np.max(y_tr_mul), np.max(y_val_mul))) + 1,
            nb_filters=nb_filters, kernel_size=kernel_size,
            lstm_units_1=lstm_units_1, lstm_units_2=lstm_units_2, dropout=dropout,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-8)

        best_val_loss    = float('inf')
        patience_counter = 0
        PATIENCE         = 10
        val_losses       = []
        best_state       = None

        for epoch in range(CNN_LSTM_MAX_EPOCHS):
            model.train()
            for X_b, yb_b, ym_b in train_dl:
                X_b, yb_b, ym_b = X_b.to(device), yb_b.to(device), ym_b.to(device)
                optimizer.zero_grad()
                out_bin, out_mul = model(X_b)
                loss = loss_bin(out_bin, yb_b) + loss_mul(out_mul, ym_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for X_b, yb_b, ym_b in val_dl:
                    X_b, yb_b, ym_b = X_b.to(device), yb_b.to(device), ym_b.to(device)
                    out_bin, out_mul = model(X_b)
                    val_loss_epoch += (loss_bin(out_bin, yb_b) + loss_mul(out_mul, ym_b)).item()

            val_loss_epoch /= len(val_dl)
            val_losses.append(val_loss_epoch)
            scheduler.step(val_loss_epoch)

            if verbose != 0:
                print(f"  Epoch {epoch+1:3d} | val_loss: {val_loss_epoch:.4f}")

            if val_loss_epoch < best_val_loss:
                best_val_loss    = val_loss_epoch
                patience_counter = 0
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    if verbose != 0:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

            if trial is not None:
                trial.report(val_loss_epoch, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), model_path)
        print(f"✅  Saved CNN-LSTM model for trial {FINAL_LABEL}")

    # ── Evaluation ────────────────────────────────────────────────────
    model.eval()
    all_pred_bin, all_pred_mul = [], []
    all_prob_bin, all_prob_mul = [], []
    all_true_bin, all_true_mul = [], []
    eval_loss_total = 0.0

    with torch.no_grad():
        for X_b, yb_b, ym_b in val_dl:
            X_b, yb_b, ym_b = X_b.to(device), yb_b.to(device), ym_b.to(device)
            out_bin, out_mul = model(X_b)
            eval_loss_total += (loss_bin(out_bin, yb_b) + loss_mul(out_mul, ym_b)).item()
            prob_bin = torch.softmax(out_bin, dim=1).cpu().numpy()
            prob_mul = torch.softmax(out_mul, dim=1).cpu().numpy()
            all_pred_bin.extend(prob_bin.argmax(axis=1))
            all_pred_mul.extend(prob_mul.argmax(axis=1))
            all_prob_bin.append(prob_bin)
            all_prob_mul.append(prob_mul)
            all_true_bin.extend(yb_b.cpu().numpy())
            all_true_mul.extend(ym_b.cpu().numpy())

    best_val_loss = eval_loss_total / len(val_dl)
    pred_bin = np.array(all_pred_bin)
    pred_mul = np.array(all_pred_mul)
    prob_bin = np.vstack(all_prob_bin)
    prob_mul = np.vstack(all_prob_mul)
    true_bin = np.array(all_true_bin)
    true_mul = np.array(all_true_mul)

    mcc_bin  = matthews_corrcoef(true_bin, pred_bin)
    mcc_mul  = matthews_corrcoef(true_mul, pred_mul)
    mean_mcc = (mcc_bin + mcc_mul) / 2
    cm_bin   = confusion_matrix(true_bin, pred_bin)
    cm_mul   = confusion_matrix(true_mul, pred_mul)

    metrics_first_output  = print_metrics(true_bin, pred_bin, prob_bin,
                                           f'Trial {FINAL_LABEL} - Binary Task',
                                           class_names=['Benign', 'Malicious'], verbose=verbose)
    metrics_second_output = print_metrics(true_mul, pred_mul, prob_mul,
                                           f'Trial {FINAL_LABEL} - Multiclass Task',
                                           class_names=label_classes, verbose=verbose)

    # Feature importance
    if reloaded and old_feature_importance is not None:
        feature_importance = old_feature_importance
    else:
        feature_importance = None
        try:
            feat_names = (list(feature_names) if feature_names is not None
                          else [f'f{i}' for i in range(n_features)])
            base_acc   = (pred_bin == true_bin).mean()
            importances = {}
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            for i in range(n_features):
                X_perm          = X_val_t.clone()
                idx             = torch.randperm(X_val_t.shape[0])
                X_perm[:, :, i] = X_val_t[idx, :, i]
                with torch.no_grad():
                    out_b, _ = model(X_perm)
                    p        = out_b.argmax(dim=1).cpu().numpy()
                importances[feat_names[i]] = float(base_acc - (p == true_bin).mean())
            max_imp            = max(importances.values()) or 1.0
            feature_importance = {k: max(v, 0) / max_imp for k, v in importances.items()}
        except Exception as e:
            print(f"  ⚠️ Feature importance failed (trial {FINAL_LABEL}): {e}")

    if verbose != 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'])
        axes[0].set_title(f'Trial {FINAL_LABEL} — Binary CM')
        sns.heatmap(cm_mul, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                    xticklabels=label_classes[:cm_mul.shape[1]],
                    yticklabels=label_classes[:cm_mul.shape[0]])
        axes[1].set_title(f'Trial {FINAL_LABEL} — Multiclass CM')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{PATH_IMG}/cnn_lstm_trial{FINAL_LABEL}_cm.png', dpi=150, bbox_inches='tight')
        plt.close()

        if val_losses is not None:
            best_epoch = int(np.argmin(val_losses))
            fig_l, ax_l = plt.subplots(figsize=(8, 4))
            ax_l.plot(val_losses, marker='o', markersize=3, linewidth=1.5, color='steelblue')
            ax_l.scatter([best_epoch], [val_losses[best_epoch]], color='red', zorder=5,
                         label=f'Best: epoch {best_epoch}, loss={val_losses[best_epoch]:.4f}')
            ax_l.set_xlabel('Epoch'); ax_l.set_ylabel('Val Loss')
            ax_l.set_title(f'Trial {FINAL_LABEL} — Validation Loss Curve')
            ax_l.legend(); ax_l.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{PATH_IMG}/cnn_lstm_trial{FINAL_LABEL}_loss.png', dpi=150, bbox_inches='tight')
            plt.close()

    object_to_store = {
        'model':              model_path,
        'model_type':         'CNN-LSTM',
        'label':              str(FINAL_LABEL),
        'mcc_bin':            mcc_bin,
        'mcc_mul':            mcc_mul,
        'mean_mcc':           mean_mcc,
        'val_loss':           best_val_loss,
        'cm_bin':             cm_bin.tolist(),
        'cm_mul':             cm_mul.tolist(),
        'feature_importance': feature_importance,
        'params': trial.params if trial is not None else {
            'nb_filters': nb_filters, 'kernel_size': kernel_size,
            'lstm_units_1': lstm_units_1, 'lstm_units_2': lstm_units_2,
            'dropout': dropout, 'lr': lr, 'batch_size': batch_size,
        },
        'metrics_binary':     metrics_first_output,
        'metrics_multiclass': metrics_second_output,
        'correlation_matrix': corr_matrix,
    }

    with open(artifacts_path, 'w') as f:
        json.dump(object_to_store, f, indent=4, default=json_serializer)

    f1_mean = ((metrics_first_output['f1'] + metrics_second_output['f1']) / 2)
    print(f"Trial {FINAL_LABEL}: MCC_bin={mcc_bin:.4f}, MCC_mul={mcc_mul:.4f}, mean={mean_mcc:.4f}")
    return f1_mean

# =============================================================================
# STEP 2 — Reload trial metadata from disk
# =============================================================================

def reload_all_trial_metadata(models_dir='models'):
    import re
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith('_artifacts.json'):
            continue
        match = re.match(r'(tabnet|cnn_lstm)_trial_(.+)_artifacts\.json', fname)
        if not match:
            continue
        model_type = match.group(1)
        label_str  = match.group(2)
        try:
            label = int(label_str)
        except ValueError:
            label = label_str
        with open(os.path.join(models_dir, fname)) as f:
            art = json.load(f)
        if art.get('cm_bin') is not None:
            art['cm_bin'] = np.array(art['cm_bin'])
        if art.get('cm_mul') is not None:
            art['cm_mul'] = np.array(art['cm_mul'])
        art['_live_model'] = None
        if model_type == 'tabnet':
            tabnet_trial_artifacts[label] = art
        elif model_type == 'cnn_lstm':
            cnn_lstm_trial_artifacts[label] = art

# =============================================================================
# STEP 3 — Dataset preparation with PuckTrick noise injection
# =============================================================================
# NOTA: in questa variante "labels-only" la *column_to_insert_noise* puo'
# essere 'Label' o 'label_generic' (le 2 colonne target del dataset).
# Il metodo PuckTrick "labels" corrompe la colonna target indicata; dopo
# l'encoding, la colonna sporca diventa la corrispondente y_train.
#   - 'Label'         -> dopo encoding -> y_tr_mul (multiclass target)
#   - 'label_generic' -> dopo encoding -> y_tr_bin (binary target)

def prepare_whole_dataset_from_scratch(column_to_insert_noise, percentage, noise_type):
    global CAT_IDXS, CAT_DIMS, label_classes

    imp_df            = pd.read_csv('models/important_features.csv')
    important_features = imp_df['feature'].tolist()

    KEEP_ALWAYS = {'Label', 'label_generic', 'Timestamp'}

    sdf_full  = spark.read.parquet(f'{PATH}/all_elaborated.parquet')
    all_cols  = set(sdf_full.columns)
    cols_to_keep = [c for c in sdf_full.columns
                    if c in KEEP_ALWAYS or c in important_features]
    sdf_full  = sdf_full.select(*cols_to_keep)

    FEATURE_COLS = [c for c in important_features if c in set(cols_to_keep)]
    CAT_COLS  = [c for c in CATEGORICAL_FEATURES if c in FEATURE_COLS]
    BIN_COLS  = [c for c in BINARY_FEATURES      if c in FEATURE_COLS]
    CONT_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS and c not in BIN_COLS]

    ts_dtype = dict(sdf_full.dtypes).get('Timestamp', 'string')
    if ts_dtype == 'string':
        sdf_full = sdf_full.withColumn(
            'Timestamp',
            F.to_timestamp(F.col('Timestamp'), 'dd/MM/yyyy HH:mm:ss')
        )

    dtypes_map = dict(sdf_full.dtypes)
    for c in FEATURE_COLS:
        if dtypes_map.get(c) not in ('double', 'float'):
            sdf_full = sdf_full.withColumn(c, F.col(c).cast('double'))

    n_before = sdf_full.count()
    sdf_full = sdf_full.filter(F.year(F.col('Timestamp')) > 1970)
    n_dropped = n_before - sdf_full.count()
    if n_dropped:
        print(f"🗑️  Removed {n_dropped:,} rows with year 1970")

    if PERCENTAGE_TO_USE < 1.0:
        fractions = {
            row['label_generic']: PERCENTAGE_TO_USE
            for row in sdf_full.select('label_generic').distinct().collect()
        }
        sdf_sampled = sdf_full.sampleBy('label_generic', fractions=fractions,
                                        seed=CURRENT_RANDOM_SEED)
    else:
        sdf_sampled = sdf_full

    sdf_sampled = sdf_sampled.orderBy('Timestamp')

    from pyspark.sql.window import Window
    w_all       = Window.orderBy('Timestamp')
    sdf_sampled = sdf_sampled.withColumn('_row_id', F.row_number().over(w_all) - 1)
    sdf_sampled = sdf_sampled.withColumn('_group',  (F.col('_row_id') % 3).cast('int'))

    train_clean = sdf_sampled.filter(F.col('_group') < 2).drop('_row_id', '_group')
    temp_clean  = sdf_sampled.filter(F.col('_group') == 2).drop('_row_id', '_group')

    w_temp      = Window.orderBy('Timestamp')
    temp_clean  = temp_clean.withColumn('_temp_id', F.row_number().over(w_temp) - 1)
    val_clean   = temp_clean.filter((F.col('_temp_id') % 2) == 0).drop('_temp_id')
    test_clean  = temp_clean.filter((F.col('_temp_id') % 2) == 1).drop('_temp_id')

    indexer       = StringIndexer(inputCol='Label', outputCol='Label_enc', handleInvalid='keep')
    indexer_model = indexer.fit(sdf_sampled.drop('_row_id', '_group'))
    label_classes = list(indexer_model.labels)

    def apply_label_encoding(sdf):
        sdf = sdf.withColumn('label_generic_enc', col('label_generic').cast('int'))
        sdf = indexer_model.transform(sdf)
        sdf = sdf.withColumn('Label_enc', col('Label_enc').cast('int'))
        return sdf

    def make_strategy_labels(affected, percentage):
        """
        Strategia specializzata per il metodo PuckTrick "labels".
        Sporca la colonna target *affected* (es. 'Label' o 'label_generic').
        """
        return {
            "selection_criteria": "all",
            "percentage": percentage,
            "mode": "new",
            "perturbate_data": {"distribution": "random", "param": {}},
            "affected_features": affected,
        }

    # Solo metodo "labels" — gli altri rami sono rimossi in questa variante.
    if noise_type != "labels":
        raise ValueError(
            f"In questa variante e' supportato SOLO noise_type='labels' (ricevuto: {noise_type})"
        )

    # ─────────────────────────────────────────────────────────────────────
    # FIX: PuckTrick "labels" richiede una colonna NUMERICA (internamente
    # esegue un cast a BIGINT — vedi pucktrick/backends/spark_backend/
    # utils_spark.py:196). 'Label' nel dataset originale e' una STRING
    # ('Benign', 'DDoS', ...), quindi sporcarla direttamente fa esplodere
    # Spark con CAST_INVALID_INPUT.
    # Soluzione: applichiamo il label encoding PRIMA del noise e poi
    # sporchiamo la colonna ENCODED corrispondente:
    #   'Label'         -> 'Label_enc'          (int multiclass)
    #   'label_generic' -> 'label_generic_enc'  (int binary)
    # La semantica e' identica a quella richiesta: stiamo sporcando gli
    # indici di classe che diventeranno y_tr_mul / y_tr_bin.
    # ─────────────────────────────────────────────────────────────────────
    TARGET_TO_ENCODED = {
        'Label':         'Label_enc',
        'label_generic': 'label_generic_enc',
    }
    if column_to_insert_noise not in TARGET_TO_ENCODED:
        raise ValueError(
            f"Per il metodo 'labels' la colonna deve essere una tra "
            f"{list(TARGET_TO_ENCODED.keys())}, ricevuto: {column_to_insert_noise}"
        )
    encoded_target_col = TARGET_TO_ENCODED[column_to_insert_noise]

    # 1) Encoding PRIMA del noise (su tutti e tre gli split).
    train_clean_enc = apply_label_encoding(train_clean)
    val_clean       = apply_label_encoding(val_clean)
    test_clean      = apply_label_encoding(test_clean)

    # 2) Applica il noise PuckTrick "labels" sulla colonna ENCODED (int).
    strategy    = make_strategy_labels(encoded_target_col, percentage)
    OBJ         = PuckTrick(dataframe=train_clean_enc, engine=Engine.SPARK)
    _, dirty_train = OBJ.labels(OBJ.original, strategy=strategy)

    dirty_train = dirty_train.drop('_pucktrick_row_id')
    dirty_train = dirty_train.orderBy('Timestamp')
    val_clean   = val_clean.orderBy('Timestamp')
    test_clean  = test_clean.orderBy('Timestamp')
    print(f"sporcato TRAIN con pucktrick labels su y_target='{column_to_insert_noise}' "
          f"(colonna encoded: '{encoded_target_col}') al {percentage*100:.1f}%")

    # NOTA: NON ri-applichiamo apply_label_encoding su dirty_train, perche'
    # sovrascriverebbe la colonna *_enc sporca con i valori puliti ricavati
    # dall'indexer_model (che a sua volta legge la colonna 'Label' originale
    # STRING, rimasta intatta).

    dirty_train = dirty_train.drop('Timestamp')
    val_clean   = val_clean.drop('Timestamp')
    test_clean  = test_clean.drop('Timestamp')

    pdf_full_clean, cat_encoders, cat_dims_dict = preprocess_to_pandas(
        sdf_sampled.drop('_row_id', '_group'), CONT_COLS, CAT_COLS, BIN_COLS
    )

    def preprocess_with_encoders(sdf, continuous_features, categorical_features,
                                  binary_features, encoders):
        pdf      = sdf.toPandas()
        available = set(pdf.columns)
        cont_cols = [c for c in continuous_features  if c in available]
        cat_cols  = [c for c in categorical_features if c in available]
        bin_cols  = [c for c in binary_features      if c in available]
        for c in cont_cols:
            pdf[c] = pd.to_numeric(pdf[c], errors='coerce')
        pdf[cont_cols] = pdf[cont_cols].fillna(0.0)
        for c in cat_cols:
            le = encoders.get(c)
            if le is None:
                le = LabelEncoder()
                pdf[c] = le.fit_transform(pdf[c].astype(str))
            else:
                try:
                    pdf[c] = le.transform(pdf[c].astype(str))
                except ValueError:
                    mapping = {cls: i for i, cls in enumerate(le.classes_)}
                    pdf[c]  = pdf[c].astype(str).map(mapping).fillna(0).astype(int)
        for c in bin_cols:
            pdf[c] = pd.to_numeric(pdf[c], errors='coerce').fillna(0).astype(int)
        return pdf

    pdf_train = preprocess_with_encoders(dirty_train, CONT_COLS, CAT_COLS, BIN_COLS, cat_encoders)
    pdf_val   = preprocess_with_encoders(val_clean,   CONT_COLS, CAT_COLS, BIN_COLS, cat_encoders)
    pdf_test  = preprocess_with_encoders(test_clean,  CONT_COLS, CAT_COLS, BIN_COLS, cat_encoders)

    CAT_IDXS = [FEATURE_COLS.index(c) for c in CAT_COLS]
    CAT_DIMS = [cat_dims_dict[c] for c in CAT_COLS]

    X_train_2d = pdf_train[FEATURE_COLS].values.astype(np.float32)
    X_val_2d   = pdf_val[FEATURE_COLS].values.astype(np.float32)
    X_test_2d  = pdf_test[FEATURE_COLS].values.astype(np.float32)

    for arr in [X_train_2d, X_val_2d, X_test_2d]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(arr, -np.finfo(np.float32).max, np.finfo(np.float32).max, out=arr)

    # ATTENZIONE: in questa variante le y_train sono "sporche" (PuckTrick
    # labels ha modificato la colonna target prima dell'encoding).
    y_tr_bin_2d  = pdf_train['label_generic_enc'].values.astype(int)
    y_tr_mul_2d  = pdf_train['Label_enc'].values.astype(int)
    y_val_bin_2d = pdf_val['label_generic_enc'].values.astype(int)
    y_val_mul_2d = pdf_val['Label_enc'].values.astype(int)
    y_te_bin_2d  = pdf_test['label_generic_enc'].values.astype(int)
    y_te_mul_2d  = pdf_test['Label_enc'].values.astype(int)

    # Filtra classi rare
    classes, counts = np.unique(y_tr_mul_2d, return_counts=True)
    rare = classes[counts < MIN_SAMPLES_PER_CLASS]
    if len(rare) > 0:
        rare_labels = [label_classes[c] for c in rare if c < len(label_classes)]
        print(f"⚠️  Dropping {len(rare)} rare classes (< {MIN_SAMPLES_PER_CLASS} samples): {rare_labels}")
        keep_tr = ~np.isin(y_tr_mul_2d, rare)
        keep_va = ~np.isin(y_val_mul_2d, rare)
        keep_te = ~np.isin(y_te_mul_2d, rare)
        X_train_2d, y_tr_bin_2d, y_tr_mul_2d = X_train_2d[keep_tr], y_tr_bin_2d[keep_tr], y_tr_mul_2d[keep_tr]
        X_val_2d,   y_val_bin_2d, y_val_mul_2d = X_val_2d[keep_va], y_val_bin_2d[keep_va], y_val_mul_2d[keep_va]
        X_test_2d,  y_te_bin_2d,  y_te_mul_2d  = X_test_2d[keep_te], y_te_bin_2d[keep_te], y_te_mul_2d[keep_te]

    train_classes = np.unique(y_tr_mul_2d)
    keep_va_cls   = np.isin(y_val_mul_2d, train_classes)
    keep_te_cls   = np.isin(y_te_mul_2d,  train_classes)
    if not keep_va_cls.all():
        X_val_2d, y_val_bin_2d, y_val_mul_2d = X_val_2d[keep_va_cls], y_val_bin_2d[keep_va_cls], y_val_mul_2d[keep_va_cls]
    if not keep_te_cls.all():
        X_test_2d, y_te_bin_2d, y_te_mul_2d = X_test_2d[keep_te_cls], y_te_bin_2d[keep_te_cls], y_te_mul_2d[keep_te_cls]

    sorted_classes = sorted(train_classes)
    remap_mul      = {int(old): new for new, old in enumerate(sorted_classes)}
    y_tr_mul_2d    = np.vectorize(remap_mul.get)(y_tr_mul_2d).astype(int)
    y_val_mul_2d   = np.vectorize(remap_mul.get)(y_val_mul_2d).astype(int)
    y_te_mul_2d    = np.vectorize(remap_mul.get)(y_te_mul_2d).astype(int)
    label_classes  = [label_classes[c] for c in sorted_classes if c < len(label_classes)]
    print(f"📋  Remapped {len(sorted_classes)} multiclass labels to 0..{len(sorted_classes)-1}")

    cont_idxs = [FEATURE_COLS.index(c) for c in CONT_COLS]
    if cont_idxs:
        scaler = StandardScaler()
        scaler.fit(X_train_2d[:, cont_idxs])
        X_train_2d[:, cont_idxs] = scaler.transform(X_train_2d[:, cont_idxs])
        X_val_2d[:,   cont_idxs] = scaler.transform(X_val_2d[:,   cont_idxs])
        X_test_2d[:,  cont_idxs] = scaler.transform(X_test_2d[:,  cont_idxs])

    def build_windows_from_arrays(X, yb, ym):
        Xw, ybw, ymw = [], [], []
        for s in range(0, len(X) - WINDOW_SIZE + 1, STEP_SIZE):
            Xw.append(X[s:s + WINDOW_SIZE])
            ybw.append(yb[s + WINDOW_SIZE - 1])
            ymw.append(ym[s + WINDOW_SIZE - 1])
        return np.array(Xw, dtype=np.float32), np.array(ybw), np.array(ymw)

    X_train_3d, y_tr_bin_3d, y_tr_mul_3d  = build_windows_from_arrays(X_train_2d, y_tr_bin_2d, y_tr_mul_2d)
    X_val_3d,   y_val_bin_3d, y_val_mul_3d = build_windows_from_arrays(X_val_2d,   y_val_bin_2d, y_val_mul_2d)

    print(f"\n📐  TabNet  → train {X_train_2d.shape}, val {X_val_2d.shape}")
    print(f"📐  CNN-LSTM → train {X_train_3d.shape}, val {X_val_3d.shape}")

    corr_cols   = FEATURE_COLS + ['label_generic_enc']
    corr_matrix = pdf_train[[c for c in corr_cols if c in pdf_train.columns]].corr().round(4).to_dict()
    print(f"📊  Correlation matrix computed ({len(corr_cols)} cols)")

    return (
        FEATURE_COLS,
        X_train_2d, X_val_2d,
        y_tr_bin_2d, y_tr_mul_2d,
        y_val_bin_2d, y_val_mul_2d,
        X_train_3d, X_val_3d,
        y_tr_bin_3d, y_tr_mul_3d,
        y_val_bin_3d, y_val_mul_3d,
        corr_matrix,
    )

# =============================================================================
# STEP 4 — Run a single experiment
# =============================================================================

def _run_training_pipeline(dataset_tuple, label_model):
    """
    Esegue TabNet + CNN-LSTM su una tupla di dataset gia' pronta.
    Centralizza il blocco di training per evitare duplicazione.
    """
    global tabnet_trial_artifacts, cnn_lstm_trial_artifacts
    tabnet_trial_artifacts   = {}
    cnn_lstm_trial_artifacts = {}
    reload_all_trial_metadata()

    best_tabnet_trial_num   = 7
    best_cnn_lstm_trial_num = 1
    best_tabnet   = tabnet_trial_artifacts[best_tabnet_trial_num]["params"]
    best_cnn_lstm = cnn_lstm_trial_artifacts[best_cnn_lstm_trial_num]["params"]

    (FEATURE_NAMES,
     X_base_train_2d, X_base_val_2d,
     y_base_tr_bin_2d, y_base_tr_mul_2d,
     y_base_val_bin_2d, y_base_val_mul_2d,
     X_base_train, X_base_val,
     y_base_tr_bin, y_base_tr_mul,
     y_base_val_bin, y_base_val_mul,
     corr_matrix) = dataset_tuple

    new_batch_size = 4096

    tabnet_multitask_objective(
        X_base_train_2d, X_base_val_2d,
        y_base_tr_bin_2d, y_base_tr_mul_2d,
        y_base_val_bin_2d, y_base_val_mul_2d,
        best_tabnet['N_a'], best_tabnet['N_steps'], best_tabnet['gamma'],
        (best_tabnet['lambda_sparse'] * ((0.001 / PERCENTAGE_TO_USE) ** 0.5)),
        best_tabnet['lr'],
        new_batch_size,
        best_tabnet['mask_type'],
        verbose=0, trial=None,
        label_model=label_model,
        feature_names=FEATURE_NAMES,
        corr_matrix=corr_matrix
    )

    cnn_lstm_multitask_objective(
        X_base_train, X_base_val,
        y_base_tr_bin, y_base_tr_mul,
        y_base_val_bin, y_base_val_mul,
        best_cnn_lstm['nb_filters'], best_cnn_lstm['kernel_size'],
        best_cnn_lstm['lstm_units_1'], best_cnn_lstm['lstm_units_2'],
        best_cnn_lstm['dropout'], best_cnn_lstm['lr'] / 2,
        batch_size=new_batch_size,
        verbose=0,
        label_model=label_model,
        feature_names=FEATURE_NAMES,
        corr_matrix=corr_matrix
    )


def run_single_experiment_load_or_create(colonna_da_sporcare, metodo, pct):
    """
    Variante "smart" per la modalita' JUST_COMPILE_DATASETS = False:
      1) Se il dataset e' gia' su disco -> lo carica
      2) Altrimenti -> lo crea al volo (IN RAM, SENZA salvarlo su disco)
      3) In entrambi i casi, esegue training di TabNet e CNN-LSTM.
    """
    label_model = f'Experiment_{metodo}_{colonna_da_sporcare.replace("/", "_")}_{pct*100:.1f}'

    if dataset_already_saved(label_model):
        print(f"📥  Loading pre-compiled dataset from disk: {label_model}")
        dataset_tuple = load_dataset_from_disk(label_model)
    else:
        print(f"⚙️  Dataset '{label_model}' non presente su disco — "
              f"creazione al volo (IN RAM, senza salvataggio)")
        dataset_tuple = prepare_whole_dataset_from_scratch(
            colonna_da_sporcare, pct, metodo
        )

    _run_training_pipeline(dataset_tuple, label_model)


# =============================================================================
# STEP 5 — Main experiment loop
# =============================================================================

if __name__ == '__main__':

    # ── Colonne y_target da sporcare con PuckTrick "labels" ───────────────
    # Sono le 2 etichette di classificazione del dataset CAVAS:
    #   - 'Label'         → label multiclass (poi diventa y_tr_mul)
    #   - 'label_generic' → label binary    (poi diventa y_tr_bin)
    Y_TARGET_COLUMNS_TO_DIRTY = ['Label', 'label_generic']

    # In questa variante usiamo SOLO il metodo "labels"
    PUCKTRICK_METHOD = 'labels'

    PERCENTAGES = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75]

    for seed in RANDOM_SEEDS:

        # ── Aggiorna i global per-seed ────────────────────────────────
        CURRENT_RANDOM_SEED = seed
        CURRENT_RS_DIR = (
            f"{BASE_DIR}/complete_experiments/experiment_rs{seed}/"
            f"{LABELS_EXPERIMENT_SUBDIR}"
        )
        PATH_IMG = f"{CURRENT_RS_DIR}/images"

        # Crea le sottocartelle di questo seed
        os.makedirs(CURRENT_RS_DIR, exist_ok=True)
        os.makedirs(PATH_IMG,       exist_ok=True)

        # Fissa tutti i seed per riproducibilita'
        pl.seed_everything(seed)

        print(f"\n{'='*60}")
        print(f"  RANDOM SEED: {seed}  ->  {CURRENT_RS_DIR}")
        if JUST_COMPILE_DATASETS:
            print(f"  MODE: JUST_COMPILE_DATASETS — solo creazione e salvataggio dataset")
        else:
            print(f"  MODE: TRAINING — carica dataset da disco o crea al volo, "
                  f"poi addestra i modelli")
        print(f"  PUCKTRICK METHOD: {PUCKTRICK_METHOD} (solo questo)")
        print(f"  Y-TARGET COLUMNS: {Y_TARGET_COLUMNS_TO_DIRTY}")
        print(f"{'='*60}\n")

        # ── Helper interno per costruire i label degli esperimenti ─────
        def _label(colonna, metodo, pct):
            return f'Experiment_{metodo}_{colonna.replace("/", "_")}_{pct*100:.1f}'

        # =====================================================================
        # BRANCH A — Crea e salva tutti i dataset su disco
        # =====================================================================
        if JUST_COMPILE_DATASETS:

            for colonna_da_sporcare in Y_TARGET_COLUMNS_TO_DIRTY:
                for pct in PERCENTAGES:

                    label = _label(colonna_da_sporcare, PUCKTRICK_METHOD, pct)

                    if dataset_already_saved(label):
                        print(f"[SKIP] Dataset gia' presente: {label}")
                        continue

                    try:
                        dataset_tuple = prepare_whole_dataset_from_scratch(
                            colonna_da_sporcare, pct, PUCKTRICK_METHOD
                        )
                        save_dataset_to_disk(label, dataset_tuple)
                    except Exception as e:
                        print(f"[rs={seed}] Errore dataset: {PUCKTRICK_METHOD} su "
                              f"y='{colonna_da_sporcare}' al {pct*100:.1f}%: {e}")
                    finally:
                        clear_memory()

        # =====================================================================
        # BRANCH B — Carica dataset da disco (o crea al volo) e addestra i modelli
        # =====================================================================
        else:

            for colonna_da_sporcare in Y_TARGET_COLUMNS_TO_DIRTY:
                for pct in PERCENTAGES:
                    
                    print("="*44)
                    print(_label(colonna_da_sporcare, PUCKTRICK_METHOD, pct))
                    print("="*44)

                    if experiment_already_exists(_label(colonna_da_sporcare, PUCKTRICK_METHOD, pct)):
                        print(f"[SKIP] Esperimento gia' esistente: "
                              f"{PUCKTRICK_METHOD} su y='{colonna_da_sporcare}' "
                              f"al {pct*100:.1f}%")
                        continue

                    try:
                        run_single_experiment_load_or_create(
                            colonna_da_sporcare, PUCKTRICK_METHOD, pct
                        )
                    except Exception as e:
                        print(f"[rs={seed}] Errore training: {PUCKTRICK_METHOD} su "
                              f"y='{colonna_da_sporcare}' al {pct*100:.1f}%: {e}")
                    finally:
                        clear_memory()

    print("\n  All labels-only experiments completed (all seeds).")
    spark.stop()