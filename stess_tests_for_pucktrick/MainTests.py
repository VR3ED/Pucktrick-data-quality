from pucktrick import PuckTrick, Engine, get_spark_session
import pandas as pd
import time
import copy
import math
import json

from pathlib import Path
from datetime import datetime, timezone
import csv


# ─────────────────────────────────────────────────────────────────
# Stopwatch
# ─────────────────────────────────────────────────────────────────
class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.stop_time = time.perf_counter()

    def elapsed(self):
        if self.start_time and self.stop_time:
            return self.stop_time - self.start_time
        return 0.0


# ─────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "data_esecuzione",
    "metodo",
    "backend",
    "num_righe",
    "stopwatch_sec",
    "strategy_json",
]


def now_iso_local():
    return datetime.now(timezone.utc).astimezone().isoformat()


def strategy_to_json(strategy: dict) -> str:
    return json.dumps(strategy, ensure_ascii=False, sort_keys=True)


def append_stopwatch_row(*, metodo, backend, num_righe, elapsed_sec, strategy, out_path):
    file_exists = out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists or f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "data_esecuzione": now_iso_local(),
            "metodo":          metodo,
            "backend":         backend,
            "num_righe":       int(num_righe),
            "stopwatch_sec":   float(elapsed_sec),
            "strategy_json":   strategy_to_json(strategy),
        })
    return out_path


# ─────────────────────────────────────────────────────────────────
# Checkpoint: legge CSV esistente, pulisce righe -1, restituisce
# un set di test già completati con successo.
#
# La chiave di ogni test è: (metodo, backend, col, num_righe)
# dove col viene estratta dal campo affected_features di strategy_json.
# ─────────────────────────────────────────────────────────────────
def load_completed(csv_path: Path) -> set:
    """
    Legge il CSV se esiste, rimuove le righe con stopwatch_sec == -1
    (riscrivendo il file pulito), e restituisce un set di tuple
    (metodo, backend, col, num_righe) già completate con successo.
    """
    if not csv_path.exists():
        print(f"[Checkpoint] Nessun CSV precedente trovato: {csv_path.name}")
        return set()

    df = pd.read_csv(csv_path)
    n_total  = len(df)
    n_errors = (df["stopwatch_sec"] == -1).sum()

    # Rimuove le righe di errore e riscrive il file pulito
    df_clean = df[df["stopwatch_sec"] != -1].copy()
    df_clean.to_csv(csv_path, index=False)

    print(f"[Checkpoint] {csv_path.name}: {n_total} righe lette, "
          f"{n_errors} errori rimossi, {len(df_clean)} valide mantenute.")

    # Costruisce il set delle chiavi già completate
    completed = set()
    for _, row in df_clean.iterrows():
        try:
            col = json.loads(row["strategy_json"])["affected_features"][0]
        except Exception:
            col = "__unknown__"
        completed.add((row["metodo"], row["backend"], col, int(row["num_righe"])))

    print(f"[Checkpoint] Test già completati: {len(completed)}")
    return completed


# ─────────────────────────────────────────────────────────────────
# Configurazione
# ─────────────────────────────────────────────────────────────────
CSV_PART1   = Path.cwd() / "stopwatch_runs.csv"
CSV_PART2   = Path.cwd() / "stopwatch_runs_16M.csv"

TARGET_ROWS = 16_000_000

BASE_STRATEGY = {
    "affected_features": ["f1"],
    "selection_criteria": "all",
    "percentage": 0.5,
    "mode": "new",
    "perturbate_data": {"distribution": "random"},
}

ALL_METHODS = ["duplicated", "missing", "noise", "outlier", "labels"]
COLUMNS     = ["f1", "f2", "f3", "f4", "f5", "date", "target"]

MASTER_PRIVATE_IP = "10.0.1.8"
MASTER_URL        = f"spark://{MASTER_PRIVATE_IP}:7077"

cluster_config = PuckTrick.make_remote_cluster_config(
    master_url      = MASTER_URL,
    num_executors   = 4,
    executor_memory = "15g",
    driver_memory   = "8g",
    driver_host     = MASTER_PRIVATE_IP,
    executor_cores=4 
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def run_method(obj, method_name, train_df, strategy):
    return getattr(obj, method_name)(train_df, strategy=strategy)


def materialize_if_spark(obj):
    targets = obj if isinstance(obj, (tuple, list)) else [obj]
    for item in targets:
        if hasattr(item, "_jdf"):
            item.count()
            return


def expand_to_target(df: pd.DataFrame, target: int) -> pd.DataFrame:
    n = len(df)
    if n >= target:
        return df.iloc[:target].reset_index(drop=True)
    repeats  = math.ceil(target / n)
    expanded = pd.concat([df] * repeats, ignore_index=True)
    return expanded.iloc[:target].reset_index(drop=True)


def init_spark(dataset_pd: pd.DataFrame):
    """
    Crea SparkSession e wrapper PuckTrick.
    Stampa l'errore reale in caso di fallimento.
    Restituisce (spark, obj_sp) oppure (None, None) se fallisce.
    """
    try:
        spark = get_spark_session(remote_cluster=cluster_config)
        dataset_sp = spark.createDataFrame(dataset_pd)
        obj_sp = PuckTrick(
            dataframe=dataset_sp,
            engine=Engine.SPARK,
            remote_cluster=cluster_config,
        )
        print("[Spark] Sessione inizializzata correttamente.")
        return spark, obj_sp
    except Exception as e:
        print(f"[Spark] ERRORE di inizializzazione: {e}")
        return None, None


def run_experiment(dataset_pd, spark, obj_sp, out_path, label, completed: set):
    """
    Esegue il loop metodi x colonne su entrambi i backend.

    - Salta i test già presenti in `completed` (chiave: metodo, backend, col, num_righe).
    - Non chiama mai spark.stop() a metà loop (vedi commenti inline).
    - In caso di errore Spark invalida solo obj_sp, mai la sessione.
    """
    N = len(dataset_pd)
    print(f"\n{'#'*65}")
    print(f"  {label}  |  {N:,} righe  ->  {out_path.name}")
    print(f"{'#'*65}")

    for metodo in ALL_METHODS:
        print(f"\n{'='*60}")
        print(f"METODO: {metodo}")
        print(f"{'='*60}")

        for col in COLUMNS:
            local_strategy = copy.deepcopy(BASE_STRATEGY)
            local_strategy["affected_features"] = [col]
            print(f"--- Colonna: {col} ---")

            # ── PANDAS ────────────────────────────────────────────────────
            key_pd = (metodo, "PANDAS", col, N)
            if key_pd in completed:
                print(f"[PANDAS/{metodo}/{col}] gia completato, skip.")
            else:
                try:
                    obj_pd = PuckTrick(dataframe=dataset_pd, engine=Engine.PANDAS)
                    sw = Stopwatch()
                    sw.start()
                    run_method(obj_pd, metodo, dataset_pd, local_strategy)
                    sw.stop()
                    elapsed = sw.elapsed()
                    append_stopwatch_row(
                        metodo=metodo, backend="PANDAS", num_righe=N,
                        elapsed_sec=elapsed, strategy=local_strategy, out_path=out_path,
                    )
                    completed.add(key_pd)
                    print(f"[PANDAS/{metodo}/{col}] {elapsed:.4f}s")
                except Exception as e:
                    print(f"[PANDAS/{metodo}/{col}] ERRORE: {e}")
                    append_stopwatch_row(
                        metodo=metodo, backend="PANDAS", num_righe=N,
                        elapsed_sec=-1, strategy=local_strategy, out_path=out_path,
                    )

            # ── SPARK ─────────────────────────────────────────────────────
            key_sp = (metodo, "SPARK", col, N)
            if key_sp in completed:
                print(f"[SPARK/{metodo}/{col}] gia completato, skip.")
            else:
                try:
                    # Caso 1: spark non e mai partito -> ritenta init completa
                    if spark is None:
                        spark, obj_sp = init_spark(dataset_pd)

                    # Caso 2: spark vivo ma wrapper crashato -> ricrea solo wrapper
                    elif obj_sp is None:
                        dataset_sp = spark.createDataFrame(dataset_pd)
                        obj_sp = PuckTrick(
                            dataframe=dataset_sp,
                            engine=Engine.SPARK,
                            remote_cluster=cluster_config,
                        )

                    if obj_sp is not None:
                        sw = Stopwatch()
                        sw.start()
                        out_sp = run_method(obj_sp, metodo, obj_sp.original, local_strategy)
                        materialize_if_spark(out_sp)
                        sw.stop()
                        elapsed = sw.elapsed()
                        append_stopwatch_row(
                            metodo=metodo, backend="SPARK", num_righe=N,
                            elapsed_sec=elapsed, strategy=local_strategy, out_path=out_path,
                        )
                        completed.add(key_sp)
                        print(f"[SPARK/{metodo}/{col}] {elapsed:.4f}s")
                    else:
                        print(f"[SPARK/{metodo}/{col}] Spark non disponibile, skip.")

                except Exception as e:
                    print(f"[SPARK/{metodo}/{col}] ERRORE: {e}")
                    append_stopwatch_row(
                        metodo=metodo, backend="SPARK", num_righe=N,
                        elapsed_sec=-1, strategy=local_strategy, out_path=out_path,
                    )
                    # Non stoppare Spark: invalida solo il wrapper.
                    obj_sp = None

    return spark, obj_sp


# ═════════════════════════════════════════════════════════════════
# PARTE 1 — Dataset originale
# ═════════════════════════════════════════════════════════════════
print("\nCaricamento dataset originale...")
dataset_pd = pd.read_csv("fake_dataset.csv")
print(f"Dataset caricato: {len(dataset_pd):,} righe")
print(f"Metodi: {ALL_METHODS}")
print(f"Colonne: {COLUMNS}")

# Legge CSV esistente, pulisce errori, recupera test già fatti
completed_part1 = load_completed(CSV_PART1)

spark, obj_sp = init_spark(dataset_pd)

spark, obj_sp = run_experiment(
    dataset_pd = dataset_pd,
    spark      = spark,
    obj_sp     = obj_sp,
    out_path   = CSV_PART1,
    label      = "PARTE 1 — dataset originale",
    completed  = completed_part1,
)
print(f"\nParte 1 completata -> {CSV_PART1}")


# ═════════════════════════════════════════════════════════════════
# PARTE 2 — Dataset espanso a 16 M righe
# ═════════════════════════════════════════════════════════════════
print("\n\nRicaricamento dataset da zero per la parte 2...")
dataset_pd = pd.read_csv("fake_dataset.csv")
print(f"Dataset ricaricato: {len(dataset_pd):,} righe")
print(f"Espansione a {TARGET_ROWS:,} righe...")
dataset_pd_16M = expand_to_target(dataset_pd, TARGET_ROWS)
print(f"Dataset espanso: {len(dataset_pd_16M):,} righe")

# Legge CSV esistente della parte 2, pulisce errori, recupera test già fatti
completed_part2 = load_completed(CSV_PART2)

# Ricrea il wrapper Spark sul dataset espanso riutilizzando la sessione attiva
try:
    if spark is None:
        spark, obj_sp = init_spark(dataset_pd_16M)
    else:
        dataset_sp_16M = spark.createDataFrame(dataset_pd_16M)
        obj_sp = PuckTrick(
            dataframe=dataset_sp_16M,
            engine=Engine.SPARK,
            remote_cluster=cluster_config,
        )
        print("[Spark] Wrapper ricreato sul dataset espanso.")
except Exception as e:
    print(f"[Spark] ERRORE setup parte 2: {e}")
    obj_sp = None

spark, obj_sp = run_experiment(
    dataset_pd = dataset_pd_16M,
    spark      = spark,
    obj_sp     = obj_sp,
    out_path   = CSV_PART2,
    label      = f"PARTE 2 — dataset espanso ({TARGET_ROWS:,} righe)",
    completed  = completed_part2,
)
print(f"\nParte 2 completata -> {CSV_PART2}")


# ─────────────────────────────────────────────────────────────────
if spark is not None:
    spark.stop()

print("\nTutti gli esperimenti completati.")
print(f"  Parte 1 -> {CSV_PART1}")
print(f"  Parte 2 -> {CSV_PART2}")