from pucktrick import PuckTrick, Engine, get_spark_session
import time

from pathlib import Path
from datetime import datetime, timezone
import csv
import json

from pyspark.sql import functions as F


# -----------------------------
# Stopwatch
# -----------------------------
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


# -----------------------------
# Config stress test
# -----------------------------
MASTER_PRIVATE_IP = "10.0.1.8"
MASTER_URL = f"spark://{MASTER_PRIVATE_IP}:7077"

TIME_LIMIT_SECONDS = (60 * 60) / 4
START_ROWS = 1_000_000
MAX_ITERS = 15

metodo = "duplicated"
strategy = {
    "affected_features": ["f1"],
    "selection_criteria": "all",
    "percentage": 0.5,
    "mode": "new",
    "perturbate_data": {"distribution": "random"},
}


# -----------------------------
# CSV logging
# -----------------------------
CSV_PATH = Path.cwd() / "spark_stress_runs.csv"

CSV_FIELDS = [
    "data_esecuzione",
    "metodo",
    "backend",
    "iterazione",
    "num_righe",
    "stopwatch_sec_iter",
    "stopwatch_sec_tot",
    "strategy_json",
]


def now_iso_local():
    return datetime.now(timezone.utc).astimezone().isoformat()


def strategy_to_json(s: dict) -> str:
    return json.dumps(s, ensure_ascii=False, sort_keys=True)


def append_row(*, iterazione: int, num_righe: int, elapsed_iter: float, elapsed_tot: float):
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if (not file_exists) or (f.tell() == 0):
            writer.writeheader()
        writer.writerow({
            "data_esecuzione": now_iso_local(),
            "metodo": metodo,
            "backend": "SPARK",
            "iterazione": int(iterazione),
            "num_righe": int(num_righe),
            "stopwatch_sec_iter": float(elapsed_iter),
            "stopwatch_sec_tot": float(elapsed_tot),
            "strategy_json": strategy_to_json(strategy),
        })


def make_dataset(n_rows: int):
    """
    Dataset sintetico 100% Spark: id + f1 random uniforme [0,1). [web:46]
    """
    return (
        spark.range(0, n_rows)
            .withColumn("f1", F.rand(seed=42))
    )


# -----------------------------
# Remote cluster config (per PuckTrick) + Spark session
# -----------------------------
cluster_config = PuckTrick.make_remote_cluster_config(
    master_url=MASTER_URL,
    num_executors=4,
    executor_memory="13g",
    driver_memory="8g",
    driver_host=MASTER_PRIVATE_IP,
)

# Create or obtain SparkSession via the library (ensures consistent config)
spark = get_spark_session(remote_cluster=cluster_config)


# -----------------------------
# Stress loop
# -----------------------------
total_elapsed = 0.0
rows = START_ROWS
OBJ = None

print(f"Start stress test: limit={TIME_LIMIT_SECONDS}s, start_rows={START_ROWS}, master={MASTER_URL}")
print(f"CSV log: {CSV_PATH}")

for i in range(1, MAX_ITERS + 1):
    if total_elapsed >= TIME_LIMIT_SECONDS:
        break

    df = make_dataset(rows)

    # Assicura che il dataframe abbia la colonna di identificazione richiesta
    # dalla libreria (`PuckTrick.PUCK_ID`) prima di passarlo ai metodi.
    df = df.withColumn(PuckTrick.PUCK_ID, F.monotonically_increasing_id())

    # Materializza per non misurare solo lazy planning
    df = df.cache()
    _ = df.count()

    # Crea una singola istanza di PuckTrick alla prima iterazione e la riusa.
    if OBJ is None:
        OBJ = PuckTrick(dataframe=df, engine=Engine.SPARK, remote_cluster=cluster_config)
    else:
        # Aggiorna l'`original_df` interno dell'oggetto con il nuovo dataframe
        # (già dotato di `PUCK_ID`) in modo che i metodi possano lavorare.
        OBJ.original_df = df

    sw = Stopwatch()
    sw.start()

    _ = OBJ.outlier(OBJ.original, strategy=strategy)

    sw.stop()
    elapsed_iter = sw.elapsed()
    total_elapsed += elapsed_iter

    append_row(iterazione=i, num_righe=rows, elapsed_iter=elapsed_iter, elapsed_tot=total_elapsed)
    print(f"[iter {i}] rows={rows} iter={elapsed_iter:.3f}s total={total_elapsed:.3f}s")

    # Cleanup cache per non saturare RAM tra iterazioni
    df.unpersist(blocking=False)

    rows *= 2

print(f"STOP: total_elapsed={total_elapsed:.3f}s (limit={TIME_LIMIT_SECONDS}s), last_rows={rows//2}")

spark.stop()
