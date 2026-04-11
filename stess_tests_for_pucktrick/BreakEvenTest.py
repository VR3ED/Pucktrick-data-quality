"""
BreakEvenTest.py
================
For each data-dirtying method (duplicated, missing, noise, outlier, labels),
measures execution time on both Pandas and Spark backends across iteratively
growing synthetic datasets (rows doubled each iteration).

The loop stops 2 iterations after ALL methods have crossed their individual
break-even point (i.e. the first iteration where Spark is faster than Pandas).

CSV log: breakeven_runs.csv  (one row per method × backend × iteration)
"""

from pucktrick import PuckTrick, Engine, get_spark_session
import pandas as pd
import numpy as np
import time
import copy

from pathlib import Path
from datetime import datetime, timezone
import csv
import json

from pyspark.sql import functions as F


# ─────────────────────────────────────────
# Stopwatch
# ─────────────────────────────────────────
class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.stop_time = time.perf_counter()

    def elapsed(self) -> float:
        if self.start_time is not None and self.stop_time is not None:
            return self.stop_time - self.start_time
        return 0.0


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
MASTER_PRIVATE_IP = "10.0.1.8"
MASTER_URL        = f"spark://{MASTER_PRIVATE_IP}:7077"

START_ROWS = 1_000_000
MAX_ITERS  = 60          # hard upper bound; break-even logic will stop earlier

ALL_METHODS = ["duplicated", "missing", "noise", "outlier", "labels"]

BASE_STRATEGY = {
    "affected_features": ["f3"],
    "selection_criteria": "all",
    "percentage": 0.5,
    "mode": "new",
    "perturbate_data": {"distribution": "random"},
}

# Possible values for the categorical column f3 (3 classes)
F3_CATEGORIES = ["A", "B", "C"]

# How many extra iterations to run once ALL methods have crossed their
# individual break-even point before stopping.
EXTRA_ITERS_AFTER_ALL_BREAKEVEN = 2


# ─────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────
CSV_PATH = Path.cwd() / "breakeven_runs.csv"

CSV_FIELDS = [
    "data_esecuzione",
    "metodo",
    "backend",
    "iterazione",
    "num_righe",
    "stopwatch_sec",
    "breakeven_reached",   # True on the exact iteration Spark first beat Pandas
    "strategy_json",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _strategy_json(s: dict) -> str:
    return json.dumps(s, ensure_ascii=False, sort_keys=True)


def append_row(
    *,
    metodo: str,
    backend: str,
    iterazione: int,
    num_righe: int,
    elapsed_sec: float,
    breakeven_reached: bool,
    strategy: dict,
) -> None:
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists or f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "data_esecuzione":   _now_iso(),
            "metodo":            metodo,
            "backend":           backend,
            "iterazione":        int(iterazione),
            "num_righe":         int(num_righe),
            "stopwatch_sec":     float(elapsed_sec),
            "breakeven_reached": breakeven_reached,
            "strategy_json":     _strategy_json(strategy),
        })


# ─────────────────────────────────────────
# Dataset factories
# ─────────────────────────────────────────
def make_spark_dataset(spark, n_rows: int):
    """
    Purely Spark-native synthetic dataset: id + f3 (categorical, 3 classes).
    f3 is derived from a uniform random value mapped to one of F3_CATEGORIES
    so that classes are roughly balanced (~33 % each).
    The PuckTrick.PUCK_ID column is added so library methods work correctly.
    The result is cached and materialised (count) before timing begins.
    """
    df = (
        spark.range(0, n_rows)
             .withColumn(
                 "f3",
                 F.element_at(
                     F.array(*[F.lit(c) for c in F3_CATEGORIES]),
                     (F.rand(seed=42) * len(F3_CATEGORIES)).cast("int") + 1,
                 ),
             )
             .withColumn(PuckTrick.PUCK_ID, F.monotonically_increasing_id())
    )
    df = df.cache()
    df.count()   # force materialisation – avoids counting lazy planning time
    return df


def make_pandas_dataset(n_rows: int) -> pd.DataFrame:
    """
    Equivalent synthetic dataset generated entirely with NumPy/Pandas –
    avoids a potentially very expensive toPandas() call on large Spark DFs.
    f3 is a categorical column with 3 balanced classes (A, B, C).
    """
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id":               np.arange(n_rows, dtype=np.int64),
        # dtype=object (plain strings) invece di pd.Categorical:
        # PuckTrick scrive internamente valori sentinella arbitrari nella
        # colonna (es. "puck was here") e pd.Categorical con categorie fisse
        # li rifiuta con un errore. Con object dtype la colonna resta flessibile.
        "f3":               rng.choice(F3_CATEGORIES, size=n_rows).astype(object),
        PuckTrick.PUCK_ID:  np.arange(n_rows, dtype=np.int64),
    })


# ─────────────────────────────────────────
# Execution helpers
# ─────────────────────────────────────────
def run_method(obj: PuckTrick, method_name: str, train_df, strategy: dict):
    return getattr(obj, method_name)(train_df, strategy=strategy)


def materialize_if_spark(obj) -> None:
    """Force a Spark action so lazy execution is included in the timing."""
    targets = obj if isinstance(obj, (tuple, list)) else [obj]
    for item in targets:
        if hasattr(item, "_jdf"):
            item.count()
            return


# ─────────────────────────────────────────
# Spark cluster config
# ─────────────────────────────────────────
cluster_config = PuckTrick.make_remote_cluster_config(
    master_url      = MASTER_URL,
    num_executors   = 4,
    executor_cores  = 4,
    executor_memory = "15g",
    driver_memory   = "8g",
    driver_host     = MASTER_PRIVATE_IP,
)

spark = get_spark_session(remote_cluster=cluster_config)

# ─────────────────────────────────────────
# Break-even tracking state
# ─────────────────────────────────────────
# breakeven_found[method] = True once Spark first beats Pandas for that method
breakeven_found: dict[str, bool] = {m: False for m in ALL_METHODS}

# Iteration index at which ALL methods had crossed their break-even
all_breakeven_iter: int | None = None

# Reuse PuckTrick Spark objects across iterations (update original_df each time)
spark_puck_objects: dict[str, PuckTrick | None] = {m: None for m in ALL_METHODS}

# ─────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────
rows = START_ROWS

print(f"\n{'#'*65}")
print(f"  BreakEvenTest  |  start_rows={START_ROWS:,}  |  master={MASTER_URL}")
print(f"  Methods: {ALL_METHODS}")
print(f"  CSV log: {CSV_PATH}")
print(f"{'#'*65}\n")

for i in range(1, MAX_ITERS + 1):

    print(f"\n{'='*65}")
    print(f"  ITERATION {i:>3}  |  rows = {rows:>14,}")
    print(f"{'='*65}")

    # ── Build datasets for this iteration ────────────────────────────────
    df_sp = make_spark_dataset(spark, rows)
    df_pd = make_pandas_dataset(rows)

    iter_breakevens_this_round: list[str] = []   # methods that cross BE this iter

    # ── Run every method on both backends ────────────────────────────────
    for metodo in ALL_METHODS:
        local_strategy = copy.deepcopy(BASE_STRATEGY)
        print(f"\n  [{metodo}]")

        # ── Pandas ──────────────────────────────────────────────────────
        elapsed_pd: float | None = None
        try:
            obj_pd = PuckTrick(dataframe=df_pd, engine=Engine.PANDAS)
            sw = Stopwatch()
            sw.start()
            run_method(obj_pd, metodo, df_pd, local_strategy)
            sw.stop()
            elapsed_pd = sw.elapsed()
            print(f"    PANDAS : {elapsed_pd:>8.3f} s")
        except Exception as exc:
            print(f"    PANDAS ERROR: {exc}")
            elapsed_pd = None

        append_row(
            metodo=metodo,
            backend="PANDAS",
            iterazione=i,
            num_righe=rows,
            elapsed_sec=elapsed_pd if elapsed_pd is not None else -1.0,
            breakeven_reached=False,   # break-even is a Spark-side event
            strategy=local_strategy,
        )

        # ── Spark ───────────────────────────────────────────────────────
        elapsed_sp: float | None = None
        be_this_iter = False
        try:
            # Reuse the PuckTrick object; just swap in the new dataframe
            if spark_puck_objects[metodo] is None:
                spark_puck_objects[metodo] = PuckTrick(
                    dataframe=df_sp,
                    engine=Engine.SPARK,
                    remote_cluster=cluster_config,
                )
            else:
                spark_puck_objects[metodo].original_df = df_sp

            obj_sp = spark_puck_objects[metodo]
            sw = Stopwatch()
            sw.start()
            out_sp = run_method(obj_sp, metodo, obj_sp.original, local_strategy)
            materialize_if_spark(out_sp)   # force Spark action for accurate timing
            sw.stop()
            elapsed_sp = sw.elapsed()
            print(f"    SPARK  : {elapsed_sp:>8.3f} s")

        except Exception as exc:
            print(f"    SPARK ERROR: {exc}")
            elapsed_sp = None
            # Invalidate the object so it is recreated on the next iteration
            spark_puck_objects[metodo] = None

        # ── Break-even detection ─────────────────────────────────────────
        if (
            elapsed_pd is not None
            and elapsed_sp is not None
            and not breakeven_found[metodo]
            and elapsed_sp < elapsed_pd
        ):
            breakeven_found[metodo] = True
            be_this_iter = True
            iter_breakevens_this_round.append(metodo)
            print(
                f"    *** BREAK-EVEN reached: Spark ({elapsed_sp:.3f}s)"
                f" < Pandas ({elapsed_pd:.3f}s) ***"
            )

        append_row(
            metodo=metodo,
            backend="SPARK",
            iterazione=i,
            num_righe=rows,
            elapsed_sec=elapsed_sp if elapsed_sp is not None else -1.0,
            breakeven_reached=be_this_iter,
            strategy=local_strategy,
        )

    # ── End-of-iteration summary ─────────────────────────────────────────
    still_pending = [m for m, found in breakeven_found.items() if not found]
    print(f"\n  Break-even status: {dict(breakeven_found)}")
    if still_pending:
        print(f"  Still waiting on:  {still_pending}")

    # ── Clean up Spark cache before next iteration ────────────────────────
    df_sp.unpersist(blocking=False)

    # ── Stopping condition ───────────────────────────────────────────────
    if all(breakeven_found.values()):
        if all_breakeven_iter is None:
            # First iteration where everything has crossed
            all_breakeven_iter = i
            print(
                f"\n  >>> ALL methods crossed break-even at iteration {i}"
                f" (rows={rows:,})."
                f" Running {EXTRA_ITERS_AFTER_ALL_BREAKEVEN} more iteration(s)…"
            )
        else:
            extra_done = i - all_breakeven_iter
            print(
                f"\n  >>> Extra iteration {extra_done}/{EXTRA_ITERS_AFTER_ALL_BREAKEVEN}"
                f" after all break-evens were found."
            )
            if extra_done >= EXTRA_ITERS_AFTER_ALL_BREAKEVEN:
                print(
                    f"\n  >>> Reached {EXTRA_ITERS_AFTER_ALL_BREAKEVEN} extra"
                    f" iterations.  Stopping."
                )
                rows *= 2
                break

    rows *= 2   # double for next round

# ─────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────
print(f"\n{'#'*65}")
print(f"  DONE  |  last dataset size = {rows // 2:,} rows")
if all_breakeven_iter is not None:
    print(f"  All methods crossed break-even at iteration {all_breakeven_iter}")
else:
    methods_not_found = [m for m, found in breakeven_found.items() if not found]
    print(f"  Methods that never crossed break-even: {methods_not_found}")
print(f"  Full results saved to: {CSV_PATH}")
print(f"{'#'*65}\n")

spark.stop()
