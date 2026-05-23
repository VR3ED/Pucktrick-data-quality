from pyspark.sql import functions as F

from pucktrick.core.utils_common import (
    abbreviate_text,
    replace_punctuation,
    remove_or_replace,
    random_upper_lower,
)
from pucktrick.backends.spark_backend.utils_spark import shuffle_words_spark

# Funzioni testuali supportate
_TEXT_FUNCTIONS = frozenset([
    "class", "upper_lower", "replace_punctuation",
    "remove_replace", "abbreviate_text", "shuffle_words",
])


# ---------------------------------------------------------------------------
# Helper privati
# ---------------------------------------------------------------------------

def _count_existing_duplicates(df):
    """Conta i duplicati in df con un singolo Spark job.

    Evita il pattern costoso dropDuplicates().count() (shuffle globale + 2 job)
    usando una coppia count / countDistinct su hash MD5 per-riga.
    """
    row_hash = F.md5(F.concat_ws("||", *[F.col(c).cast("string") for c in df.columns]))
    row = df.select(
        F.count("*").alias("total"),
        F.countDistinct(row_hash).alias("distinct"),
    ).collect()[0]
    return row["total"] - row["distinct"]


def _build_value_filter(affected_features, value, condition_logic):
    """Restituisce una Column expression per il filtraggio per valore, o None.

    Costruisce l'espressione in modo lazy (nessuna Action), combinando le
    condizioni con AND (default) o OR in base a condition_logic.
    """
    if not value or not any(v is not None for v in value):
        return None

    exprs = []
    for col, val in zip(affected_features, value):
        exprs.append(F.col(col) == val if val is not None else F.col(col).isNotNull())

    if not exprs:
        return None

    combined = exprs[0]
    use_or   = (condition_logic == "or")
    for expr in exprs[1:]:
        combined = (combined | expr) if use_or else (combined & expr)

    return combined


def _apply_text_function(df, col, function):
    """Applica la trasformazione testuale sulla colonna indicata (lazy)."""
    if function == "shuffle_words":
        return df.withColumn(col, shuffle_words_spark()(F.col(col)))
    if function == "abbreviate_text":
        return df.withColumn(col, abbreviate_text(F.col(col)))
    if function == "replace_punctuation":
        return df.withColumn(col, replace_punctuation(F.col(col)))
    if function == "remove_replace":
        return df.withColumn(col, remove_or_replace(F.col(col)))
    if function == "upper_lower":
        return df.withColumn(col, random_upper_lower(F.col(col)))
    return df  # "class" o non riconosciuta: righe duplicate senza modifica


# ---------------------------------------------------------------------------
# Entry point pubblico
# ---------------------------------------------------------------------------

def duplicate(train_df, strategy={}, original_df=None):
    """Versione snellita di duplicate() per Spark.

    Feature supportate rispetto alla dummy base:
    - mode "new" / "extended" (con conteggio incrementale)
    - selection_criteria per filtrare le righe eleggibili
    - function per trasformazioni testuali sulle righe duplicate
    - filtraggio per valore sulle affected_features (perturbate_data.value)

    Principi di ottimizzazione:
    - Ogni DataFrame viene contato al massimo una volta; il valore viene
      riusato nelle espressioni successive (nessun .count() ridondante).
    - Il pool viene messo in cache solo perché è l'unico DF letto due volte
      (.count() + .sample()); tutto il resto rimane lazy.
    - _count_existing_duplicates viene chiamato solo nel caso extended + "all"
      senza original_df, l'unico scenario che richiede il conteggio dei duplicati
      già presenti nel dataframe in ingresso.
    - Le trasformazioni testuali vengono applicate solo sulle righe campionate,
      non sull'intero dataframe.
    """
    percentage         = strategy.get("percentage", 0.0)
    mode               = strategy.get("mode", "new")
    selection_criteria = strategy.get("selection_criteria")
    affected_features  = strategy.get("affected_features")
    function           = strategy.get("function")
    perturbate_config  = strategy.get("perturbate_data", {})
    value              = perturbate_config.get("value", [])
    condition_logic    = perturbate_config.get("condition_logic", None)

    if percentage <= 0:
        return 0, train_df

    if affected_features and not isinstance(affected_features, list):
        affected_features = [affected_features]

    # ------------------------------------------------------------------
    # 1. Pool — righe eleggibili per il campionamento (tutto lazy)
    # ------------------------------------------------------------------
    use_all_criteria = not selection_criteria or selection_criteria == "all"

    pool = train_df if use_all_criteria else train_df.filter(selection_criteria)

    if affected_features and value:
        value_filter = _build_value_filter(affected_features, value, condition_logic)
        if value_filter is not None:
            pool = pool.filter(value_filter)

    # ------------------------------------------------------------------
    # 2. Calcolo rows_to_add — minimo numero di Action
    #
    #    pool viene cachato qui perché è l'unico DF usato sia per .count()
    #    che per .sample(); tutti gli altri DF vengono contati una volta sola.
    # ------------------------------------------------------------------
    pool       = pool.cache()
    pool_count = pool.count()   # Action #1 — unica lettura del pool

    if pool_count <= 0:
        pool.unpersist()
        return 0, train_df

    if mode == "extended":
        # Riferimento: quante righe "dovrebbero esserci" nell'originale
        if original_df is not None:
            ref_df    = original_df if use_all_criteria else original_df.filter(selection_criteria)
            ref_count = ref_df.count()                       # Action #2 (solo extended)
        else:
            ref_count = pool_count  # nessun reference esterno

        target = int(ref_count * percentage)

        if use_all_criteria and original_df is None:
            # Caso speciale: dobbiamo scoprire quanti duplicati esistono già
            # nel train_df senza un reference esterno
            already_added = _count_existing_duplicates(pool)  # Action #2 alternativa
        else:
            # Righe già aggiunte = differenza tra pool attuale e reference
            already_added = pool_count - ref_count

        rows_to_add = target - already_added
    else:
        rows_to_add = int(pool_count * percentage)

    if rows_to_add <= 0:
        pool.unpersist()
        return 1, train_df

    # ------------------------------------------------------------------
    # 3. Campionamento esatto
    #
    #    fraction leggermente superiore a quanto necessario (×1.1) per
    #    compensare la natura stocastica di .sample(); .limit() garantisce
    #    poi il conteggio esatto senza ulteriori Action.
    # ------------------------------------------------------------------
    safe_fraction = min((rows_to_add / pool_count) * 1.1, 1.0)
    sampled_rows  = pool.sample(withReplacement=True, fraction=safe_fraction, seed=42) \
                        .limit(rows_to_add)

    pool.unpersist()

    # ------------------------------------------------------------------
    # 4. Trasformazioni testuali (solo sulle righe campionate — lazy)
    # ------------------------------------------------------------------
    if function in _TEXT_FUNCTIONS and affected_features:
        for col in affected_features:
            sampled_rows = _apply_text_function(sampled_rows, col, function)

    # ------------------------------------------------------------------
    # 5. Unione con il dataframe originale
    # ------------------------------------------------------------------
    try:
        noise_df = train_df.unionByName(sampled_rows)
    except Exception:
        sampled_rows = sampled_rows.select(*train_df.columns)
        noise_df = train_df.union(sampled_rows)

    return 0, noise_df