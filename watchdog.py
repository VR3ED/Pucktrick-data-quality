#!/usr/bin/env python3
# =============================================================================
# WATCHDOG — Rilancia automaticamente il main script in caso di crash
#
# Uso:
#   python3 watchdog.py
#   python3 watchdog.py --script path/to/4-4b-cavas_model_Experiment_cluster.py
#   python3 watchdog.py --max-restarts 50 --wait 30
#
# Il watchdog si ferma solo quando lo script stampa la stringa di completamento
# oppure quando viene raggiunto il numero massimo di riavvii.
# =============================================================================

import argparse
import subprocess
import sys
import time
import os
from datetime import datetime

# ── Configurazione default ────────────────────────────────────────────────────
DEFAULT_SCRIPT      = "4-4c-cavas_model_Experiment_localhost.py"
COMPLETION_STRING   = "All experiments completed (all seeds)."
DEFAULT_MAX_RESTART = 100       # numero massimo di riavvii prima di arrendersi
DEFAULT_WAIT_SEC    = 15        # secondi di attesa tra un riavvio e l'altro
LOG_FILE            = "watchdog.log"


def log(msg: str, also_print: bool = True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_script(script_path: str) -> bool:
    """
    Lancia lo script come subprocess, stampa l'output in tempo reale e
    ritorna True se lo script ha stampato la stringa di completamento.
    """
    completed = False

    cmd = [sys.executable, script_path]
    log(f"Avvio: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # unifica stderr in stdout
        text=True,
        bufsize=1,                  # line-buffered
        cwd=os.path.dirname(os.path.abspath(script_path)) or ".",
        # start_new_session=True chiama os.setsid() nel child prima di exec:
        # il child ottiene una nuova sessione Unix e un nuovo process group.
        # L'OOM killer di Linux non risale oltre i confini di sessione,
        # quindi colpisce solo il child (e i suoi sottoprocessi) senza
        # toccare questo watchdog.
        start_new_session=True,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line)
        if COMPLETION_STRING in line:
            completed = True

    proc.wait()
    exit_code = proc.returncode
    log(f"Script terminato con exit code {exit_code}.")

    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Watchdog: rilancia lo script finché non completa con successo."
    )
    parser.add_argument(
        "--script",
        default=DEFAULT_SCRIPT,
        help=f"Path dello script da monitorare (default: {DEFAULT_SCRIPT})"
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=DEFAULT_MAX_RESTART,
        help=f"Numero massimo di riavvii (default: {DEFAULT_MAX_RESTART})"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=DEFAULT_WAIT_SEC,
        help=f"Secondi di attesa tra un riavvio e il successivo (default: {DEFAULT_WAIT_SEC})"
    )
    args = parser.parse_args()

    script_path = os.path.abspath(args.script)
    if not os.path.exists(script_path):
        print(f"ERRORE: script non trovato: {script_path}")
        sys.exit(1)

    log("=" * 60)
    log(f"WATCHDOG START")
    log(f"  Script      : {script_path}")
    log(f"  Max restarts: {args.max_restarts}")
    log(f"  Wait (sec)  : {args.wait}")
    log(f"  Log file    : {os.path.abspath(LOG_FILE)}")
    log("=" * 60)

    attempt = 0
    while attempt <= args.max_restarts:
        attempt += 1
        log(f"--- Tentativo {attempt}/{args.max_restarts + 1} ---")

        completed = run_script(script_path)

        if completed:
            log("=" * 60)
            log("WATCHDOG: completamento rilevato. Uscita.")
            log("=" * 60)
            sys.exit(0)

        if attempt > args.max_restarts:
            break

        log(f"WATCHDOG: crash/interruzione rilevata. Riavvio tra {args.wait}s ...")
        time.sleep(args.wait)

    log("=" * 60)
    log(f"WATCHDOG: raggiunto il limite di {args.max_restarts} riavvii. Uscita.")
    log("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()