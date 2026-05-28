#!/usr/bin/env python3
# =============================================================================
# WATCHDOG — Rilancia automaticamente il main script in caso di crash.
#
# Strategia di isolamento:
#   Ogni run dello script target viene avviato in una tmux session DEDICATA
#   (es. "exp_run_1", "exp_run_2", ...). Il watchdog gira nella sua sessione
#   tmux separata ("watchdog"). Se il kernel OOM-killa la sessione del target,
#   la sessione del watchdog non viene toccata e rilancia automaticamente.
#
#   L'output del target viene rediretto su un log per run (run_N.log) e
#   il watchdog lo monitora in tail per rilevare il completamento o il crash.
#
# Prerequisiti:
#   - tmux installato  (sudo pacman -S tmux)
#   - virtual env gia' attivato prima di lanciare il watchdog
#
# Uso (da dentro una sessione tmux, con venv attivo):
#   python3 watchdog.py
#   python3 watchdog.py --script path/to/script.py --max-restarts 50 --wait 30
# =============================================================================

import argparse
import subprocess
import sys
import time
import os
import shutil
from datetime import datetime

# ── Configurazione default ────────────────────────────────────────────────────
DEFAULT_SCRIPT       = "5-cavas_model_experiment_targets.py"
COMPLETION_STRING    = "All experiments completed (all seeds)."
DEFAULT_MAX_RESTARTS = 100      # numero massimo di riavvii prima di arrendersi
DEFAULT_WAIT_SEC     = 15       # secondi di pausa tra un riavvio e l'altro
WATCHDOG_LOG         = "watchdog.log"
POLL_INTERVAL        = 2        # secondi tra un controllo e l'altro sul log del run
TMUX_SESSION_PREFIX  = "exp_run"


# ── Utility ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def kill_tmux_session(session_name: str):
    if tmux_session_exists(session_name):
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ── Core ──────────────────────────────────────────────────────────────────────

def run_in_tmux_session(session_name: str, script_path: str, run_log: str) -> bool:
    """
    Crea una nuova tmux session dedicata, avvia lo script al suo interno
    redirigendo stdout+stderr su `run_log`, poi monitora il log finche':
      - la sessione tmux sparisce  -> crash/OOM  -> ritorna False
      - il log contiene COMPLETION_STRING        -> ritorna True

    La session viene sempre ripulita all'uscita.
    """
    python_bin = sys.executable
    script_dir = os.path.dirname(os.path.abspath(script_path))

    # Comando che gira dentro tmux:
    # usa il python del venv corrente ed esegue lo script loggando tutto su file.
    inner_cmd = (
        f"cd {script_dir!r} && "
        f"{python_bin!r} {script_path!r} "
        f"> {run_log!r} 2>&1"
    )

    # Crea la sessione detached
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "-x", "220", "-y", "50"],
        check=True,
    )

    # Invia il comando alla sessione
    subprocess.run(
        ["tmux", "send-keys", "-t", session_name, inner_cmd, "Enter"],
        check=True,
    )

    log(f"Sessione tmux '{session_name}' avviata. Log run: {run_log}")

    # Attendi che il file di log venga creato
    for _ in range(30):
        if os.path.exists(run_log):
            break
        time.sleep(1)

    completed = False
    last_size = 0

    try:
        while True:
            time.sleep(POLL_INTERVAL)

            # 1. Leggi le righe nuove dal log e stampale in tempo reale
            if os.path.exists(run_log):
                current_size = os.path.getsize(run_log)
                if current_size > last_size:
                    with open(run_log, "r", errors="replace") as f:
                        f.seek(last_size)
                        new_lines = f.read()
                    last_size = current_size

                    for line in new_lines.splitlines():
                        print(line, flush=True)
                        with open(WATCHDOG_LOG, "a") as wf:
                            wf.write(line + "\n")

                    if COMPLETION_STRING in new_lines:
                        completed = True
                        break

            # 2. Controlla se la sessione e' ancora viva
            if not tmux_session_exists(session_name):
                log(f"Sessione tmux '{session_name}' non trovata (crash/OOM).")
                break

    finally:
        kill_tmux_session(session_name)

    return completed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Watchdog: rilancia lo script in sessioni tmux isolate finche' non completa."
    )
    parser.add_argument(
        "--script",
        default=DEFAULT_SCRIPT,
        help=f"Path dello script da monitorare (default: {DEFAULT_SCRIPT})"
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=DEFAULT_MAX_RESTARTS,
        help=f"Numero massimo di riavvii (default: {DEFAULT_MAX_RESTARTS})"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=DEFAULT_WAIT_SEC,
        help=f"Secondi di attesa tra un riavvio e il successivo (default: {DEFAULT_WAIT_SEC})"
    )
    args = parser.parse_args()

    if not tmux_available():
        print("ERRORE: tmux non trovato. Installalo con: sudo pacman -S tmux")
        sys.exit(1)

    script_path = os.path.abspath(args.script)
    if not os.path.exists(script_path):
        print(f"ERRORE: script non trovato: {script_path}")
        sys.exit(1)

    log("=" * 60)
    log("WATCHDOG START")
    log(f"  Script       : {script_path}")
    log(f"  Max restarts : {args.max_restarts}")
    log(f"  Wait (sec)   : {args.wait}")
    log(f"  Watchdog log : {os.path.abspath(WATCHDOG_LOG)}")
    log("=" * 60)

    for attempt in range(1, args.max_restarts + 2):
        session_name = f"{TMUX_SESSION_PREFIX}_{attempt}"
        run_log      = os.path.abspath(f"run_{attempt}.log")

        log(f"--- Tentativo {attempt}/{args.max_restarts + 1} "
            f"| sessione tmux: '{session_name}' ---")

        # Pulisci eventuale sessione residua con lo stesso nome
        kill_tmux_session(session_name)

        completed = run_in_tmux_session(session_name, script_path, run_log)

        if completed:
            log("=" * 60)
            log("WATCHDOG: completamento rilevato. Uscita.")
            log("=" * 60)
            sys.exit(0)

        if attempt > args.max_restarts:
            break

        log(f"WATCHDOG: riavvio tra {args.wait}s ...")
        time.sleep(args.wait)

    log("=" * 60)
    log(f"WATCHDOG: raggiunto il limite di {args.max_restarts} riavvii. Uscita.")
    log("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()