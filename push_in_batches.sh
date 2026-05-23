#!/bin/bash
# push_in_batches.sh
# Splits unpushed commits into small batches and pushes them one at a time.
# Safe to re-run if interrupted: already-pushed batches are skipped automatically.

set -e

BATCH_SIZE=50
REPO_DIR="$(dirname "$(realpath "$0")")"
cd "$REPO_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "ERROR: Not a git repository"
    exit 1
fi

REMOTE="origin"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "=================================================="
echo "  push_in_batches.sh"
echo "=================================================="
echo "Branch : $BRANCH"
echo "Remote : $REMOTE/$BRANCH"
echo ""

# Check connection to remote
echo "Verifying connection to remote..."
git fetch "$REMOTE" "$BRANCH" --quiet 2>/dev/null || {
    echo "ERROR: Cannot reach remote $REMOTE. Check your network/credentials."
    exit 1
}

# Show unpushed commits before reset
AHEAD=$(git rev-list --count "$REMOTE/$BRANCH"..HEAD)
echo "Commits ahead of remote: $AHEAD"
if [ "$AHEAD" -gt 0 ]; then
    echo ""
    echo "These commits will be dissolved and re-split into batches:"
    git log --oneline "$REMOTE/$BRANCH"..HEAD
fi
echo ""

# ── Confirm ────────────────────────────────────────────────────────────────────
echo "This script will:"
echo "  1. Reset the $AHEAD unpushed commit(s) (file content preserved)"
echo "  2. Re-commit everything in batches of $BATCH_SIZE files"
echo "  3. Push each batch immediately"
echo ""
read -rp "Continue? (y/N) " CONFIRM
[[ "$CONFIRM" == "y" || "$CONFIRM" == "Y" ]] || { echo "Aborted."; exit 0; }

# ── Reset unpushed commits (mixed: keeps changes in working tree) ──────────────
if [ "$AHEAD" -gt 0 ]; then
    echo ""
    echo "Resetting $AHEAD commit(s) to working tree..."
    git reset "$REMOTE/$BRANCH"
    echo "Reset done."
fi

# ── Collect all files to commit ────────────────────────────────────────────────
echo ""
echo "Collecting files to commit..."

# modified tracked files + untracked non-ignored files, null-separated
mapfile -d '' -t ALL_FILES < <(
    git ls-files -z --modified --others --exclude-standard
)

TOTAL=${#ALL_FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to commit. Working tree is clean relative to $REMOTE/$BRANCH."
    exit 0
fi

NUM_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Files to commit : $TOTAL"
echo "Batch size      : $BATCH_SIZE"
echo "Total batches   : $NUM_BATCHES"
echo ""
read -rp "Proceed with $NUM_BATCHES commits + pushes? (y/N) " CONFIRM2
[[ "$CONFIRM2" == "y" || "$CONFIRM2" == "Y" ]] || { echo "Aborted."; exit 0; }

# ── Main loop ──────────────────────────────────────────────────────────────────
FAILED_BATCHES=()

for (( i=0; i<TOTAL; i+=BATCH_SIZE )); do
    BATCH_NUM=$(( i/BATCH_SIZE + 1 ))
    BATCH=("${ALL_FILES[@]:$i:$BATCH_SIZE}")

    echo ""
    echo "── Batch $BATCH_NUM / $NUM_BATCHES (${#BATCH[@]} files) ──"

    # Stage
    git add -- "${BATCH[@]}"

    # Skip if nothing staged (e.g. file was deleted and already gone)
    if git diff --cached --quiet; then
        echo "  Nothing staged in this batch, skipping."
        continue
    fi

    # Commit
    git commit -m "push-batch $BATCH_NUM/$NUM_BATCHES"

    # Push with retry (up to 3 attempts)
    PUSHED=false
    for attempt in 1 2 3; do
        if git push "$REMOTE" "$BRANCH"; then
            PUSHED=true
            echo "  ✓ Pushed batch $BATCH_NUM"
            break
        else
            echo "  Push attempt $attempt failed. Retrying in 5s..."
            sleep 5
        fi
    done

    if ! $PUSHED; then
        echo "  ✗ Batch $BATCH_NUM failed after 3 attempts."
        echo "    The commit is local. Re-run the script to retry from here."
        FAILED_BATCHES+=("$BATCH_NUM")
        break  # Stop on push failure to avoid a long queue of unpushable commits
    fi
done

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
if [ ${#FAILED_BATCHES[@]} -eq 0 ]; then
    REMAINING=$(git rev-list --count "$REMOTE/$BRANCH"..HEAD 2>/dev/null || echo "?")
    if [ "$REMAINING" -eq 0 ] 2>/dev/null; then
        echo "  All done! Repository is fully up to date."
    else
        echo "  Script finished. $REMAINING commit(s) still ahead of remote."
        echo "  Re-run the script to push them."
    fi
else
    echo "  Stopped at batch ${FAILED_BATCHES[*]} due to push failure."
    echo "  Fix the issue and re-run this script — it will resume safely."
fi
echo "=================================================="
