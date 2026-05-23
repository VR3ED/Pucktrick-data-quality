#!/bin/bash
# =============================================================================
# SLURM submission script — CAVAS / dirtify experiments
# Cluster: DatAI Research Lab (datai01 partition)
# Directory: /scratch_share/datai/fcavallini/dirtify/
# =============================================================================

#SBATCH --account=datai
#SBATCH --partition=datai01
#SBATCH --job-name=dirtify-cavas

# ── Risorse ──────────────────────────────────────────────────────────────────
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# ── Log ──────────────────────────────────────────────────────────────────────
#SBATCH --output=/scratch_share/datai/fcavallini/dirtify/job_logs/out_%x_%j.log
#SBATCH --error=/scratch_share/datai/fcavallini/dirtify/job_logs/error_%x_%j.log

set -x

# =============================================================================
# DEFINIZIONI PERCORSI
# =============================================================================
export BASEDIR="/scratch_share/datai/fcavallini/dirtify"
export TMPDIR="${BASEDIR}/tmp_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "$TMPDIR"
mkdir -p "${BASEDIR}/job_logs"

cd "$BASEDIR"

# =============================================================================
# INFO DI SISTEMA (finisce nel log)
# =============================================================================
pwd; hostname; date
echo "SLURM_JOB_ID:   $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "CPUs allocate:  $SLURM_CPUS_PER_TASK"
echo "Memoria:        $SLURM_MEM_PER_NODE MB"
nvidia-smi || echo "nvidia-smi non disponibile"

# =============================================================================
# CARICA MODULI E ATTIVA AMBIENTE CONDA
# =============================================================================
module purge
module load amd/slurm

source /opt/share/sw/amd/gcc-8.5.0/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda activate /scratch_share/datai/fcavallini/envs/dirtify

# Verifica che Python e le librerie chiave siano disponibili
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python3 -c "import pyspark; print('PySpark:', pyspark.__version__)"

# =============================================================================
# LANCIA LO SCRIPT PYTHON
# =============================================================================
python3 "${BASEDIR}/4-4-cavas_model_Experiment_cluster.py"

# =============================================================================
# CLEANUP
# =============================================================================
echo "Job completato: $(date)"
rm -rf "$TMPDIR"
