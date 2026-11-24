#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=neologisms-defs
#SBATCH -o experiments/obscure_sorrows/logs/JOB%j.out

set -uo

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/obscure_sorrows/logs

JOB_ID="${SLURM_JOB_ID:-local}"
echo "job $JOB_ID running on ${SLURM_NODELIST:-localhost}"
echo "start: $(date)"

source .venv/bin/activate

PYTHONPATH="$PWD" .venv/bin/python experiments/obscure_sorrows/train.py "$@"

echo "end: $(date)"
