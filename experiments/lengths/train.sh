#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=neologisms-len
#SBATCH -o experiments/lengths/logs/JOB%j.out

set -uo

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/lengths/logs

JOB_ID="${SLURM_JOB_ID:-local}"
echo "job $JOB_ID running on ${SLURM_NODELIST:-localhost}"
echo "start: $(date)"

source .venv/bin/activate

PYTHONPATH="$PWD" .venv/bin/python experiments/lengths/train.py \
    --model "meta-llama/Llama-3.2-3B-Instruct"\
    "$@"

echo "end: $(date)"
