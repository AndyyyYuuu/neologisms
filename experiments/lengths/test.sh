#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=neologisms-len-test
#SBATCH -o experiments/lengths/logs/JOB%j.out

set -uo

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/lengths/logs

JOB_ID="${SLURM_JOB_ID:-local}"
echo "job $JOB_ID running on ${SLURM_NODELIST:-localhost}"
echo "start: $(date)"

source .venv/bin/activate

PYTHONPATH="$PWD" .venv/bin/python experiments/lengths/test.py \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --embed "experiments/lengths/embeds/llama-3.2-3b_5-50.pt" \
    --sample-size 10 \

echo "end: $(date)"
