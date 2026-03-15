#!/bin/bash
#SBATCH --account=rrg-smolesky
#SBATCH --job-name=rgb_e2e_cnn_sweep
#SBATCH --output=experiments/jobs/outputs/%x_%A_%a.out
#SBATCH --error=experiments/jobs/outputs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/jobs/outputs

module load StdEnv/2023
module load python/3.12.4
module load cuda/12.9

source .venv/bin/activate

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DISTANCES=(100 200 300 400 500 600 700 800 900 1000)
DISTANCE_UM="${DISTANCES[$SLURM_ARRAY_TASK_ID]}"
ARTIFACTS_DIR="$ROOT_DIR/experiments/artifacts/rgb_e2e_cnn_distance_${DISTANCE_UM}"

srun python -u experiments/rgb_e2e_cnn.py \
  --distance-um "$DISTANCE_UM" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  "$@"
