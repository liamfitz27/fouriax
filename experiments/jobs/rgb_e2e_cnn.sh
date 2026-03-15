#!/bin/bash
#SBATCH --account=rrg-smolesky
#SBATCH --job-name=rgb_e2e_cnn
#SBATCH --output=experiments/jobs/outputs/%x_%j.out
#SBATCH --error=experiments/jobs/outputs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/jobs/outputs

module load StdEnv/2023
module load python/3.12.4
module load cuda/12.9

source .venv/bin/activate

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

srun python -u experiments/rgb_e2e_cnn.py "$@"
