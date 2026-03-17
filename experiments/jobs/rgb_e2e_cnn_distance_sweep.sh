#!/bin/bash
#SBATCH --account=rrg-smolesky
#SBATCH --job-name=rgb_e2e_cnn_sweep
#SBATCH --output=experiments/jobs/outputs/%x_%A_%a.out
#SBATCH --error=experiments/jobs/outputs/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load StdEnv/2023
module load python/3.12.4
module load cuda/12.9

source .venv/bin/activate

NOISE_LEVELS=(0.00 0.005 0.01 0.02 0.04 0.08)

for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
  ARTIFACTS_DIR="experiments/artifacts/rgb_e2e_cnn_noise_${NOISE_LEVEL}"
  srun python -u experiments/rgb_e2e_cnn.py \
    --noise-level "$NOISE_LEVEL" \
    --artifacts-dir "$ARTIFACTS_DIR"
done
