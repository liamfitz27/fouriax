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

module load StdEnv/2023
module load python/3.12.4
module load cuda/12.9

source .venv/bin/activate

srun python -u experiments/rgb_e2e_cnn.py --batch-size 16 \
    --epochs 50 --optical-lr 5e-3 --decoder-lr 1e-3 --sensor-dx-um 0.7 \
    --meta-dx-um 0.7 --distance-um 100 --noise-level 0 \
    --sensor-size-px 128 --cnn-base-channels 64 \
    --artifacts-dir experiments/artifacts/rgb_e2e_cnn_128_p7sensor_p7meta_100um
