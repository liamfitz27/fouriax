#!/bin/bash
#SBATCH --account=rrg-smolesky
#SBATCH --job-name=rgb_e2e_cnn_probe
#SBATCH --output=experiments/jobs/outputs/%x_%j.out
#SBATCH --error=experiments/jobs/outputs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


module load StdEnv/2023
module load python/3.12.4
module load cuda/12.9

source .venv/bin/activate

# name|distance_um|wavelength_min_um|wavelength_max_um|epochs|batch_size|optical_lr|decoder_lr
CONFIGS=(
  "cool_short_fast|450|0.85|1.05|30|64|2e-3|1e-3"
  "baseline_center_long|650|0.95|1.15|50|64|1e-3|1e-3"
  "best_basin_low_lr|800|1.00|1.20|60|128|5e-4|1e-3"
  "best_basin_high_lr|800|1.00|1.20|35|128|2e-3|1e-3"
  "mid_red_balanced|950|1.10|1.30|45|128|1e-3|5e-4"
  "mid_broad_band|1000|1.00|1.40|50|64|7e-4|1e-3"
  "far_red_long|1150|1.20|1.40|50|64|1e-3|1e-3"
  "far_red_aggressive|1250|1.30|1.50|35|64|2e-3|5e-4"
  "deep_red_edge|1400|1.40|1.60|30|128|5e-4|1e-3"
  "nir_edge_short|600|0.80|1.00|35|128|2e-3|1e-3"
  "library_edge_red|900|1.50|1.70|30|64|5e-4|1e-3"
  "max_span_probe|1200|0.80|1.70|20|64|1e-3|5e-4"
)

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r \
    name \
    distance_um \
    wavelength_min_um \
    wavelength_max_um \
    epochs \
    batch_size \
    optical_lr \
    decoder_lr <<<"${config}"

  artifacts_dir="experiments/artifacts/rgb_e2e_cnn_probe_${name}"

  echo "launch name=${name} distance_um=${distance_um} wavelengths_um=${wavelength_min_um}-${wavelength_max_um} epochs=${epochs} batch_size=${batch_size} optical_lr=${optical_lr} decoder_lr=${decoder_lr}"

  srun python -u experiments/rgb_e2e_cnn.py \
    --distance-um "${distance_um}" \
    --wavelength-min-um "${wavelength_min_um}" \
    --wavelength-max-um "${wavelength_max_um}" \
    --num-wavelengths 3 \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --optical-lr "${optical_lr}" \
    --decoder-lr "${decoder_lr}" \
    --artifacts-dir "${artifacts_dir}"
done
