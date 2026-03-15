#!/bin/bash
#SBATCH --account=rrg-smolesky
#SBATCH --job-name=rgb_e2e_cnn_stage1
#SBATCH --output=experiments/jobs/outputs/%x_%j.out
#SBATCH --error=experiments/jobs/outputs/%x_%j.err
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load StdEnv/2023
module load python/3.12.4
module load cuda/12.9

source .venv/bin/activate

DISTANCES_UM=(600 700 800 900 1000 1100 1200)
WAVELENGTH_WINDOWS_NM=(
  "900 1100"
  "1000 1200"
  "1100 1300"
  "1200 1400"
  "1300 1500"
)

MAX_PARALLEL=2
ACTIVE_JOBS=0

run_point() {
  local distance_um=$1
  local wavelength_min_nm=$2
  local wavelength_max_nm=$3
  local wavelength_min_um
  local wavelength_max_um
  local artifacts_dir

  printf -v wavelength_min_um "%d.%03d" \
    "$((wavelength_min_nm / 1000))" \
    "$((wavelength_min_nm % 1000))"
  printf -v wavelength_max_um "%d.%03d" \
    "$((wavelength_max_nm / 1000))" \
    "$((wavelength_max_nm % 1000))"
  printf -v artifacts_dir \
    "experiments/artifacts/rgb_e2e_cnn_stage1_wl_%04d_%04d_distance_%04d" \
    "$wavelength_min_nm" \
    "$wavelength_max_nm" \
    "$distance_um"

  echo "launch distance_um=${distance_um} wavelength_nm=${wavelength_min_nm}-${wavelength_max_nm} artifacts_dir=${artifacts_dir}"

  srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 \
    python -u experiments/rgb_e2e_cnn.py \
    --epochs 50 \
    --optical-lr 1e-3 \
    --distance-um "${distance_um}" \
    --wavelength-min-um "${wavelength_min_um}" \
    --wavelength-max-um "${wavelength_max_um}" \
    --num-wavelengths 3 \
    --artifacts-dir "${artifacts_dir}"
}

for window in "${WAVELENGTH_WINDOWS_NM[@]}"; do
  read -r wavelength_min_nm wavelength_max_nm <<<"${window}"
  for distance_um in "${DISTANCES_UM[@]}"; do
    run_point "${distance_um}" "${wavelength_min_nm}" "${wavelength_max_nm}" &
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
    if [[ "${ACTIVE_JOBS}" -ge "${MAX_PARALLEL}" ]]; then
      wait -n
      ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
    fi
  done
done

wait
