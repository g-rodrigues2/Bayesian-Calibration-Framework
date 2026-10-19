#!/bin/bash
#----------------------------------------------------
# Radiotherapy parameter-sharing scenarios V1-V8 of the delayed exponential decay model, 8 processes on
# one 128-core node with 16 cores each. Growth fixed at the thesis values; fractions of 2 Gy on days 0-5.
#
#   sbatch shared_scenarios.slurm
#
# For the scenario reported in the thesis, convergence was checked with
#   python ../01_control_growth/run_until_converged.py --stage . \
#       --backend reinitiate_files/reinitiate__ExponentialDecayDelay_V2.h5 \
#       --env CALIB_SCENARIOS=2 CALIB_GROWTH_SOURCE=fixed CALIB_DOSE_DAYS=0,1,2,3,4,5 CALIB_DOSE=2
#----------------------------------------------------
#SBATCH -J radiotherapy_sharing
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -A <your-allocation>

export CALIB_GROWTH_SOURCE=fixed
export CALIB_DOSE_DAYS=0,1,2,3,4,5
export CALIB_DOSE=2
export CALIB_BACKEND_ONLY=1
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=16
export CALIB_CHAIN_SIZE=30000
export CALIB_WALKERS_PER_DIM=4
export CALIB_INIT=ball
export CALIB_DROP_STUCK=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"
unset CALIB_TEST

mkdir -p logs
echo "Start: $(date)"

for s in 1 2 3 4 5 6 7 8; do
    CALIB_SCENARIOS=$s python global.py > "logs/V${s}.log" 2>&1 &
    sleep 0.5
done
wait

echo "End: $(date)"
python ../01_control_growth/backend_progress.py reinitiate_files --target "$CALIB_CHAIN_SIZE"
