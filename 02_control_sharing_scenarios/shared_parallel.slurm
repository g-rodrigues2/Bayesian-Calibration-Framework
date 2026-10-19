#!/bin/bash
#----------------------------------------------------
# Parameter sharing between the control groups: _exp (scenarios 1, 4) and _log (scenarios 1-4),
# 6 processes on one 128-core node with 20 cores each.
#
#   sbatch shared_parallel.slurm
#----------------------------------------------------
#SBATCH -J control_sharing
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -A <your-allocation>

export CALIB_BACKEND_ONLY=1
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=20
export CALIB_CHAIN_SIZE=30000
export CALIB_WALKERS_PER_DIM=4
export CALIB_INIT=ball
export CALIB_DROP_STUCK=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"
unset CALIB_TEST

mkdir -p logs
echo "Start: $(date)"

declare -A SCENARIO_LIST
SCENARIO_LIST[_exp]="1 4"
SCENARIO_LIST[_log]="1 2 3 4"

for m in _exp _log; do
    for s in ${SCENARIO_LIST[$m]}; do
        CALIB_MODELS=$m CALIB_SCENARIOS=$s python global.py > "logs/${m}_s${s}.log" 2>&1 &
        sleep 1
    done
done
wait

echo "End: $(date)"
python ../01_control_growth/backend_progress.py reinitiate_files --target "$CALIB_CHAIN_SIZE"
