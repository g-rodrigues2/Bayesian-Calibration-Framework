#!/bin/bash
#----------------------------------------------------
# Growth models of the control groups, global mode: 7 models x 2 groups = 14 processes on one
# 128-core node, each with a pool of 9 cores over the walkers.
#
#   sbatch control_parallel.slurm
#
# Adjust the partition, allocation and time to your cluster. The backends are resumable: resubmitting
# completes partial chains.
#----------------------------------------------------
#SBATCH -J control_global
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -A <your-allocation>

export CALIB_MODE=global
export CALIB_BACKEND_ONLY=1        # sampling only; run global.py without it afterwards for outputs
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=9
export CALIB_CHAIN_SIZE=30000
export CALIB_WALKERS_PER_DIM=4
export CALIB_INIT=ball
export CALIB_DROP_STUCK=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"
unset CALIB_TEST CALIB_MICE

mkdir -p logs
echo "Start: $(date)"

# (do not name a bash array GROUPS: it is a read-only built-in)
MODEL_LIST=(_exp _mendel _bert _lin _gomp _surf _log)
GROUP_LIST=(control_sensitive control_resistant)

for m in "${MODEL_LIST[@]}"; do
    for g in "${GROUP_LIST[@]}"; do
        CALIB_MODELS=$m CALIB_GROUPS=$g python global.py > "logs/${g}${m}.log" 2>&1 &
        sleep 1
    done
done
wait

echo "End: $(date)"
python backend_progress.py reinitiate_files --target "$CALIB_CHAIN_SIZE"
