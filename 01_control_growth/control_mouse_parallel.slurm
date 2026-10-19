#!/bin/bash
#----------------------------------------------------
# Growth models of the control groups, mouse-specific mode: 7 models x 2 groups = 14 processes on one
# 128-core node, each with a pool of 9 cores over the mice (serial sampler per mouse; with ndim = 3-5
# parallelizing the walkers would be dominated by communication costs).
#
#   sbatch control_mouse_parallel.slurm
#----------------------------------------------------
#SBATCH -J control_mouse
#SBATCH -o output_mouse.%j
#SBATCH -e error_mouse.%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -A <your-allocation>

export CALIB_MODE=mouse_specific
export CALIB_BACKEND_ONLY=1
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=9
export CALIB_CHAIN_SIZE=30000
export CALIB_WALKERS_PER_DIM=4
export CALIB_INIT=ball
export CALIB_DROP_STUCK=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"
unset CALIB_TEST CALIB_MICE        # CALIB_MICE="c1_m2,c1_m7" restricts to a subset of mice

mkdir -p logs
echo "Start: $(date)"

MODEL_LIST=(_exp _mendel _bert _lin _gomp _surf _log)
GROUP_LIST=(control_sensitive control_resistant)

for m in "${MODEL_LIST[@]}"; do
    for g in "${GROUP_LIST[@]}"; do
        CALIB_MODELS=$m CALIB_GROUPS=$g python global.py > "logs/mouse_${g}${m}.log" 2>&1 &
        sleep 1
    done
done
wait

echo "End: $(date)"
python backend_progress.py reinitiate_files --target "$CALIB_CHAIN_SIZE"
