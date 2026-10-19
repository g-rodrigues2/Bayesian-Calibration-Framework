#!/bin/bash
#----------------------------------------------------
# Immunotherapy models: the 12 formulations of the thesis x 1 group (io_sensitive, 3 mice), global mode.
# Growth fixed at the thesis values; 7 administrations (days 0, 3, ..., 18) of a normalized dose.
#
#   sbatch immuno_parallel.slurm
#----------------------------------------------------
#SBATCH -J immunotherapy
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -A <your-allocation>

export CALIB_MODE=global
export CALIB_GROWTH_SOURCE=fixed
export CALIB_DOSE_DAYS=0,3,6,9,12,15,18
export CALIB_DOSE=1
export CALIB_BACKEND_ONLY=1
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=10
export CALIB_CHAIN_SIZE=30000
export CALIB_WALKERS_PER_DIM=6
export CALIB_INIT=ball
export CALIB_DROP_STUCK=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"
unset CALIB_TEST

mkdir -p logs
echo "Start: $(date)"

MODEL_LIST=(_ExponentialDecay _CumulativeDose _AccelED _AccelCD _MichaelisMenten _dePillisRadunskayaLaw \
            _CDwithLogFactor _CDwithExpFactor _CDwithSinFactor \
            _EDwithLogFactor _EDwithExpFactor _EDwithSinFactor)

for m in "${MODEL_LIST[@]}"; do
    CALIB_MODELS=$m python global.py > "logs/${m}.log" 2>&1 &
    sleep 0.5
done
wait

echo "End: $(date)"
python ../01_control_growth/backend_progress.py reinitiate_files --target "$CALIB_CHAIN_SIZE"
