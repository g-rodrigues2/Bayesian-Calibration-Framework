#!/bin/bash
#----------------------------------------------------
# Radiotherapy models: 19 models x 2 groups, global mode. Growth (r, cc) is fixed at the values of the
# thesis and the effect is anchored at the fraction days (2 Gy on days 0-5). Processes are launched in
# waves of 16 (16 x 8 = 128 cores).
#
#   sbatch radiotherapy_parallel.slurm
#----------------------------------------------------
#SBATCH -J radiotherapy
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -A <your-allocation>

export CALIB_MODE=global
export CALIB_GROWTH_SOURCE=fixed
export CALIB_DOSE_DAYS=0,1,2,3,4,5
export CALIB_DOSE=2
export CALIB_BACKEND_ONLY=1
export CALIB_COMPRESS_H5=1
export CALIB_USE_POOL=1
export CALIB_N_CORES=8
export CALIB_CHAIN_SIZE=30000
export CALIB_WALKERS_PER_DIM=4
export CALIB_INIT=ball
export CALIB_DROP_STUCK=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONWARNINGS="ignore::UserWarning"
unset CALIB_TEST CALIB_MICE

mkdir -p logs
echo "Start: $(date)"

MODEL_LIST=(_LQ _ExponentialDecay _CumulativeDose _AccelED _AccelCD \
            _CDwithLogFactor _CDwithExpFactor _CDwithSinFactor \
            _EDwithLogFactor _EDwithExpFactor _EDwithSinFactor \
            _ExponentialDecayDelay _EDEFDelay _CumulativeDoseDelay _CDEFDelay \
            _ACDDelay _AEDDelay _CDSFDelay _EDSFDelay)
GROUP_LIST=(radiation_sensitive radiation_resistant)

launched=0
for m in "${MODEL_LIST[@]}"; do
    for g in "${GROUP_LIST[@]}"; do
        CALIB_MODELS=$m CALIB_GROUPS=$g python global.py > "logs/${g}${m}.log" 2>&1 &
        launched=$((launched + 1))
        if (( launched % 16 == 0 )); then wait; fi
        sleep 0.5
    done
done
wait

echo "End: $(date)"
python ../01_control_growth/backend_progress.py reinitiate_files --target "$CALIB_CHAIN_SIZE"
