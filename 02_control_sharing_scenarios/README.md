# Stage 2: sharing growth parameters between phenotypes

Tests whether the growth rate `r` and carrying capacity `K` of the selected models can be shared
between `control_sensitive` and `control_resistant`. Both groups are calibrated jointly with a single
residual standard deviation; each parameter is either global (one value) or group-specific.

| Scenario | `_log`: r | `_log`: K | `_exp`: r |
|---|---|---|---|
| s1 | group | group | group |
| s2 | group | global | – |
| s3 | global | group | – |
| s4 | global | global | global |

Scenario s2 (group-specific `r`, shared `K`) is the configuration used by the treatment stages.

## Usage

```bash
CALIB_MODELS=_log CALIB_SCENARIOS=2 CALIB_CHAIN_SIZE=30000 CALIB_WALKERS_PER_DIM=4 python global.py
CALIB_MODELS=_log CALIB_CHAIN_SIZE=30000 python global.py      # outputs and scenario table
python thesis_figures.py --figures fig4,fig5,fig6,figS1        # thesis figures (Output_Calibration/thesis_figures)
python ../01_control_growth/run_until_converged.py --stage . --backend reinitiate_files/reinitiate__log_s1.h5 \
    --env CALIB_MODELS=_log CALIB_SCENARIOS=1 CALIB_WALKERS_PER_DIM=4 CALIB_SOLVER=analytic \
    --start 30000 --step 10000 --max-steps 300000
sbatch shared_parallel.slurm
python -m pytest tests -q
```

`CALIB_SCENARIOS` (or `CALIB_SCENARIOS_LOG`, `CALIB_SCENARIOS_EXP`) selects scenarios; the other settings
are those of stage 1. Outputs: `multi_chain_{model}_s{N}.gz`, `multi_ll_pars_{model}_s{N}.npz` and
`scenario_selection{model}.csv` in `Output_Calibration/`.

All four `_log` chains satisfy the convergence criteria (at least 50 integrated autocorrelation times after
burn-in and split R-hat < 1.05 for every parameter): s2, used by the treatment stages, has 30000 steps and
72 walkers (at least 61 autocorrelation times, R-hat ≤ 1.02); s1, whose carrying capacity of the sensitive
group is weakly identified, needed 155485 steps with 76 walkers (65 autocorrelation times).

## Modules

```
global.py            calibration driver
scenarios.py         scenario definitions and partition of theta
bounds_shared.py     priors and labels of the multi-group theta
inference_shared.py  joint likelihood
metrics_shared.py    scenario selection
thesis_figures.py    thesis figures: posterior violins of the scenarios, histograms and fits of s2
```
