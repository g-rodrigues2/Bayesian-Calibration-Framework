# Stage 4: sharing radiotherapy parameters between phenotypes

Eight scenarios for the parameters of the delayed exponential decay model, calibrated jointly on both
radiotherapy groups with a single residual standard deviation. Growth is fixed (logistic, stage 2
scenario 2) and fractions of 2 Gy are given on days 0–5.

| Scenario | α | β | δ |
|---|---|---|---|
| V1 | group | group | group |
| V2 | global | group | global |
| V3 | group | global | global |
| V4 | global | global | global |
| V5 | group | group | global |
| V6 | group | global | group |
| V7 | global | group | group |
| V8 | global | global | group |

Priors (thesis Table 4.1): α ~ U(0, 1), β ~ U(0, 1.5), δ ~ U(0, 8) days; initial volumes
U(0, 2·max observed initial volume); std ~ U(0.1, 1000).

Scenario V2 is reported in the thesis. Its chain was extended until at least 50 integrated
autocorrelation times after burn-in and split R-hat < 1.05:

```bash
python ../01_control_growth/run_until_converged.py --stage . \
    --backend reinitiate_files/reinitiate__ExponentialDecayDelay_V2.h5 \
    --env CALIB_SCENARIOS=2 CALIB_GROWTH_SOURCE=fixed CALIB_DOSE_DAYS=0,1,2,3,4,5 CALIB_DOSE=2 \
          CALIB_WALKERS_PER_DIM=2 --start 40000 --step 10000 --max-steps 90000
```

## Usage

```bash
CALIB_SCENARIOS=2 CALIB_GROWTH_SOURCE=fixed CALIB_CHAIN_SIZE=30000 python global.py
sbatch shared_scenarios.slurm
python -m pytest tests -q
CALIB_GROWTH_SOURCE=fixed CALIB_DOSE_DAYS=0,1,2,3,4,5 python thesis_figures.py   # fig7, fig8a-d, fig9, figS2
CALIB_GROWTH_SOURCE=fixed CALIB_DOSE_DAYS=0,1,2,3,4,5 python legacy_scenario_chains.py \
    --legacy <folder with the original V1..V8 runs> --scenarios 1,3,4,5,6,7,8 --out Output_Calibration/legacy
```

The violins of all scenarios (fig7) use the converged V2 chain and, for the other scenarios, the chains
of the original calibrations converted by `legacy_scenario_chains.py` (same burn-in and stuck-walker
removal). Those runs used the original priors: upper bounds 0.1 for α (0.15 in V4), 0.1 for β of the
sensitive group and 1.5 or 2.0 for β of the resistant group or when shared, and 10 days for δ (15 in V4).

Outputs: `multi_chain__ExponentialDecayDelay_V{N}.gz`, `multi_ll_pars__ExponentialDecayDelay_V{N}.npz`
and `scenario_selection__ExponentialDecayDelay.csv` in `Output_Calibration/`.

## Modules

```
global.py               calibration driver
scenarios_rad.py        scenario definitions (reuses the partition of stage 2)
bounds_scen_rad.py      priors and labels of the multi-group theta
inference_scen_rad.py   joint likelihood
metrics_scen_rad.py     scenario selection
thesis_figures.py       thesis figures: scenario violins, V2 histograms, fits and scatter plots
legacy_scenario_chains.py  converts the chains of the original scenario runs
```
