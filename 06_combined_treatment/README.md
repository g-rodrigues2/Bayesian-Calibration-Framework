# Radiotherapy + immunotherapy: prediction, SUV biomarker and leave-one-out (Sections 4.5–4.7)

Analyses of the combined radiotherapy + immunotherapy cohort (`rad+io_sensitive`, 15 mice, 4T1). Growth,
radiotherapy and immunotherapy parameters are **not** re-estimated: they are fixed at the values of the
previous calibration stages (`thesis_params.py`), and only the initial tumor volumes, the residual standard
deviation and, when present, the parameters of the SUV function are calibrated.

## Model

$$\frac{dT}{dt} = rT\Bigl(1-\frac{T}{K}\Bigr) - \alpha T\sum_i D_i\,e^{-\beta(t-\tau_i-\delta)}H(t-\tau_i-\delta) - T\sum_j \phi\,\frac{D_j^{\lambda}}{sT^{\lambda}+D_j^{\lambda}}H(t-\tau_j)$$

| Model | Lysis coefficient φ | Calibrated |
|---|---|---|
| `_RadImmunoFixed` | `d` (immunotherapy estimate) | initial volumes, std |
| `_SUVLinear` | `a·SUV + b` | `a`, `b`, initial volumes, std |
| `_SUVMichaelisMenten` | `a·SUV / (b + SUV)` | idem |
| `_SUVLogistic` | `a / (1 + exp(−b·SUV))` | idem |
| `_SUVSatExp` | `a·(1 − exp(−b·SUV))` | idem |

Radiotherapy: 2 Gy on days 0–5; immunotherapy: unit dose on days 9, 12, …, 24 (`CALIB_RAD_DAYS`,
`CALIB_IO_DAYS`). The SUV of each mouse is read from the data file name.

## Usage

```bash
python calibrate.py prediction      # fixed parameters; initial volumes and std calibrated
python calibrate.py suv             # BIC of the SUV functions
python posterior_suv_params.py      # posterior of a and b of the linear SUV function
python loo.py train                 # 30 training folds (15 mice x 2 formulations)
python loo.py strict                # held-out log-likelihood table
python loo.py conditional           # held-out initial volume and std of each fold
python thesis_figures.py            # thesis figures: fig12, fig13, fig14a-b and the LOO figures
python validate_legacy.py --legacy <original results>        # reproduction check without MCMC
```

`loo.py train` and `loo.py conditional` accept `--models` and `--mice` to run single folds. Each chain of
the thesis was extended until convergence, for example

```bash
python ../01_control_growth/run_until_converged.py --stage . \
    --backend reinitiate_files/rad+io_sensitive_EXCL-c0_m9_SUVLinear.h5 \
    --command "loo.py train --models _SUVLinear --mice c0_m9" \
    --env CALIB_WALKERS_PER_DIM=2 CALIB_USE_POOL=0 --start 20000 --step 10000 --max-steps 150000
```

Environment: `CALIB_CHAIN_SIZE` (total steps, default 10000; resumable), `CALIB_COND_CHAIN_SIZE` (overrides
it for the conditional chains), `CALIB_WALKERS_PER_DIM` (2; 8 for the conditional chains of the thesis),
`CALIB_INIT` (`ball`, around a MAP estimate, or `uniform`), `CALIB_USE_POOL`, `CALIB_N_CORES`,
`CALIB_BACKEND_ONLY` (sampling only) and `CALIB_RI_SOLVER` (`compiled`, default, or `odeint`).

Outputs: `Output_Calibration/{prediction,suv,loo}/` — `chain_<stem>.gz` (post burn-in, stuck walkers
removed), `estimate_<stem>.npz` (maximum-likelihood sample), `trace_<stem>.png`, model-selection and
LOO tables, figures. HDF5 backends in `reinitiate_files/`.

## Solver

The right-hand side is discontinuous at every administration. `rad_immuno_fast.py` integrates the model
in compiled code with an adaptive Dormand–Prince 5(4) scheme (relative tolerance 1e-9), restarting at each
switching time and stopping exactly at the observation times; the whole cohort likelihood is one compiled
call (about 0.9 ms against 46 ms with `odeint`). Against a DOP853 reference with tolerance 1e-13, its
largest volume error on posterior samples is 1.4e-6 mm³ (3.2e-3 mm³ for `odeint`), and log-likelihoods
differ from those of `odeint` by less than 3e-4 (`tests/test_rad_immuno.py`). The figures keep using
`rad_immuno_models.solve_mouse` (`odeint`) on the plotting grid.

## Reproduction of the thesis results

The results of Sections 4.5–4.7 were first produced by the original scripts, which this pipeline replaces.
`validate_legacy.py` evaluates the saved maximum-likelihood samples of those runs with the likelihood
implemented here: all 64 checks agree (prediction, three SUV functions, 30 LOO training folds and the 30
strict held-out log-likelihoods, whose totals with the original chains are −2852.649 with SUV and
−3178.240 without SUV).

Every chain was then extended or rerun until convergence (at least 50 integrated autocorrelation times after
burn-in and split R-hat < 1.05; `Output_Calibration/convergence_*.csv`):

- prediction: 29,517 steps, 32 walkers (65 autocorrelation times, R-hat 1.02);
- linear SUV: 50,000 steps, 36 walkers, 2 stuck walkers removed (52 autocorrelation times, R-hat ≤ 1.03);
  a 2.86 [1.68, 4.04], b −2.56 [−4.00, −1.04];
- LOO training folds: 20,010–30,030 steps without SUV and 40,020–70,520 steps with SUV;
- held-out conditional chains: restarted with 8 walkers per parameter (5,000–10,000 steps).

With the converged folds the strict held-out totals are −2871.865 with SUV and −3179.813 without SUV, and
the SUV-based formulation is better in 10 of the 15 mice (11 with the original chains; mouse c0_m9 changes
side because the maximum-likelihood sample of its longer training chain differs).

Differences with respect to the original runs that affect new calibrations:

- initialization around a MAP estimate with removal of stuck walkers (the original runs used uniform
  initialization; in the linear SUV calibration 2 of 36 walkers were stuck in a low-likelihood region and
  produced the positive-`b` tail of that chain);
- `_SUVLogistic` implements the logistic function of the thesis table; the original "logistic" script did
  not, so its result is not reproduced;
- priors follow the original code: initial volume in (0, 2·max initial volume), std in (0.1, 1000) for the
  prediction and (0.1, 300) for the SUV and LOO analyses.

## Modules

```
thesis_params.py         inherited growth, radiotherapy and immunotherapy parameters
data_loader_ri.py        data files, SUV and treatment schedules
rad_immuno_models.py     model equation, lysis functions and odeint solution
rad_immuno_fast.py       compiled solver and cohort log-likelihood
bounds_ri.py             priors
inference_ri.py          likelihood and posterior
mcmc_ri.py               resumable emcee runner
postprocess_ri.py        posterior trajectories, CCC/PCC, figures
calibrate.py             prediction and SUV analyses
loo.py                   leave-one-out analysis
posterior_suv_params.py  posterior summary of a and b
thesis_figures.py        thesis figures of Sections 4.5–4.7 and appendix
validate_legacy.py       reproduction check against the original results (requires the original runs)
```

Shared modules (`initialization`, `walkers`, `metrics`, `model_selection`, `postprocess`,
`growth_params`, `buffered_backend`) are imported from `../01_control_growth/` through `_path_setup.py`.
