# Stage 1: tumor growth of the control groups

Bayesian calibration (emcee) of seven growth models on the untreated groups `control_sensitive`
(4T1, 9 mice) and `control_resistant` (RR-4T1, 5 mice), and model selection. This folder also holds the
modules shared by all stages.

| Model | dT/dt | Parameters |
|---|---|---|
| `_exp` | r T | r |
| `_mendel` | r T^b | r, b |
| `_bert` | r T^(2/3) − b T | r, b |
| `_lin` | r T / (T + K) | r, K |
| `_gomp` | r T ln(b / (T + c)) | r, b, c |
| `_surf` | r T / (T + K)^(1/3) | r, K |
| `_log` | r T (1 − T/K) | r, K |

## Modes

| | `global` (default) | `mouse_specific` |
|---|---|---|
| Sampler per | (model, group) | (model, group, mouse) |
| theta | `[parameters, ic1..icM, std]` | `[parameters, ic, std]` |
| Pool parallelizes | walkers | mice (serial sampler per mouse) |
| Outputs | `Output_Calibration/` | `Output_Calibration_per_mouse/` |

Both modes use a single `inference.log_likelihood` parametrized by `mouse_index`; the joint likelihood
equals the sum of the individual ones (`tests/test_inference.py`).

## Usage

```bash
CALIB_TEST=1 python global.py                                   # smoke test (50 steps, TEST_ files)
CALIB_MODELS=_log CALIB_GROUPS=control_resistant CALIB_CHAIN_SIZE=2000 python global.py
CALIB_MODE=mouse_specific CALIB_MICE=c1_m2,c1_m7 CALIB_MODELS=_log python global.py
python model_selection.py --metrics aicc,bic,waic,loo           # selection from saved outputs
python convergence_check.py --models _log
python run_until_converged.py --stage . --backend reinitiate_files/reinitiate_control_resistant_log.h5 \
    --env CALIB_MODELS=_log CALIB_GROUPS=control_resistant
python thesis_scatter_figure.py --model _log --out Output_Calibration/thesis_figures/fig3.png
python thesis_chain_traces.py        # chain traces and convergence table of the selected models and LOO
python -m pytest tests -q

sbatch control_parallel.slurm         # global mode, 7 models x 2 groups
sbatch control_mouse_parallel.slurm   # mouse-specific mode
```

## Settings

| Variable | Default | Effect |
|---|---|---|
| `CALIB_MODE` | `global` | `global` or `mouse_specific` |
| `CALIB_MODELS` | all 7 | comma-separated models |
| `CALIB_GROUPS` | both control groups | |
| `CALIB_MICE` | all | subset of mice (mouse-specific mode) |
| `CALIB_CHAIN_SIZE` | `1000` | total number of steps (resumable) |
| `CALIB_WALKERS_PER_DIM` | `2` | walkers per dimension (4 used in the thesis runs) |
| `CALIB_INIT` | `ball` | `ball` (around a MAP estimate) or `uniform` |
| `CALIB_DROP_STUCK` | `1` | remove stuck walkers before saving the chain |
| `CALIB_USE_POOL`, `CALIB_N_CORES` | `1`, all cores | multiprocessing |
| `CALIB_BACKEND_ONLY` | `0` | `1`: sampling only (no outputs) |
| `CALIB_COMPRESS_H5`, `CALIB_COMPRESS_LEVEL` | `0`, `4` | gzip compression of the backends |
| `CALIB_SELECT`, `CALIB_METRICS` | `1`, `aic,aicc,bic,dic,wbic` | model selection at the end |
| `CALIB_BAND`, `CALIB_BAND_KIND`, `CALIB_N_BAND` | `1`, `both`, `200` | credible and predictive bands |
| `CALIB_CENTRAL` | `median` | solid line of the fits: `median` or `maxll` |
| `CALIB_SOLVER` | `compiled` | closed form of `_exp` and `_log` and compiled Dormand–Prince solver (`compiled_ode.py`) for the other laws; also used by stages 3 and 5. `analytic` (closed forms, `odeint` otherwise), `odeint`, or `solve_ivp` (`CALIB_SOLVER_METHOD`) |
| `CALIB_TEST` | `0` | smoke test |

`HDF5_USE_FILE_LOCKING=FALSE` is set by the modules that read backends; export it before inspecting a
backend by hand while a calibration writes to it.

## Methods

- **Initialization.** Uniform initialization over the prior box left about 7% of the walkers stuck in a
  high-`std` mode (`initialization.py`, `walkers.py`). Walkers start in a small ball around a MAP
  estimate, and walkers whose mean log-likelihood lies more than 50 units below the median walker are
  removed before the chain is saved (no removal if more than half would be flagged).
- **Model selection** (`model_selection.py`). AIC, AICc (preferred to AIC: n/k is 8–15), BIC, DIC,
  WAIC, PSIS-LOO and WBIC, with differences to the best model and weights. `w_BIC` are BIC weights, not
  WBIC. Flags: `p_D < 0` (DIC unreliable), `k̂ > 0.7` (LOO unreliable for that observation), low WBIC
  effective sample size. Burn-in: first 25% of the steps.
- **Convergence** (`convergence_check.py`, `run_until_converged.py`). Integrated autocorrelation time,
  effective sample size, acceptance fraction, lower tail of the log-likelihood and split R-hat with the
  walkers as chains.
- **Figures** (`postprocess.py`, `posterior_histogram.py`, `group_comparison.py`). Fits of each mouse
  with 95% credible bands of the curve and 95% posterior predictive intervals, CCC and PCC with credible
  intervals, posterior histograms, box plots with Mann–Whitney U tests between groups. The thesis figures
  of all stages use `thesis_style.py` (fonts, axes, bands, histograms, violins and traces of the thesis).

## Modules

```
global.py                calibration driver (both modes)
data_loader.py           control data files
models.py                growth models and ODE solution
bounds.py                uniform priors and labels
inference.py             log-prior, log-likelihood, log-probability
initialization.py        MAP search and walker initialization
walkers.py               stuck-walker detection and removal
metrics.py               CCC, PCC, NRMSE, MAPE
model_selection.py       information criteria
convergence_check.py     convergence diagnostics of the backends
run_until_converged.py   extends a chain until explicit convergence criteria are met
backend_progress.py      steps of each backend (used by the .slurm files)
buffered_backend.py      emcee HDF5 backend that writes the chain in blocks
postprocess.py           posterior summaries, bands and fit figures
posterior_histogram.py   posterior histograms
group_comparison.py      box plots and Mann-Whitney U tests between groups
growth_params.py         growth parameters inherited by the treatment stages
thesis_style.py          plotting style shared by the thesis figures of all stages
thesis_scatter_figure.py observed versus predicted volumes of the control groups (fig3)
thesis_chain_traces.py   chain traces and convergence table of the thesis appendix
```
