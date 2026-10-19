# Stage 3: radiotherapy-effect models

Calibration and selection of 19 formulations of the radiotherapy term on `radiation_sensitive`
(15 mice) and `radiation_resistant` (9 mice):

$$\frac{dT}{dt} = rT\Bigl(1-\frac{T}{K}\Bigr) - R(t)\,T$$

Growth (`r`, `K`) is fixed at the values of the control stages (`CALIB_GROWTH_SOURCE`); the effect is
anchored at the fraction days (2 Gy on days 0–5; `CALIB_DOSE_DAYS`, `CALIB_DOSE`).

| Family | Models |
|---|---|
| Linear-quadratic surviving fraction (discrete jump at each fraction) | `_LQ` |
| Exponential decay (ED) and cumulative dose (CD) | `_ExponentialDecay`, `_CumulativeDose` |
| Accelerated decay | `_AccelED`, `_AccelCD` |
| ED and CD with logistic, exponential or sinusoidal onset factors | `_EDwithLogFactor`, `_EDwithExpFactor`, `_EDwithSinFactor`, `_CDwithLogFactor`, `_CDwithExpFactor`, `_CDwithSinFactor` |
| Delayed variants | `_ExponentialDecayDelay`, `_CumulativeDoseDelay`, `_ACDDelay`, `_AEDDelay`, `_EDEFDelay`, `_CDEFDelay`, `_EDSFDelay`, `_CDSFDelay` |

The delayed exponential decay model, `α Σ_i D_i exp(−β(t − τ_i − δ)) H(t − τ_i − δ)`, is carried to
stage 4.

Note: in `model_EDEFDelay` and `model_CDEFDelay` the signature order is (delay, rate), whereas the labels
and bounds in `bounds_rad.py` list (rate, delay); the bounds are those of the runs reported in the thesis.

## Usage

```bash
CALIB_GROWTH_SOURCE=fixed CALIB_MODELS=_ExponentialDecayDelay CALIB_CHAIN_SIZE=30000 python global.py
CALIB_MODE=mouse_specific CALIB_GROWTH_SOURCE=fixed CALIB_MODELS=_ExponentialDecayDelay python global.py
python prior_predictive_rad.py --models _ExponentialDecayDelay --n 300
sbatch radiotherapy_parallel.slurm
python -m pytest tests -q
```

Settings are those of stage 1 plus `CALIB_GROWTH_SOURCE` (`fixed`, `auto`, `control_log`,
`shared:<sN>`), `CALIB_DOSE_DAYS`, `CALIB_DOSE` and the budget of the MAP search (`CALIB_INIT_NRANDOM`,
`CALIB_INIT_MAXITER`, `CALIB_INIT_MAXFEV`).

## Modules

```
global.py                calibration driver (both modes)
data_loader_rad.py       radiotherapy data files and dose schedule
radiation_models.py      radiotherapy models and solvers (continuous and piecewise LQ)
bounds_rad.py            priors and labels
inference_rad.py         likelihood with fixed growth
model_selection_rad.py   information criteria
postprocess_rad.py       bands, CCC/PCC and figures
prior_predictive_rad.py  prior predictive check
```
