# Stage 5: immunotherapy-effect models

Calibration and selection of the 12 formulations of the immunotherapy term (thesis Table S3.1) on
`io_sensitive` (4T1, 3 mice):

$$\frac{dT}{dt} = rT\Bigl(1-\frac{T}{K}\Bigr) - I(T,t)\,T$$

Growth is fixed (`CALIB_GROWTH_SOURCE=fixed`); anti-PD-1/anti-CTLA-4 is given on days 0, 3, …, 18 with a
normalized unit dose (`CALIB_DOSE_DAYS`, `CALIB_DOSE`).

| Model | Effect per administration j (active for t ≥ τ_j) | Parameters |
|---|---|---|
| `_ExponentialDecay` | α D_j exp(−β(t − τ_j)) | α, β |
| `_CumulativeDose` | α D_cum exp(−β(t − τ_j)) | α, β |
| `_AccelED`, `_AccelCD` | as above with (t − τ_j)² | α, β |
| `_MichaelisMenten` | α D_j / (1 + β D_j) | α, β |
| `_dePillisRadunskayaLaw` | d D_j^λ / (s T^λ + D_j^λ) | d, s, λ |
| `_EDwith{Log,Exp,Sin}Factor`, `_CDwith{Log,Exp,Sin}Factor` | ED or CD with an onset factor | α, β, and steepness, rate or frequency |

The de Pillis–Radunskaya law has the lowest BIC. The upper bounds of β for `_CumulativeDose` (15) and
`_AccelCD` (100) were widened after the prior predictive check showed observations above the 5–95%
prior band with the original bounds (8 and 5); see `bounds_imm.py`.

## Usage

```bash
CALIB_GROWTH_SOURCE=fixed CALIB_MODELS=_dePillisRadunskayaLaw CALIB_CHAIN_SIZE=30000 python global.py
CALIB_MODE=mouse_specific CALIB_GROWTH_SOURCE=fixed CALIB_MODELS=_AccelCD python global.py
CALIB_GROWTH_SOURCE=fixed python ppc_grid.py --draws 20000 --seed 0   # prior predictive figure of the thesis
CALIB_GROWTH_SOURCE=fixed python profile_likelihood.py --model _dePillisRadunskayaLaw --extend d=50
CALIB_GROWTH_SOURCE=fixed python profile_likelihood.py --plot-only   # redraw the profile figure from its table
CALIB_GROWTH_SOURCE=fixed python thesis_figures.py                   # fig10a-c and fig11
python posterior_diagnostics.py --backend reinitiate_files/reinitiate_io_sensitive_dePillisRadunskayaLaw.h5 \
    --labels d,s,lam,ic1,ic2,ic3,std --lower 0.01,0.01,0.01,60,60,60,0 --upper 2,2,2,287.44,287.44,287.44,200
sbatch immuno_parallel.slurm
python -m pytest tests -q
```

`CALIB_MS_PARAMS` (global mode) calibrates the listed parameters per mouse and the others globally, e.g.
`CALIB_MS_PARAMS=d`. Model selection compares every model with saved outputs and writes
`Output_Calibration/model_selection_imm_{mode}.csv`.

## Identifiability of the de Pillis–Radunskaya parameters

With a constant dose the effect of each administration is d / (1 + s T^λ). The data constrain the
resulting kill rate but not d, s and λ separately: the profile log-likelihood of d is flat over a wide
range, so the posterior of d is shaped by its prior bounds. `profile_likelihood.py` and
`posterior_diagnostics.py` document this; the posterior predictions (fits, CCC) remain well defined.

## Modules

```
global.py                calibration driver (both modes, mixed parameterization)
data_loader_imm.py       immunotherapy data files, schedule and SUV
immuno_models.py         the 12 immunotherapy models and ODE solution
bounds_imm.py            priors and labels
inference_imm.py         likelihood with fixed growth
model_selection_imm.py   information criteria
postprocess_imm.py       bands, CCC/PCC and figures
prior_predictive.py      prior predictive check per model
ppc_grid.py              prior predictive figure of the 12 models
posterior_diagnostics.py autocorrelation, ESS, R-hat, mass near bounds, correlations
profile_likelihood.py    profile log-likelihood of the treatment parameters
thesis_figures.py        thesis figures of the de Pillis-Radunskaya calibration
```
