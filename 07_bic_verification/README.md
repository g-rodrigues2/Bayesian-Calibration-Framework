# Stage 7: verification of the maximum log-likelihoods of the BIC tables

The BIC of a model uses only its maximum log-likelihood, which the calibrations take from the best MCMC sample.
This stage checks every cell of the BIC tables of Chapter 4 (growth models, growth sharing scenarios, radiotherapy
models and sharing scenarios, immunotherapy models and SUV functions): 670 cells, group and mouse-specific
calibrations.

For each cell, `verify_bic.py`
1. re-evaluates the stored maximum-likelihood sample (`estimates/<cell>.npz`) with the likelihood of the stage that
   produced it (reproduction check), and
2. maximizes the log-likelihood inside the prior box with bounded Powell searches, restarted until they stop
   improving, from the stored sample, from four perturbations of it and, for the mouse-specific problems, from six
   uniform draws over the box.

The refined log-likelihood is never below that of the stored sample, so a BIC can only decrease.

For the radiotherapy models (all except the linear-quadratic one) the searches run over the radiotherapy
parameters only. These models are Bernoulli equations, dT/dt = T [r (1 - T/K) - g(t)], whose solution is
T(t) = T0 A(t) / (1 + T0 B(t)) with A and B independent of the initial volume, so for fixed radiotherapy
parameters the initial volume of each mouse and the common std are maximized exactly, without new ODE
solutions (`fast_profile.py`). The refined vector is then evaluated with the likelihood of the calibrations.
On the group calibrations this gives the same or higher maxima as the direct search over all parameters,
in minutes instead of hours.
`summarize_bic.py` recomputes each table with the refined values and reports the best model of every row and the
BIC weights of the group calibrations before and after.

```bash
python verify_bic.py run --scope global --workers 14      # group and scenario calibrations
python verify_bic.py run --scope mouse --workers 14       # mouse-specific calibrations
python verify_bic.py run --tables S2.3                    # one table
python summarize_bic.py                                   # results/summary.md and results/table_<label>.csv
python -m pytest tests -q
```

Results are appended to `results/cells.csv` (cells already present are skipped, so an interrupted run resumes);
refined parameter vectors go to `results/pars/`. The whole verification takes about an hour on a 16-core workstation.

`estimates/` holds the maximum-likelihood samples of the calibrations reported in the thesis. They were copied from
the original runs with `python verify_bic.py import --legacy <folder of the original calibrations>`; the cell list
and the source of each estimate are in `entries.py`.

## Findings of the reproduction check

- **Linear-quadratic model.** The volume measured on a fraction day is the volume before irradiation and the fraction
  of day 0 is applied, as in the calibrations of the thesis (`03_radiotherapy/radiation_models.solve_model_lq`).
- **Mouse-specific radiotherapy calibrations** used wider priors than the group calibrations: 1 for every
  radiotherapy parameter of the models without delay, and 2 for α, β and the onset parameter of the delayed models,
  with the delay up to 8 days (`03_radiotherapy/bounds_rad._MOUSE_SPECIFIC_UPPER`). The mouse-specific
  immunotherapy calibrations used initial volumes above 60 mm³ and std in (0.1, 200)
  (`05_immunotherapy/bounds_imm`).
- **Delayed onset-factor models (EDEF and CDEF with delay, Table S2.3).** The original calibrations activated the
  effect at the fraction day instead of fraction day + delay, so during the delay the onset factor was negative. The
  verification evaluates the model of the thesis (effect from τᵢ + δ), starting also from the estimates of the
  delayed models without the factor.
- **Group immunotherapy calibrations of eight models (Table S3.2)** were run with growth parameters slightly different
  from those fixed in the thesis; their log-likelihoods differ by about 1 and are recomputed on the common basis.
- **Accelerated immunotherapy terms with large decay rates (Table S3.2).** With the widened priors of the cumulative
  dose (CD) and accelerated cumulative dose (ACD) models, the effect of each administration can last a small
  fraction of a day. scipy's `odeint` steps over these pulses and underestimates the tumor kill, which produced
  apparently better maxima at the upper prior limits. The verification integrates the models with the compiled
  solver of `01_control_growth/compiled_ode.py`, restarted at every administration, which agrees with a DOP853
  reference restarted at the administrations (`tests/test_compiled_ode.py`); with it, the de Pillis-Radunskaya law
  has the lowest global BIC.
- **Radiotherapy sharing scenarios.** The original scenario runs used δ up to 10 days (15 in scenario 4); scenario 8
  has δ = 9.95 for the sensitive group, outside the prior of the thesis (0–8 days).
