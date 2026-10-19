# Bayesian calibration of a biology-based model using [89Zr]-CD8 PET imaging and tumor volume changes to characterize therapeutic response to radiation and immunotherapy in triple-negative breast cancer

<!-- DOI badge: replace XXXXXXXX with the Zenodo record once the deposit is published -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX)

This repository holds the code and data of the article of the same name and of **Chapter 4** of the
doctoral thesis

> G. Rodrigues, *Mathematical models to describe tumoral behaviors under treatment regimens*,
> Graduate Program in Biometrics, Institute of Biosciences, São Paulo State University (UNESP),
> Botucatu, 2026. Advisor: P. F. A. Mancera. Co-advisor: E. A. B. F. Lima.

Both describe the same analysis, and every figure and table of either can be redrawn from this
repository. It calibrates, selects and validates models of the form

```
dT/dt = G(T,t) − R(T,t) − I(T,t)
```

for the tumor burden `T` of 4T1 (radiation-sensitive) and RR-4T1 (radiation-resistant) murine tumors,
with `G` the untreated growth, `R` the radiotherapy effect and `I` the immunotherapy effect, using the
longitudinal tumor volumes and the [89Zr]-CD8 PET images of Song et al.

The analysis is sequential: each stage calibrates one term and fixes the parameters estimated by the
previous stages.

| Stage | Folder | Thesis section | Data | Estimated |
|---|---|---|---|---|
| 1 | [`01_control_growth/`](01_control_growth/) | 4.2 | control, 9 + 5 mice | 7 growth models (global and mouse-specific) |
| 2 | [`02_control_sharing_scenarios/`](02_control_sharing_scenarios/) | 4.2 | control | sharing of `r` and `K` between phenotypes |
| 3 | [`03_radiotherapy/`](03_radiotherapy/) | 4.3 | radiotherapy, 15 + 9 mice | 19 radiotherapy-effect models |
| 4 | [`04_radiotherapy_sharing_scenarios/`](04_radiotherapy_sharing_scenarios/) | 4.3 | radiotherapy | sharing of `α`, `β`, `δ` between phenotypes |
| 5 | [`05_immunotherapy/`](05_immunotherapy/) | 4.4 | immunotherapy, 3 mice | 12 immunotherapy-effect models, prior predictive check, profile likelihood |
| 6 | [`06_combined_treatment/`](06_combined_treatment/) | 4.5–4.7 | radiotherapy + immunotherapy, 15 mice | prediction, SUV-based lysis, leave-one-out |
| 7 | [`07_bic_verification/`](07_bic_verification/) | 4.2–4.6 | all | verification of the maximum log-likelihoods of every BIC table (670 cells) |

`01_control_growth` also holds the modules shared by all stages: walker initialization around a MAP
estimate, stuck-walker removal, information criteria (AIC, AICc, BIC, DIC, WAIC, PSIS-LOO, WBIC),
convergence diagnostics, posterior bands and the plotting style of the figures. The other stages import
them through `_path_setup.py`.

[`REPRODUCE.md`](REPRODUCE.md) gives the command that produces each figure and table. The thinned
posterior chains used by those commands are included here. The walker-resolved chains, from which the
convergence diagnostics of the appendix are computed, are deposited separately because of their size;
the DOI of that deposit is given in the description of this record.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`arviz>=0.23` is required: earlier versions do not import with `matplotlib` 3.11.

## Running a stage

Every stage is configured with `CALIB_*` environment variables and runs `global.py` (stage 6:
`calibrate.py` and `loo.py`). Chains are stored in resumable HDF5 backends (`reinitiate_files/`, not
deposited), so re-running a stage with a larger `CALIB_CHAIN_SIZE` continues the chain. Outputs
(thinned chains without burn-in and stuck walkers, maximum-likelihood samples, figures and tables) are
written to `Output_Calibration/` (global mode) and `Output_Calibration_per_mouse/` (mouse-specific).

```bash
cd 01_control_growth
CALIB_TEST=1 python global.py                                   # 50-step smoke test
CALIB_MODELS=_log CALIB_CHAIN_SIZE=30000 python global.py       # one model, both control groups
python -m pytest tests -q
```

The `.slurm` files run each stage on one 128-core node; set the partition and allocation for your
cluster. A typical workflow samples on the cluster with `CALIB_BACKEND_ONLY=1` and produces the outputs
locally by running `global.py` again with the same `CALIB_CHAIN_SIZE` (the sampler is skipped).

In the treatment stages, `CALIB_GROWTH_SOURCE=fixed` uses the growth parameters reported in the thesis
(logistic model, scenario 2) and `auto` reads them from the outputs of stages 1 and 2. The scripts that
redraw the figures of the thesis always use the first.

## Convergence

`01_control_growth/run_until_converged.py` extends a chain until every parameter has at least 50
integrated autocorrelation times after burn-in and split R-hat below 1.05, and records the diagnostics
of each extension. `05_immunotherapy/posterior_diagnostics.py` and `profile_likelihood.py` report the
autocorrelation, effective sample size, R-hat and profile likelihood used to assess practical
identifiability. `01_control_growth/thesis_chain_traces.py` draws the traces of the selected chains and
writes their convergence table (thesis Appendix).

## Performance

Long extensions are dominated by the likelihood and by file access, not by the sampler:

- stage 6 integrates the combined model with a compiled Dormand–Prince solver restarted at every
  treatment administration (`06_combined_treatment/rad_immuno_fast.py`, default; about 50 times faster
  than `odeint`, log-likelihood differences below 3e-4);
- stages 1 to 5 use the closed-form solutions of the exponential and logistic models and, for the other
  models, a compiled Dormand–Prince solver restarted at every effect onset
  (`01_control_growth/compiled_ode.py`, default `CALIB_SOLVER=compiled`; 10 to 50 times faster per solve
  than `odeint`, volumes within 1e-6 of a tight `odeint` reference, `tests/test_compiled_ode.py`);
  `CALIB_SOLVER=odeint` restores scipy;
- the BIC verification (stage 7) maximizes the radiotherapy log-likelihoods through their exact profile
  over the initial volumes and the std (`07_bic_verification/fast_profile.py`), so the searches run over
  the radiotherapy parameters only;
- the HDF5 backend writes the chain in blocks (`01_control_growth/buffered_backend.py`); files remain
  readable by `emcee.backends.HDFBackend`.

With a fast likelihood, run one serial sampler per chain (`CALIB_USE_POOL=0`) and as many chains in
parallel as there are cores. Do not open a backend while a calibration writes to it (on Windows this
aborts the run).

## Figures of the thesis

Every figure of Chapter 4 and of its appendix is produced by the `thesis_*.py` scripts of the stages,
which share the style of `01_control_growth/thesis_style.py`: Times New Roman text, Computer Modern
mathematics and canvases sized by `thesis_style.canvas` so that the point sizes of the scripts print
between 7 and 9 pt at the width each figure occupies on the page.

## MCMC chains

`export_chains.py` writes the chains of the selected models and of the leave-one-out analysis (every
10th step, float32, with the burn-in and the stuck walkers used in the thesis) for the data deposit; the
convergence diagnostics of the Appendix are reproduced from these files (`read_exported_chain`).

## Data

See [`data/README.md`](data/README.md).

## License

MIT, see [LICENSE](LICENSE). If you use this code or data, please cite the deposit, the thesis and the
article listed in [CITATION.cff](CITATION.cff).
