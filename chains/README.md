# MCMC chains

Posterior samples of the models selected in Chapter 4 and of the leave-one-out analysis of Section 4.7,
written by [`export_chains.py`](../export_chains.py) from the resumable HDF5 backends, which are not
published. Chains of the models that were not selected are not here; their maximum-likelihood samples
are in `Output_Calibration/` of each stage.

| Folder | Chains |
|---|---|
| `01_control_growth/` | logistic growth of the two control groups |
| `02_control_sharing_scenarios/` | sharing scenario 2 between phenotypes |
| `04_radiotherapy_sharing_scenarios/` | delayed exponential decay, sharing scenario 2 |
| `05_immunotherapy/` | de Pillis-Radunskaya law (`d` and `s` sampled on the log scale) |
| `06_combined_treatment/` | combined-therapy prediction with fixed parameters and with the linear SUV-based lysis rate, the 45 training folds (`EXCL-*`) of the three formulations compared in Section 4.7 and the 30 held-out chains (`PRED-*`) of the conditional analysis |

`manifest.json` lists every file with the number of steps, the thinning interval, the numbers of walkers
and parameters, and the file size.

## Format

Each file is a compressed NumPy archive, read by `read_exported_chain` in `export_chains.py`:

| Key | Content |
|---|---|
| `chain` | array (stored steps, walkers, parameters), float32, every `thin`-th step of the chain |
| `log_prob` | array (stored steps, walkers), float32, log-posterior of the stored samples |
| `thin` | thinning interval |
| `n_steps` | number of steps of the original chain |
| `burn_in` | burn-in used in the thesis (first quarter of the steps), in original steps |
| `stuck` | boolean mask of the walkers discarded as stuck |
| `labels` | parameter names, in the sampling coordinates |

The parameter order is that of the calibration code (`bounds*.py` of each stage). The posterior samples
used in the thesis are `chain[burn_in // thin:, ~stuck, :]`. Thinning by ten keeps at least thirty
samples per integrated autocorrelation time in every chain, so the convergence diagnostics of the
Appendix (steps per autocorrelation time and split R-hat) are recomputed from these files.
