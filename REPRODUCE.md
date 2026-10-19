# Reproducing the figures and tables of Chapter 4

Every command runs from the folder of its stage. Figures are written with the file names used in the
thesis (`Figuras/cap4_tnbc/`). Treatment stages use `CALIB_GROWTH_SOURCE=fixed` (growth parameters of
the thesis); the scripts that redraw the figures of the thesis ask for those values themselves. Test
suites: `python -m pytest tests -q` in each stage (WAIC and PSIS-LOO require `arviz`).

**What a command needs.** The figure and table commands read the thinned chains
(`Output_Calibration/*.gz`) and the maximum-likelihood samples (`*.npz`), which are **part of this
deposit**: every figure and table can be redrawn without any other download, in minutes. What is not
deposited are the resumable HDF5 backends of `reinitiate_files/`, which only serve to extend a chain,
and the walker-resolved chains, which are needed solely for the convergence diagnostics of the appendix
and are in the separate data deposit. A command that starts a calibration therefore samples it from
scratch, which takes hours per chain; a command that redraws a figure does not.

## Stage 1 — control growth (`01_control_growth`)

| Thesis item | Command | Output |
|---|---|---|
| BIC of the growth models (global and mouse-specific) | `CALIB_MODELS=_exp,_mendel,_bert,_lin,_gomp,_surf,_log python global.py` then `python model_selection.py --metrics bic` | `Output_Calibration/` |
| Observed vs predicted volumes, logistic model | `python thesis_scatter_figure.py --model _log --out Output_Calibration/thesis_figures/fig3.png` | `fig3.png` |
| Appendix: walker traces and convergence tables | `python thesis_chain_traces.py` (`--table-only` rewrites the tables) | `Output_Calibration/thesis_figures/chains/` |

## Stage 2 — growth sharing scenarios (`02_control_sharing_scenarios`)

| Thesis item | Command | Output |
|---|---|---|
| BIC of the four scenarios | `CALIB_MODELS=_log python global.py` | `Output_Calibration/scenario_selection_log.csv` |
| Posterior violins of r and K | `python thesis_figures.py --figures fig4` | `fig4.png` |
| Posterior histograms, scenario 2 | `python thesis_figures.py --figures fig5` | `fig5a.png`–`fig5c.png` |
| Best and worst mice, scenario 2 | `python thesis_figures.py --figures fig6` | `fig6.png` |
| Appendix: every control mouse | `python thesis_figures.py --figures figS1` | `figS1_1.png`, `figS1_2.png` |

## Stage 3 — radiotherapy models (`03_radiotherapy`)

| Thesis item | Command | Output |
|---|---|---|
| BIC of the radiotherapy-effect models | `CALIB_GROWTH_SOURCE=fixed python global.py` then `python model_selection_rad.py` | `Output_Calibration/` |
| Prior predictive check | `CALIB_GROWTH_SOURCE=fixed python prior_predictive_rad.py` | `Output_Calibration/` |

## Stage 4 — radiotherapy sharing scenarios (`04_radiotherapy_sharing_scenarios`)

Prefix every command with `CALIB_GROWTH_SOURCE=fixed CALIB_DOSE_DAYS=0,1,2,3,4,5`.

| Thesis item | Command | Output |
|---|---|---|
| BIC of the eight scenarios | `python global.py` | `Output_Calibration/scenario_selection__ExponentialDecayDelay.csv` |
| Chains of the original scenario runs (V1, V3–V8) | `python legacy_scenario_chains.py --legacy <original runs> --scenarios 1,3,4,5,6,7,8 --out Output_Calibration/legacy` | `Output_Calibration/legacy/` |
| Posterior violins of α, β and δ | `python thesis_figures.py --figures fig7` | `fig7.png` |
| Posterior histograms, scenario V2 | `python thesis_figures.py --figures fig8` | `fig8a.png`–`fig8d.png` |
| Best and worst mice, scenario V2 | `python thesis_figures.py --figures fig9` | `fig9.png` |
| Appendix: every radiotherapy mouse | `python thesis_figures.py --figures figS2` | `figS2_1.png`, `figS2_2.png` |

## Stage 5 — immunotherapy (`05_immunotherapy`)

Prefix every command with `CALIB_GROWTH_SOURCE=fixed`.

| Thesis item | Command | Output |
|---|---|---|
| BIC of the 12 immunotherapy models | `python global.py` then `python model_selection_imm.py` | `Output_Calibration/model_selection_imm_global.csv` |
| Prior predictive check of the 12 models | `python ppc_grid.py --draws 20000 --seed 0` | `Output_Calibration/` |
| de Pillis–Radunskaya chain (d and s on the log scale) | `CALIB_MODELS=_dePillisRadunskayaLaw CALIB_LOG_PARAMS=d,s python global.py` | `Output_Calibration/` |
| Posterior histograms of d, s and λ | `python thesis_figures.py --figures fig10` | `fig10a.png`–`fig10c.png` |
| Global and mouse-specific fits | `python thesis_figures.py --figures fig11` | `fig11.png` |
| Appendix: profile log-likelihood | `python profile_likelihood.py --chain <posterior chain> --extend d=50 --lower lam=0.3` (`--plot-only` redraws) | `figS3_1.png` |

## Stage 6 — radiotherapy + immunotherapy (`06_combined_treatment`)

| Thesis item | Command | Output |
|---|---|---|
| Prediction with fixed parameters | `python calibrate.py prediction` then `python thesis_figures.py --figures fig12` | `global_fixedpars_PostPredBand.png`, `_ALL.png` |
| BIC of the SUV-based lysis functions | `python calibrate.py suv` | `Output_Calibration/suv/model_selection.csv` |
| SUV-based predictions | `python thesis_figures.py --figures fig13` | `global_SUV_based_PostPredBand.png`, `_ALL.png` |
| Posterior of a and b | `python posterior_suv_params.py` and `python thesis_figures.py --figures fig14` | `fig14a.png`, `fig14b.png` |
| Leave-one-out: held-out log-likelihood table | `python loo.py train`, `python loo.py strict` | `Output_Calibration/loo/strict_loo_summary.csv` |
| Leave-one-out figures | `python loo.py conditional` then `python thesis_figures.py --figures loo` | `SUV_contribution.png`, `LOO_held_out_SUV_based_ALL.png`, `fig_LOO_semSUV_ALL.png` |

## Stage 7 — verification of the BIC tables (`07_bic_verification`)

| Thesis item | Command | Output |
|---|---|---|
| Maximum log-likelihoods of every BIC table of Chapter 4 (growth models, sharing scenarios, radiotherapy models and scenarios, immunotherapy models, SUV functions) | `python verify_bic.py run --scope global` and `--scope mouse`, then `python summarize_bic.py` | `results/summary.md`, `results/table_<label>.csv` |

## Convergence

Each chain of the thesis was extended with `01_control_growth/run_until_converged.py` until every
parameter had at least 50 integrated autocorrelation times after burn-in and split R-hat below 1.05 (see
the READMEs of the stages for the exact commands and `Output_Calibration/convergence_*.csv` for the
diagnostics of each extension). Many chains can run at once: use `CALIB_USE_POOL=0` and one process per
chain. The chains exported for the data deposit (`export_chains.py`) reproduce the Appendix
tables of convergence diagnostics.
