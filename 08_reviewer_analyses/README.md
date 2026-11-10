# Stage 8: analyses requested by the reviewers

This stage collects the analyses added in response to the peer review of the article
*Bayesian calibration of a biology-based mathematical model using [89Zr]-CD8 PET imaging and tumor
volume changes to characterize therapeutic response to radiation and immunotherapy in triple-negative
breast cancer* (Mathematical Biosciences, first round). Each analysis answers a specific comment and
leaves its numbers in `results/`, so that every figure the revision quotes can be recomputed from this
deposit alone.

Stages 1 to 7 are unchanged by the review except where noted below. Analyses that extend an existing
pipeline live with that pipeline and are listed here with a pointer.

| Reviewer comment | Analysis | Where |
|---|---|---|
| R1.1, R1.9 — compare the SUV and non-SUV formulations in the same folds, with a positive parameterization | `_SUVExp` (`phi = exp(a z + b)`, standardized SUV) added to the leave-one-out comparison | `06_combined_treatment/loo.py` |
| R1.1 — compare SUV with other prespecified covariates | `_VolExp`, the same form with the initial tumor volume in place of SUV | `06_combined_treatment/loo.py` |
| R1.1 — a non-SUV comparator refitted on the same basis | `_RadImmunoFitD` (M0), one lysis rate re-estimated in each training fold; the counts of the leave-one-out table are against it | `06_combined_treatment/loo.py`, `fast_fit.py` |
| R1.4 — correlated and heteroscedastic residuals, and their effect on the BIC | `likelihood_sensitivity.py` | here |
| R1.5 — dependence of the leave-one-out comparison on the summary of the immunotherapy posterior that is inherited | `vector_sensitivity.py`, the five formulations refitted in the same folds under the marginal medians and under an earlier vector | here |
| R1.6, R1.16 — leave-one-out score integrated over the posterior, paired per-mouse differences, interval coverage | `loo_posterior.py`, Q = 6000 draws per fold, paired differences against M0 and counts against M0 and the inherited rate | `06_combined_treatment/loo_posterior.py` |

## `likelihood_sensitivity.py`

The likelihood of the calibrations treats every tumor-volume observation as independent and of
constant variance. Reviewer 1 observes that repeated measurements of the same animal are correlated
and that the residual spread may grow with tumor size, so that the intervals can be too narrow and the
BIC, whose penalty depends on the sample size, can be affected. The script quantifies both effects on
the fitted models, without repeating the model selection:

1. residuals of the global combined-treatment fit at the posterior median, mouse by mouse;
2. lag-1 autocorrelation of those residuals within each mouse, pooled over mice;
3. dependence of the residual spread on the fitted volume (Spearman correlation);
4. the effective sample size of an AR(1) sequence with the pooled autocorrelation,
   `n_eff = n (1 - rho) / (1 + rho)`, and the BIC of the parameter-sharing scenarios recomputed with
   `n_eff` in place of `n`, which shows whether the selection survives the correction.

```bash
python likelihood_sensitivity.py
```

Inputs: the global SUV chain of stage 6 (`06_combined_treatment/Output_Calibration/suv/`) and the
verified log-likelihoods of stage 7 (`07_bic_verification/results/cells.csv`). Outputs:
`results/residual_structure.csv` (lag-1 autocorrelation per mouse), `results/residual_summary.csv`
(pooled, mean and range of that autocorrelation, the Spearman correlation and its p-value, and the
factor `(1 - rho)/(1 + rho)`) and `results/bic_neff.csv`
(each scenario with `n`, `n_eff`, `k`, the log-likelihood and the BIC and weights under both sample
sizes). The run takes a few seconds.

Values obtained with the deposited chains: pooled lag-1 autocorrelation 0.693 (mean over mice 0.660,
range 0.347 to 0.848); Spearman correlation between the absolute residual and the fitted volume 0.284
(p = 1.8e-8); effective sample size 18.1% of the nominal one, so the credible intervals are about 2.4
times narrower than a formulation with an explicit within-mouse covariance would give. Under `n_eff`
the control groups go from n = 284 to 51 and the radiotherapy groups from n = 505 to 91, and Scenario 2
keeps the lowest BIC in both, with its weight falling from 0.940 to 0.874 and from 0.815 to 0.643.
The correction therefore weakens the evidence without reversing any selection.

## `vector_sensitivity.py`

The combined-treatment stage inherits the immunotherapy vector `(d, s, lam)` from stage 5, and the
maximum lysis rate of that calibration is only weakly identified. The script asks whether the
leave-one-out comparison of Chapter 4 depends on which summary of that posterior is carried forward,
repeating the strict comparison, all five formulations in the same fifteen folds, under the three
summaries one could reasonably inherit:

1. `max_likelihood`, the sample of largest likelihood, which is the vector of the leave-one-out table
   (`06_combined_treatment/thesis_params.py`);
2. `marginal_median`, the marginal medians of the converged immunotherapy chain;
3. `submitted`, the vector an earlier version inherited, kept in the script as a constant because it
   is neither the maximum nor the median of any stored chain and traces to the legacy calibration.

Under the two alternative vectors every fold is refitted as the leave-one-out table is, by the profiled
maximum of `06_combined_treatment/fast_fit.py` with its multi-start refinement, and the held-out mouse
is scored as `loo.py strict` scores it, 150 folds in all. Under `max_likelihood` nothing is refitted:
the fold estimates of stage 6 are read, and the per-mouse scores are checked against
`strict_loo_per_mouse.csv` to six decimals; if the check fails the script stops and asks for
`loo.py strict` to be rerun.

```bash
python vector_sensitivity.py                 # one process per core
python vector_sensitivity.py --processes 8
python vector_sensitivity.py --check-only    # only the max_likelihood check, in seconds
```

Inputs: the immunotherapy chain of stage 5
(`05_immunotherapy/Output_Calibration/multi_chain_io_sensitive_dePillisRadunskayaLaw.gz`), the fold
estimates of stage 6 and `06_combined_treatment/Output_Calibration/loo/strict_loo_per_mouse.csv`.
Outputs: `results/vector_sensitivity_per_fold.csv` (training and held-out log-likelihood of every fold
under every vector) and `results/vector_sensitivity_summary.csv` (per vector, the total held-out
log-likelihood of each formulation, the count of mice above M0, the ordering of the five, and the
advantage of the best formulation with SUV over the best without it). That advantage is 486, 406 and
350 under the three vectors, the values Section 4.7 of the thesis quotes. The refits take about 27
CPU-hours; the formulations with a lysis function to search take minutes per fold, so they are
scheduled first.
