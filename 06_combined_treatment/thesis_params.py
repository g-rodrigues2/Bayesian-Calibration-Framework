"""Parameters inherited from the sequential calibration stages (Chapter 4 of the thesis).

The combined radiotherapy + immunotherapy analyses keep growth, radiotherapy and immunotherapy
parameters fixed and estimate only initial volumes, the residual standard deviation and, when
present, the parameters of the SUV function. This module is the single source of those values:

    growth         logistic model, sharing scenario s2 of the control groups (Section 4.2)
    radiotherapy   delayed exponential decay, scenario 2, radiation-sensitive group (Section 4.3)
    immunotherapy  de Pillis-Radunskaya response, global calibration of the immunotherapy-only
                   group (Section 4.4)

Each inherited vector is one posterior sample, the one of largest likelihood, rather than a set of
marginal summaries, so that it is a parameter vector the model was actually evaluated at. The
immunotherapy vector is the maximum-likelihood sample of

    ../05_immunotherapy/Output_Calibration/multi_ll_pars_io_sensitive_dePillisRadunskayaLaw.npz

which is the converged calibration of that stage. Only the formulation without SUV reads `d`, since
the others replace the lysis coefficient by a function of the covariate, but `s` and `lam` enter the
saturating factor of every formulation, so the three are inherited together and come from the same
sample. `tests/test_rad_immuno.py` re-derives them from that file, so they cannot drift away from the
calibration they claim to come from.
"""
import _path_setup  # noqa: F401
from growth_params import _FIXED_BY_PHENOTYPE

RADIOTHERAPY = {'sensitive': dict(a_rad=1.52716650e-02, b_rad=6.91980858e-02, delay=4.76117778e+00)}
IMMUNOTHERAPY = {'sensitive': dict(d=1.921404956124263, s=0.011735717455240095,
                                   lam=1.9733508998165368)}


def upstream_params(phenotype):
    """Dict with r, cc, a_rad, b_rad, delay, d, s and lam for a phenotype."""
    if phenotype not in RADIOTHERAPY or phenotype not in IMMUNOTHERAPY:
        raise ValueError(f'No calibrated upstream parameters for phenotype {phenotype!r}.')
    return {**_FIXED_BY_PHENOTYPE[phenotype], **RADIOTHERAPY[phenotype], **IMMUNOTHERAPY[phenotype]}
