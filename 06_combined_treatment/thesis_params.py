"""Parameters inherited from the sequential calibration stages (Chapter 4 of the thesis).

The combined radiotherapy + immunotherapy analyses keep growth, radiotherapy and immunotherapy
parameters fixed and estimate only initial volumes, the residual standard deviation and, when
present, the parameters of the SUV function. This module is the single source of those values:

    growth         logistic model, sharing scenario s2 of the control groups (Section 4.2)
    radiotherapy   delayed exponential decay, scenario 2, radiation-sensitive group (Section 4.3)
    immunotherapy  de Pillis-Radunskaya response, global calibration of the immunotherapy-only
                   group (Section 4.4); d is used only by the formulation without SUV
"""
import _path_setup  # noqa: F401
from growth_params import _FIXED_BY_PHENOTYPE

RADIOTHERAPY = {'sensitive': dict(a_rad=1.52716650e-02, b_rad=6.91980858e-02, delay=4.76117778e+00)}
IMMUNOTHERAPY = {'sensitive': dict(d=1.5243135102396321, s=0.015152682754421672, lam=1.928733156897561)}


def upstream_params(phenotype):
    """Dict with r, cc, a_rad, b_rad, delay, d, s and lam for a phenotype."""
    if phenotype not in RADIOTHERAPY or phenotype not in IMMUNOTHERAPY:
        raise ValueError(f'No calibrated upstream parameters for phenotype {phenotype!r}.')
    return {**_FIXED_BY_PHENOTYPE[phenotype], **RADIOTHERAPY[phenotype], **IMMUNOTHERAPY[phenotype]}
