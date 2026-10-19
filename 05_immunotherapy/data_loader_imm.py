"""Loading of the immunotherapy data.

Files: io_{phenotype}_c{cohort}_m{mouse}[_t...][_SUV{value}].txt with two columns (day, volume in mm^3).

Administration days are given with `dose_days` (CALIB_DOSE_DAYS) and are not read from the file names.
The default for this dataset is the administration schedule of the experiment, 0,3,6,9,12,15,18.

SUV: the value after '_SUV' in the file name is stored in mouse['SUV'] (None if absent). It is not used
in this stage; the SUV-based models are in 06_combined_treatment.
"""
import glob
import os
import re

import numpy as np

PHENOTYPES = ('sensitive', 'resistant')
# Matches io_{phenotype}_c{N}_m{N}; the rest (_t..., _SUV...) is ignored.
_FNAME_RE = re.compile(r'^io_(sensitive|resistant)_c(\d+)_m(\d+)(?:_.*)?$')
_SUV_RE = re.compile(r'_SUV([0-9.]+)')

# Administration schedule of this dataset (file names _t0_t3_t6_t9_t12_t15_t18, the same for the 3
# mice). Override with CALIB_DOSE_DAYS for other data.
DEFAULT_DOSE_DAYS = [0, 3, 6, 9, 12, 15, 18]


def _parse_suv(base):
    m = _SUV_RE.search(base)
    return float(m.group(1)) if m else None


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed'))


def load_data(data_dir=DATA_DIR, phenotypes=None, dose_days=None):
    """{group: [mice...]} with group = io_{phenotype}. Each record has 'name', 'data' (Nx2),
    'treatment'='immuno', 'treatment_days', 'SUV', 'phenotype' and 'cohort'."""
    if phenotypes is not None:
        phenotypes = set(phenotypes)
    days = np.asarray(DEFAULT_DOSE_DAYS if dose_days is None else dose_days,
                      dtype=int)

    full_data = {}
    for path in sorted(glob.glob(os.path.join(data_dir, '*.txt'))):
        base = os.path.basename(path)[:-4]
        m = _FNAME_RE.match(base)
        if m is None:
            continue
        phen, cohort, mouse = m.groups()
        if phenotypes is not None and phen not in phenotypes:
            continue
        full_data.setdefault(f'io_{phen}', []).append({
            'name': f'c{cohort}_m{mouse}',
            'data': np.loadtxt(path, ndmin=2),
            'treatment': 'immuno',
            'treatment_days': days.copy(),
            'SUV': _parse_suv(base),   # stored; not used in this stage
            'phenotype': phen,
            'cohort': cohort,
        })
    return full_data


def load_immuno_data(data_dir=DATA_DIR, dose_days=None):
    """Immunotherapy groups present in the data."""
    return load_data(data_dir, phenotypes=PHENOTYPES, dose_days=dose_days)


def parse_dose_days_env(env_value):
    """'CALIB_DOSE_DAYS' ('0,3,6,9,12,15,18') -> array; empty -> None."""
    env_value = (env_value or '').strip()
    if not env_value:
        return None
    return np.array([int(x) for x in env_value.split(',') if x.strip()],
                    dtype=int)
