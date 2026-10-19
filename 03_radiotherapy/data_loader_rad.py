"""Loading of the radiotherapy data.

Files: radiation_{phenotype}_c{cohort}_m{mouse}[...].txt with two columns (day, volume in mm^3).

The dose days (radiotherapy fractions) are not read from the file names: they are given with
`dose_days` (CALIB_DOSE_DAYS). The default is the schedule of the experiment: six daily 2 Gy fractions
on days 0-5 (the _t0_..._t5 suffix of the file names).
"""
import glob
import os
import re

import numpy as np

PHENOTYPES = ('sensitive', 'resistant')
# Matches the name up to the mouse number; any suffix (_t..., etc.) is ignored.
_FNAME_RE = re.compile(r'^radiation_(sensitive|resistant)_c(\d+)_m(\d+)(?:_.*)?$')

# Default schedule: fractions on days 0-5.
DEFAULT_DOSE_DAYS = [0, 1, 2, 3, 4, 5]


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed'))


def load_data(data_dir=DATA_DIR, phenotypes=None, dose_days=None):
    """{group: [mice...]} with group = radiation_{phenotype}. Each record has 'name', 'data' (Nx2),
    'treatment'='radiation', 'treatment_days', 'phenotype' and 'cohort'.

    dose_days: dose days applied to all mice; None -> DEFAULT_DOSE_DAYS (days 0-5).
    """
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
        data = np.loadtxt(path, ndmin=2)
        full_data.setdefault(f'radiation_{phen}', []).append({
            'name': f'c{cohort}_m{mouse}',
            'data': data,
            'treatment': 'radiation',
            'treatment_days': days.copy(),
            'phenotype': phen,
            'cohort': cohort,
        })
    return full_data


def load_radiation_data(data_dir=DATA_DIR, dose_days=None):
    """Both radiotherapy groups (present even if empty).

    dose_days sets the dose schedule (None -> days 0-5)."""
    full = load_data(data_dir, phenotypes=PHENOTYPES, dose_days=dose_days)
    for p in PHENOTYPES:
        full.setdefault(f'radiation_{p}', [])
    return full


def parse_dose_days_env(env_value):
    """'CALIB_DOSE_DAYS' ('0,1,2,3,4,5' or empty) -> array, or None when not given."""
    env_value = (env_value or '').strip()
    if not env_value:
        return None
    return np.array([int(x) for x in env_value.split(',') if x.strip() != ''],
                    dtype=int)
