"""Control-group data of the triple-negative breast cancer (TNBC) experiments.

Files in ``data/processed/`` follow the pattern
    control_{phenotype}_c{cohort}_m{mouse}_t0_t1_..._t5.txt
with two columns: day and tumor volume (mm^3).

The phenotype (``sensitive`` / ``resistant``) defines the calibration group. The ``_t0_t1_..._t5``
suffix encodes the treatment/imaging days of the experimental design; for the control groups it is
the same in every file, is kept in ``treatment_days`` and does not enter the likelihood.

``load_data()`` filters by phenotype; ``load_control_data()`` returns both control groups.
"""
import glob
import os
import re

import numpy as np


# Known phenotypes -> calibration group
PHENOTYPES = ('sensitive', 'resistant')

# File-name pattern: control_sensitive_c0_m1_t0_t1_t2_t3_t4_t5
_FNAME_RE = re.compile(r'^control_(sensitive|resistant)_c(\d+)_m(\d+)_t(.+)$')


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed'))


def load_data(data_dir=DATA_DIR, phenotypes=None):
    """Loads the mice of the selected phenotypes.

    Parameters:
      data_dir    - folder with the .txt files.
      phenotypes  - phenotypes to include (e.g. ['resistant']); None includes all.

    Returns ``{group_name: [mouse_record, ...]}`` with group name ``control_{phenotype}``; each record
    has the keys 'name' (e.g. 'c0_m1'), 'data' (N x 2 array), 'treatment', 'treatment_days',
    'phenotype' and 'cohort'.

    Mice are ordered by ``sorted()`` on the file path; this order defines ic1..icN in the global mode.
    """
    if phenotypes is not None:
        phenotypes = set(phenotypes)

    full_data = {}
    for path in sorted(glob.glob(os.path.join(data_dir, '*.txt'))):
        base = os.path.basename(path)[:-4]
        m = _FNAME_RE.match(base)
        if m is None:
            continue
        phenotype, cohort, mouse, t_suffix = m.group(1), m.group(2), m.group(3), m.group(4)

        if phenotypes is not None and phenotype not in phenotypes:
            continue

        # 't0_t1_t2_t3_t4_t5' -> [0, 1, 2, 3, 4, 5]
        t_days = np.array([int(t) for t in t_suffix.split('_t')], dtype=int)

        data = np.loadtxt(path, ndmin=2)
        group = f'control_{phenotype}'
        full_data.setdefault(group, []).append({
            'name': f'c{cohort}_m{mouse}',
            'data': data,
            'treatment': 'control',
            'treatment_days': t_days,
            'phenotype': phenotype,
            'cohort': cohort,
        })

    return full_data


def load_control_data(data_dir=DATA_DIR):
    """Both control groups, with the keys ``'control_sensitive'`` and ``'control_resistant'``
    guaranteed even if empty."""
    full = load_data(data_dir, phenotypes=PHENOTYPES)
    for p in PHENOTYPES:
        full.setdefault(f'control_{p}', [])
    return full
