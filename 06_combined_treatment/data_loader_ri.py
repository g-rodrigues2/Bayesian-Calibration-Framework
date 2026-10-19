"""Data of the radiotherapy + immunotherapy cohort.

Files: rad+io_<phenotype>_c<cohort>_m<mouse>[_t...]_SUV<value>.txt, two columns (day, volume in mm^3).
The [89Zr]-CD8 PET SUV is read from the file name. Treatment schedules are the same for every
mouse (Table 4.x of the thesis) and can be overridden with CALIB_RAD_DAYS and CALIB_IO_DAYS:
radiotherapy on days 0-5 (2 Gy per fraction), immunotherapy on days 9, 12, ..., 24 (unit dose).
"""
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAD_DAYS = [0, 1, 2, 3, 4, 5]
IO_DAYS = [9, 12, 15, 18, 21, 24]
DOSE_RAD = 2.0
DOSE_IO = 1.0
_NAME = re.compile(r'^rad\+io_(sensitive|resistant)_c(\d+)_m(\d+)(?:_.*)?$')
_SUV = re.compile(r'_SUV([0-9.]+)$')


def _days(env, default):
    value = os.environ.get(env, '').strip()
    return np.array([float(x) for x in value.split(',')] if value else default, dtype=float)


def load_data(data_dir=None):
    """{group: [mouse records]} with group = 'rad+io_<phenotype>', mice in file-name order."""
    rad_days, io_days = _days('CALIB_RAD_DAYS', RAD_DAYS), _days('CALIB_IO_DAYS', IO_DAYS)
    groups = {}
    for path in sorted(glob.glob(os.path.join(data_dir or os.path.join(HERE, '..', 'data', 'processed'), '*.txt'))):
        base = os.path.basename(path)[:-4]
        match = _NAME.match(base)
        if match is None:
            continue
        phenotype, cohort, mouse = match.groups()
        suv = _SUV.search(base)
        groups.setdefault(f'rad+io_{phenotype}', []).append(dict(
            name=f'c{cohort}_m{mouse}', data=np.loadtxt(path, ndmin=2), SUV=float(suv.group(1)) if suv else None,
            rad_days=rad_days, io_days=io_days, phenotype=phenotype))
    return groups


def max_initial_volume(mice):
    return max(float(m['data'][0, 1]) for m in mice)
