"""Adds the folders of stages 1-6 to sys.path. Module names are unique across stages, so the likelihoods,
bounds and data loaders of every stage can be imported side by side. Import this module first.
"""
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
TNBC = os.path.abspath(os.path.join(HERE, '..'))
DATA_DIR = os.path.join(TNBC, 'data', 'processed')
STAGES = ('01_control_growth', '02_control_sharing_scenarios', '03_radiotherapy',
          '04_radiotherapy_sharing_scenarios', '05_immunotherapy', '06_combined_treatment')
for _stage in reversed(STAGES):
    _path = os.path.join(TNBC, _stage)
    if _path not in sys.path:
        sys.path.insert(0, _path)
