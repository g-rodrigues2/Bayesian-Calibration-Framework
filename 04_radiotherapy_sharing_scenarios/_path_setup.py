"""Adds to sys.path:
  - 03_radiotherapy/                (radiation_models, data_loader_rad, bounds_rad)
  - 01_control_growth/              (initialization, walkers, model_selection, growth_params)
  - 02_control_sharing_scenarios/   (scenarios: partition of theta)

Import this module before the other modules of this stage.
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_BASE = os.path.abspath(os.path.join(_HERE, '..', '03_radiotherapy'))
_CONTROL = os.path.abspath(os.path.join(_HERE, '..', '01_control_growth'))
_SHARED = os.path.abspath(os.path.join(_HERE, '..', '02_control_sharing_scenarios'))

for _p in (_BASE, _CONTROL, _SHARED):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE_DIR = _BASE
CONTROL_DIR = _CONTROL
SHARED_DIR = _SHARED
DATA_DIR = os.path.abspath(os.path.join(_HERE, '..', 'data', 'processed'))
