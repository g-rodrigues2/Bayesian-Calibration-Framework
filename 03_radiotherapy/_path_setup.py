"""Adds the control-growth stages to sys.path so that initialization, walkers, model_selection,
postprocess, metrics and growth_params are reused.

Import this module before any `from initialization import ...`.
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_CONTROL = os.path.abspath(os.path.join(_HERE, '..', '01_control_growth'))
_SHARED = os.path.abspath(os.path.join(_HERE, '..', '02_control_sharing_scenarios'))

for _p in (_CONTROL, _SHARED):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONTROL_DIR = _CONTROL
SHARED_DIR = _SHARED
