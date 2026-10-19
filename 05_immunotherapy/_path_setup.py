"""Adds 01_control_growth/ to sys.path so that initialization, walkers, model_selection, postprocess and
growth_params are reused. Import this module before the other modules of this stage.
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_CONTROL = os.path.abspath(os.path.join(_HERE, '..', '01_control_growth'))
if _CONTROL not in sys.path:
    sys.path.insert(0, _CONTROL)

CONTROL_DIR = _CONTROL
