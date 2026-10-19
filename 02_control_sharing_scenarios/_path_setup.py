"""Adds ``01_control_growth/`` to sys.path so that models, bounds, inference, data_loader,
postprocess and convergence_check are reused.

Import this module before any ``from models import ...`` in this stage.
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_CONTROL = os.path.abspath(os.path.join(_HERE, '..', '01_control_growth'))

if _CONTROL not in sys.path:
    sys.path.insert(0, _CONTROL)

# Processed tumor-volume data
DATA_DIR = os.path.abspath(os.path.join(_HERE, '..', 'data', 'processed'))
