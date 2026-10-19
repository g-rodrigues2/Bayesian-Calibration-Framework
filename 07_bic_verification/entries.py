"""Cells of the BIC tables of Chapter 4 and the files holding their original maximum-likelihood samples.

Every cell is one (table, block, row, column) with
  table   thesis table label: S1.2, 4, S2.2, S2.3, 5, S3.2, SUV
  block   calibration group(s) of the table section
  row     mouse name for mouse-specific calibrations, 'Global' for the group calibration, or a scenario
  column  model name as printed in the thesis
  model   model identifier of the stage code
  kind    likelihood family (see likelihoods.py)
  source  (base, relative path) of the original estimate: base 'legacy' is the folder of the original
          calibrations (not public), base 'tnbc' this repository; None when no estimate exists

`import_estimates()` copies the original estimates to estimates/<cell id>.npz (pars, max_ll, source), so
that the verification runs from this repository alone.
"""
import os

import _stage_paths  # noqa: F401
import numpy as np

from data_loader import load_control_data
from data_loader_imm import load_immuno_data
from data_loader_rad import load_radiation_data

HERE = _stage_paths.HERE
ESTIMATES = os.path.join(HERE, 'estimates')

CONTROL_MODELS = [('von Bertalanffy', '_bert'), ('Exponential', '_exp'), ('Gompertz', '_gomp'),
                  ('Linear', '_lin'), ('Logistic', '_log'), ('Mendelsohn', '_mendel'), ('Surface', '_surf')]
# (thesis column, model, name in the original calibrations)
RT_MODELS = [('ACD', '_AccelCD', 'AccelCD'), ('AED', '_AccelED', 'AccelDRT'),
             ('CDEF', '_CDwithExpFactor', 'CDwithExpFactor'), ('CDLF', '_CDwithLogFactor', 'CDwithLogFactor'),
             ('CDSF', '_CDwithSinFactor', 'CDwithSinFactor'), ('CD', '_CumulativeDose', 'CumulativeDose'),
             ('EDEF', '_EDwithExpFactor', 'DRTwithExpFactor'), ('EDLF', '_EDwithLogFactor', 'DRTwithLogFactor'),
             ('EDSF', '_EDwithSinFactor', 'DRTwithSinFactor'), ('ED', '_ExponentialDecay', 'DamageRepairTime'),
             ('LQ', '_LQ', 'LQ')]
RT_DELAY_MODELS = [('ACD w/ delay', '_ACDDelay', 'ACDDelay'), ('AED w/ delay', '_AEDDelay', 'ADRTDelay'),
                   ('CDEF w/ delay', '_CDEFDelay', 'CDEFDelay'), ('CDSF w/ delay', '_CDSFDelay', 'CDSFDelay'),
                   ('CD w/ delay', '_CumulativeDoseDelay', 'CumulativeDoseDelay'),
                   ('EDEF w/ delay', '_EDEFDelay', 'DRTEFDelay'), ('EDSF w/ delay', '_EDSFDelay', 'DRTSFDelay'),
                   ('ED w/ delay', '_ExponentialDecayDelay', 'DamageRepairTimeDelay')]
IO_MODELS = [('ACD', '_AccelCD'), ('AED', '_AccelED'), ('CDEF', '_CDwithExpFactor'), ('CDLF', '_CDwithLogFactor'),
             ('CDSF', '_CDwithSinFactor'), ('CD', '_CumulativeDose'), ('EDEF', '_EDwithExpFactor'),
             ('EDLF', '_EDwithLogFactor'), ('EDSF', '_EDwithSinFactor'), ('ED', '_ExponentialDecay'),
             ('MM', '_MichaelisMenten'), ('PR', '_dePillisRadunskayaLaw')]
SUV_MODELS = [('Linear', '_SUVLinear', 'RadImmuno'), ('Michaelis-Menten', '_SUVMichaelisMenten', 'RadImmuno2'),
              ('Logistic', '_SUVLogistic', None), ('Saturating exponential', '_SUVSatExp', 'RadImmuno4')]

_CONTROL_MS = 'Control_Calibration/mouse_specific_v1/Output_Calibration_per_mouse'
_CONTROL_GL = 'Control_Calibration/global/Output_Calibration'
_RT_MS = 'Radiation_Calibration/radiation_mouse_specific/mouse_specific_new/Output_Calibration'
_RT_DELAY_MS = 'Radiation_Calibration/radiation_mouse_specific/mouse_specific_delay_new/Output_Calibration'
_RT_GL = 'Radiation_Calibration/radiation_global/Resultados Organizados/global_{phenotype}/Output_Calibration'
_RT_DELAY_GL = ('Radiation_Calibration/radiation_global/Resultados Organizados/global_{phenotype}_delay/'
                'Output_Calibration_Delay')
_IO_MS = 'Imuno_Calibration/imuno_each_mouse_nova_versao/Output_Calibration'
_SUV_GL = ('Immuno+rad_calibration/global/immuno_dePillisRadunskaya_effect/global_SUV_io_rad+io_SUVfunction/'
           'Output_Calibration')


def cell_id(cell):
    return '__'.join([cell['table'], cell['block'], cell['row'], cell['model']]).replace('+', 'p')


def _cell(table, block, row, column, model, kind, source, **extra):
    c = dict(table=table, block=block, row=row, column=column, model=model, kind=kind, source=source, **extra)
    c['id'] = cell_id(c)
    return c


def all_cells():
    control = load_control_data()
    radiation = load_radiation_data()
    immuno = load_immuno_data()
    cells = []

    for group in ('control_sensitive', 'control_resistant'):
        for column, model in CONTROL_MODELS:
            for mouse in control[group]:
                cells.append(_cell('S1.2', group, mouse['name'], column, model, 'control',
                                   ('legacy', f'{_CONTROL_MS}/ll_pars_{group}_{mouse["name"]}{model}.npz')))
            cells.append(_cell('S1.2', group, 'Global', column, model, 'control',
                               ('legacy', f'{_CONTROL_GL}/multi_ll_pars_{group}{model}.npz')))

    for n in (1, 2, 3, 4):
        cells.append(_cell('4', 'control', f's{n}', f'Scenario {n}', '_log', 'control_sharing',
                           ('tnbc', f'02_control_sharing_scenarios/Output_Calibration/multi_ll_pars__log_s{n}.npz'),
                           scenario=n))

    for table, models, ms_dir, gl_dir in (('S2.2', RT_MODELS, _RT_MS, _RT_GL),
                                          ('S2.3', RT_DELAY_MODELS, _RT_DELAY_MS, _RT_DELAY_GL)):
        for group in ('radiation_sensitive', 'radiation_resistant'):
            phenotype = group.split('_')[1]
            for column, model, legacy in models:
                for mouse in radiation[group]:
                    cells.append(_cell(table, group, mouse['name'], column, model, 'radiation',
                                       ('legacy', f'{ms_dir}/ll_pars_{group}_{mouse["name"]}_{legacy}.npz')))
                cells.append(_cell(table, group, 'Global', column, model, 'radiation',
                                   ('legacy', f'{gl_dir.format(phenotype=phenotype)}/multi_ll_pars_{group}_{legacy}.npz')))

    for n in range(1, 9):
        folder = 'Output_Calibration' if n == 2 else 'Output_Calibration/legacy'
        cells.append(_cell('5', 'radiation', f'V{n}', f'Scenario {n}', '_ExponentialDecayDelay', 'radiation_sharing',
                           ('tnbc', f'04_radiotherapy_sharing_scenarios/{folder}/'
                                    f'multi_ll_pars__ExponentialDecayDelay_V{n}.npz'), scenario=n))

    group = 'io_sensitive'
    for column, model in IO_MODELS:
        for mouse in immuno[group]:
            rerun = f'05_immunotherapy/Output_Calibration_per_mouse/ll_pars_{group}_{mouse["name"]}{model}.npz'
            source = (('tnbc', rerun) if os.path.exists(os.path.join(_stage_paths.TNBC, rerun))
                      else ('legacy', f'{_IO_MS}/ll_pars_{group}_{mouse["name"]}{model}.npz'))
            cells.append(_cell('S3.2', group, mouse['name'], column, model, 'immuno', source))
        cells.append(_cell('S3.2', group, 'Global', column, model, 'immuno',
                           ('tnbc', f'05_immunotherapy/Output_Calibration/multi_ll_pars_{group}{model}.npz')))

    for column, model, legacy in SUV_MODELS:
        source = None if legacy is None else ('legacy', f'{_SUV_GL}/multi_ll_pars_rad+io_sensitive_{legacy}.npz')
        cells.append(_cell('SUV', 'rad+io_sensitive', 'Global', column, model, 'suv', source))
    return cells


def estimate_path(cell):
    return os.path.join(ESTIMATES, cell['id'] + '.npz')


def import_estimates(cells, legacy_root):
    """Copies the original estimates into estimates/. Returns the ids of the cells without an estimate."""
    os.makedirs(ESTIMATES, exist_ok=True)
    missing = []
    for cell in cells:
        if cell['source'] is None:
            continue
        base, rel = cell['source']
        root = legacy_root if base == 'legacy' else _stage_paths.TNBC
        if root is None:
            missing.append(cell['id'])
            continue
        path = os.path.join(root, *rel.split('/'))
        if not os.path.exists(path):
            missing.append(cell['id'])
            continue
        z = np.load(path, allow_pickle=True)
        np.savez(estimate_path(cell), pars=np.asarray(z['pars'], dtype=float), max_ll=float(z['max_ll']),
                 source=f'{base}:{rel}')
    return missing


def load_estimate(cell):
    path = estimate_path(cell)
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return np.asarray(z['pars'], dtype=float), float(z['max_ll'])
