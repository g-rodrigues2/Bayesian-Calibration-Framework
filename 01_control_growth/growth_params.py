"""Growth parameters (r, cc) inherited from the control groups and fixed in the treatment stages.

Treatment-agnostic: each treatment group '{treatment}_{phenotype}' (radiation_sensitive, io_sensitive,
...) inherits from 'control_{phenotype}'. The inheritance is automatic, from the outputs of the previous
stages:
  1) the growth model with the lowest BIC in the control groups (01_control_growth), and
  2) the sharing scenario with the lowest BIC (02_control_sharing_scenarios),
whose r and cc are loaded. The treatment ODEs use logistic growth, r V (1 - V/cc), so the selected
control model must be `_log`; otherwise a warning is printed.

Source (CALIB_GROWTH_SOURCE):
  'auto' (default) -> selected model and scenario; falls back to 'control_log', then 'fixed'.
  'shared:<sN>'    -> a specific sharing scenario.
  'control_log'    -> separate logistic fits of the control groups (cc per group).
  'fixed'          -> the values reported in the thesis (logistic model, scenario s2).

Overrides: CALIB_GROWTH_MODEL (e.g. _log), CALIB_GROWTH_SCENARIO (e.g. s2), CALIB_SELECT_METRIC (bic).
"""
import glob
import os

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
CONTROL_DIR = _HERE
SHARED_DIR = os.path.abspath(os.path.join(_HERE, '..', '02_control_sharing_scenarios'))
DATA_DIR = os.path.abspath(os.path.join(_HERE, '..', 'data', 'processed'))

# Treatment-agnostic: any group '{treatment}_{phenotype}' inherits from 'control_{phenotype}'.
DEFAULT_GROUPS = ('radiation_sensitive', 'radiation_resistant')

# Values of the thesis (logistic model, scenario s2), indexed by phenotype.
_FIXED_BY_PHENOTYPE = {
    'sensitive': dict(r=8.89429860e-02, cc=3.69671552e+03),
    'resistant': dict(r=1.15631515e-01, cc=3.69671552e+03),
}
_LOGISTIC = '_log'


def phenotype_of(group):
    """Phenotype of a treatment group: 'radiation_sensitive' -> 'sensitive'."""
    return group.rsplit('_', 1)[-1]


def _fixed_for(groups):
    return {g: dict(_FIXED_BY_PHENOTYPE[phenotype_of(g)]) for g in groups}


def _bic(max_ll, k, n):
    return k * np.log(n) - 2.0 * max_ll if n > 0 else float('inf')


def _npz(path):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def _control_obs():
    """Number of observations per control group (for the BIC used in the discovery)."""
    try:
        from data_loader import load_control_data
        fd = load_control_data(DATA_DIR)
        return {g: sum(m['data'].shape[0] for m in mice) for g, mice in fd.items()}
    except Exception:
        return {}


# === Selected growth model of the control groups ===

def discover_control_model(metric='bic'):
    """Growth model with the lowest BIC summed over both control groups.

    Reads multi_ll_pars_control_{group}_{model}.npz. None if no estimates are found.
    """
    out_dir = os.path.join(CONTROL_DIR, 'Output_Calibration')
    n_obs = _control_obs()
    totals = {}
    for ctag in ('sensitive', 'resistant'):
        group = f'control_{ctag}'
        n = n_obs.get(group, 0)
        for p in glob.glob(os.path.join(out_dir, f'multi_ll_pars_{group}_*.npz')):
            model = '_' + os.path.basename(p).rsplit('_', 1)[1].rsplit('.', 1)[0]
            npz = _npz(p)
            if npz is None:
                continue
            b = _bic(float(npz['max_ll']), int(len(npz['pars'])), n)
            totals[model] = totals.get(model, 0.0) + b
    if not totals:
        return None
    return min(totals, key=totals.get)


# === Selected sharing scenario ===

def _scenario_obs():
    try:
        from data_loader import load_control_data
        fd = load_control_data(DATA_DIR)
        return sum(m['data'].shape[0] for mice in fd.values() for m in mice)
    except Exception:
        return 0


def discover_shared_scenario(model=_LOGISTIC, metric='bic'):
    """Tag of the sharing scenario with the lowest BIC (e.g. 's2').

    Reads multi_ll_pars_{model}_s*.npz of the sharing-scenarios stage. None if absent.
    """
    out_dir = os.path.join(SHARED_DIR, 'Output_Calibration')
    n = _scenario_obs()
    best, best_bic = None, float('inf')
    for p in sorted(glob.glob(os.path.join(out_dir,
                                           f'multi_ll_pars_{model}_s*.npz'))):
        tag = os.path.basename(p).rsplit('_', 1)[1].rsplit('.', 1)[0]  # 'sN'
        npz = _npz(p)
        if npz is None:
            continue
        b = _bic(float(npz['max_ll']), int(len(npz['pars'])), n)
        if b < best_bic:
            best, best_bic = tag, b
    return best


# === r and cc from an estimate file ===

def _extract_r_cc(npz, group_tag):
    """r and cc of a group from an estimate file with global or group-specific labels
    ('r_global', 'r_{tag}' or 'r', and likewise for cc)."""
    labels = [str(x) for x in npz['labels']]
    pars = npz['pars']
    idx = {lab: i for i, lab in enumerate(labels)}

    def pick(base):
        for cand in (f'{base}_global', f'{base}_{group_tag}', base):
            if cand in idx:
                return float(pars[idx[cand]])
        raise KeyError(base)

    return pick('r'), pick('cc')


def _from_shared_scenario(model, tag, groups):
    """r and cc per group from sharing scenario `tag` (e.g. 's2')."""
    p = os.path.join(SHARED_DIR, 'Output_Calibration',
                     f'multi_ll_pars_{model}_{tag}.npz')
    npz = _npz(p)
    if npz is None:
        return None
    out = {}
    for group in groups:
        try:
            r, cc = _extract_r_cc(npz, phenotype_of(group))
        except KeyError:
            return None
        out[group] = dict(r=r, cc=cc)
    return out


def _from_control_log(model, groups):
    """r and cc per group from the separate logistic fits of the control groups."""
    out = {}
    for group in groups:
        ctag = phenotype_of(group)
        p = os.path.join(CONTROL_DIR, 'Output_Calibration',
                         f'multi_ll_pars_control_{ctag}{model}.npz')
        npz = _npz(p)
        if npz is None or 'labels' not in npz:
            return None
        try:
            r, cc = _extract_r_cc(npz, ctag)
        except KeyError:
            return None
        out[group] = dict(r=r, cc=cc)
    return out


def load_growth_params(groups=None, source=None, verbose=True):
    """{group: {'r': ..., 'cc': ...}} for treatment groups, each inheriting from the control group of
    the same phenotype. See the module docstring for the sources."""
    groups = list(groups) if groups is not None else list(DEFAULT_GROUPS)
    source = (source or os.environ.get('CALIB_GROWTH_SOURCE', 'auto')).lower()
    metric = os.environ.get('CALIB_SELECT_METRIC', 'bic').lower()
    model = os.environ.get('CALIB_GROWTH_MODEL', '').strip()
    scenario_npz = os.environ.get('CALIB_GROWTH_SCENARIO', '').strip()

    gp = None
    used = None

    if source.startswith('shared:'):
        model = model or _LOGISTIC
        tag = source.split(':', 1)[1]
        gp = _from_shared_scenario(model, tag, groups)
        used = f'shared:{tag} ({model})'
        if gp is None:
            raise FileNotFoundError(f'Scenario {tag} of {model} not found.')
    elif source == 'control_log':
        gp = _from_control_log(model or _LOGISTIC, groups)
        used = f'control_log ({model or _LOGISTIC})'
        if gp is None:
            raise FileNotFoundError('Separate _log fits of the control groups not found.')
    elif source == 'fixed':
        gp, used = _fixed_for(groups), 'fixed'
    else:  # auto: selected control model and sharing scenario
        model = model or discover_control_model(metric) or _LOGISTIC
        if model != _LOGISTIC and verbose:
            print(f'[growth] WARNING: the selected control model is {model}, but the '
                  f'treatment ODEs assume logistic growth (_log). '
                  f'r and cc are used as logistic parameters; check compatibility.')
        tag = scenario_npz or discover_shared_scenario(model, metric)
        if tag is not None:
            gp = _from_shared_scenario(model, tag, groups)
            used = f'auto -> shared {tag} ({model})'
        if gp is None:
            gp = _from_control_log(model, groups)
            used = f'auto -> control_log ({model})'
        if gp is None:
            gp, used = _fixed_for(groups), 'auto -> fixed (fallback)'

    if verbose:
        det = ', '.join(f"{g.replace('radiation_','')}: r={v['r']:.4g}, cc={v['cc']:.0f}"
                        for g, v in gp.items())
        print(f'[growth] source={used} | {det}')
    return gp
