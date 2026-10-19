"""Effect of the uncertainty of the previous calibration stages on the prediction of the combined cohort.

The prediction of Section 4.5 keeps the growth, radiotherapy and immunotherapy parameters fixed at the
maximum-likelihood samples of the stages that estimated them, and propagates only the posterior of the
mouse-specific initial volumes. This script quantifies what that choice leaves out: it draws the
upstream parameters from their own posteriors (growth from the sharing scenario 2 of stage 2,
radiotherapy from the scenario 2 of stage 4, immunotherapy from the de Pillis-Radunskaya chain of
stage 5, independently, as the sequential scheme implies), solves the combined model for every mouse of
the radiotherapy + immunotherapy cohort, and compares the resulting spread with the one obtained with
the upstream parameters fixed.

    python upstream_uncertainty.py [--draws 200] [--seed 42]

Writes Output_Calibration/upstream_uncertainty.csv (per mouse and time, the two bands) and prints the
summary reported in the thesis: median width of the 95% band at the observation times, and the
posterior of the cohort concordance under each scheme.
"""
import argparse
import os

import numpy as np

import _path_setup  # noqa: F401
from metrics import ccc, pcc

from data_loader_ri import load_data
from rad_immuno_models import solve_mouse
from thesis_params import upstream_params

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = 'rad+io_sensitive'
OUT = os.path.join(HERE, 'Output_Calibration')
GROWTH = os.path.join(HERE, '..', '02_control_sharing_scenarios', 'Output_Calibration')
RADIO = os.path.join(HERE, '..', '04_radiotherapy_sharing_scenarios', 'Output_Calibration')
IMMUNO = os.path.join(HERE, '..', '05_immunotherapy', 'Output_Calibration')
PREDICTION = os.path.join(OUT, 'prediction')


def columns(chain_path, npz_path, names):
    """Columns `names` of a chain, in the order of the labels of its .npz."""
    labels = [str(x) for x in np.load(npz_path, allow_pickle=True)['labels']]
    chain = np.loadtxt(chain_path, ndmin=2)
    return chain[:, [labels.index(n) for n in names]]


def upstream_draws(n, rng):
    """`n` draws of (r, cc, a_rad, b_rad, delay, d, s, lam) from the posteriors of stages 2, 4 and 5."""
    growth = columns(os.path.join(GROWTH, 'multi_chain__log_s2.gz'),
                     os.path.join(GROWTH, 'multi_ll_pars__log_s2.npz'), ['r_sensitive', 'cc_global'])
    radio = columns(os.path.join(RADIO, 'multi_chain__ExponentialDecayDelay_V2.gz'),
                    os.path.join(RADIO, 'multi_ll_pars__ExponentialDecayDelay_V2.npz'),
                    ['alpha_rad_global', 'beta_rad_sensitive', 'delay_global'])
    immuno = columns(os.path.join(IMMUNO, 'multi_chain_io_sensitive_dePillisRadunskayaLaw.gz'),
                     os.path.join(IMMUNO, 'multi_ll_pars_io_sensitive_dePillisRadunskayaLaw.npz'),
                     ['d', 's', 'lam'])
    pick = [table[rng.choice(len(table), size=n, replace=False)] for table in (growth, radio, immuno)]
    return np.hstack(pick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mice = load_data()[GROUP]
    fixed = upstream_params('sensitive')

    # initial volumes: posterior of the prediction chain of Section 4.5
    labels = [str(x) for x in np.load(os.path.join(PREDICTION, f'estimate_{GROUP}_RadImmunoFixed.npz'),
                                      allow_pickle=True)['labels']]
    chain = np.loadtxt(os.path.join(PREDICTION, f'chain_{GROUP}_RadImmunoFixed.gz'), ndmin=2)
    ic_cols = [labels.index(f'ic{i + 1}') for i in range(len(mice))]
    ic_draws = chain[rng.choice(len(chain), size=args.draws, replace=False)][:, ic_cols]

    theta = upstream_draws(args.draws, rng)
    names = ('r', 'cc', 'a_rad', 'b_rad', 'delay', 'd', 's', 'lam')
    observed = [m['data'][:, 1] for m in mice]
    obs = np.concatenate(observed)

    rows, ccc_fixed, ccc_full = [], [], []
    width_fixed = {m['name']: [] for m in mice}
    width_full = {m['name']: [] for m in mice}
    traj_fixed = {m['name']: [] for m in mice}
    traj_full = {m['name']: [] for m in mice}

    for k in range(args.draws):
        up_draw = dict(zip(names, theta[k]))
        pred_fixed, pred_full = [], []
        for j, mouse in enumerate(mice):
            ic = ic_draws[k, j]
            pred_fixed.append(solve_mouse('_RadImmunoFixed', (), ic, mouse, fixed))
            pred_full.append(solve_mouse('_RadImmunoFixed', (), ic, mouse, up_draw))
            traj_fixed[mouse['name']].append(pred_fixed[-1])
            traj_full[mouse['name']].append(pred_full[-1])
        ccc_fixed.append(ccc(np.concatenate(pred_fixed), obs))
        ccc_full.append(ccc(np.concatenate(pred_full), obs))

    for mouse in mice:
        for tag, store, width in (('fixed', traj_fixed, width_fixed), ('upstream', traj_full, width_full)):
            band = np.percentile(np.vstack(store[mouse['name']]), [2.5, 97.5], axis=0)
            width[mouse['name']] = band[1] - band[0]
            for t, lo, hi in zip(mouse['data'][:, 0], band[0], band[1]):
                rows.append((mouse['name'], tag, t, lo, hi))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'upstream_uncertainty.csv')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('mouse,scheme,day,lower,upper\n')
        for r in rows:
            fh.write('%s,%s,%.4f,%.6f,%.6f\n' % r)

    wf = np.concatenate([width_fixed[m['name']] for m in mice])
    wu = np.concatenate([width_full[m['name']] for m in mice])
    print(f'{args.draws} draws, {len(mice)} mice')
    print('median width of the 95%% band (mm3): initial volumes only %.1f, with upstream parameters %.1f '
          '(ratio %.1f)' % (np.median(wf), np.median(wu), np.median(wu) / max(np.median(wf), 1e-9)))
    for tag, values in (('initial volumes only', ccc_fixed), ('with upstream parameters', ccc_full)):
        q = np.percentile(values, [50, 2.5, 97.5])
        print('cohort CCC, %s: %.3f [%.3f, %.3f]' % (tag, q[0], q[1], q[2]))
    print('written:', path)


if __name__ == '__main__':
    main()
