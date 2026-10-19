"""Tables of the BIC verification: stored and refined BIC of every cell, the best model of every row and the
BIC weights of the group calibrations, before and after refinement.

    python summarize_bic.py            # reads results/cells.csv; writes results/summary.md and
                                       # results/table_<label>.csv

For the tables of mouse-specific and group calibrations (S1.2, S2.2, S2.3, S3.2, SUV) the models are the
columns and the rows are mice or 'Global'; for the sharing scenarios (4, 5) the scenarios are the rows of a
single column, compared with each other.
"""
import os

import _stage_paths  # noqa: F401
import numpy as np
import pandas as pd

from entries import all_cells

RESULTS = os.path.join(_stage_paths.HERE, 'results')
SCENARIO_TABLES = ('4', '5')
TABLE_ORDER = ('S1.2', '4', 'S2.2', 'S2.3', '5', 'S3.2', 'SUV')


def bic_weights(values):
    v = np.asarray(values, dtype=float)
    d = v - np.nanmin(v)
    w = np.where(np.isfinite(d), np.exp(-d / 2), 0.0)
    return w / w.sum()


def _best(series):
    s = series.dropna()
    return s.idxmin() if len(s) else None


def block_summary(df, table, block):
    """(wide table, text lines) of one table block."""
    sub = df[(df['table'] == table) & (df['block'] == block)]
    lines = []
    if table in SCENARIO_TABLES:
        wide = sub.set_index('row')[['column', 'stored_ll', 'refined_ll', 'bic_stored', 'bic_refined']].copy()
        wide['w_stored'] = bic_weights(wide['bic_stored'])
        wide['w_refined'] = bic_weights(wide['bic_refined'])
        before, after = _best(wide['bic_stored']), _best(wide['bic_refined'])
        lines.append(f'- lowest BIC: {wide.loc[before, "column"]} before, {wide.loc[after, "column"]} after'
                     + ('' if before == after else '  **(changed)**'))
        return wide, lines

    stored = sub.pivot(index='row', columns='column', values='bic_stored')
    refined = sub.pivot(index='row', columns='column', values='bic_refined')
    changed_rows = []
    for row in refined.index:
        b, a = _best(stored.loc[row]), _best(refined.loc[row])
        if b != a:
            changed_rows.append(f'{row}: {b} -> {a}')
    if 'Global' in refined.index:
        ws, wr = bic_weights(stored.loc['Global']), bic_weights(refined.loc['Global'])
        b, a = _best(stored.loc['Global']), _best(refined.loc['Global'])
        lines.append(f'- global lowest BIC: {b} (weight {ws[list(stored.columns).index(b)]:.3f}) before, '
                     f'{a} (weight {wr[list(refined.columns).index(a)]:.3f}) after'
                     + ('' if a == b else '  **(changed)**'))
    per_mouse = [r for r in changed_rows if not r.startswith('Global')]
    n_mice = len([r for r in refined.index if r != 'Global'])
    if n_mice:
        lines.append(f'- mice whose lowest-BIC model changes: {len(per_mouse)} of {n_mice}'
                     + (f' ({"; ".join(per_mouse)})' if per_mouse else ''))
    wide = pd.concat({'stored': stored, 'refined': refined}, axis=1)
    return wide, lines


def main():
    path = os.path.join(RESULTS, 'cells.csv')
    df = pd.read_csv(path)
    order = {c['id']: i for i, c in enumerate(all_cells())}
    df = df.assign(order=df['id'].map(order)).sort_values('order')
    missing = len(order) - len(df)

    out = ['# Verification of the maximum log-likelihoods of the BIC tables', '',
           f'{len(df)} of {len(order)} cells verified' + (f' ({missing} pending).' if missing else '.'), '',
           'Reproduction: |reproduced - stored| log-likelihood above 1e-3 in '
           f'{int((df["reproduced_ll"] - df["stored_ll"]).abs().gt(1e-3).sum())} cells; '
           f'stored sample outside the prior box in {int((~df["stored_inside_box"].astype(bool)).sum())} cells.', '',
           'Refinement: largest increase of the log-likelihood over the reproduced value '
           f'{df["delta_ll"].max():.3f}; cells improved by more than 1 unit: '
           f'{int(df["delta_ll"].gt(1).sum())}.', '']
    for table in TABLE_ORDER:
        tdf = df[df['table'] == table]
        if tdf.empty:
            continue
        out.append(f'## Table {table}')
        frames = []
        for block in tdf['block'].unique():
            wide, lines = block_summary(df, table, block)
            out.append(f'### {block}')
            out.extend(lines)
            big = tdf[(tdf['block'] == block) & tdf['delta_ll'].gt(1)]
            for _, r in big.iterrows():
                out.append(f'  - {r["row"]} {r["column"]}: log L {r["reproduced_ll"]:.2f} -> {r["refined_ll"]:.2f}')
            frames.append(wide.assign(block=block) if table in SCENARIO_TABLES else
                          pd.concat({block: wide}, names=['block']))
            out.append('')
        pd.concat(frames).to_csv(os.path.join(RESULTS, f'table_{table}.csv'), float_format='%.4f')
    with open(os.path.join(RESULTS, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
