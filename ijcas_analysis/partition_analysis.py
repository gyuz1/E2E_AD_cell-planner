"""Partition analysis for the paper (Table 4, Fig. 2). CPU only, annotations only.

For every moving command it compares three partitions of the target-point space
at equal cell count K:
  dp        constrained DP of tools/fit_cell_layout.py (the proposed one)
  uniform   equal-width cells over the same outer extent
  quantile  equal-occupancy cells (training quantiles; also the DP's start point)
2-column commands (TURN_LEFT/RIGHT) keep 2 lateral columns and vary forward rows.

Quantity reported for a partition P and a frame set S:
  D(P,S)   = mean over frames of ||tau - mean_cell(tau)||^2 over the 12-dim
             cumulative 3-s trajectory (m^2) -- the DP's objective per frame
  L2(P,S)  = the planning-L2 (mean of 1/2/3-s cumulative averages) of the
             cell-mean predictor
On 'val' the cell means are the TRAINING means (generalization), on 'train' they
are the in-sample means. The cell-mean predictor is a target-point-only planner
with no perception; it is an analysis device, not a compared planner.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/../tools')
import fit_cell_layout as fcl  # noqa: E402

NAMES = ['LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT']
COLS = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2}
GRID = 0.5
CONS = dict(min_frames=50, min_scenes=3, min_width=1.0)
KS = {0: [1, 2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 150],
      1: [1, 2, 4, 6, 8, 11, 14, 18, 24, 30],
      2: [1, 2, 4, 6, 8, 11, 14, 18, 24, 30],
      3: [1, 2, 3, 4, 6, 8, 10, 12],
      4: [1, 2, 3, 4, 6, 8, 10, 12]}


def evaluate(F, L, tr, va):
    nl = len(L) - 1
    K = (len(F) - 1) * nl
    out = {}
    fr, lc = fcl.assign(tr[0], F, L)
    g = fr * nl + lc
    cnt = np.bincount(g, minlength=K)
    means = np.zeros((K, tr[1].shape[1]))
    for k in range(K):
        if cnt[k]:
            means[k] = tr[1][g == k].mean(0)
    sc = [len(np.unique(tr[2][g == k])) for k in range(K)]
    out['min_frames'] = int(cnt.min())
    out['min_scenes'] = int(min(sc))
    out['empty_cells'] = int((cnt == 0).sum())
    for tag, (tp, traj, _) in (('train', tr), ('val', va)):
        f2, l2 = fcl.assign(tp, F, L)
        pred = means[f2 * nl + l2]
        out[f'D_{tag}'] = float(((traj - pred) ** 2).sum(1).mean())
        out[f'L2_{tag}'] = fcl.official_l2(pred, traj)
    return out


def edges(kind, tp, n, lo, hi):
    if kind == 'uniform':
        return list(np.linspace(lo, hi, n + 1))
    q = np.quantile(tp, np.arange(1, n) / n)
    return [lo, *q, hi]


def main(cache, out_json, stop_rule='forward'):
    c = np.load(cache)
    split = {s: (c[f'{s}_tp'], c[f'{s}_cmd'], c[f'{s}_traj'], c[f'{s}_scene']) for s in ('train', 'val')}
    res = {'grid': GRID, 'constraints': CONS, 'stop_rule': stop_rule, 'rows': []}
    for k, name in enumerate(NAMES):
        tr = fcl.moving_frames(*split['train'], k, stop_rule, 1.0)
        va = fcl.moving_frames(*split['val'], k, stop_rule, 1.0)
        x0, x1, y0, y1 = fcl.BOUNDS[k]
        nl = COLS[k]
        print(f'\n== {name}: train {len(tr[0])} frames ({len(np.unique(tr[2]))} scenes), '
              f'val {len(va[0])} frames ({len(np.unique(va[2]))} scenes), lateral columns {nl}')
        for K in KS[k]:
            nf = K // nl if nl > 1 else K
            if nl > 1 and K % nl:
                continue
            if nl > 1 and nf < 1:
                continue
            t0 = time.perf_counter()
            if K == nl and nl == 1:
                F, L = [x0, x1], [y0, y1]
            else:
                F, L = fcl.fit(tr[0], tr[1], tr[2], nf, nl, fcl.BOUNDS[k], GRID, **CONS)
            dt = time.perf_counter() - t0
            row = {'command': name, 'K': nf * nl}
            if F is None:
                row['dp'] = 'INFEASIBLE'
            else:
                row['dp'] = evaluate(F, L, tr, va) | {'fit_seconds': dt}
            for kind in ('uniform', 'quantile'):
                Fb = edges(kind, tr[0][:, 0], nf, x0, x1)
                Lb = edges(kind, tr[0][:, 1], nl, y0, y1) if nl > 1 else [y0, y1]
                row[kind] = evaluate(Fb, Lb, tr, va)
            res['rows'].append(row)
            fmt = lambda r: ('INFEASIBLE' if r == 'INFEASIBLE' else
                             f"D {r['D_train']:6.2f}/{r['D_val']:6.2f}  L2 {r['L2_train']:.3f}/{r['L2_val']:.3f}  "
                             f"min n {r['min_frames']:4d} sc {r['min_scenes']:2d} empty {r['empty_cells']}")
            print(f"K={row['K']:4d}  dp[{fmt(row['dp'])}]" + (f" {dt:.2f}s" if F is not None else ''))
            print(f"        uni[{fmt(row['uniform'])}]")
            print(f"        qnt[{fmt(row['quantile'])}]")
    with open(out_json, 'w') as fh:
        json.dump(res, fh, indent=1)
    print('\nwrote', out_json)


if __name__ == '__main__':
    main(*sys.argv[1:])
