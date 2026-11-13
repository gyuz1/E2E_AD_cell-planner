"""Constraint sensitivity and STOP-rule effect for the DP partition (LANE_KEEP).

LANE_KEEP holds 80% of moving training frames; it is the command whose cell
count the paper varies (15 vs 60). Same D / L2 definitions as partition_analysis.py.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/../tools')
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fit_cell_layout as fcl  # noqa: E402
from partition_analysis import evaluate  # noqa: E402

c = np.load(sys.argv[1])
S = {s: (c[f'{s}_tp'], c[f'{s}_cmd'], c[f'{s}_traj'], c[f'{s}_scene']) for s in ('train', 'val')}
rows = []


def run(K, mf, ms, w, rule='forward', step=0.5, k=0, tag=''):
    tr = fcl.moving_frames(*S['train'], k, rule, 1.0)
    va = fcl.moving_frames(*S['val'], k, rule, 1.0)
    F, L = fcl.fit(tr[0], tr[1], tr[2], K, 1, fcl.BOUNDS[k], step, mf, ms, w)
    r = dict(K=K, min_frames=mf, min_scenes=ms, min_width=w, stop_rule=rule, grid=step, tag=tag,
             n_train=len(tr[0]), n_val=len(va[0]))
    if F is None:
        r['result'] = 'INFEASIBLE'
        print(f'{tag:10s} K={K:3d} frames>={mf:4d} scenes>={ms:2d} width>={w:3.1f} grid={step} {rule:7s}: INFEASIBLE')
    else:
        e = evaluate(F, [-20, 17] if k == 0 else fcl.BOUNDS[k][2:], tr, va)
        r.update(e, edges=F)
        print(f'{tag:10s} K={K:3d} frames>={mf:4d} scenes>={ms:2d} width>={w:3.1f} grid={step} {rule:7s}: '
              f'D {e["D_train"]:6.2f}/{e["D_val"]:6.2f}  L2 {e["L2_train"]:.3f}/{e["L2_val"]:.3f}  '
              f'min n {e["min_frames"]:4d} sc {e["min_scenes"]:2d}  narrowest {min(np.diff(F)):.1f} m')
    rows.append(r)


for K in (15, 60):
    for mf in (1, 20, 50, 100, 200, 400):
        run(K, mf, 3, 1.0, tag='frames')
    for ms in (1, 3, 5, 10):
        run(K, 50, ms, 1.0, tag='scenes')
    for w in (0.5, 1.0, 2.0, 4.0):
        run(K, 50, 3, w, step=0.5, tag='width')
    run(K, 50, 3, 1.0, step=1.0, tag='grid')
    run(K, 50, 3, 1.0, rule='norm', tag='stoprule')

# how many frames the two STOP rules disagree on, per command
tp, cmd = S['train'][0], S['train'][1]
for split in ('train', 'val'):
    tp, cmd = S[split][0], S[split][1]
    fwd = tp[:, 0] < 1.0
    nrm = np.linalg.norm(tp, axis=-1) < 1.0
    for k, n in enumerate(['LK', 'LCL', 'LCR', 'TL', 'TR']):
        m = cmd == k
        print(f'stop-rule disagreement {split} {n}: {int((fwd[m] != nrm[m]).sum())} of {int(m.sum())} frames')

json.dump(rows, open(sys.argv[2], 'w'), indent=1)
print('wrote', sys.argv[2])
