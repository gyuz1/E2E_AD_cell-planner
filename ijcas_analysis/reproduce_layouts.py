"""Find the fitting settings under which each committed layout is reproduced exactly.

The LANE_KEEP 60x1 layout is known to come back with grid 0.5 m, >=50 frames,
>=3 scenes, >=1 m width, forward stop rule. The other four commands were fitted
earlier with wider minimum widths (4-5 m, SHARED_CONTEXT 2026-09) and integer
edges; this searches grid step x min width x min frames x min scenes for each.
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/../tools')
import fit_cell_layout as fcl  # noqa: E402

COMMITTED = {  # VADLAW_etri_tiny_clean_cellplanner(.py / _lk60.py)
    'LK15': (0, [1, 12, 22, 31, 38, 45, 51, 56, 61, 66, 72, 82, 94, 101, 106, 117], [-20, 17]),
    'LK60': (0, [1.0, 3.0, 6.5, 9.0, 12.0, 15.0, 17.0, 19.0, 21.0, 23.5, 25.5, 27.5, 29.5, 31.0,
                 32.5, 34.5, 36.5, 38.0, 39.5, 41.5, 43.5, 45.0, 46.0, 47.5, 49.0, 50.5, 52.0,
                 53.5, 55.0, 56.5, 58.0, 59.0, 60.0, 61.5, 62.5, 63.5, 64.5, 65.5, 67.0, 68.5,
                 70.0, 72.0, 74.5, 77.0, 79.5, 82.5, 87.0, 89.5, 92.0, 94.5, 96.5, 98.0, 99.5,
                 101.5, 103.0, 104.5, 106.0, 107.5, 109.0, 111.0, 117.0], [-20, 17]),
    'LCL11': (1, [1, 20, 25, 36, 44, 51, 58, 66, 78, 94, 103, 116], [-16, 12]),
    'LCR14': (2, [1, 23, 30, 35, 41, 45, 49, 53, 59, 64, 74, 81, 89, 100, 115], [-12, 18]),
    'TL4x2': (3, [1, 16, 21, 30, 48], [-2, 10, 27]),
    'TR6x2': (4, [1, 12, 16, 20, 26, 33, 43], [-26, -11.5, 2]),
}

c = np.load(sys.argv[1])
tp, cmd, traj, scene = c['train_tp'], c['train_cmd'], c['train_traj'], c['train_scene']
for name, (k, F0, L0) in COMMITTED.items():
    mtp, mtraj, msc = fcl.moving_frames(tp, cmd, traj, scene, k, 'forward', 1.0)
    nf, nl = len(F0) - 1, len(L0) - 1
    hits = []
    for step, w, mf, ms in itertools.product((0.5, 1.0), (1, 2, 3, 4, 5, 6),
                                             (50,), (3,)):
        F, L = fcl.fit(mtp, mtraj, msc, nf, nl, fcl.BOUNDS[k], step, mf, ms, w)
        if F is not None and np.allclose(F, F0) and np.allclose(L, L0):
            hits.append(f'grid={step} width>={w} frames>={mf} scenes>={ms}')
    print(f'{name:6s} {len(mtp):6d} frames  reproduced by: {hits or "NONE of the tried settings"}')
