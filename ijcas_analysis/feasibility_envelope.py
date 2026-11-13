"""Data envelope U of eq. (3): 99.9th percentiles over training ground truth.

Only frames whose 6 future steps are all valid (gt_ego_fut_masks) are used.
Reports the bounds, how often hold-out ground truth leaves them (a sanity check:
it should be small but not zero), and the 99th / max for context.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from kinematics import quantities, violations, V_MIN  # noqa: E402

c = np.load(sys.argv[1])
P = 99.9
res = {'percentile': P, 'v_min_for_curvature': V_MIN}
qs = {}
for s in ('train', 'val'):
    ok = c[f'{s}_mask'].min(1) > 0
    qs[s] = quantities(c[f'{s}_traj'][ok])
    res[f'{s}_frames_used'] = int(ok.sum())
    res[f'{s}_frames_total'] = int(len(ok))
    print(f'{s}: {ok.sum()} of {len(ok)} frames have all 6 future steps valid')

q = qs['train']
fin = lambda x: x[np.isfinite(x)]
U = {
    'kappa': float(np.percentile(np.abs(fin(q['kappa'])), P)),
    'a_lat': float(np.percentile(np.abs(fin(q['a_lat'])), P)),
    'a_max': float(np.percentile(q['a_lon'], P)),
    'a_min': float(np.percentile(q['a_lon'], 100 - P)),
    'jerk': float(np.percentile(np.abs(q['jerk']), P)),
}
res['U'] = U
print('\nenvelope U (train ground truth, 99.9th percentile):')
for name, x, two in (('|kappa| [1/m]', np.abs(fin(q['kappa'])), False),
                     ('|a_lat| [m/s^2]', np.abs(fin(q['a_lat'])), False),
                     ('a_lon [m/s^2]', q['a_lon'].ravel(), True),
                     ('|jerk| [m/s^3]', np.abs(q['jerk']).ravel(), False)):
    p99, p999, mx = np.percentile(x, 99), np.percentile(x, P), x.max()
    lo = f'  lower 0.1%: {np.percentile(x, 100 - P):.3f}' if two else ''
    print(f'  {name:16s} 99%: {p99:.4f}  99.9%: {p999:.4f}  max: {mx:.3f}{lo}   (n={x.size})')
nan_frac = float(np.isnan(q['kappa']).mean())
res['kappa_undefined_fraction_train'] = nan_frac
print(f'  curvature undefined (segment speed < {V_MIN} m/s): {100*nan_frac:.1f}% of steps')
print(f'  speed: median {np.median(q["v"]):.2f} m/s, 99.9%: {np.percentile(q["v"], P):.2f} m/s')

for s in ('train', 'val'):
    v = violations(qs[s], U)
    res[f'{s}_violation_rate'] = {k: float(x.mean()) for k, x in v.items()}
    print(f'\n{s} ground-truth trajectories outside U: ' +
          ', '.join(f'{k} {100*x.mean():.2f}%' for k, x in v.items()))

json.dump(res, open(sys.argv[2], 'w'), indent=1)
print('\nwrote', sys.argv[2])
