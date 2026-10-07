"""How far can perfect ego velocity / acceleration estimation alone get us?

Scores kinematic oracles against the same hold-out val split and the same
challenge L2 windowing the real evaluator uses, so their numbers sit on the
same axis as the model's. No GPU, no model -- these are closed-form
extrapolations from GT ego state, which is exactly what a hypothetical
perfect vision-based v/a estimator would hand the planner.

The point is to bound the payoff before spending GPU time on better motion
estimation: if the constant-acceleration oracle already scores worse than
the model does today, then no amount of v/a accuracy explains the remaining
gap, and the headroom is in predicting the future maneuver instead.

  const-velocity  : p(t) = v * t              (perfect v, a assumed 0)
  const-accel     : p(t) = v*t + 0.5*a*t^2    (perfect v AND a)
"""
import argparse
import pickle

import numpy as np

FUT_TS = 6
DT = 0.5


def challenge_l2(pred_cum, gt_cum):
    """Mean of the competition's overlapping 1s/2s/3s window L2s."""
    dist = np.linalg.norm(pred_cum - gt_cum, axis=-1)
    return float(np.mean([dist[:2].mean(), dist[:4].mean(), dist[:6].mean()]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann-file', required=True)
    args = p.parse_args()

    with open(args.ann_file, 'rb') as handle:
        infos = pickle.load(handle)['infos']

    t = np.arange(1, FUT_TS + 1) * DT           # 0.5 .. 3.0 s
    rows = {'const-velocity': [], 'const-accel': [], 'zero (stay put)': []}
    cmd_rows = {}

    for info in infos:
        if not info.get('fut_valid_flag', False):
            continue
        gt = np.asarray(info['gt_ego_fut_trajs'], dtype=np.float64)
        if gt.shape != (FUT_TS, 2):
            continue
        gt_cum = gt.cumsum(0)

        lcf = np.asarray(info['gt_ego_lcf_feat'], dtype=np.float64)
        v = lcf[0:2]
        a = lcf[2:4]

        cv = v[None, :] * t[:, None]
        ca = v[None, :] * t[:, None] + 0.5 * a[None, :] * (t[:, None] ** 2)
        zero = np.zeros_like(gt_cum)

        vals = {'const-velocity': challenge_l2(cv, gt_cum),
                'const-accel': challenge_l2(ca, gt_cum),
                'zero (stay put)': challenge_l2(zero, gt_cum)}
        for k, val in vals.items():
            rows[k].append(val)

        ci = int(np.asarray(info['gt_ego_fut_cmd']).argmax())
        cmd_rows.setdefault(ci, {k: [] for k in vals})
        for k, val in vals.items():
            cmd_rows[ci][k].append(val)

    n = len(rows['const-velocity'])
    print('%d hold-out samples, same L2 windowing as the challenge\n' % n)
    print('%-22s %10s' % ('oracle', 'L2'))
    print('-' * 34)
    for k in ('zero (stay put)', 'const-velocity', 'const-accel'):
        print('%-22s %10.4f' % (k, np.mean(rows[k])))

    CMD = ['LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT',
           'TURN_RIGHT', 'U_TURN', 'STOP']
    print('\nby command')
    print('%-16s %8s %14s %12s' % ('', 'n', 'const-velocity', 'const-accel'))
    print('-' * 54)
    for ci in sorted(cmd_rows):
        r = cmd_rows[ci]
        print('%-16s %8d %14.4f %12.4f'
              % (CMD[ci], len(r['const-velocity']),
                 np.mean(r['const-velocity']), np.mean(r['const-accel'])))


if __name__ == '__main__':
    main()
