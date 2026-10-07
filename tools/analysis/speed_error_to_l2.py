"""How accurate does the vision ego-status estimate have to be?

LANE_KEEP is 85% of our error and behaves as the integral of speed-estimate
error (the --zero-ego-lcf control collapsed it 39x). So the question "how do
we improve" reduces to a quantitative one: what speed and acceleration error
does a given L2 require?

This answers it directly on the real val split. It builds the same
constant-acceleration oracle tools/kinematic_oracle_ceiling.py measures,
then corrupts the v and a it is given with controlled noise and re-measures
under the competition's own L2 windowing. The result is a budget: to land at
some L2, the estimator has to be at least this good.

Reference points already measured on this split:
    perfect v, a assumed 0     0.5965
    perfect v AND a            0.2708
    A student v1               0.4218   (between them, no noise model)
    compliant baseline         0.4885

Usage:
    python tools/speed_error_to_l2.py [--ann <val pkl>] [--trials N]
"""
import argparse
import pickle

import numpy as np


FUT_TS_INTERVAL_S = 0.5


def l2_windows(pred_delta, gt_delta):
    """Competition metric: mean of the 1s/2s/3s window means."""
    d = np.linalg.norm(pred_delta.cumsum(1) - gt_delta.cumsum(1), axis=-1)
    return np.stack([d[:, :2].mean(1), d[:, :4].mean(1), d[:, :6].mean(1)], 1)


def rollout(v, a, T=6, dt=FUT_TS_INTERVAL_S):
    """Constant-acceleration per-step displacements from [N,2] v and a."""
    steps = []
    for i in range(T):
        t0, t1 = i * dt, (i + 1) * dt
        # position(t) = v*t + a*t^2/2, so the step is the difference
        steps.append(v * (t1 - t0) + a * (t1 ** 2 - t0 ** 2) / 2.0)
    return np.stack(steps, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann', default='data/etri/.causal_regen_split_301_75_10hz/'
                                    'vad_etri_infos_temporal_val_split.pkl')
    p.add_argument('--trials', type=int, default=3)
    args = p.parse_args()

    with open(args.ann, 'rb') as fh:
        infos = pickle.load(fh)['infos']
    gt = np.stack([np.asarray(i['gt_ego_fut_trajs'], dtype=np.float64)
                   .reshape(-1, 2)[:6] for i in infos])
    lcf = np.stack([np.asarray(i['gt_ego_lcf_feat'], dtype=np.float64)
                    for i in infos])
    v = lcf[:, 0:2].copy()
    a = lcf[:, 2:4].copy()
    print(f'{len(infos)} val samples\n')

    base = l2_windows(rollout(v, a), gt).mean()
    print(f'perfect v+a oracle : {base:.4f}   '
          f'(compare with 0.2708 from kinematic_oracle_ceiling.py)')
    print(f'perfect v, a=0     : {l2_windows(rollout(v, np.zeros_like(a)), gt).mean():.4f}\n')

    rng = np.random.default_rng(0)

    def measure(sv, sa):
        out = []
        for _ in range(args.trials):
            vv = v + rng.normal(0, sv, v.shape)
            aa = a + rng.normal(0, sa, a.shape)
            out.append(l2_windows(rollout(vv, aa), gt).mean())
        return float(np.mean(out))

    print('speed error only (acceleration assumed perfect)')
    print(f"{'speed RMSE (m/s)':>18} {'L2':>9}")
    for sv in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0):
        print(f'{sv:>18.2f} {measure(sv, 0.0):>9.4f}')

    print('\nacceleration error only (speed assumed perfect)')
    print(f"{'accel RMSE (m/s^2)':>20} {'L2':>9}")
    for sa in (0.0, 0.1, 0.25, 0.46, 1.0):
        note = '  <- the real std of a' if abs(sa - 0.46) < 1e-9 else ''
        print(f'{sa:>20.2f} {measure(0.0, sa):>9.4f}{note}')

    print('\nboth wrong -- an error budget for a target L2')
    print(f"{'speed RMSE':>12} {'accel RMSE':>12} {'L2':>9}")
    for sv, sa in ((0.1, 0.1), (0.25, 0.25), (0.5, 0.3),
                   (0.5, 0.46), (1.0, 0.46), (1.5, 0.46), (2.0, 0.46)):
        print(f'{sv:>10.2f} {sa:>12.2f} {measure(sv, sa):>9.4f}')

    print('\nnote: on the train split v has std 5.70 m/s and a has 0.46 m/s^2.')
    print('an acceleration RMSE of 0.46 means knowing nothing about acceleration.')


if __name__ == '__main__':
    main()
