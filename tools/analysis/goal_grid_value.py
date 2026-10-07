"""How much is knowing the 5s goal cell worth on the scored 3s L2, per grid design?

Offline, no network. For each design, fits ridge regressions from the current
ego state [vx, vy, ax, ay, yaw_rate, speed, 1] to the cumulative 3s trajectory,
one model per (command, goal cell) on the train split, and scores the val split
with the official L2 (mean of the 1s/2s/3s cumulative-distance averages).

  no cell            -> what state + command alone give (kinematic floor)
  oracle cell        -> upper bound on what a perfect cell classifier adds
  kinematic cell     -> the cell chosen by constant-acceleration extrapolation
                        from the same state, i.e. no information beyond the
                        state (should be ~no gain; a sanity control)
  oracle TP (cont.)  -> the target point itself as a regressor, the ceiling of
                        the whole idea

A design is only worth it if oracle-cell gain is large AND the cell is
learnable from what the network sees; the second part is not measured here.
"""
import pickle
from itertools import product

import numpy as np

D = 'data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_{}_split.pkl'


def load(sp):
    d = pickle.load(open(D.format(sp), 'rb'))
    infos = d['infos'] if isinstance(d, dict) else d
    infos = [i for i in infos if i.get('fut_valid_flag', False)]
    lcf = np.stack([np.asarray(i['gt_ego_lcf_feat'], np.float64) for i in infos])
    X = np.concatenate([lcf[:, [0, 1, 2, 3, 4, 7]], np.ones((len(lcf), 1))], 1)
    Y = np.stack([np.asarray(i['gt_ego_fut_trajs'], np.float64).cumsum(0)
                  for i in infos]).reshape(len(infos), -1)
    cmd = np.array([int(np.asarray(i['gt_ego_fut_cmd']).argmax()) for i in infos])
    tp = np.stack([np.asarray(i['gt_ego_target_point'], np.float64).reshape(-1)[:2]
                   for i in infos])
    return X, Y, cmd, tp, lcf


def official_l2(P, Y):
    d = np.linalg.norm(P.reshape(-1, 6, 2) - Y.reshape(-1, 6, 2), axis=-1)
    return float(np.mean([d[:, :2].mean(), d[:, :4].mean(), d[:, :6].mean()]))


def fit(X, Y, lam=1e-2):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def cells(tp, rng, g):
    x0, x1, y0, y1 = rng
    gx, gy = g
    xi = np.clip(np.floor((tp[:, 0] - x0) / (x1 - x0) * gx), 0, gx - 1)
    yi = np.clip(np.floor((tp[:, 1] - y0) / (y1 - y0) * gy), 0, gy - 1)
    return (xi * gy + yi).astype(int)


def kin_tp(lcf, t=5.0):
    vx, vy, ax, ay = lcf[:, 0], lcf[:, 1], lcf[:, 2], lcf[:, 3]
    return np.stack([vx * t + 0.5 * ax * t * t, vy * t + 0.5 * ay * t * t], 1)


def grouped(Xtr, Ytr, gtr, Xte, gte, min_n=30):
    glob = fit(Xtr, Ytr)
    P = Xte @ glob
    for key in np.unique(gte):
        m_tr = gtr == key
        W = fit(Xtr[m_tr], Ytr[m_tr]) if m_tr.sum() >= min_n else glob
        m = gte == key
        P[m] = Xte[m] @ W
    return P


def main():
    Xtr, Ytr, ctr, ttr, ltr = load('train')
    Xte, Yte, cte, tte, lte = load('val')
    print(f'train {len(Xtr)} / val {len(Xte)} (fut_valid)')

    base = official_l2(grouped(Xtr, Ytr, ctr, Xte, cte), Yte)
    Xtr_tp = np.concatenate([Xtr, ttr], 1)
    Xte_tp = np.concatenate([Xte, tte], 1)
    ctp = official_l2(grouped(Xtr_tp, Ytr, ctr, Xte_tp, cte), Yte)
    print(f'\n{"design":<34}{"L2 oracle":>10}{"gain":>8}{"L2 kin-cell":>12}'
          f'{"kin hit":>9}{"out%":>7}{"cells used":>11}')
    print(f'{"command only (no cell)":<34}{base:>10.4f}')
    print(f'{"command + oracle TP (continuous)":<34}{ctp:>10.4f}{base - ctp:>+8.4f}')

    designs = []
    for rng in [(-5, 110, -25, 25), (-5, 80, -25, 25), (-5, 60, -20, 20),
                (0, 120, -30, 30)]:
        for g in [(5, 5), (7, 3), (5, 3), (5, 1), (10, 1), (10, 3), (15, 1),
                  (10, 5), (20, 1)]:
            designs.append((rng, g))
    for rng, g in designs:
        k = g[0] * g[1]
        gtr = ctr * k + cells(ttr, rng, g)
        gte = cte * k + cells(tte, rng, g)
        P = grouped(Xtr, Ytr, gtr, Xte, gte)
        l_or = official_l2(P, Yte)
        kc = cells(kin_tp(lte), rng, g)
        gk = cte * k + kc
        l_kin = official_l2(grouped(Xtr, Ytr, gtr, Xte, gk), Yte)
        hit = float((kc == cells(tte, rng, g)).mean())
        x0, x1, y0, y1 = rng
        out = float(((tte[:, 0] < x0) | (tte[:, 0] >= x1) |
                     (tte[:, 1] < y0) | (tte[:, 1] >= y1)).mean())
        used = len(np.unique(cells(ttr, rng, g)))
        name = f'{g[0]}x{g[1]}  x{x0}..{x1} y{y0}..{y1}'
        print(f'{name:<34}{l_or:>10.4f}{base - l_or:>+8.4f}{l_kin:>12.4f}'
              f'{hit:>9.3f}{100 * out:>7.2f}{used:>6}/{k}')


if __name__ == '__main__':
    main()
