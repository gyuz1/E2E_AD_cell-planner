"""Is the projection gain just smoothing? Compare, on the same windows, the
selected plan against plain polynomial smoothers of it (no vehicle model):
  poly3 / poly4 in time per axis, through the origin, no state;
  poly4 with the initial velocity pinned to the given state (obs or meas).
Single process (cheap)."""
import sys, json, numpy as np
t = np.arange(1, 7) * 0.5

def off(P, gt):
    e = np.linalg.norm(P - gt, axis=-1); return np.mean([e[:2].mean(), e[:4].mean(), e.mean()])

def fit(P, deg, v0=None):
    cols = [t ** p for p in range(1 if v0 is None else 2, deg + 1)]
    A = np.stack(cols, 1); out = np.empty_like(P)
    for ax in range(2):
        y = P[:, ax] - (0 if v0 is None else v0[ax] * t)
        c, *_ = np.linalg.lstsq(A, y, rcond=None); out[:, ax] = A @ c + (0 if v0 is None else v0[ax] * t)
    return out

def main():
  d = np.load(sys.argv[1]); idx = np.where(d['mode'] <= 4)[0]
  res = {}
  for name, f in {'none': lambda i: d['pred'][i],
                  'poly3': lambda i: fit(d['pred'][i], 3), 'poly4': lambda i: fit(d['pred'][i], 4),
                  'poly4_v0obs': lambda i: fit(d['pred'][i], 4, d['state_pred'][i, :2]),
                  'poly4_v0meas': lambda i: fit(d['pred'][i], 4, d['ego_lcf'][i, :2])}.items():
      res[name] = float(np.mean([off(f(i), d['gt'][i]) for i in idx]))
  print(json.dumps(res, indent=1))
  json.dump(res, open(sys.argv[2], 'w'), indent=1)

if __name__ == '__main__':
    main()
