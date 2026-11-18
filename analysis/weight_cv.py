"""Rate-term weights chosen by 5-fold scene cross-validation on the held-out scenes
(choose on 4/5 of the scenes, evaluate on the remaining 1/5), network-estimated state."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
from smoothing_baseline import off
d = np.load(sys.argv[1]); idx = np.where(d['mode'] <= 4)[0]
sp, pred, gt, scene = d['state_pred'][idx], d['pred'][idx].astype(float), d['gt'][idx], d['scene'][idx]
grid = [(wa, wk) for wa in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0) for wk in (0.2, 1.0, 5.0)]
E = {}
for g in grid:
    E[g] = np.array([off(project_state(pred[i], sp[i], 'est', wa=g[0], wk=g[1]), gt[i]) for i in range(len(idx))])
us = np.unique(scene); perm = np.random.default_rng(0).permutation(us); folds = np.array_split(perm, 5)
cv = np.empty(len(idx)); picks = []
for f in folds:
    te = np.isin(scene, f); best = min(grid, key=lambda g: E[g][~te].mean()); picks.append(best); cv[te] = E[best][te]
none = np.array([off(pred[i], gt[i]) for i in range(len(idx))])
rng = np.random.default_rng(0); per = {s: (cv - none)[scene == s] for s in us}
bs = [np.concatenate([per[s] for s in rng.choice(us, len(us))]).mean() for _ in range(2000)]
rep = dict(grid={f'{g[0]},{g[1]}': float(E[g].mean()) for g in grid}, picks=[list(p) for p in picks], E_cv=float(cv.mean()), E_none=float(none.mean()),
           delta_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_improved=float((cv < none).mean()))
print(json.dumps(rep, indent=1)); json.dump(rep, open(__file__.rsplit('/', 1)[0] + f'/results/weight_cv_{sys.argv[2]}.json', 'w'), indent=1)
