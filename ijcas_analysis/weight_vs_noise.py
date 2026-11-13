"""E_plan over (rate weight rho_a) x (speed noise sigma): the best weight as a
function of state accuracy.  Measured state + Gaussian speed noise, seed 0."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from projection_ablation import run
from smoothing_baseline import off
d = np.load(sys.argv[1]); idx = np.where(d['mode'] <= 4)[0]
lcf, pred, gt = d['ego_lcf'][idx], d['pred'][idx].astype(float), d['gt'][idx]
E = lambda P: float(np.mean([off(P[i], gt[i]) for i in range(len(gt))]))
v = lcf[:, 7]; W = (0.05, 0.2, 0.5, 1.0, 2.0, 5.0); S = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
tab = {}
for sg in S:
    vv = np.maximum(v + np.random.default_rng(0).normal(0, sg, len(v)), 0) if sg else v
    for wa in W:
        tab[f'{sg},{wa}'] = E(run(pred, np.c_[lcf[:, :7], vv], (7, 2, 4), wa=wa, wk=1.0))
    print(sg, [round(tab[f'{sg},{wa}'], 4) for wa in W], flush=True)
none = E(pred)
best = {str(sg): dict(wa=min(W, key=lambda w: tab[f'{sg},{w}']), E=min(tab[f'{sg},{w}'] for w in W)) for sg in S}
out = dict(weights=W, sigmas=S, table=tab, best=best, none=none)
json.dump(out, open(__file__.rsplit('/', 1)[0] + f'/results/weight_vs_noise_{sys.argv[2]}.json', 'w'), indent=1); print(json.dumps(best))
