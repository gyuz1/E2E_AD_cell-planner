"""Speed-error sensitivity with the cross-validated weights (wa=2, wk=1)."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from projection_ablation import run
from smoothing_baseline import off
d = np.load(sys.argv[1]); idx = np.where(d['mode'] <= 4)[0]
sp, lcf, pred, gt = d['state_pred'][idx], d['ego_lcf'][idx], d['pred'][idx].astype(float), d['gt'][idx]
E = lambda P: float(np.mean([off(P[i], gt[i]) for i in range(len(gt))]))
v = lcf[:, 7]; kw = dict(wa=2.0, wk=1.0); out = {'est': E(run(pred, sp, (5, 2, 4), **kw)), 'meas': E(run(pred, lcf, (7, 2, 4), **kw)), 'sens': []}
for sg in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
    es = [E(run(pred, np.c_[lcf[:, :7], np.maximum(v + np.random.default_rng(s).normal(0, sg, len(v)), 0)], (7, 2, 4), **kw)) for s in range(1 if sg == 0 else 3)]
    out['sens'].append(dict(kind='sigma', value=sg, E=float(np.mean(es)), E_sd=float(np.std(es))))
for sc in (0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.05, 1.07, 1.1):
    out['sens'].append(dict(kind='scale', value=sc, E=E(run(pred, np.c_[lcf[:, :7], v * sc], (7, 2, 4), **kw))))
print(json.dumps(out, indent=0)); json.dump(out, open(__file__.rsplit('/', 1)[0] + f'/results/sens_cvweights_{sys.argv[2]}.json', 'w'), indent=1)
