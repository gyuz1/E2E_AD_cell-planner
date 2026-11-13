"""Ablations, state-error sensitivity and the linearized prediction of it.
Uses the compiled solver (fast_projection).  Weights were fixed a priori; the
weight sweep is reported as sensitivity, not used to pick the method."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project, project_state, rollout, solve, LO, HI, ALAT, JMAX
from smoothing_baseline import off
R = __file__.rsplit('/', 1)[0] + '/results/'

def run(pred, st, cols, course=True, **kw):
    src = 'est' if cols[0] == 5 else 'meas'
    return np.stack([project_state(pred[i], st[i], src, course=course, **kw) for i in range(len(pred))])

def main(dump, tag):
    d = np.load(dump); idx = np.where(d['mode'] <= 4)[0]
    sp, lcf = d['state_pred'][idx], d['ego_lcf'][idx]; pred = d['pred'][idx].astype(float); gt = d['gt'][idx]
    OBS, MEAS = (5, 2, 4), (7, 2, 4)
    E = lambda P: float(np.mean([off(P[i], gt[i]) for i in range(len(gt))]))
    SQ = lambda P: float(((P - gt) ** 2).sum(-1).sum(-1).mean())
    rep = {'n': int(len(idx)), 'none': E(pred), 'none_sq': SQ(pred)}
    inf = np.full(12, np.inf)
    abl = {}
    for src, cols, st in (('obs', OBS, sp), ('meas', MEAS, lcf)):
        abl[src] = {'full': E(run(pred, st, cols)),
                    'no_bounds_no_soft': E(run(pred, st, cols, lo=-inf, hi=inf, ws=0.0)),
                    'no_rate_terms': E(run(pred, st, cols, wa=0.0, wk=0.0)),
                    'heading_zero': E(run(pred, st, cols, course=False)),
                    'no_current_accel_yawrate': E(run(pred, np.c_[st[:, :2], np.zeros(len(st)), st[:, 3], np.zeros(len(st)), st[:, 5:]] if src == 'obs' else np.c_[st[:, :2], np.zeros(len(st)), st[:, 3], np.zeros(len(st)), st[:, 5:]], cols))}
        print(src, abl[src], flush=True)
    rep['ablation'] = abl
    sweep = []
    for wa in (0.01, 0.05, 0.2, 1.0):
        for wk in (0.2, 1.0, 5.0):
            sweep.append(dict(wa=wa, wk=wk, E_obs=E(run(pred, sp, OBS, wa=wa, wk=wk))))
    rep['weight_sweep_obs'] = sweep; print('sweep done', flush=True)
    # sensitivity to speed error, measured state + noise / scale
    v = lcf[:, 7]; sens = []
    for sg in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        es, qs = [], []
        for seed in range(3):
            vv = np.maximum(v + np.random.default_rng(seed).normal(0, sg, len(v)), 0)
            P = run(pred, np.c_[lcf[:, :7], vv], MEAS); es.append(E(P)); qs.append(SQ(P))
            if sg == 0: break
        sens.append(dict(kind='sigma', value=sg, E=float(np.mean(es)), E_sd=float(np.std(es)), sq=float(np.mean(qs))))
    for sc in (0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.05, 1.07, 1.1):
        P = run(pred, np.c_[lcf[:, :7], v * sc], MEAS); sens.append(dict(kind='scale', value=sc, E=E(P), sq=SQ(P)))
    rep['sensitivity'] = sens; print('sens done', flush=True)
    rep = json.loads(json.dumps(rep, default=float))
    json.dump(rep, open(R + f'projection_ablation_{tag}.json', 'w'), indent=1)
    print(json.dumps(rep['ablation'], indent=1))
    print(json.dumps(rep['sensitivity'], indent=0))
    print(json.dumps(rep['weight_sweep_obs'], indent=0))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
