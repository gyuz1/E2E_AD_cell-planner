"""Locked tuning/test split repeated over 10 random scene splits (planning error only):
how stable are the projection gain and the selector's further gain? Candidate projections are
computed once and reused. usage: locked_repeat.py <heldout_dump> <tag>"""
import sys, json, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
from smoothing_baseline import off
from deep_analysis import boot
R = __file__.rsplit('/', 1)[0] + '/results/'
TAU = 0.5
GRID = [(wa, wk) for wa in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0) for wk in (0.2, 1.0, 5.0)]
WA = (0.05, 0.2, 0.5, 1, 2, 5, 10, 20)
def main(dump, tag):
    d = np.load(dump); idx = np.where(d['mode'] <= 4)[0]
    P = d['pred'][idx].astype(float); Y = d['gt'][idx].astype(float); cmd = d['mode'][idx]; sc = d['scene'][idx]
    n = len(P); EP = np.array([off(P[i], Y[i]) for i in range(n)]); us = np.unique(sc); rep = dict(tag=tag)
    for src, st in (('meas', d['ego_lcf'][idx].astype(float)), ('est', d['state_pred'][idx].astype(float))):
        Eg = {g: np.array([off(project_state(P[i], st[i], src, wa=g[0], wk=g[1]), Y[i]) for i in range(n)]) for g in GRID}
        vc = 7 if src == 'meas' else 5; v = st[:, vc]; a = st[:, 2]; w = st[:, 4]; K = len(WA) + 1
        Q = np.empty((K, n, 6, 2)); Q[0] = P
        for k, wa in enumerate(WA):
            Q[k + 1] = np.stack([project_state(P[i], st[i], src, wa=wa, wk=5.0) for i in range(n)])
        E = np.array([[off(Q[k][i], Y[i]) for i in range(n)] for k in range(K)])
        corr = np.sqrt(((Q - P[None]) ** 2).sum((-1, -2)))
        alpha = (np.linalg.norm(P[:, 0], axis=-1) / TAU - v) / (TAU / 2); ev = np.linalg.norm(P[:, 0], axis=-1) / TAU - (v + a * TAU / 2)
        X = np.column_stack([np.abs(alpha), ev, v, w, np.eye(5)[np.clip(cmd, 0, 4)], corr[1:].T, np.linalg.norm(P[:, -1], axis=-1)])
        rows = []
        for seed in range(10):
            perm = np.random.default_rng(seed).permutation(us); tu = np.isin(sc, perm[:len(us) // 2]); te = ~tu
            best = min(GRID, key=lambda g: Eg[g][tu].mean()); Ef = Eg[best]
            pred = np.array([HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                             random_state=0).fit(X[tu], E[k, tu]).predict(X[te]) for k in range(K)])
            ti = np.where(te)[0]; Es = E[pred.argmin(0), ti]
            ci_p = boot((Ef - EP)[te], sc[te], 1000); ci_s = boot(Es - Ef[te], sc[te], 1000)
            rows.append(dict(seed=seed, weights=list(best), none=float(EP[te].mean()), fixed=float(Ef[te].mean()), selector=float(Es.mean()),
                             proj_sig=bool(ci_p[1] < 0), sel_better=bool(Es.mean() < Ef[te].mean()), sel_sig=bool(ci_s[1] < 0), sel_worse_sig=bool(ci_s[0] > 0)))
        rep[src] = dict(rows=rows, proj_sig=sum(r['proj_sig'] for r in rows), sel_better=sum(r['sel_better'] for r in rows),
                        sel_sig=sum(r['sel_sig'] for r in rows), sel_worse_sig=sum(r['sel_worse_sig'] for r in rows),
                        mean_rel_proj=float(np.mean([(r['fixed'] - r['none']) / r['none'] for r in rows])),
                        mean_rel_sel=float(np.mean([(r['selector'] - r['fixed']) / r['fixed'] for r in rows])))
        print(tag, src, {k: v for k, v in rep[src].items() if k != 'rows'}, flush=True)
    json.dump(rep, open(R + f'locked_repeat_{tag}.json', 'w'), indent=1)
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
