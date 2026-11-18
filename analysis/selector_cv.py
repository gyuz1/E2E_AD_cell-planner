"""Run-time selector of the projection weight (or no projection) from signals available at
inference: |alpha_t|, speed mismatch of the first interval, speed, yaw rate, command, and the
correction size ||P - Q_k|| of every candidate. One gradient-boosting regressor per candidate
predicts E_plan; the candidate with the lowest prediction is used. Evaluated by five-fold scene
cross-validation (same folds as elsewhere), against the fixed weight chosen on the same training
folds. No ground truth and no measured-minus-estimated state difference is used as input.
usage: selector_cv.py <heldout_dump> <tag>"""
import sys, json, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
from smoothing_baseline import off
from deep_analysis import boot
R = __file__.rsplit('/', 1)[0] + '/results/'
TAU = 0.5
WA = (0.05, 0.2, 0.5, 1, 2, 5, 10, 20)

def main(dump, tag):
    d = np.load(dump); f = np.load(R + f'final_{tag}.npz'); idx = f['idx']
    P = d['pred'][idx].astype(float); Y = d['gt'][idx].astype(float); cmd = d['mode'][idx]; scene = d['scene'][idx]
    rep = dict(tag=tag)
    us = np.unique(scene); folds = np.array_split(np.random.default_rng(0).permutation(us), 5)
    for src, st in (('meas', d['ego_lcf'][idx].astype(float)), ('est', d['state_pred'][idx].astype(float))):
        vc = 7 if src == 'meas' else 5
        v = st[:, vc]; a = st[:, 2]; w = st[:, 4]
        n = len(P); K = len(WA) + 1
        Q = np.empty((K, n, 6, 2)); Q[0] = P
        for k, wa in enumerate(WA):
            Q[k + 1] = np.stack([project_state(P[i], st[i], 'est' if src == 'est' else 'meas', wa=wa, wk=5.0) for i in range(n)])
        E = np.array([[off(Q[k][i], Y[i]) for i in range(n)] for k in range(K)])          # (K, n)
        corr = np.sqrt(((Q - P[None]) ** 2).sum((-1, -2)))                                   # (K, n)
        alpha = (np.linalg.norm(P[:, 0], axis=-1) / TAU - v) / (TAU / 2)
        ev = np.linalg.norm(P[:, 0], axis=-1) / TAU - (v + a * TAU / 2)
        X = np.column_stack([np.abs(alpha), ev, v, w, np.eye(5)[np.clip(cmd, 0, 4)], corr[1:].T,
                             np.linalg.norm(P[:, -1], axis=-1)])
        sel = np.empty(n); fixed = np.empty(n); picks = []; Qsel = np.empty_like(P); Qfix = np.empty_like(P); chosen = np.empty(n, int)
        for fo in folds:
            te = np.isin(scene, fo); tr = ~te
            kb = int(E[:, tr].mean(1).argmin()); fixed[te] = E[kb, te]; picks.append(kb); Qfix[te] = Q[kb][te]
            pred = np.empty((K, te.sum()))
            for k in range(K):
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                  min_samples_leaf=40, random_state=0).fit(X[tr], E[k, tr])
                pred[k] = m.predict(X[te])
            ch = pred.argmin(0); ti = np.where(te)[0]; sel[te] = E[ch, ti]; Qsel[ti] = Q[ch, ti]; chosen[ti] = ch
        rep[src] = dict(E_none=float(E[0].mean()), E_fixed_cv=float(fixed.mean()),
                        fixed_picks=[('none' if k == 0 else WA[k - 1]) for k in picks],
                        E_selector_cv=float(sel.mean()), E_oracle=float(E.min(0).mean()),
                        delta_selector_vs_fixed=float((sel - fixed).mean()), ci=boot(sel - fixed, scene),
                        frac_windows_better=float((sel < fixed - 1e-9).mean()), frac_windows_worse=float((sel > fixed + 1e-9).mean()))
        rep[src]['chosen_share'] = {('none' if k == 0 else str(WA[k - 1])): float((chosen == k).mean()) for k in range(K)}
        np.savez_compressed(R + f'selector_{tag}_{src}.npz', Q_sel=Qsel, Q_fixed=Qfix, chosen=chosen, idx=idx)
        print(tag, src, rep[src], flush=True)
    json.dump(rep, open(R + f'selector_{tag}.json', 'w'), indent=1)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
