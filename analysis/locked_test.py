"""Locked test: the 75 held-out scenes are split once (seed 0) into a tuning half and a test half.
Everything that is chosen (projection weights, selector regressors, controller gains) is chosen on
the tuning half only; the test half is evaluated once. Measured and estimated state.
usage: locked_test.py <heldout_dump> <tag>"""
import sys, json, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
from smoothing_baseline import off
from deep_analysis import boot
import closedloop_sim as cl
R = __file__.rsplit('/', 1)[0] + '/results/'
TAU = 0.5
GRID = [(wa, wk) for wa in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0) for wk in (0.2, 1.0, 5.0)]
WA = (0.05, 0.2, 0.5, 1, 2, 5, 10, 20)

def main(dump, tag):
    d = np.load(dump); idx = np.where(d['mode'] <= 4)[0]
    P = d['pred'][idx].astype(float); Y = d['gt'][idx].astype(float); cmd = d['mode'][idx]; sc = d['scene'][idx]
    L = d['ego_lcf'][idx].astype(float); S = d['state_pred'][idx].astype(float); n = len(P)
    us = np.unique(sc); perm = np.random.default_rng(0).permutation(us); tune_s = perm[:len(us) // 2]
    tu = np.isin(sc, tune_s); te = ~tu
    EP = np.array([off(P[i], Y[i]) for i in range(n)])
    rep = dict(tag=tag, n_tune=int(tu.sum()), n_test=int(te.sum()), scenes_tune=int(len(tune_s)), scenes_test=int(len(us) - len(tune_s)),
               E_none_test=float(EP[te].mean()))
    refs = {}
    for src, st in (('meas', L), ('est', S)):
        Eg = {g: np.array([off(project_state(P[i], st[i], src, wa=g[0], wk=g[1]), Y[i]) for i in range(n)]) for g in GRID}
        best = min(GRID, key=lambda g: Eg[g][tu].mean())
        Qf = np.stack([project_state(P[i], st[i], src, wa=best[0], wk=best[1]) for i in range(n)])
        # selector
        vc = 7 if src == 'meas' else 5; v = st[:, vc]; a = st[:, 2]; w = st[:, 4]; K = len(WA) + 1
        Q = np.empty((K, n, 6, 2)); Q[0] = P
        for k, wa in enumerate(WA):
            Q[k + 1] = np.stack([project_state(P[i], st[i], src, wa=wa, wk=5.0) for i in range(n)])
        E = np.array([[off(Q[k][i], Y[i]) for i in range(n)] for k in range(K)])
        corr = np.sqrt(((Q - P[None]) ** 2).sum((-1, -2)))
        alpha = (np.linalg.norm(P[:, 0], axis=-1) / TAU - v) / (TAU / 2); ev = np.linalg.norm(P[:, 0], axis=-1) / TAU - (v + a * TAU / 2)
        X = np.column_stack([np.abs(alpha), ev, v, w, np.eye(5)[np.clip(cmd, 0, 4)], corr[1:].T, np.linalg.norm(P[:, -1], axis=-1)])
        pred = np.empty((K, te.sum()))
        for k in range(K):
            pred[k] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                                    random_state=0).fit(X[tu], E[k, tu]).predict(X[te])
        ch = pred.argmin(0); ti = np.where(te)[0]; Qs = P.copy(); Qs[ti] = Q[ch, ti]
        Ef = Eg[best]; Es = np.array([off(Qs[i], Y[i]) for i in range(n)])
        rep[src] = dict(weights=list(best), E_fixed_test=float(Ef[te].mean()), E_selector_test=float(Es[te].mean()),
                        ci_fixed_vs_none=boot((Ef - EP)[te], sc[te]), ci_selector_vs_fixed=boot((Es - Ef)[te], sc[te]),
                        rel_fixed=float((Ef[te].mean() - EP[te].mean()) / EP[te].mean()))
        refs[src] = (Qf, Qs)
        print(tag, src, rep[src], flush=True)
    # controller gains tuned on the tuning half only (recorded trajectories), executed error on the test half
    x0 = cl.initial(L); ti = np.where(tu)[0]
    sc_g, gains = cl.tune_gains(Y[ti], x0[ti]); rep['gains_tune'] = list(gains)
    te_i = np.where(te)[0]; M = {}
    for name, W in (('plan', P), ('meas_fixed', refs['meas'][0]), ('meas_selector', refs['meas'][1]), ('est_fixed', refs['est'][0]), ('est_selector', refs['est'][1])):
        m, *_ = cl.metrics(W[te_i], Y[te_i], x0[te_i], gains); M[name] = m
    cl_rep = {k: dict(E_exec=float(m['E_exec'].mean()), track=float(m['track_rms'].mean()), jerk=float(m['jerk_rms'].mean())) for k, m in M.items()}
    for k in M:
        if k != 'plan':
            cl_rep[k]['ci_vs_plan'] = boot(M[k]['E_exec'] - M['plan']['E_exec'], sc[te_i])
    cl_rep['meas_selector']['ci_vs_fixed'] = boot(M['meas_selector']['E_exec'] - M['meas_fixed']['E_exec'], sc[te_i])
    rep['closedloop_test'] = cl_rep
    json.dump(rep, open(R + f'locked_test_{tag}.json', 'w'), indent=1); print(json.dumps(rep, indent=1))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
