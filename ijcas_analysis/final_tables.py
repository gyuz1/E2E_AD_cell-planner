"""All paper numbers for one dump, with the compiled projection (the method).
usage: final_tables.py <dump.npz> <tag>"""
import sys, json, time, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project, project_state
from smoothing_baseline import fit, off
from kinematics import quantities, violations
from trackability_precheck import U
R = __file__.rsplit('/', 1)[0] + '/results/'

def main(dump, tag):
    d = np.load(dump); all_l2 = np.array([off(d['pred'][i], d['gt'][i]) for i in range(len(d['pred']))])
    idx = np.where(d['mode'] <= 4)[0]
    sp, lcf = d['state_pred'][idx], d['ego_lcf'][idx]; pred = d['pred'][idx].astype(float); gt = d['gt'][idx]
    cmd, scene = d['mode'][idx], d['scene'][idx]; n = len(idx)
    Pobs = np.stack([project_state(pred[i], sp[i], 'est') for i in range(n)])
    Pmeas = np.stack([project_state(pred[i], lcf[i], 'meas') for i in range(n)])
    Pv0 = np.stack([fit(pred[i], 4, sp[i, :2]) for i in range(n)])
    Ps = np.stack([fit(pred[i], 4) for i in range(n)])
    tt = np.arange(1, 7) * 0.5
    CAm = lcf[:, None, 0:2] * tt[None, :, None] + 0.5 * lcf[:, None, 2:4] * tt[None, :, None] ** 2
    CAe = sp[:, None, 0:2] * tt[None, :, None] + 0.5 * sp[:, None, 2:4] * tt[None, :, None] ** 2
    E = {k: np.array([off(P[i], gt[i]) for i in range(n)]) for k, P in
         (('none', pred), ('poly4', Ps), ('poly4_v0_obs', Pv0), ('proj_obs', Pobs), ('proj_meas', Pmeas), ('CA_meas', CAm), ('CA_est', CAe))}
    # tuned weights: 5-fold scene CV over (rho_a, rho_kappa), separately for each state source
    grid = [(wa, wk) for wa in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0) for wk in (0.2, 1.0, 5.0)]
    us0 = np.unique(scene); folds = np.array_split(np.random.default_rng(0).permutation(us0), 5); picks = {}
    for src, st in (('obs', sp), ('meas', lcf)):
        Eg = {g: np.array([off(project_state(pred[i], st[i], 'est' if src == 'obs' else 'meas', wa=g[0], wk=g[1]), gt[i]) for i in range(n)]) for g in grid}
        cv = np.empty(n); picks[src] = []
        for f in folds:
            te = np.isin(scene, f); b = min(grid, key=lambda g: Eg[g][~te].mean()); picks[src].append(list(b)); cv[te] = Eg[b][te]
        E[f'proj_{src}_cv'] = cv
    np.savez_compressed(R + f'final_{tag}.npz', idx=idx, proj_obs=Pobs, proj_meas=Pmeas, **{f'E_{k}': v for k, v in E.items()})
    rep = dict(cv_picks=picks, tag=tag, checkpoint=str(d['checkpoint']), n_all=int(len(all_l2)), E_all=float(all_l2.mean()), n_moving=n,
               E={k: float(v.mean()) for k, v in E.items()})
    rng = np.random.default_rng(0); us = np.unique(scene)
    for k in ('proj_obs', 'proj_meas', 'poly4_v0_obs', 'proj_obs_cv', 'proj_meas_cv', 'CA_meas'):
        diff = E[k] - E['none']; per = {s: diff[scene == s] for s in us}
        bs = [np.concatenate([per[s] for s in rng.choice(us, len(us))]).mean() for _ in range(2000)]
        rep[f'delta_{k}'] = dict(mean=float(diff.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                 rel=float(diff.mean() / E['none'].mean()), frac_improved=float((diff < 0).mean()))
    names = ['LK', 'LCL', 'LCR', 'TL', 'TR']
    rep['by_cmd'] = {names[c]: dict(n=int((cmd == c).sum()), **{k: float(v[cmd == c].mean()) for k, v in E.items()}) for c in range(5)}
    v = lcf[:, 7]; m = v > 2
    def init(P):
        a0 = (quantities(P)['v'][:, 0] - v) / 0.25
        return dict(rms=float(np.sqrt((a0[m] ** 2).mean())), frac_gt2=float((np.abs(a0[m]) > 2).mean()))
    rep['initial_accel'] = {'plan': init(pred), 'proj_obs': init(Pobs), 'gt': init(gt)}
    e = sp[:, 5] - v
    rep['observer'] = dict(speed_rmse=float(np.sqrt((e ** 2).mean())), speed_bias=float(e.mean()),
                           speed_scale=float((sp[m, 5] @ v[m]) / (v[m] @ v[m])),
                           ax_rmse=float(np.sqrt(((sp[:, 2] - lcf[:, 2]) ** 2).mean())),
                           yaw_rate_rmse=float(np.sqrt(((sp[:, 4] - lcf[:, 4]) ** 2).mean())))
    U2 = {k: x * 1.02 for k, x in U.items()}
    rep['violation_2pct'] = {k: float(violations(quantities(P), U2)['any'].mean()) for k, P in (('plan', pred), ('proj_obs', Pobs), ('gt', gt))}
    q = lambda P: float(np.sqrt(np.nanmean(quantities(P)['jerk'] ** 2)))
    rep['jerk_rms'] = {'plan': q(pred), 'proj_obs': q(Pobs), 'gt': q(gt)}
    ts = []
    for i in range(0, n, 5):
        t0 = time.perf_counter(); project_state(pred[i], sp[i], 'est'); ts.append(time.perf_counter() - t0)
    rep['solve_ms'] = dict(mean=1e3 * float(np.mean(ts)), p95=1e3 * float(np.percentile(ts, 95)), max=1e3 * float(np.max(ts)))
    json.dump(rep, open(R + f'final_{tag}.json', 'w'), indent=1); print(json.dumps(rep, indent=1))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
