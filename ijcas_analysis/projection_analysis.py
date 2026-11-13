"""Analyses of the kinematic projection on the stride-5 dump.
 A. observer accuracy (vision-estimated vs measured state)
 B. when does projection hurt?  binned by speed, observer speed error, plan-model
    residual (how far the plan is from the model's reachable set), command;
    gating rule chosen by 5-fold scene cross-validation on the residual
 C. sensitivity: projection from measured state + speed noise sigma, and speed scale bias
 D. feasibility / jerk of plans before and after projection; solve time per window
"""
import sys, json, time, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from trackability_precheck import track, U
from smoothing_baseline import off
from kinematics import quantities, violations

R = __file__.rsplit('/', 1)[0] + '/results/'
def job(a): return track(*a)[1]

def main(dump):
    d = np.load(dump); p = np.load(R + 'projection_s5.npz')
    idx = p['idx']; sp, lcf = d['state_pred'][idx], d['ego_lcf'][idx]
    pred, gt = d['pred'][idx].astype(float), d['gt'][idx]
    cmd, scene = d['mode'][idx], d['scene'][idx]
    E0, Eo, Em = p['E_none'], p['E_proj_obs'], p['E_proj_meas']
    rep = {}
    # A. observer
    v_hat, v = sp[:, 5], lcf[:, 7]
    obs = {}
    for name, a, b in (('speed', sp[:, 5], lcf[:, 7]), ('vx', sp[:, 0], lcf[:, 0]), ('vy', sp[:, 1], lcf[:, 1]),
                       ('ax', sp[:, 2], lcf[:, 2]), ('ay', sp[:, 3], lcf[:, 3]), ('yaw_rate', sp[:, 4], lcf[:, 4])):
        e = a - b
        obs[name] = dict(rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()), std_true=float(b.std()),
                         corr=float(np.corrcoef(a, b)[0, 1]))
    mv = v > 2
    obs['speed_rel_bias_v>2'] = float(((v_hat - v) / v)[mv].mean())
    obs['speed_scale_fit'] = float((v_hat[mv] @ v[mv]) / (v[mv] @ v[mv]))
    rep['A_observer'] = obs
    # B. when does it hurt
    dE = Eo - E0
    resid = np.sqrt(((p['proj_obs'] - pred) ** 2).sum(-1).mean(-1))     # plan vs model-reachable (obs)
    verr = np.abs(v_hat - v)
    def binned(x, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum(): out.append(dict(range=[float(lo), float(hi)], n=int(m.sum()), dE_mean=float(dE[m].mean()),
                                        frac_hurt=float((dE[m] > 0).mean()), E_none=float(E0[m].mean())))
        return out
    q = lambda x: np.unique(np.quantile(x, np.linspace(0, 1, 6)))
    rep['B_by_speed'] = binned(v_hat, np.r_[q(v_hat)[:-1], np.inf])
    rep['B_by_obs_speed_err'] = binned(verr, np.r_[q(verr)[:-1], np.inf])
    rep['B_by_residual'] = binned(resid, np.r_[q(resid)[:-1], np.inf])
    rep['B_corr'] = {'resid': float(np.corrcoef(resid, dE)[0, 1]), 'obs_speed_err': float(np.corrcoef(verr, dE)[0, 1]),
                     'speed': float(np.corrcoef(v_hat, dE)[0, 1])}
    # gating: project only if resid > thr; thr chosen on 4/5 of scenes, applied to 1/5
    us = np.unique(scene); rng = np.random.default_rng(0); perm = rng.permutation(us); folds = np.array_split(perm, 5)
    grid = np.quantile(resid, np.linspace(0, 0.95, 40)); gated = np.empty_like(E0); thrs = []
    for f in folds:
        te = np.isin(scene, f); tr = ~te
        best = min(grid, key=lambda t: np.where(resid[tr] > t, Eo[tr], E0[tr]).mean())
        thrs.append(float(best)); gated[te] = np.where(resid[te] > best, Eo[te], E0[te])
    rep['B_gating_cv'] = dict(thresholds=thrs, E_gated=float(gated.mean()), E_none=float(E0.mean()), E_proj=float(Eo.mean()),
                              frac_projected=float(np.mean([(resid[np.isin(scene, f)] > t).mean() for f, t in zip(folds, thrs)])))
    # C. sensitivity to state error (measured state + perturbation)
    sens = []
    rng = np.random.default_rng(1)
    conds = [('sigma', s) for s in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)] + [('scale', s) for s in (0.9, 0.95, 1.05, 1.1)]
    with ProcessPoolExecutor(6) as ex:
        for kind, s in conds:
            vv = v + rng.normal(0, s, len(v)) if kind == 'sigma' else v * s
            vv = np.maximum(vv, 0)
            args = [(pred[i], vv[i], lcf[i, 2], lcf[i, 4] / max(vv[i], 2.0)) for i in range(len(idx))]
            P = list(ex.map(job, args, chunksize=16))
            sens.append(dict(kind=kind, value=s, E=float(np.mean([off(P[i], gt[i]) for i in range(len(idx))]))))
            print(sens[-1], flush=True)
    rep['C_sensitivity'] = sens
    rep['C_ref'] = dict(E_none=float(E0.mean()), E_proj_obs=float(Eo.mean()), obs_speed_rmse=obs['speed']['rmse'])
    # D. feasibility, jerk, timing
    def feas(P):
        qq = quantities(P); vi = violations(qq, U)
        return dict(viol_any=float(vi['any'].mean()), jerk_rms=float(np.sqrt(np.nanmean(qq['jerk'] ** 2))),
                    max_abs_jerk_mean=float(np.nanmax(np.abs(qq['jerk']), 1).mean()))
    rep['D_feasibility'] = {'plan': feas(pred), 'proj_obs': feas(p['proj_obs']), 'proj_meas': feas(p['proj_meas']), 'gt': feas(gt)}
    ts = []
    for i in range(0, len(idx), 19):
        t0 = time.perf_counter(); track(pred[i], v_hat[i], sp[i, 2], sp[i, 4] / max(v_hat[i], 2.0)); ts.append(time.perf_counter() - t0)
    rep['D_solve_ms'] = dict(n=len(ts), mean=1e3 * float(np.mean(ts)), p95=1e3 * float(np.percentile(ts, 95)),
                             note='scipy least_squares, python rollout, single CPU core, unoptimised')
    rep = json.loads(json.dumps(rep, default=float))
    json.dump(rep, open(R + 'projection_analysis_s5.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))

if __name__ == '__main__':
    main(sys.argv[1])
