"""Analysis-paper candidates on existing dumps (CPU only, no training).
 (2) longitudinal/lateral decomposition of plan error w.r.t. the recorded path
 (3) does the plan's initial-speed error follow the network's state error?
 (4) consistency on training vs held-out scenes
 (10) projection-theorem check: gain vs correction size and GT distance to R
 (14) per-window oracle weight vs fixed weight (upper bound for adaptive weighting)
 (6/7) transmission of state perturbations through the implemented projection
usage: deep_analysis.py <heldout_dump> <train_dump> <tag>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project, project_state
from smoothing_baseline import off
R = __file__.rsplit('/', 1)[0] + '/results/'
TAU = 0.5; TT = np.arange(1, 7) * TAU

def frenet(P, gt):
    """signed along-track (+ = plan ahead) and cross-track error per waypoint, tangent of the recorded path"""
    Y = np.concatenate([np.zeros((len(gt), 1, 2)), gt], 1)
    tan = np.empty_like(gt)
    for j in range(6):
        d = Y[:, min(j + 2, 6)] - Y[:, j]          # central difference around waypoint j+1
        tan[:, j] = d
    nrm = np.linalg.norm(tan, axis=-1, keepdims=True); ok = nrm[..., 0] > 0.05
    tan = np.where(nrm > 0.05, tan / np.maximum(nrm, 1e-9), np.array([1.0, 0.0]))
    e = P - gt
    lon = (e * tan).sum(-1); lat = e[..., 0] * -tan[..., 1] + e[..., 1] * tan[..., 0]
    return lon, lat, ok

def agg(x):  # E_plan-style averaging of a per-waypoint magnitude
    return np.mean([x[:, :2].mean(1), x[:, :4].mean(1), x.mean(1)], 0)

def alpha(P, v):
    return (np.linalg.norm(P[:, 0], axis=-1) / TAU - v) / (TAU / 2)

def boot(diff, scene, n=2000):
    us = np.unique(scene); per = {s: diff[scene == s] for s in us}; rng = np.random.default_rng(0)
    bs = [np.concatenate([per[s] for s in rng.choice(us, len(us))]).mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def proj_all(pred, st, src, picks, scene, **kw):
    us0 = np.unique(scene); folds = np.array_split(np.random.default_rng(0).permutation(us0), 5)
    out = np.empty_like(pred)
    for fo, (wa, wk) in zip(folds, picks):
        for i in np.where(np.isin(scene, fo))[0]:
            out[i] = project_state(pred[i], st[i], src, wa=wa, wk=wk, **kw)
    return out

def main(hd, td, tag):
    rep = dict(tag=tag)
    d = np.load(hd); fj = json.load(open(R + f'final_{tag}.json')); f = np.load(R + f'final_{tag}.npz')
    idx = f['idx']; pred = d['pred'][idx].astype(float); gt = d['gt'][idx].astype(float)
    lcf = d['ego_lcf'][idx].astype(float); sp = d['state_pred'][idx].astype(float); scene = d['scene'][idx]; cmd = d['mode'][idx]
    v = lcf[:, 7]; n = len(idx)
    Pm = proj_all(pred, lcf, 'meas', fj['cv_picks']['meas'], scene)
    Pe = proj_all(pred, sp, 'est', fj['cv_picks']['obs'], scene)
    E = {k: np.array([off(P[i], gt[i]) for i in range(n)]) for k, P in (('none', pred), ('meas_cv', Pm), ('est_cv', Pe))}
    rep['E'] = {k: float(x.mean()) for k, x in E.items()}
    # (2) decomposition
    dec = {}
    for k, P in (('none', pred), ('meas_cv', Pm), ('est_cv', Pe), ('gt_zero', np.zeros_like(gt))):
        lon, lat, ok = frenet(P, gt)
        dec[k] = dict(E_lon=float(agg(np.abs(lon)).mean()), E_lat=float(agg(np.abs(lat)).mean()),
                      lon_bias_3s=float(lon[:, 5].mean()), lon_abs_by_t=np.abs(lon).mean(0).round(4).tolist(),
                      lat_abs_by_t=np.abs(lat).mean(0).round(4).tolist())
        if k == 'none': dec['tangent_valid_frac'] = float(ok.mean())
    for k in ('meas_cv', 'est_cv'):
        dec[f'{k}_share_of_gain'] = dict(lon=dec['none']['E_lon'] - dec[k]['E_lon'], lat=dec['none']['E_lat'] - dec[k]['E_lat'])
    by_cmd = {}
    for c, name in ((0, 'LK'), (1, 'LCL'), (2, 'LCR'), (3, 'TL'), (4, 'TR')):
        m = cmd == c
        if m.sum() == 0: continue
        row = {}
        for k, P in (('none', pred), ('meas_cv', Pm)):
            lon, lat, _ = frenet(P[m], gt[m]); row[k] = [float(agg(np.abs(lon)).mean()), float(agg(np.abs(lat)).mean())]
        row['n'] = int(m.sum()); by_cmd[name] = row
    dec['by_cmd_[E_lon,E_lat]'] = by_cmd
    rep['decomposition'] = dec
    # (3) plan initial-speed error vs network state error
    vexp = v + lcf[:, 2] * TAU / 2                        # mean speed over first interval implied by measured v, a
    ev_plan = np.linalg.norm(pred[:, 0], axis=-1) / TAU - vexp
    ev_gt = np.linalg.norm(gt[:, 0], axis=-1) / TAU - vexp
    dv_net = sp[:, 5] - v; da_net = sp[:, 2] - lcf[:, 2]
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    rep['initial_speed'] = dict(std_ev_plan=float(ev_plan.std()), std_ev_gt=float(ev_gt.std()), std_dv_net=float(dv_net.std()),
                                mean_ev_plan=float(ev_plan.mean()), mean_ev_gt=float(ev_gt.mean()),
                                corr_evplan_dvnet=c(ev_plan, dv_net), corr_evplan_evgt=c(ev_plan, ev_gt),
                                corr_evplan_danet=c(ev_plan, da_net),
                                corr_abs_evplan_E_none=c(np.abs(ev_plan), E['none']),
                                corr_abs_evplan_gain_meas=c(np.abs(ev_plan), E['none'] - E['meas_cv']),
                                corr_v_gain_meas=c(v, E['none'] - E['meas_cv']))
    # gain by |alpha| bins
    al = np.abs(alpha(pred, v)); bins = [0, 0.5, 1, 2, 4, 1e9]; gb = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (al >= lo) & (al < hi)
        gb.append(dict(alpha=f'[{lo},{hi})', n=int(m.sum()), E_none=float(E['none'][m].mean()),
                       gain_meas=float((E['none'] - E['meas_cv'])[m].mean()), gain_est=float((E['none'] - E['est_cv'])[m].mean())))
    rep['gain_by_alpha'] = gb
    vb = [0, 2, 5, 10, 15, 100]; gv = []
    for lo, hi in zip(vb[:-1], vb[1:]):
        m = (v >= lo) & (v < hi)
        gv.append(dict(v=f'[{lo},{hi})', n=int(m.sum()), E_none=float(E['none'][m].mean()), frac_alpha_gt2=float((al[m] > 2).mean()),
                       gain_meas=float((E['none'] - E['meas_cv'])[m].mean()), gain_est=float((E['none'] - E['est_cv'])[m].mean())))
    rep['gain_by_speed'] = gv
    # (4) train vs held-out consistency
    t = np.load(td); ti = np.where(t['mode'] <= 4)[0]
    tp, tg, tl = t['pred'][ti].astype(float), t['gt'][ti].astype(float), t['ego_lcf'][ti].astype(float)
    Et = np.array([off(tp[i], tg[i]) for i in range(len(ti))])
    rep['train_vs_heldout'] = dict(
        train=dict(n=int(len(ti)), E_none=float(Et.mean()), frac_alpha_gt2_plan=float((np.abs(alpha(tp, tl[:, 7])) > 2).mean()),
                   frac_alpha_gt2_gt=float((np.abs(alpha(tg, tl[:, 7])) > 2).mean()),
                   std_ev_plan=float((np.linalg.norm(tp[:, 0], axis=-1) / TAU - tl[:, 7] - tl[:, 2] * TAU / 2).std())),
        heldout=dict(n=n, E_none=float(E['none'].mean()), frac_alpha_gt2_plan=float((al > 2).mean()),
                     frac_alpha_gt2_gt=float((np.abs(alpha(gt, v)) > 2).mean()), std_ev_plan=float(ev_plan.std())))
    # (10) projection-theorem check with the unregularized projection from the measured state
    P0 = np.stack([project_state(pred[i], lcf[i], 'meas', wa=0.0, wk=0.0, ws=0.0) for i in range(n)])
    G0 = np.stack([project_state(gt[i], lcf[i], 'meas', wa=0.0, wk=0.0, ws=0.0) for i in range(n)])
    fro = lambda A, B: np.sqrt(((A - B) ** 2).sum((-1, -2)))
    a_, b_, c_, dgt = fro(pred, gt), fro(P0, gt), fro(pred, P0), fro(G0, gt)
    gain2 = a_ ** 2 - b_ ** 2
    rep['theory_check_unregularized'] = dict(
        E_none=float(E['none'].mean()), E_pure=float(np.mean([off(P0[i], gt[i]) for i in range(n)])),
        frac_gain2_ge_corr2_all=float((gain2 >= c_ ** 2 - 1e-9).mean()),
        frac_gain2_ge_corr2_gt_in_R=float((gain2[dgt < 0.05] >= c_[dgt < 0.05] ** 2 - 1e-9).mean()), n_gt_in_R=int((dgt < 0.05).sum()),
        frac_triangle_bound=float((b_ <= a_ + dgt + 1e-9).mean()),
        corr_gain2_corr2=c(gain2, c_ ** 2), median_dgt=float(np.median(dgt)), mean_dgt=float(dgt.mean()),
        gain_by_dgt=[dict(d=f'[{lo},{hi})', n=int(((dgt >= lo) & (dgt < hi)).sum()),
                          mean_gain_fro=float((a_ - b_)[(dgt >= lo) & (dgt < hi)].mean()),
                          frac_harm=float(((b_ > a_)[(dgt >= lo) & (dgt < hi)]).mean()))
                     for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1), (1, 1e9)) if ((dgt >= lo) & (dgt < hi)).sum() > 0])
    np.savez_compressed(R + f'deep_{tag}.npz', P_meas_cv=Pm, P_est_cv=Pe, P_pure=P0, G_pure=G0, idx=idx)
    # (14) oracle per-window weight
    grid = [(wa, wk) for wa in (0.0, 0.05, 0.2, 0.5, 1, 2, 5, 10, 20, 50) for wk in (0.2, 1, 5, 20)]
    orc = {}
    for src, st in (('meas', lcf), ('est', sp)):
        Eg = np.stack([[off(project_state(pred[i], st[i], src, wa=wa, wk=wk), gt[i]) for i in range(n)] for wa, wk in grid], 1)
        best_fixed = Eg.mean(0).argmin()
        orc[src] = dict(fixed_best=list(grid[best_fixed]), E_fixed=float(Eg[:, best_fixed].mean()), E_oracle=float(Eg.min(1).mean()),
                        E_oracle_incl_none=float(np.minimum(Eg.min(1), E['none']).mean()), E_none=float(E['none'].mean()),
                        corr_absalpha_bestwa=c(al, np.log10(np.array([grid[k][0] for k in Eg.argmin(1)]) + 0.01)))
        # simple inference-time gate: project only if |alpha| > thr (thr chosen by scene CV)
        us0 = np.unique(scene); folds = np.array_split(np.random.default_rng(0).permutation(us0), 5); cv = np.empty(n)
        thrs = [0, 0.25, 0.5, 1, 1.5, 2, 3]; Ef = Eg[:, best_fixed]; picks = []
        for fo in folds:
            te = np.isin(scene, fo)
            sc = [np.where(al[~te] > th, Ef[~te], E['none'][~te]).mean() for th in thrs]; th = thrs[int(np.argmin(sc))]; picks.append(th)
            cv[te] = np.where(al[te] > th, Ef[te], E['none'][te])
        orc[src]['alpha_gate_cv'] = dict(E=float(cv.mean()), thr_picks=picks)
    rep['oracle_weight'] = orc
    # (6/7) transmission of state perturbations through the implemented projection (measured state, CV weights)
    wa, wk = fj['cv_picks']['meas'][0]; sub = np.random.default_rng(1).choice(n, 800, replace=False); tr = {}
    for name, col, dlt in (('v', 7, 0.1), ('a', 2, 0.2), ('omega', 4, 0.01)):
        st2 = lcf.copy(); st2[:, col] += dlt
        if name == 'v': st2[:, 0] += dlt
        dE = [off(project_state(pred[i], st2[i], 'meas', wa=wa, wk=wk), project_state(pred[i], lcf[i], 'meas', wa=wa, wk=wk)) / dlt for i in sub]
        tr[name] = dict(delta=dlt, transmission_mean=float(np.mean(dE)), transmission_median=float(np.median(dE)))
    lin = np.zeros((6, 2)); tr['model_v_straight'] = float(off(np.c_[TT, 0 * TT], lin))
    tr['model_a_straight'] = float(off(np.c_[TT ** 2 / 2, 0 * TT], lin))
    tr['model_omega_at_v'] = {f'{vv}': float(off(np.c_[0 * TT, vv * TT ** 2 / 2], lin)) for vv in (5, 10, 15)}
    rep['transmission'] = tr
    json.dump(rep, open(R + f'deep_{tag}.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))

if __name__ == '__main__':
    main(*sys.argv[1:4])
