"""(A) Split of the squared-error change into correction and alignment terms for each corrected plan,
     per planner, command and d_Y bin; alignment further split into along-track / cross-track parts.
(B) Arc-length comparison: path-shape error (distance of each plan waypoint to the recorded path)
     and timing error (arc-length difference along the recorded path) for learned and projected plans.
usage: split_terms.py <heldout_dump> <tag>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from smoothing_baseline import off
from deep_analysis import frenet, boot
R = __file__.rsplit('/', 1)[0] + '/results/'

def agg(x):
    return np.mean([x[:, :2].mean(1), x[:, :4].mean(1), x.mean(1)], 0)

def tangents(gt):
    Y = np.concatenate([np.zeros((len(gt), 1, 2)), gt], 1); tan = np.empty_like(gt)
    for j in range(6):
        tan[:, j] = Y[:, min(j + 2, 6)] - Y[:, j]
    n = np.linalg.norm(tan, axis=-1, keepdims=True)
    return np.where(n > 0.05, tan / np.maximum(n, 1e-9), np.array([1.0, 0.0]))

def path_proj(pts, poly):
    """closest point of each pt (k,2) on polyline poly (m,2): returns distance and arc length"""
    seg = poly[1:] - poly[:-1]; L = np.linalg.norm(seg, axis=1); s0 = np.r_[0, np.cumsum(L)][:-1]
    out_d = np.empty(len(pts)); out_s = np.empty(len(pts))
    for i, p in enumerate(pts):
        t = np.clip(((p - poly[:-1]) * seg).sum(1) / np.maximum(L ** 2, 1e-12), 0, 1)
        q = poly[:-1] + t[:, None] * seg; dist = np.linalg.norm(p - q, axis=1); k = dist.argmin()
        out_d[i] = dist[k]; out_s[i] = s0[k] + t[k] * L[k]
    return out_d, out_s

def arclength(P, gt):
    """path error (unsigned distance to the recorded path) and timing error (signed arc-length difference)"""
    n = len(gt); dpath = np.empty((n, 6)); dtime = np.empty((n, 6))
    for i in range(n):
        Y = np.vstack([[0, 0], gt[i]]); seg = np.diff(Y, axis=0); ok = np.linalg.norm(seg, axis=1) > 1e-3
        if ok.sum() == 0:
            u_end = np.array([1.0, 0]); u_start = u_end
        else:
            u_end = seg[np.where(ok)[0][-1]]; u_end = u_end / np.linalg.norm(u_end)
            u_start = seg[np.where(ok)[0][0]]; u_start = u_start / np.linalg.norm(u_start)
        poly = np.vstack([-30 * u_start, Y, Y[-1] + 60 * u_end])     # extended at both ends
        d1, s1 = path_proj(P[i], poly); d0, s0 = path_proj(gt[i], poly)
        dpath[i] = d1; dtime[i] = s1 - s0
    return dpath, dtime

def main(dump, tag):
    d = np.load(dump); f = np.load(R + f'final_{tag}.npz'); dp = np.load(R + f'deep_{tag}.npz')
    idx = f['idx']; P = d['pred'][idx].astype(float); Y = d['gt'][idx].astype(float); cmd = d['mode'][idx]; scene = d['scene'][idx]
    tan = tangents(Y); nrm = np.stack([-tan[..., 1], tan[..., 0]], -1)
    dY = np.sqrt(((dp['G_pure'] - Y) ** 2).sum((-1, -2)))
    EP = np.array([off(P[i], Y[i]) for i in range(len(P))])
    refs = dict(meas_cv=dp['P_meas_cv'], est_cv=dp['P_est_cv'], meas_fixed=f['proj_meas'], est_fixed=f['proj_obs'], meas_unreg=dp['P_pure'])
    rep = dict(tag=tag, n=int(len(P)))
    split = {}
    for k, Q in refs.items():
        D = ((P - Y) ** 2).sum((-1, -2)) - ((Q - Y) ** 2).sum((-1, -2))
        C = ((P - Q) ** 2).sum((-1, -2)); A = D - C
        PQ, QY = P - Q, Q - Y
        A_lon = 2 * ((PQ * tan).sum(-1) * (QY * tan).sum(-1)).sum(-1); A_lat = 2 * ((PQ * nrm).sum(-1) * (QY * nrm).sum(-1)).sum(-1)
        EQ = np.array([off(Q[i], Y[i]) for i in range(len(P))])
        def summ(m):
            return dict(n=int(m.sum()), mean_gain_sq=float(D[m].mean()), mean_corr=float(C[m].mean()), mean_align=float(A[m].mean()),
                        mean_align_lon=float(A_lon[m].mean()), mean_align_lat=float(A_lat[m].mean()),
                        frac_align_neg=float((A[m] < 0).mean()), frac_harm_sq=float((D[m] < 0).mean()),
                        frac_harm_Eplan=float((EQ[m] > EP[m]).mean()), mean_gain_Eplan=float((EP - EQ)[m].mean()),
                        median_corr=float(np.median(C[m])), median_align=float(np.median(A[m])))
        s = dict(all=summ(np.ones(len(P), bool)))
        s['sign_agreement_sq_vs_Eplan'] = float((np.sign(D) == np.sign(EP - EQ)).mean())
        s['by_cmd'] = {nm: summ(cmd == c) for c, nm in ((0, 'LK'), (1, 'LCL'), (2, 'LCR'), (3, 'TL'), (4, 'TR')) if (cmd == c).sum()}
        s['by_dY'] = {f'[{lo},{hi})': summ((dY >= lo) & (dY < hi)) for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1), (1, 1e9)) if ((dY >= lo) & (dY < hi)).sum()}
        s['ci_mean_align'] = boot(A, scene); s['ci_mean_corr'] = boot(C, scene)
        split[k] = s
    rep['split'] = split
    arc = {}
    for k, Q in (('none', P), ('meas_cv', refs['meas_cv']), ('est_cv', refs['est_cv']), ('meas_fixed', refs['meas_fixed'])):
        dpath, dtime = arclength(Q, Y)
        row = dict(E_path=float(agg(dpath).mean()), E_timing=float(agg(np.abs(dtime)).mean()), timing_bias_3s=float(dtime[:, 5].mean()),
                   path_by_t=dpath.mean(0).round(4).tolist())
        row['by_cmd'] = {nm: [float(agg(dpath[cmd == c]).mean()), float(agg(np.abs(dtime[cmd == c])).mean())]
                         for c, nm in ((0, 'LK'), (1, 'LCL'), (2, 'LCR'), (3, 'TL'), (4, 'TR')) if (cmd == c).sum()}
        arc[k] = row; arc[k + '_dpath'] = None
        np.save(R + f'arc_{tag}_{k}_dpath.npy', dpath)
    arc = {k: v for k, v in arc.items() if v is not None}
    for k in ('meas_cv', 'est_cv'):
        a0 = np.load(R + f'arc_{tag}_none_dpath.npy'); a1 = np.load(R + f'arc_{tag}_{k}_dpath.npy')
        arc[k]['ci_change_E_path'] = boot(agg(a1) - agg(a0), scene)
    rep['arclength'] = arc
    json.dump(rep, open(R + f'split_{tag}.json', 'w'), indent=1); print(json.dumps(rep, indent=1)[:6000])

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
