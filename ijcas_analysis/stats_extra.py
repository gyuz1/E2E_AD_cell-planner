"""Uncertainty and control analyses requested in review (CPU only).
 1 scene-bootstrap CI of the abrupt-start share (v>2 m/s): learned vs recorded
 2 closed-loop metrics with CIs (tracking, jerk, saturation) incl. constant acceleration and selector
 3 gain distribution: share of the mean gain from |alpha|>=4, median gain, share of scenes improved
 4 ablation with bounds and penalties removed separately
 5 speed-noise sweep over 5 seeds with rho_k=5, scene CI of the change at the CV weight
 6 transmission per window with scene CI
 7 train vs held-out with matched sampling and command/speed reweighting
 8 per-command change of E_plan with scene CI
usage: stats_extra.py <heldout_dump> <train_dump> <tag> [parts]"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
from smoothing_baseline import off
from deep_analysis import boot, alpha
import closedloop_sim as cl
R = __file__.rsplit('/', 1)[0] + '/results/'
TAU = 0.5

def bootmean(x, scene, n=2000):
    return boot(x, scene, n)

def main(hd, td, tag, parts):
    d = np.load(hd); f = np.load(R + f'final_{tag}.npz'); fj = json.load(open(R + f'final_{tag}.json')); dp = np.load(R + f'deep_{tag}.npz')
    idx = f['idx']; P = d['pred'][idx].astype(float); Y = d['gt'][idx].astype(float); L = d['ego_lcf'][idx].astype(float)
    S = d['state_pred'][idx].astype(float); scene = d['scene'][idx]; cmd = d['mode'][idx]; v = L[:, 7]; n = len(P)
    EP = np.array([off(P[i], Y[i]) for i in range(n)]); Pm = dp['P_meas_cv']; Em = np.array([off(Pm[i], Y[i]) for i in range(n)])
    wa, wk = fj['cv_picks']['meas'][0]
    try:
        rep = json.load(open(R + f'stats_extra_{tag}.json'))
    except FileNotFoundError:
        rep = dict(tag=tag)
    if '1' in parts:
        m = v > 2; ap = (np.abs(alpha(P, v)) > 2).astype(float); ag = (np.abs(alpha(Y, v)) > 2).astype(float)
        rep['abrupt'] = dict(n=int(m.sum()), plan=float(ap[m].mean()), recorded=float(ag[m].mean()),
                             ci_plan=boot(ap[m], scene[m]), ci_recorded=boot(ag[m], scene[m]), ci_diff=boot((ap - ag)[m], scene[m]))
    if '2' in parts:
        x0 = cl.initial(L); g = json.load(open(R + 'closedloop_gains.json'))['gains']
        tt = np.arange(1, 7) * TAU
        CA = L[:, None, 0:2] * tt[None, :, None] + 0.5 * L[:, None, 2:4] * tt[None, :, None] ** 2
        refs = dict(plan=P, fixed=f['proj_meas'], cv=Pm, ca=CA)
        try:
            refs['selector'] = np.load(R + f'selector_{tag}_meas.npz')['Q_sel']
        except FileNotFoundError:
            pass
        M = {}
        for k, W in refs.items():
            mm, *_ = cl.metrics(W, Y, x0, g); M[k] = dict(E_exec=mm['E_exec'], track=mm['track_rms'], jerk=mm['jerk_rms'], sat=(mm['sat_frac'] > 0).astype(float))
        cr = {}
        for k in M:
            cr[k] = {q: dict(mean=float(M[k][q].mean()), ci=boot(M[k][q], scene)) for q in M[k]}
            if k != 'plan':
                cr[k]['diff_vs_plan'] = {q: dict(mean=float((M[k][q] - M['plan'][q]).mean()), ci=boot(M[k][q] - M['plan'][q], scene)) for q in M[k]}
        rep['closedloop_ci'] = cr
    if '3' in parts:
        al = np.abs(alpha(P, v)); gain = EP - Em; top = al >= 4
        per = np.array([gain[scene == s].mean() for s in np.unique(scene)])
        rep['gain_distribution'] = dict(mean=float(gain.mean()), median=float(np.median(gain)), frac_worse=float((gain < 0).mean()),
                                        share_from_alpha_ge4=float(gain[top].sum() / gain.sum()), n_alpha_ge4=int(top.sum()),
                                        mean_without_alpha_ge4=float(gain[~top].mean()), ci_without=boot(gain[~top], scene[~top]),
                                        frac_scenes_improved=float((per > 0).mean()), n_scenes=int(len(per)),
                                        pct=[float(np.percentile(gain, q)) for q in (10, 25, 50, 75, 90)])
    if '4' in parts:
        big = np.full(12, 1e3); ab = {}
        for src, st in (('est', S), ('meas', L)):
            row = {}
            for name, kw in (('full', {}), ('no_bounds_keep_penalties', dict(lo=-big, hi=big)), ('no_penalties_keep_bounds', dict(ws=0.0)),
                             ('no_bounds_no_penalties', dict(lo=-big, hi=big, ws=0.0))):
                Q = [project_state(P[i], st[i], src, wa=0.05, wk=1.0, **kw) for i in range(n)]
                row[name] = float(np.mean([off(Q[i], Y[i]) for i in range(n)]))
            ab[src] = row
        rep['ablation_split'] = ab
    if '5' in parts:
        W = (0.05, 0.2, 0.5, 1, 2, 5); SG = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5); seeds = range(5); tab = {}
        diffcv = {}
        for sg in SG:
            for sd in (seeds if sg > 0 else [0]):
                vv = v + np.random.default_rng(sd).normal(0, sg, n) if sg > 0 else v.copy()
                st = np.c_[L[:, :7], np.maximum(vv, 0), L[:, 8:]]
                for w_ in W:
                    E = np.array([off(project_state(P[i], st[i], 'meas', wa=w_, wk=5.0), Y[i]) for i in range(n)])
                    tab.setdefault(f'{sg},{w_}', []).append(float(E.mean()))
                    if w_ == 5: diffcv.setdefault(str(sg), []).append(E - EP)
        rep['noise'] = dict(table={k: dict(mean=float(np.mean(x)), sd=float(np.std(x)), nseed=len(x)) for k, x in tab.items()},
                            cv_weight_change={k: dict(mean=float(np.mean(x)), ci=boot(np.mean(x, 0), scene)) for k, x in diffcv.items()}, none=float(EP.mean()))
    if '6' in parts:
        sub = np.random.default_rng(1).choice(n, 800, replace=False); tr = {}
        for name, col, dl in (('v', 7, 0.1), ('a', 2, 0.2), ('omega', 4, 0.01)):
            st2 = L.copy(); st2[:, col] += dl
            if name == 'v': st2[:, 0] += dl
            x = np.array([off(project_state(P[i], st2[i], 'meas', wa=wa, wk=wk), project_state(P[i], L[i], 'meas', wa=wa, wk=wk)) / dl for i in sub])
            tr[name] = dict(mean=float(x.mean()), ci=boot(x, scene[sub]), p10=float(np.percentile(x, 10)), p50=float(np.median(x)), p90=float(np.percentile(x, 90)))
        rep['transmission_ci'] = tr
    if '7' in parts:
        t = np.load(td); ti = np.where(t['mode'] <= 4)[0]
        tp_, tg, tl, tc = t['pred'][ti].astype(float), t['gt'][ti].astype(float), t['ego_lcf'][ti].astype(float), t['mode'][ti]
        Et = np.array([off(tp_[i], tg[i]) for i in range(len(ti))]); at = (np.abs(alpha(tp_, tl[:, 7])) > 2).astype(float)
        ah = (np.abs(alpha(P, v)) > 2).astype(float); fr = d['frame'][idx]
        mstride = (fr % 10) == (fr.min() % 10)
        vb = np.digitize(v, [2, 5, 10, 15]); vbt = np.digitize(tl[:, 7], [2, 5, 10, 15])
        key_h = cmd * 10 + vb; key_t = tc * 10 + vbt
        w = np.zeros(n)
        for k in np.unique(key_h):
            pt = (key_t == k).mean(); ph = (key_h == k).mean(); w[key_h == k] = pt / ph if ph > 0 else 0
        rep['train_heldout_controlled'] = dict(
            train=dict(n=int(len(ti)), E=float(Et.mean()), abrupt=float(at.mean())),
            heldout_all=dict(n=n, E=float(EP.mean()), abrupt=float(ah.mean())),
            heldout_stride10=dict(n=int(mstride.sum()), E=float(EP[mstride].mean()), abrupt=float(ah[mstride].mean())),
            heldout_reweighted_to_train_cmd_speed=dict(E=float((w * EP).sum() / w.sum()), abrupt=float((w * ah).sum() / w.sum())),
            train_cmd_share={int(c): float((tc == c).mean()) for c in np.unique(tc)}, heldout_cmd_share={int(c): float((cmd == c).mean()) for c in np.unique(cmd)})
    if '8' in parts:
        pc = {}
        for c, nm in ((0, 'LK'), (1, 'LCL'), (2, 'LCR'), (3, 'TL'), (4, 'TR')):
            m = cmd == c
            if m.sum() == 0: continue
            pc[nm] = dict(n=int(m.sum()), n_scenes=int(len(np.unique(scene[m]))), none=float(EP[m].mean()), cv=float(Em[m].mean()),
                          ci_change=boot((Em - EP)[m], scene[m]))
        rep['per_command'] = pc
    json.dump(rep, open(R + f'stats_extra_{tag}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('noise',)}, indent=1)[:5000])

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else '12345678')
