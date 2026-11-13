"""Pre-check for a feasibility/trackability-aware selection topic.

Question: among the candidates of the commanded manoeuvre, is there one that the
vehicle can follow much more easily than the one the cell rule selects?  If not,
a trackability-aware selector has nothing to improve.

Tracking cost V_k: kinematic vehicle model (x, y, psi, v), inputs (a, kappa)
piecewise constant per 0.5 s, integrated at 0.1 s, z0 = (0, 0, 0, v0) where v0 is
the observer speed (planner-legal) or the measured speed (reference only).
Hard bounds on a and kappa (box constraints of U), soft penalties on lateral
acceleration and jerk, small input-rate term.  Reported V = RMS position error
(m) between the best trackable path and the candidate over the 6 waypoints.
"""
import sys, json, numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from kinematics import quantities, violations

U = json.load(open(__file__.rsplit('/', 1)[0] + '/results/feasibility_envelope.json'))['U']
DT, SUB, H = 0.1, 5, 6


def rollout(u, v0):
    a, k = u[:H], u[H:]
    x = y = psi = 0.0; v = v0; P = []
    for j in range(H):
        for _ in range(SUB):
            x += DT * v * np.cos(psi); y += DT * v * np.sin(psi)
            psi += DT * v * k[j]; v = max(v + DT * a[j], 0.0)
        P.append((x, y))
    return np.array(P)


def track(P_ref, v0, a0=0.0, k0=0.0):
    def res(u):
        P = rollout(u, v0)
        a, k = u[:H], u[H:]
        r_pos = (P - P_ref).ravel()
        vv = np.maximum(v0 + np.cumsum(a) * 0.5, 0)
        r_lat = 3.0 * np.maximum(np.abs(vv ** 2 * k) - U['a_lat'], 0)
        r_jerk = 3.0 * np.maximum(np.abs(np.diff(np.r_[a0, a])) / 0.5 - U['jerk'], 0)
        r_du = 0.05 * np.r_[np.diff(np.r_[a0, a]), 20 * np.diff(np.r_[k0, k])]
        return np.r_[r_pos, r_lat, r_jerk, r_du]
    lo = np.r_[np.full(H, U['a_min']), np.full(H, -U['kappa'])]
    hi = np.r_[np.full(H, U['a_max']), np.full(H, U['kappa'])]
    u0 = np.clip(np.zeros(2 * H), lo, hi)
    s = least_squares(res, u0, bounds=(lo, hi), max_nfev=200)
    P = rollout(s.x, v0)
    return float(np.sqrt(((P - P_ref) ** 2).sum(-1).mean())), P


def job(args):
    refs, v0, a0, k0 = args
    return [track(r, v0, a0, k0) for r in refs]


def main(path):
    d = np.load(path)
    mv = d['mode'] <= 4
    idx = np.where(mv)[0]
    sp = d['state_pred']; lcf = d['ego_lcf']
    jobs, meta = [], []
    for i in idx:
        cs = np.where(d['cand_cmd'][i] == d['mode'][i])[0]
        C = d['cand'][i, cs].astype(float)
        sel = int(np.argmin(np.abs(C - d['pred'][i]).max((1, 2))))
        for src, v0, a0, w in (('obs', sp[i, 5], sp[i, 2], sp[i, 4]), ('meas', lcf[i, 7], lcf[i, 2], lcf[i, 4])):
            k0 = w / max(v0, 2.0)
            jobs.append((np.concatenate([C, d['gt'][i][None]]), v0, a0, k0))
            meta.append((i, src, sel, cs))
    with ProcessPoolExecutor(14) as ex:
        out = list(ex.map(job, jobs, chunksize=4))
    rows = {'obs': [], 'meas': []}
    for (i, src, sel, cs), r in zip(meta, out):
        V = np.array([x[0] for x in r[:-1]]); Vgt = r[-1][0]
        C = d['cand'][i, cs].astype(float); gt = d['gt'][i]
        l2 = lambda P: np.linalg.norm(P - gt, axis=-1)
        def off(P):
            e = l2(P); return np.mean([e[:2].mean(), e[:4].mean(), e.mean()])
        kb = int(np.argmin(V))
        rows[src].append(dict(i=int(i), cmd=int(d['mode'][i]), n=len(cs), V_sel=V[sel], V_min=V[kb], V_gt=Vgt,
                              sel_is_min=kb == sel, l2_sel=off(C[sel]), l2_minV=off(C[kb]),
                              l2_tracked_sel=off(r[sel][1]), V_med=float(np.median(V))))
    # static envelope check, selected vs all candidates of command vs GT
    selP = np.stack([d['cand'][r['i'], np.where(d['cand_cmd'][r['i']] == d['mode'][r['i']])[0]][0] for r in rows['obs']])
    allC = np.concatenate([d['cand'][i, d['cand_cmd'][i] == d['mode'][i]] for i in idx]).astype(float)
    selC = np.stack([d['pred'][i] for i in idx])
    rep = {'n_moving_windows': int(len(idx)),
           'static_violation_rate': {'selected': float(violations(quantities(selC), U)['any'].mean()),
                                      'all_candidates_of_command': float(violations(quantities(allC), U)['any'].mean()),
                                      'gt': float(violations(quantities(d['gt'][idx]), U)['any'].mean())}}
    for src, R in rows.items():
        g = lambda k: np.array([r[k] for r in R])
        rep[src] = dict(V_sel_mean=g('V_sel').mean(), V_sel_p95=np.percentile(g('V_sel'), 95),
                        V_min_mean=g('V_min').mean(), V_gt_mean=g('V_gt').mean(), V_med_mean=g('V_med').mean(),
                        frac_sel_is_min=g('sel_is_min').mean(),
                        frac_gap_gt_0p1m=(g('V_sel') - g('V_min') > 0.1).mean(),
                        frac_V_sel_gt_0p2m=(g('V_sel') > 0.2).mean(),
                        l2_sel=g('l2_sel').mean(), l2_minV=g('l2_minV').mean(), l2_tracked_sel=g('l2_tracked_sel').mean(),
                        corr_Vsel_l2sel=float(np.corrcoef(g('V_sel'), g('l2_sel'))[0, 1]))
    rep = json.loads(json.dumps(rep, default=float))
    print(json.dumps(rep, indent=1))
    out = __file__.rsplit('/', 1)[0] + '/results/trackability_precheck_s50.json'
    json.dump(rep, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
