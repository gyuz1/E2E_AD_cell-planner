"""Controller-in-the-loop evaluation of plan references (CPU only).

Each reference (6 waypoints at 0.5 s, ego frame) is handed to the same
trajectory-tracking controller, which drives a vehicle model that is
deliberately different from the kinematic model used by the projection:
dynamic single-track model with saturating linear tires, blended with a
kinematic model below 5 m/s, first-order actuator lags, actuator limits
and a steering-rate limit. The vehicle starts from the measured state.
The scene does not react (non-reactive replay): other road users are not
simulated, so this evaluates the planning-control interface only.

usage: closedloop_sim.py <dump.npz> <tag> [--tune]
"""
import sys, json, numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from smoothing_baseline import off
from fast_projection import project_state
R = __file__.rsplit('/', 1)[0] + '/results/'

# --- vehicle (generic mid-size passenger car; assumed, not identified) ---
P = dict(m=1800.0, Iz=3200.0, lf=1.30, lr=1.60, Cf=1.0e5, Cr=1.0e5, mu=0.9,
         tau_a=0.30, tau_d=0.10, a_min=-6.0, a_max=3.0, d_max=0.55, dd_max=0.60)
DT, T, CTRL = 0.005, 3.0, 4          # integration step, horizon, control every 4 steps (50 Hz)
NS = int(round(T / DT))
TT = np.arange(NS + 1) * DT


def references(W):
    """W (N,6,2) waypoints -> per-step reference p, v, a (N,NS+1,2) by a cubic spline in
    time through the origin and the waypoints (identical interface for every reference)."""
    knots = np.arange(7) * 0.5
    Y = np.concatenate([np.zeros((len(W), 1, 2)), W], 1)          # (N,7,2)
    cs = CubicSpline(knots, Y, axis=1, bc_type='not-a-knot')
    return cs(TT), cs(TT, 1), cs(TT, 2)


@njit(cache=True)
def deriv(s, a_cmd, d_cmd, prm):
    m, Iz, lf, lr, Cf, Cr, mu, tau_a, tau_d = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8]
    X, Y, psi, vx, vy, r, a, d = s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]
    L = lf + lr; g = 9.81
    vxe = max(vx, 0.5)
    af = d - np.arctan2(vy + lf * r, vxe); ar = -np.arctan2(vy - lr * r, vxe)
    Fzf = m * g * lr / L; Fzr = m * g * lf / L
    Fyf = min(max(Cf * af, -mu * Fzf), mu * Fzf); Fyr = min(max(Cr * ar, -mu * Fzr), mu * Fzr)
    dvx_dyn = a + r * vy - Fyf * np.sin(d) / m
    dvy_dyn = (Fyf * np.cos(d) + Fyr) / m - r * vx
    dr_dyn = (lf * Fyf * np.cos(d) - lr * Fyr) / Iz
    # kinematic model (low speed): relax vy, r to their kinematic values
    beta = np.arctan(lr * np.tan(d) / L); v = np.hypot(vx, vy)
    vy_k = v * np.sin(beta); r_k = v * np.cos(beta) * np.tan(d) / L
    w = min(max((vx - 3.0) / 2.0, 0.0), 1.0)
    dvx = w * dvx_dyn + (1 - w) * a
    dvy = w * dvy_dyn + (1 - w) * (vy_k - vy) / 0.05
    dr = w * dr_dyn + (1 - w) * (r_k - r) / 0.05
    out = np.empty(8)
    out[0] = vx * np.cos(psi) - vy * np.sin(psi); out[1] = vx * np.sin(psi) + vy * np.cos(psi); out[2] = r
    out[3] = dvx; out[4] = dvy; out[5] = dr
    out[6] = (a_cmd - a) / tau_a
    out[7] = 0.0                                                    # steering handled with rate limit below
    return out


@njit(cache=True)
def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


@njit(parallel=True, cache=True)
def simulate(pr, vr, ar, x0, prm, lim, gains):
    """pr,vr,ar (N,NS+1,2) reference; x0 (N,8) initial [X,Y,psi,vx,vy,r,a,delta].
    returns positions (N,NS+1,2), a_cmd requested (N,NC), a actual (N,NS+1),
    saturation flags (N,NC), steering rate (N,NS), lateral accel (N,NS+1)."""
    N, NT = pr.shape[0], pr.shape[1]
    NC = (NT - 1) // CTRL
    a_min, a_max, d_max, dd_max = lim[0], lim[1], lim[2], lim[3]
    kv, ks, ky, kpsi = gains[0], gains[1], gains[2], gains[3]
    L = prm[2] + prm[3]; tau_d = prm[8]
    pos = np.zeros((N, NT, 2)); areq = np.zeros((N, NC)); aact = np.zeros((N, NT))
    sat = np.zeros((N, NC)); ddot = np.zeros((N, NT - 1)); alat = np.zeros((N, NT))
    for i in prange(N):
        s = x0[i].copy(); ac = 0.0; dc = s[7]; c = 0
        for k in range(NT):
            pos[i, k, 0] = s[0]; pos[i, k, 1] = s[1]; aact[i, k] = s[6]; alat[i, k] = s[3] * s[5]
            if k == NT - 1:
                break
            if k % CTRL == 0:
                vxr, vyr = vr[i, k, 0], vr[i, k, 1]; sp_r = np.hypot(vxr, vyr)
                if sp_r > 0.3:
                    psir = np.arctan2(vyr, vxr)
                    kap = (vxr * ar[i, k, 1] - vyr * ar[i, k, 0]) / sp_r ** 3
                    at = (vxr * ar[i, k, 0] + vyr * ar[i, k, 1]) / sp_r
                else:
                    psir = s[2]; kap = 0.0; at = np.hypot(ar[i, k, 0], ar[i, k, 1])
                dx, dy = s[0] - pr[i, k, 0], s[1] - pr[i, k, 1]
                tx, ty = np.cos(psir), np.sin(psir)
                es = -(dx * tx + dy * ty); ey = -dx * ty + dy * tx
                v = np.hypot(s[3], s[4]); course = s[2] + np.arctan2(s[4], max(s[3], 0.1))
                epsi = wrap(course - psir)
                a_req = at + kv * (sp_r - v) + ks * es
                d_req = np.arctan(L * kap) - np.arctan(ky * ey / max(v, 3.0)) - kpsi * epsi
                areq[i, c] = a_req
                ac = min(max(a_req, a_min), a_max)
                dc = min(max(d_req, -d_max), d_max)
                sat[i, c] = 1.0 if (a_req < a_min or a_req > a_max or abs(d_req) > d_max) else 0.0
                c += 1
            # steering: first-order lag with a rate limit
            dd = (dc - s[7]) / tau_d
            dd = min(max(dd, -dd_max), dd_max)
            ddot[i, k] = dd
            # RK4 on the chassis + longitudinal actuator, steering held over the step
            k1 = deriv(s, ac, dc, prm); k2 = deriv(s + 0.5 * DT * k1, ac, dc, prm)
            k3 = deriv(s + 0.5 * DT * k2, ac, dc, prm); k4 = deriv(s + DT * k3, ac, dc, prm)
            s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            s[7] = s[7] + DT * dd
            if s[3] < 0.0:
                s[3] = 0.0
    return pos, areq, aact, sat, ddot, alat


def prm_vec():
    return np.array([P[k] for k in ('m', 'Iz', 'lf', 'lr', 'Cf', 'Cr', 'mu', 'tau_a', 'tau_d')])


def lim_vec(scale=1.0):
    return np.array([P['a_min'] * scale, P['a_max'] * scale, P['d_max'], P['dd_max']])


def initial(lcf):
    L = P['lf'] + P['lr']; vx = np.maximum(lcf[:, 0], 0.0)
    d0 = np.clip(np.arctan(L * lcf[:, 4] / np.maximum(vx, 1.0)), -P['d_max'], P['d_max'])
    z = np.zeros(len(lcf))
    return np.stack([z, z, z, vx, lcf[:, 1], lcf[:, 4], np.clip(lcf[:, 2], P['a_min'], P['a_max']), d0], 1)


def run(W, x0, gains, lim=None):
    pr, vr, ar = references(W)
    out = simulate(pr, vr, ar, x0, prm_vec(), lim_vec() if lim is None else lim, np.asarray(gains, float))
    return (pr,) + out


def metrics(W, gt, x0, gains, lim=None):
    pr, pos, areq, aact, sat, ddot, alat = run(W, x0, gains, lim)
    idx = (np.arange(1, 7) * 0.5 / DT).round().astype(int)
    ex = pos[:, idx]
    jerk = np.diff(aact, axis=1) / DT
    return dict(
        E_exec=np.array([off(ex[i], gt[i]) for i in range(len(gt))]),
        E_ref=np.array([off(W[i], gt[i]) for i in range(len(gt))]),
        track_rms=np.sqrt(((pos - pr) ** 2).sum(-1).mean(1)),
        track_max=np.sqrt(((pos - pr) ** 2).sum(-1)).max(1),
        a_req_max=np.abs(areq).max(1), a_req0=areq[:, 0],
        sat_frac=sat.mean(1),
        jerk_rms=np.sqrt((jerk ** 2).mean(1)), jerk_max=np.abs(jerk).max(1),
        steer_rate_rms=np.sqrt((ddot ** 2).mean(1)),
        alat_max=np.abs(alat).max(1),
    ), pos, aact, areq


def tune_gains(gt, x0, n=600, seed=0):
    """Controller gains chosen once, on the recorded (human) trajectories only."""
    sel = np.random.default_rng(seed).choice(len(gt), min(n, len(gt)), replace=False)
    best = None
    for kv in (0.5, 1.0, 2.0):
        for ks in (0.2, 0.5, 1.0):
            for ky in (0.5, 1.0, 2.0):
                for kp in (0.5, 1.0, 2.0):
                    m, *_ = metrics(gt[sel], gt[sel], x0[sel], (kv, ks, ky, kp))
                    sc = m['track_rms'].mean()
                    if best is None or sc < best[0]:
                        best = (sc, (kv, ks, ky, kp))
    return best


def main(dump, tag):
    d = np.load(dump); f = np.load(R + f'final_{tag}.npz'); rep0 = json.load(open(R + f'final_{tag}.json'))
    idx = f['idx']; gt = d['gt'][idx].astype(float); pred = d['pred'][idx].astype(float)
    lcf = d['ego_lcf'][idx].astype(float); sp = d['state_pred'][idx].astype(float); scene = d['scene'][idx]
    x0 = initial(lcf)
    gfile = R + 'closedloop_gains.json'
    try:
        gains = json.load(open(gfile))['gains']
    except FileNotFoundError:
        sc, gains = tune_gains(gt, x0); json.dump(dict(gains=gains, track_rms_gt=sc, source=tag), open(gfile, 'w'), indent=1)
    print('gains', gains)
    # tuned (CV) projection from the measured state, same folds as final_tables.py
    us0 = np.unique(scene); folds = np.array_split(np.random.default_rng(0).permutation(us0), 5)
    Pcv = np.empty_like(pred)
    for fo, (wa, wk) in zip(folds, rep0['cv_picks']['meas']):
        for i in np.where(np.isin(scene, fo))[0]:
            Pcv[i] = project_state(pred[i], lcf[i], 'meas', wa=wa, wk=wk)
    tt = np.arange(1, 7) * 0.5
    CA = lcf[:, None, 0:2] * tt[None, :, None] + 0.5 * lcf[:, None, 2:4] * tt[None, :, None] ** 2
    refs = dict(plan=pred, proj_obs=f['proj_obs'], proj_meas=f['proj_meas'], proj_meas_cv=Pcv, CA_meas=CA, gt=gt)
    rep = dict(tag=tag, gains=list(gains), vehicle=P, n=int(len(idx)), dt=DT, ctrl_hz=1 / (DT * CTRL))
    rng = np.random.default_rng(0); us = np.unique(scene); M = {}
    for k, W in refs.items():
        m, pos, aact, areq = metrics(W, gt, x0, gains)
        M[k] = m
        rep[k] = {q: float(v.mean()) for q, v in m.items() if q != 'a_req0'}
        rep[k]['a_req0_abs_jump'] = float(np.abs(m['a_req0'] - x0[:, 6]).mean())
        rep[k]['frac_a_req_gt3'] = float((m['a_req_max'] > 3).mean())
        rep[k]['frac_sat_any'] = float((m['sat_frac'] > 0).mean())
        np.save(R + f'cl_{tag}_{k}_pos.npy', pos[:, ::20].astype(np.float32))
        np.save(R + f'cl_{tag}_{k}_aact.npy', aact[:, ::4].astype(np.float32))
    for k in ('proj_obs', 'proj_meas', 'proj_meas_cv', 'CA_meas'):
        diff = M[k]['E_exec'] - M['plan']['E_exec']; per = {s: diff[scene == s] for s in us}
        bs = [np.concatenate([per[s] for s in rng.choice(us, len(us))]).mean() for _ in range(2000)]
        rep[f'delta_exec_{k}'] = dict(mean=float(diff.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                      rel=float(diff.mean() / M['plan']['E_exec'].mean()))
    # robustness: actuator limits tightened to comfort values
    lim_c = np.array([-3.0, 2.0, P['d_max'], P['dd_max']])
    rep['comfort_limits'] = {k: float(metrics(refs[k], gt, x0, gains, lim_c)[0]['E_exec'].mean()) for k in ('plan', 'proj_meas', 'proj_meas_cv', 'gt')}
    json.dump(rep, open(R + f'closedloop_{tag}.json', 'w'), indent=1)
    for k in refs:
        r = rep[k]
        print(f"{k:13s} E_ref {r['E_ref']:.4f} E_exec {r['E_exec']:.4f} track {r['track_rms']:.3f} "
              f"jerk {r['jerk_rms']:.2f} a_req>3 {r['frac_a_req_gt3']*100:.1f}% sat {r['frac_sat_any']*100:.1f}% "
              f"|da0| {r['a_req0_abs_jump']:.2f} alat {r['alat_max']:.2f}")
    for k in ('proj_obs', 'proj_meas', 'proj_meas_cv', 'CA_meas'):
        print(k, rep[f'delta_exec_{k}'])
    print('comfort limits', rep['comfort_limits'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
