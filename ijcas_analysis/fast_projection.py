"""Compiled kinematic projection: same problem as trackability_precheck.track
(same residuals, same box bounds) solved by projected Levenberg-Marquardt with an
exact Jacobian from forward sensitivity propagation, compiled with numba.
Checks agreement with the scipy solution and measures single-window latency."""
import sys, json, time, numpy as np, numba as nb
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from trackability_precheck import U
from smoothing_baseline import off
DT, SUB, H, TAU = 0.1, 5, 6, 0.5
LO = np.r_[np.full(H, U['a_min']), np.full(H, -U['kappa'])]
HI = np.r_[np.full(H, U['a_max']), np.full(H, U['kappa'])]
ALAT, JMAX = U['a_lat'], U['jerk']

@nb.njit(cache=True)
def resjac(u, Pref, v0, a0, k0, alat, jmax, wa=0.05, wk=1.0, ws=3.0, psi0=0.0):
    r = np.zeros(36); J = np.zeros((36, 12))
    x = 0.0; y = 0.0; psi = psi0; v = v0
    sx = np.zeros(12); sy = np.zeros(12); sp = np.zeros(12); sv = np.zeros(12)
    for j in range(H):
        a = u[j]; k = u[H + j]
        for _ in range(SUB):
            c = np.cos(psi); s = np.sin(psi)
            nx = x + DT * v * c; ny = y + DT * v * s
            nsx = sx + DT * (sv * c - v * s * sp); nsy = sy + DT * (sv * s + v * c * sp)
            npsi = psi + DT * v * k; nsp = sp + DT * sv * k; nsp[H + j] += DT * v
            nv = v + DT * a
            if nv > 0.0:
                nsv = sv.copy(); nsv[j] += DT
            else:
                nv = 0.0; nsv = np.zeros(12)
            x, y, psi, v = nx, ny, npsi, nv; sx, sy, sp, sv = nsx, nsy, nsp, nsv
        r[2 * j] = x - Pref[j, 0]; r[2 * j + 1] = y - Pref[j, 1]
        J[2 * j] = sx; J[2 * j + 1] = sy
    # soft lateral acceleration on planned speed profile vv_j = max(v0 + 0.5*cumsum(a), 0)
    cum = 0.0
    for j in range(H):
        cum += u[j]; vv = v0 + TAU * cum
        if vv > 0.0:
            lat = vv * vv * u[H + j]
            if abs(lat) > alat:
                sg = 1.0 if lat > 0 else -1.0
                r[12 + j] = ws * (abs(lat) - alat)
                J[12 + j, H + j] = ws * sg * vv * vv
                for m in range(j + 1):
                    J[12 + j, m] = ws * sg * 2.0 * vv * TAU * u[H + j]
    for j in range(H):
        prev = a0 if j == 0 else u[j - 1]
        da = u[j] - prev
        if abs(da) / TAU > jmax:
            sg = 1.0 if da > 0 else -1.0
            r[18 + j] = ws * (abs(da) / TAU - jmax)
            J[18 + j, j] = ws * sg / TAU
            if j > 0: J[18 + j, j - 1] = -ws * sg / TAU
        r[24 + j] = wa * da; J[24 + j, j] = wa
        if j > 0: J[24 + j, j - 1] = -wa
        pk = k0 if j == 0 else u[H + j - 1]
        r[30 + j] = wk * (u[H + j] - pk); J[30 + j, H + j] = wk
        if j > 0: J[30 + j, H + j - 1] = -wk
    return r, J, x

@nb.njit(cache=True)
def rollout(u, v0, psi0=0.0):
    P = np.zeros((H, 2)); x = 0.0; y = 0.0; psi = psi0; v = v0
    for j in range(H):
        for _ in range(SUB):
            x += DT * v * np.cos(psi); y += DT * v * np.sin(psi); psi += DT * v * u[H + j]
            v = max(v + DT * u[j], 0.0)
        P[j, 0] = x; P[j, 1] = y
    return P

@nb.njit(cache=True)
def solve(Pref, v0, a0, k0, lo, hi, alat, jmax, wa=0.05, wk=1.0, ws=3.0, iters=60, psi0=0.0):
    u = np.zeros(12)
    for i in range(12): u[i] = min(max(0.0, lo[i]), hi[i])
    r, J, _ = resjac(u, Pref, v0, a0, k0, alat, jmax, wa, wk, ws, psi0); cost = (r * r).sum(); mu = 1e-3
    for it in range(iters):
        g = J.T @ r; A = J.T @ J
        for i in range(12): A[i, i] += mu * (1.0 + A[i, i])
        d = np.linalg.solve(A, -g)
        un = u + d
        for i in range(12): un[i] = min(max(un[i], lo[i]), hi[i])
        rn, Jn, _ = resjac(un, Pref, v0, a0, k0, alat, jmax, wa, wk, ws, psi0); cn = (rn * rn).sum()
        if cn < cost:
            step = np.abs(un - u).max(); u, r, J = un, rn, Jn
            conv = cost - cn < 1e-10 * (1 + cost); cost = cn; mu = max(mu * 0.3, 1e-9)
            if conv or step < 1e-8: break
        else:
            mu *= 10.0
            if mu > 1e8: break
    return u, cost

def project(Pref, v0, a0, w, lo=LO, hi=HI, wa=0.05, wk=1.0, ws=3.0, psi0=0.0):
    k0 = w / max(v0, 2.0)
    u, _ = solve(np.ascontiguousarray(Pref, np.float64), float(v0), float(a0), float(k0), lo, hi, ALAT, JMAX, wa, wk, ws, 60, float(psi0))
    return rollout(u, float(v0), float(psi0))

def main(dump, tag):
    R = __file__.rsplit('/', 1)[0] + '/results/'
    d = np.load(dump); p = np.load(R + f'projection_{tag}.npz'); idx = p['idx']
    sp, lcf = d['state_pred'][idx], d['ego_lcf'][idx]; pred = d['pred'][idx].astype(float); gt = d['gt'][idx]
    project(pred[0], sp[0, 5], sp[0, 2], sp[0, 4])          # compile
    t0 = time.perf_counter(); Pf = np.stack([project(pred[i], sp[i, 5], sp[i, 2], sp[i, 4]) for i in range(len(idx))]); tt = time.perf_counter() - t0
    lat = []
    for i in range(0, len(idx), 7):
        t1 = time.perf_counter(); project(pred[i], sp[i, 5], sp[i, 2], sp[i, 4]); lat.append(time.perf_counter() - t1)
    Ef = np.array([off(Pf[i], gt[i]) for i in range(len(idx))])
    dev = np.sqrt(((Pf - p['proj_obs']) ** 2).sum(-1).mean(-1))
    rep = dict(n=int(len(idx)), E_fast=float(Ef.mean()), E_scipy=float(p['E_proj_obs'].mean()), E_none=float(p['E_none'].mean()),
               dev_vs_scipy_m=dict(mean=float(dev.mean()), p99=float(np.percentile(dev, 99)), max=float(dev.max())),
               latency_ms=dict(mean=1e3 * float(np.mean(lat)), p95=1e3 * float(np.percentile(lat, 95)), max=1e3 * float(np.max(lat))),
               total_s=tt, note='numba, single core, projected LM, exact Jacobian')
    print(json.dumps(rep, indent=1)); json.dump(rep, open(R + f'fast_projection_{tag}.json', 'w'), indent=1)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])


def project_state(Pref, st, src, course=True, **kw):
    """Projection from one state row: src='est' (state_pred: vx,vy,ax,ay,w,speed)
    or 'meas' (ego_lcf: vx,vy,ax,ay,w,L,W,speed,...). Initial heading = course
    angle of the velocity vector (direction of motion in the ego frame)."""
    vc = 5 if src == 'est' else 7
    psi0 = float(np.arctan2(st[1], max(st[0], 0.5))) if course else 0.0
    return project(Pref, max(float(st[vc]), 0.0), st[2], st[4], psi0=psi0, **kw)
