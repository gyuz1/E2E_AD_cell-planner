"""fig_qualitative.pdf: learned plan, projected plan (measured state, CV weights) and recorded driving,
drawn on the front camera image (top) and in bird's-eye view with the reachable set of the kinematic
model (bottom). Windows are chosen by rule, not by gain: among windows with v>5 m/s and |alpha|>2 m/s^2
of a command group, the one with the median change of E_plan.
usage: plot_qualitative.py <heldout_dump> <val_pkl> <image_root>"""
import sys, json, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import rollout
from trackability_precheck import U
from smoothing_baseline import off
from deep_analysis import alpha
A = __file__.rsplit('/', 1)[0]; R = A + '/results/'; F = A + '/../figures/'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6, 'legend.fontsize': 7})
dump, pkl, root = sys.argv[1:4]
d = np.load(dump); f = np.load(R + 'final_cell105.npz'); dp = np.load(R + 'deep_cell105.npz'); idx = f['idx']
P = d['pred'][idx].astype(float); Y = d['gt'][idx].astype(float); L = d['ego_lcf'][idx].astype(float); Q = dp['P_meas_cv']
cmd = d['mode'][idx]; sc = d['scene'][idx]; fr = d['frame'][idx]; v = L[:, 7]
gain = np.array([off(P[i], Y[i]) - off(Q[i], Y[i]) for i in range(len(P))]); al = np.abs(alpha(P, v))
groups = [('Lane keeping', cmd == 0), ('Lane change', (cmd == 1) | (cmd == 2)), ('Turn', (cmd == 3) | (cmd == 4))]
picks = []
for name, m in groups:
    c = np.where(m & (v > 5) & (al > 2))[0]
    if len(c) == 0: c = np.where(m & (v > 5))[0]
    k = c[np.argsort(gain[c])[len(c) // 2]]; picks.append((name, int(k), len(c)))
infos = {x['token']: x for x in pickle.load(open(pkl, 'rb'))['infos']}

def quat_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def project(cam, pts):
    Rm = quat_R(cam['sensor2ego_rotation']); t = np.array(cam['sensor2ego_translation'])
    pc = (pts - t) @ Rm                      # ego -> camera
    ok = pc[:, 2] > 1.0
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    k1, k2, p1, p2, k3 = np.array(cam['distortion'], float)
    r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = np.array(cam['cam_intrinsic_raw'], float)
    return np.c_[K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], ok

def dense(W):
    Wz = np.vstack([[0, 0], W]); t = np.arange(7); tt = np.linspace(0, 6, 61)
    return np.c_[np.interp(tt, t, Wz[:, 0]), np.interp(tt, t, Wz[:, 1])]

def reach(v0, psi0):
    pts = []
    for a in np.linspace(U['a_min'], U['a_max'], 9):
        for k in np.linspace(-U['kappa'], U['kappa'], 41):
            pts.append(rollout(np.r_[np.full(6, a), np.full(6, k)], float(v0), float(psi0)))
    return np.array(pts)                     # (n,6,2)

sty = dict(learned=('o-', '#c0392b', 'Learned plan'), proj=('s-', '#1f5fa8', 'Projected plan'), gt=('^--', '#2e7d32', 'Recorded driving'))
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw=dict(height_ratios=[1.0, 1.0]))
info_out = []
from kinematics import quantities
tw = np.arange(6) * 0.5 + 0.25
for j, (name, k, nc) in enumerate(picks):
    tok = f'{sc[k]}_{int(fr[k]):08d}'; inf = infos[tok]; cam = inf['cams']['camera_front']
    img = Image.open(f"{root}/{cam['data_path'].split('data/train/')[-1]}")
    W, H = img.size; a = ax[0, j]; a.imshow(img); a.set_xlim(W * 0.12, W * 0.88); a.set_ylim(H * 0.98, H * 0.42); a.axis('off')
    for key, Wp in (('gt', Y[k]), ('learned', P[k]), ('proj', Q[k])):
        m_, c_, lab = sty[key]
        pts = dense(Wp); uv, ok = project(cam, np.c_[pts, np.zeros(len(pts))])
        a.plot(uv[ok, 0], uv[ok, 1], '-' if key != 'gt' else '--', color=c_, lw=1.8)
        uvw, okw = project(cam, np.c_[Wp, np.zeros(6)]); a.plot(uvw[okw, 0], uvw[okw, 1], m_[0], color=c_, ms=4, mfc='white', mew=1.2)
    a.set_title(f'{name} ($v_0$={v[k]:.1f} m/s)', fontsize=8)
    b = ax[1, j]; tt = np.linspace(0, 3, 61)
    b.fill_between(tt, np.maximum(v[k] + U['a_min'] * tt, 0), v[k] + U['a_max'] * tt, color='0.88', lw=0, label='Reachable speed')
    for key, Wp in (('gt', Y[k]), ('learned', P[k]), ('proj', Q[k])):
        m_, c_, lab = sty[key]
        b.plot(tw, quantities(Wp[None])['v'][0], m_, color=c_, ms=3.8, mfc='white', lw=1.2, label=lab)
    b.plot([0], [v[k]], 'kx', ms=6, label=r'Measured speed $v_0$')
    b.set_xlabel('Time (s)'); b.set_ylabel('Speed (m/s)' if j == 0 else ''); b.set_xlim(-0.1, 3.0)
    vv = np.r_[quantities(P[k][None])['v'][0], quantities(Y[k][None])['v'][0], quantities(Q[k][None])['v'][0], v[k]]
    b.set_ylim(vv.min() - 1.5, vv.max() + 1.5)
    b.text(0.03, 0.97, f"$E_{{\\mathrm{{plan}}}}$ {off(P[k], Y[k]):.2f}$\\rightarrow${off(Q[k], Y[k]):.2f} m", transform=b.transAxes, va='top', fontsize=7)
    info_out.append(dict(name=name, token=tok, candidates=nc, v0=float(v[k]), alpha=float(al[k]), E_learned=float(off(P[k], Y[k])), E_proj=float(off(Q[k], Y[k]))))
h, l = ax[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(F + 'fig_qualitative.pdf', dpi=200); fig.savefig(F + 'fig_qualitative.png', dpi=200)
json.dump(info_out, open(R + 'fig_qualitative_cell105.json', 'w'), indent=1); print(info_out)
