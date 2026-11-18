"""Demo: kinematic projection of the three held-out windows shown in Fig. 5 of the paper.

For each window the learned plan of the 105-cell planner is projected from the measured
state with the cross-validated weights (rho_a, rho_kappa) = (5, 5). The planning error
before and after projection is printed, and the plans are drawn on the front-camera image
together with their speed profiles (demo/output/demo.png).

usage: python demo/demo.py [--planner-output demo/output/cell105_3frame.npz]

By default the learned plans stored in demo/demo_windows.npz are used. With
--planner-output, the plans come from your own run of tools/evaluate.py on
demo/demo_infos.pkl (see README).
"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'analysis'))
from fast_projection import project_state
from trackability_precheck import U
from smoothing_baseline import off
from kinematics import quantities

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--planner-output', default=None, help='npz written by tools/evaluate.py --dump-npz')
args = ap.parse_args()
d = dict(np.load(os.path.join(HERE, 'demo_windows.npz')))
if args.planner_output:
    o = np.load(args.planner_output)
    for j in range(len(d['name'])):
        k = np.where((o['scene'] == d['scene'][j]) & (o['frame'] == d['frame'][j]))[0][0]
        for key in ('pred', 'gt', 'ego_lcf', 'state_pred'):
            d[key][j] = o[key][k]
cams = json.load(open(os.path.join(HERE, 'cameras.json')))


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def to_image(cam, pts):
    """Ego-frame ground points (n,2) -> pixel coordinates of the front camera."""
    p = np.c_[pts, np.zeros(len(pts))]
    pc = (p - np.array(cam['sensor2ego_translation'])) @ quat_R(cam['sensor2ego_rotation'])
    ok = pc[:, 2] > 1.0
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    k1, k2, p1, p2, k3 = cam['distortion']
    r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = np.array(cam['cam_intrinsic'])
    return np.c_[K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], ok


def dense(W):
    Wz = np.vstack([[0, 0], W]); t = np.arange(7); tt = np.linspace(0, 6, 61)
    return np.c_[np.interp(tt, t, Wz[:, 0]), np.interp(tt, t, Wz[:, 1])]


style = {'Recorded driving': (':', '^', '#3A7D44'), 'Learned plan': ('-', 'o', '#2F5597'),
         'Projected plan': ('--', 's', '#E46C0A')}
tw = np.arange(6) * 0.5 + 0.25
fig, ax = plt.subplots(3, 2, figsize=(8, 8), gridspec_kw=dict(width_ratios=[1.4, 1.0]))
for j in range(len(d['name'])):
    P, Y, s = d['pred'][j].astype(float), d['gt'][j].astype(float), d['ego_lcf'][j].astype(float)
    Q = project_state(P, s, 'meas', wa=5.0, wk=5.0)
    v0 = float(s[7])
    print(f"{d['name'][j]:13s} v0 = {v0:5.1f} m/s   E_plan: learned {off(P, Y):.3f} m -> projected {off(Q, Y):.3f} m")

    a = ax[j, 0]; img = plt.imread(os.path.join(HERE, 'images', str(d['image'][j])))
    cam = cams[str(d['image'][j])]; a.imshow(img); a.axis('off')
    H, W = img.shape[:2]; allu = []
    for lab, Wp in (('Recorded driving', Y), ('Learned plan', P), ('Projected plan', Q)):
        ls, mk, c = style[lab]
        uv, ok = to_image(cam, dense(Wp)); a.plot(uv[ok, 0], uv[ok, 1], ls, color=c, lw=2)
        uvw, okw = to_image(cam, Wp); a.plot(uvw[okw, 0], uvw[okw, 1], mk, color=c, ms=4, mfc='white')
        allu.append(uv[ok])
    U_ = np.vstack(allu); cw, ch = W * 0.5, H * 0.42
    cx = np.clip((U_[:, 0].min() + U_[:, 0].max()) / 2, cw / 2, W - cw / 2)
    y1 = min(H, max(U_[:, 1].max() + 0.04 * H, H * 0.6))
    a.set_xlim(cx - cw / 2, cx + cw / 2); a.set_ylim(y1, y1 - ch)
    a.set_title(f"{d['name'][j]}, $v_0$ = {v0:.1f} m/s", fontsize=9)

    b = ax[j, 1]; tt = np.linspace(0, 3, 61)
    b.fill_between(tt, np.maximum(v0 + U['a_min'] * tt, 0), v0 + U['a_max'] * tt, color='0.88', lw=0,
                   label='Reachable speed')
    for lab, Wp in (('Recorded driving', Y), ('Learned plan', P), ('Projected plan', Q)):
        ls, mk, c = style[lab]
        b.plot(tw, quantities(Wp[None])['v'][0], ls + mk, color=c, ms=4, mfc='white', label=lab)
    b.plot([0], [v0], 'kx', ms=6, label='Measured speed $v_0$')
    b.set_xlim(-0.1, 3.0); b.set_ylabel('Speed (m/s)')
    b.set_title(f"$E_{{plan}}$ {off(P, Y):.2f} $\\rightarrow$ {off(Q, Y):.2f} m", fontsize=9)
ax[-1, 1].set_xlabel('Time (s)')
h, l = ax[0, 1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
os.makedirs(os.path.join(HERE, 'output'), exist_ok=True)
fig.savefig(os.path.join(HERE, 'output', 'demo.png'), dpi=150)
print('saved demo/output/demo.png')
