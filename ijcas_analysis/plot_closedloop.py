"""fig_example.pdf: two held-out windows driven by the same controller with the learned plan and with
the projected plan (measured state, weights from scene cross-validation). Windows are chosen by rule, not by gain:
among windows whose learned plan starts >1.5 m/s away from the measured speed, the one with the
median closed-loop change, for lane keeping and for a lane change.
usage: plot_closedloop.py <heldout_dump.npz> <tag>"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import closedloop_sim as cl
from kinematics import quantities
A = __file__.rsplit('/', 1)[0]; R = A + '/results/'; F = A + '/../figures/'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6, 'legend.fontsize': 7})
dump, tag = sys.argv[1], sys.argv[2]
d = np.load(dump); idx = np.where(d['mode'] <= 4)[0]
pred = d['pred'][idx].astype(float); gt = d['gt'][idx].astype(float); lcf = d['ego_lcf'][idx].astype(float); cmd = d['mode'][idx]
Pp = np.load(R + f'deep_{tag}.npz')['P_meas_cv']; g = json.load(open(R + 'closedloop_gains.json'))['gains']; x0 = cl.initial(lcf)
m0, pos0, a0, _ = cl.metrics(pred, gt, x0, g); m1, pos1, a1, _ = cl.metrics(Pp, gt, x0, g)
mis = np.abs(quantities(pred)['v'][:, 0] - lcf[:, 7]) > 1.5
dE = m1['E_exec'] - m0['E_exec']
def median_pick(mask):
    c = np.where(mask & mis)[0]; return int(c[np.argsort(dE[c])[len(c) // 2]]), len(c)
picks = [median_pick(cmd == 0), median_pick((cmd == 1) | (cmd == 2))]
t = cl.TT; tw = np.arange(6) * 0.5 + 0.25
def speed(pos):
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / cl.DT
    k = 20; vp = np.pad(v, (k // 2, k - 1 - k // 2), mode='edge'); return np.convolve(vp, np.ones(k) / k, mode='valid')
fig, ax = plt.subplots(3, 2, figsize=(7.0, 5.6))
tw6 = np.arange(1, 7) * 0.5; wi = (tw6 / cl.DT).round().astype(int)
for j, ((k, nc), title) in enumerate(zip(picks, ('Lane keeping', 'Lane change'))):
    a = ax[0, j]
    Yk = np.vstack([[0, 0], gt[k]])
    a.plot(Yk[:, 0], Yk[:, 1], '^:', color='0.55', ms=3.2, lw=0.9, mfc='white', label='Recorded driving')
    a.plot(pos0[k][:, 0], pos0[k][:, 1], '-', color='0.0', lw=1.6, alpha=0.5, label='Executed, learned plan')
    a.plot(pos1[k][:, 0], pos1[k][:, 1], '--', color='0.35', lw=1.6, alpha=0.7, label='Executed, projected plan')
    a.plot(pos0[k][wi, 0], pos0[k][wi, 1], 'o', color='0.0', ms=3.2, mfc='white')
    a.plot(pos1[k][wi, 0], pos1[k][wi, 1], 's', color='0.35', ms=3.2, mfc='white')
    a.set_xlabel('x (m)'); a.set_ylabel('y (m)'); a.set_title(title, fontsize=8)
    b = ax[1, j]
    b.plot(tw, quantities(pred[k][None])['v'][0], 'o-', color='0.0', ms=3.2, lw=0.9, mfc='white', label='Learned plan (reference)')
    b.plot(t[1:], speed(pos0[k]), '-', color='0.0', lw=1.6, alpha=0.5)
    b.plot(tw, quantities(Pp[k][None])['v'][0], 's--', color='0.35', ms=3.2, lw=0.9, mfc='white', label='Projected plan (reference)')
    b.plot(t[1:], speed(pos1[k]), '--', color='0.35', lw=1.6, alpha=0.7)
    b.plot(tw, quantities(gt[k][None])['v'][0], '^:', color='0.55', ms=3.2, lw=0.9, mfc='white')
    b.plot([0], [lcf[k, 7]], 'x', ms=6, color='0.0', label=r'Measured speed $v_0$')
    b.set_ylabel('Speed (m/s)')
    c = ax[2, j]
    e0 = np.linalg.norm(pos0[k][wi] - gt[k], axis=1); e1 = np.linalg.norm(pos1[k][wi] - gt[k], axis=1)
    c.plot(tw6, e0, 'o-', color='0.0', ms=3.2, lw=1.0, mfc='white'); c.plot(tw6, e1, 's--', color='0.35', ms=3.2, lw=1.0, mfc='white')
    c.set_xlabel('Time (s)'); c.set_ylabel('Position error (m)'); c.set_xlim(-0.1, 3.1); b.set_xlim(-0.1, 3.1)
h0, l0 = ax[0, 0].get_legend_handles_labels(); h1, l1 = ax[1, 0].get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, loc='lower center', ncol=3, frameon=False, fontsize=6.8)
for a, s_ in zip(ax.flat, ('(a)', '(b)', '(c)', '(d)', '(e)', '(f)')): a.text(-0.16, 1.02, s_, transform=a.transAxes)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(F + 'fig_example.pdf'); fig.savefig(F + 'fig_example.png', dpi=200); plt.close(fig)
info = dict(picks=[int(idx[k]) for k, _ in picks], candidates=[nc for _, nc in picks],
            E_exec_plan=[float(m0['E_exec'][k]) for k, _ in picks], E_exec_proj=[float(m1['E_exec'][k]) for k, _ in picks])
json.dump(info, open(R + f'fig_example_{tag}.json', 'w'), indent=1); print(info)
