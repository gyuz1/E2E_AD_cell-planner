"""fig_example.pdf: two held-out windows driven by the same controller with the learned plan and with
the projected plan (weights selected on training scenes). Windows are chosen by rule, not by gain:
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
Pp = np.load(R + f'proj_trainsel_{tag}.npy'); g = json.load(open(R + 'closedloop_gains.json'))['gains']; x0 = cl.initial(lcf)
m0, pos0, a0, _ = cl.metrics(pred, gt, x0, g); m1, pos1, a1, _ = cl.metrics(Pp, gt, x0, g)
mis = np.abs(quantities(pred)['v'][:, 0] - lcf[:, 7]) > 1.5
dE = m1['E_exec'] - m0['E_exec']
def median_pick(mask):
    c = np.where(mask & mis)[0]; return int(c[np.argsort(dE[c])[len(c) // 2]]), len(c)
picks = [median_pick(cmd == 0), median_pick((cmd == 1) | (cmd == 2))]
t = cl.TT; tw = np.arange(6) * 0.5 + 0.25
def speed(pos):
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / cl.DT
    k = 20; return np.convolve(v, np.ones(k) / k, mode='same')
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True)
for j, ((k, nc), title) in enumerate(zip(picks, ('Lane keeping', 'Lane change'))):
    a = ax[0, j]
    a.plot(tw, quantities(pred[k][None])['v'][0], 'o-', color='0.0', ms=3.2, lw=0.9, mfc='white', label='Learned plan (reference)')
    a.plot(t[1:], speed(pos0[k]), '-', color='0.0', lw=1.8, alpha=0.45, label='Executed, learned plan')
    a.plot(tw, quantities(Pp[k][None])['v'][0], 's--', color='0.35', ms=3.2, lw=0.9, mfc='white', label='Projected plan (reference)')
    a.plot(t[1:], speed(pos1[k]), '--', color='0.35', lw=1.8, alpha=0.45, label='Executed, projected plan')
    a.plot(tw, quantities(gt[k][None])['v'][0], '^:', color='0.55', ms=3.2, lw=0.9, mfc='white', label='Recorded driving')
    a.plot([0], [lcf[k, 7]], 'x', ms=6, color='0.0', label=r'Measured speed $v_0$')
    a.set_ylabel('Speed (m/s)')
    a.set_title(f'{title}: executed $E_{{\\mathrm{{plan}}}}$ {m0["E_exec"][k]:.2f} m $\\rightarrow$ {m1["E_exec"][k]:.2f} m, '
                f'tracking RMS {m0["track_rms"][k]*100:.0f} $\\rightarrow$ {m1["track_rms"][k]*100:.0f} cm', fontsize=7)
    b = ax[1, j]
    b.plot(t, a0[k], '-', color='0.0', lw=1.0, label='Executed, learned plan')
    b.plot(t, a1[k], '--', color='0.35', lw=1.0, label='Executed, projected plan')
    b.axhline(cl.P['a_max'], color='0.7', lw=0.6); b.axhline(cl.P['a_min'], color='0.7', lw=0.6)
    b.set_ylim(min(-3.5, a0[k].min() - 0.3, a1[k].min() - 0.3), max(3.3, a0[k].max() + 0.3))
    b.set_xlabel('Time (s)'); b.set_ylabel('Acceleration (m/s$^2$)')
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=6.8)
for a, s in zip(ax.flat, ('(a)', '(b)', '(c)', '(d)')): a.text(-0.13, 1.02, s, transform=a.transAxes)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(F + 'fig_example.pdf'); fig.savefig(F + 'fig_example.png', dpi=200); plt.close(fig)
info = dict(picks=[int(idx[k]) for k, _ in picks], candidates=[nc for _, nc in picks],
            E_exec_plan=[float(m0['E_exec'][k]) for k, _ in picks], E_exec_proj=[float(m1['E_exec'][k]) for k, _ in picks])
json.dump(info, open(R + f'fig_example_{tag}.json', 'w'), indent=1); print(info)
