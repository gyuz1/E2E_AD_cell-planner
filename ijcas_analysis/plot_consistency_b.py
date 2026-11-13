"""fig_consistency.pdf rebuilt: (a) implied initial acceleration distribution (learned, projected from
the measured state with CV weights, recorded); (b) reduction of E_plan (positive = better) by the
measured-state CV projection, grouped by quintiles Q1-Q5 of the correction size ||P - Q||, with
scene-bootstrap 95% intervals of the mean, the median, and the share of windows made worse.
usage: plot_consistency_b.py <heldout_dump>"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from kinematics import quantities
from smoothing_baseline import off
from deep_analysis import boot
A = __file__.rsplit('/', 1)[0]; R = A + '/results/'; F = A + '/../figures/'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6, 'legend.fontsize': 7})
d = np.load(sys.argv[1]); p = np.load(R + 'final_cell105.npz'); dp = np.load(R + 'deep_cell105.npz'); idx = p['idx']
P = d['pred'][idx].astype(float); Y = d['gt'][idx].astype(float); v = d['ego_lcf'][idx, 7]; sc = d['scene'][idx]; Q = dp['P_meas_cv']
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
bins = np.linspace(-6, 6, 49); m = v > 2
for W, lab, ls, c in ((P, 'Learned plan', '-', '0.0'), (Q, 'Projected (measured, CV)', '--', '0.35'), (Y, 'Recorded driving', ':', '0.55')):
    a0 = (quantities(W)['v'][:, 0] - v) / 0.25
    h, _ = np.histogram(np.clip(a0[m], -6, 6), bins, density=True)
    ax[0].step(bins[:-1], h, where='post', ls=ls, color=c, lw=1.1, label=lab)
ax[0].set_yscale('log'); ax[0].set_xlabel(r'Implied initial acceleration $\alpha_t$ (m/s$^2$)'); ax[0].set_ylabel('Density'); ax[0].legend(frameon=False, fontsize=6.5)
corr = np.sqrt(((Q - P) ** 2).sum(-1).mean(-1))
gain = np.array([off(P[i], Y[i]) - off(Q[i], Y[i]) for i in range(len(P))])
qs = np.quantile(corr, np.linspace(0, 1, 6)); B = []
for k, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
    mm = (corr >= lo) & (corr <= hi) if k == 4 else (corr >= lo) & (corr < hi)
    B.append(dict(q=f'Q{k+1}', range_cm=[100 * lo, 100 * hi], n=int(mm.sum()), mean=float(gain[mm].mean()), ci=boot(gain[mm], sc[mm]),
                  median=float(np.median(gain[mm])), frac_worse=float((gain[mm] < 0).mean())))
json.dump(B, open(R + 'correction_quintiles_cell105.json', 'w'), indent=1)
x = np.arange(5); mean = np.array([b['mean'] for b in B]) * 100; ci = np.array([b['ci'] for b in B]) * 100
ax[1].bar(x, mean, color='0.75', edgecolor='0.0', lw=0.5, width=0.6, label='Mean (95% CI)')
ax[1].errorbar(x, mean, yerr=[mean - ci[:, 0], ci[:, 1] - mean], fmt='none', ecolor='0.0', lw=0.8, capsize=2)
ax[1].plot(x, [b['median'] * 100 for b in B], 'D', color='0.0', ms=3.5, mfc='white', label='Median')
for k, b in enumerate(B):
    ax[1].text(k, -2.2, f"{100*b['frac_worse']:.0f}%", ha='center', va='top', fontsize=6.5)
ax[1].text(-0.55, -2.2, 'worse:', ha='right', va='top', fontsize=6.5)
ax[1].axhline(0, color='0.0', lw=0.6); ax[1].set_xticks(x); ax[1].set_xticklabels([b['q'] for b in B])
ax[1].set_xlabel('Quintile of the correction size $\\|P-Q\\|$'); ax[1].set_ylabel('Reduction of $E_{\\mathrm{plan}}$ (cm)')
ax[1].set_ylim(-5, max(ci[:, 1].max() + 2, 5)); ax[1].legend(frameon=False, fontsize=6.5, loc='upper left')
for a, t in zip(ax, ('(a)', '(b)')): a.text(0.02, -0.32, t, transform=a.transAxes)
fig.tight_layout(); fig.savefig(F + 'fig_consistency.pdf'); fig.savefig(F + 'fig_consistency.png', dpi=200); plt.close(fig)
print(json.dumps(B, indent=0))
