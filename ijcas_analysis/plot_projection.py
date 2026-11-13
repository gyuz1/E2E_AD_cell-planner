"""Figures for the projection paper (grayscale-safe: markers / line styles).
 fig_sensitivity.pdf : E_plan vs speed-noise sigma and vs speed scale (measured state perturbed)
 fig_consistency.pdf : (a) implied initial acceleration distribution, (b) gain vs plan-model residual quintile
 fig_example.pdf     : two held-out windows, plan / projected / ground truth
"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from kinematics import quantities
A = __file__.rsplit('/', 1)[0]; R = A + '/results/'; F = A + '/../figures/'
dump = sys.argv[1]
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6, 'legend.fontsize': 7})
pa = json.load(open(R + 'projection_analysis_s5.json'))
# --- sensitivity: (a) E vs speed noise for several rate weights, (b) E vs speed scale, a-priori and CV weights
wn = json.load(open(R + 'weight_vs_noise_cell105.json')); sc = json.load(open(R + 'sens_cvweights_cell105.json'))
fin = json.load(open(R + 'final_cell105.json')); cvj = json.load(open(R + 'weight_cv_cell105.json'))
none = fin['E']['none']; obs_rmse = fin['observer']['speed_rmse']; obs_scale = fin['observer']['speed_scale']
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
mk = {0.05: ('o', '-', '0.0'), 0.5: ('s', '--', '0.25'), 2.0: ('^', '-.', '0.45'), 5.0: ('D', ':', '0.6')}
S = wn['sigmas']
for w, (m_, ls, c) in mk.items():
    ax[0].plot(S, [wn['table'][f'{s_},{w}'] for s_ in S], marker=m_, ls=ls, color=c, ms=3.2, lw=1, mfc='white', label=rf'$\rho_a={w:g}$')
ax[0].axhline(none, ls='--', lw=0.8, color='0.5', label='no projection')
ax[0].plot([obs_rmse], [fin['E']['proj_obs']], 'x', color='0.0', ms=6, mew=1.2, label=r'estimated state, $\rho_a=0.05$')
ax[0].plot([obs_rmse], [cvj['grid']['2.0,1.0']], '+', color='0.0', ms=8, mew=1.2, label=r'estimated state, $\rho_a=2$')
ax[0].set_xlabel(r'Speed noise $\sigma_v$ (m/s)'); ax[0].set_ylabel(r'$E_{\mathrm{plan}}$ (m)'); ax[0].set_ylim(0.2, 0.34)
ax[0].legend(frameon=False, fontsize=6.2, loc='upper left', ncol=2)
abl = json.load(open(R + 'projection_ablation_cell105.json'))
for data, lab, m_, ls, c in ((abl['sensitivity'], r'$\rho_a=0.05$', 'o', '-', '0.0'), (sc['sens'], r'$\rho_a=2$', '^', '-.', '0.45')):
    s0 = [r for r in data if r['kind'] == 'sigma' and r['value'] == 0][0]
    pts = sorted([(r['value'], r['E']) for r in data if r['kind'] == 'scale'] + [(1.0, s0['E'])])
    ax[1].plot([100 * (p_[0] - 1) for p_ in pts], [p_[1] for p_ in pts], marker=m_, ls=ls, color=c, ms=3.2, lw=1, mfc='white', label=lab)
ax[1].axhline(none, ls='--', lw=0.8, color='0.5', label='no projection')
ax[1].set_xlabel('Speed scale error (%)'); ax[1].set_ylim(0.2, 0.34); ax[1].legend(frameon=False, fontsize=6.5, loc='upper center')
for a_, t in zip(ax, ('(a)', '(b)')): a_.text(-0.12, -0.28, t, transform=a_.transAxes)
fig.tight_layout(); fig.savefig(F + 'fig_sensitivity.pdf'); fig.savefig(F + 'fig_sensitivity.png', dpi=200); plt.close(fig)
# --- consistency
d = np.load(dump); p = np.load(R + 'final_cell105.npz'); idx = p['idx']; v = d['ego_lcf'][idx, 7]; m = v > 2
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
bins = np.linspace(-6, 6, 49)
for P, lab, ls, c in ((d['pred'][idx], 'Learned plan', '-', '0.0'), (p['proj_obs'], 'Projected plan', '--', '0.35'), (d['gt'][idx], 'Ground truth', ':', '0.55')):
    a0 = (quantities(P)['v'][:, 0] - v) / 0.25
    h, _ = np.histogram(np.clip(a0[m], -6, 6), bins, density=True)
    ax[0].step(bins[:-1], h, where='post', ls=ls, color=c, lw=1.1, label=lab)
ax[0].set_yscale('log'); ax[0].set_xlabel(r'Implied initial acceleration (m/s$^2$)'); ax[0].set_ylabel('Density'); ax[0].legend(frameon=False)
res = np.sqrt(((p['proj_obs'] - d['pred'][idx]) ** 2).sum(-1).mean(-1)); dEr = p['E_proj_obs'] - p['E_none']
qs = np.quantile(res, np.linspace(0, 1, 6)); B = []
for lo, hi in zip(qs[:-1], qs[1:]):
    mm = (res >= lo) & (res <= hi); B.append(dict(range=[lo, hi], dE_mean=float(dEr[mm].mean()), frac_hurt=float((dEr[mm] > 0).mean())))
json.dump(B, open(R + 'residual_bins_cell105.json', 'w'), indent=1)
x = np.arange(len(B)); ax[1].bar(x, [1e2 * b['dE_mean'] for b in B], color='0.6', edgecolor='0.0', lw=0.5, width=0.6)
ax[1].axhline(0, color='0.0', lw=0.6)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'{1e2*b["range"][0]:.1f}–{1e2*b["range"][1]:.1f}' for b in B], fontsize=6.5)
ax[1].set_xlabel(r'Plan–model residual quintile (cm)'); ax[1].set_ylabel(r'$\Delta E_{\mathrm{plan}}$ (cm)')
for a, t in zip(ax, ('(a)', '(b)')): a.text(0.02, -0.3, t, transform=a.transAxes)
fig.tight_layout(); fig.savefig(F + 'fig_consistency.pdf'); fig.savefig(F + 'fig_consistency.png', dpi=200); plt.close(fig)
# --- examples: speed profiles of the two largest gains (lane change left, lane keeping)
dE = p['E_proj_obs'] - p['E_none']; cmd = d['mode'][idx]
picks = [int(np.argmin(np.where(cmd == 1, dE, np.inf))), int(np.argmin(np.where(cmd == 0, dE, np.inf)))]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
tm = np.arange(6) * 0.5 + 0.25
for a, k, title in zip(ax, picks, ('Lane change left', 'Lane keeping')):
    i = idx[k]
    for P, lab, mk, ls, c in ((d['pred'][i], 'Learned plan', 'o', '-', '0.0'), (p['proj_obs'][k], 'Projected plan', 's', '--', '0.35'), (d['gt'][i], 'Ground truth', '^', ':', '0.55')):
        a.plot(tm, quantities(P[None])['v'][0], ls=ls, marker=mk, ms=3.5, lw=1, color=c, mfc='white', label=lab)
    a.plot([0], [d['state_pred'][i, 5]], 'D', ms=4.5, color='0.0', label=r'Vision-estimated speed $\hat v_0$')
    a.plot([0], [d['ego_lcf'][i, 7]], 'x', ms=6, color='0.0', label=r'Measured speed $v_0$')
    a.set_xlabel('Time (s)'); a.set_ylabel('Speed (m/s)')
    a.set_title(f'{title}: $E_{{\\mathrm{{plan}}}}$ {p["E_none"][k]:.2f} m $\\rightarrow$ {p["E_proj_obs"][k]:.2f} m', fontsize=7.5)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=6.8)
for a, t in zip(ax, ('(a)', '(b)')): a.text(-0.12, -0.28, t, transform=a.transAxes)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(F + 'fig_example.pdf'); fig.savefig(F + 'fig_example.png', dpi=200); plt.close(fig)
print('picked', [int(idx[k]) for k in picks], [float(dE[k]) for k in picks])
