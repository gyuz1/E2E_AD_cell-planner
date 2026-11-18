"""fig_tradeoff.pdf: executed error vs tracking RMS (left) and vs jerk RMS (right) for every reference
type and planner (controller-in-the-loop, measured state). Points from stats_extra_{tag}.json."""
import json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
A = __file__.rsplit('/', 1)[0]; R = A + '/results/'; F = A + '/../figures/'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6, 'legend.fontsize': 6.8})
ref = dict(plan=('Learned plan', '#2F5597', 'o'), fixed=('Projected, fixed', '#999999', 'v'), cv=('Projected, CV', '#E46C0A', 's'),
           selector=('Projected, selector', '#8E3B8E', 'D'), ca=('Const. acceleration', '#3A7D44', '^'))
fill = dict(cell105='full', cell15='left', nocell='none')
fig, ax = plt.subplots(2, 1, figsize=(3.45, 5.0))
for t, lab in (('cell105', '105-cell'), ('cell15', '15-cell'), ('nocell', 'No-cell')):
    c = json.load(open(R + f'stats_extra_{t}.json'))['closedloop_ci']
    for k, (name, col, mk) in ref.items():
        for j, q in enumerate(('track', 'jerk')):
            ax[j].plot(c[k][q]['mean'], c[k]['E_exec']['mean'], mk, color=col, ms=6, fillstyle=fill[t], mec=col, mew=1.0)
    for j, q in enumerate(('track', 'jerk')):
        ax[j].plot([c['plan'][q]['mean'], c['cv'][q]['mean']], [c['plan']['E_exec']['mean'], c['cv']['E_exec']['mean']], '-', color='0.75', lw=0.6, zorder=0)
h = [plt.Line2D([], [], marker=mk, ls='', color=col, ms=6, label=name) for name, col, mk in ref.values()]
h += [plt.Line2D([], [], marker='o', ls='', color='0.3', ms=6, fillstyle=f, label=l) for l, f in (('105-cell', 'full'), ('15-cell', 'left'), ('No-cell', 'none'))]
ax[0].set_xlabel('Tracking error RMS (m)'); ax[1].set_xlabel('Jerk RMS (m/s$^3$)')
for a in ax: a.set_ylabel('Executed error (m)'); a.grid(alpha=0.25, lw=0.5)
ax[0].text(0.97, 0.05, 'easier to track $\\leftarrow$', transform=ax[0].transAxes, ha='right', fontsize=7, color='0.35')
ax[0].text(0.03, 0.95, '$\\downarrow$ more accurate', transform=ax[0].transAxes, va='top', fontsize=7, color='0.35')
fig.legend(handles=h, loc='lower center', frameon=False, ncol=2, fontsize=6.3)
for a, s in zip(ax, ('(a)', '(b)')): a.text(-0.17, 1.0, s, transform=a.transAxes)
fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(F + 'fig_tradeoff.pdf'); fig.savefig(F + 'fig_tradeoff.png', dpi=200)
