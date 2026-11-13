"""Fig. 2: lane-keeping within-cell distortion vs number of cells.

Reads results/partition_sweep.json. Held-out distortion (training cell means
applied to held-out frames) solid, training distortion dashed. Grayscale-safe:
partitions differ by marker and line style, not only by shade.
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sw = json.load(open(sys.argv[1]))
rows = [r for r in sw['rows'] if r['command'] == 'LANE_KEEP' and r['K'] >= 5]
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6})
fig, ax = plt.subplots(figsize=(3.4, 2.3))
style = {'dp': ('Dynamic programming', 'o', '-', '0.0'),
         'uniform': ('Uniform', 's', '-', '0.45'),
         'quantile': ('Equal occupancy', '^', '-', '0.65')}
for key, (lab, mk, ls, col) in style.items():
    pts = [(r['K'], r[key]) for r in rows if r[key] != 'INFEASIBLE']
    K = [p[0] for p in pts]
    ax.plot(K, [p[1]['D_val'] for p in pts], ls=ls, marker=mk, ms=3.5, lw=1.0, color=col,
            label=f'{lab}', mfc='white' if key != 'dp' else col)
    ax.plot(K, [p[1]['D_train'] for p in pts], ls=':', lw=0.8, color=col)
    if key == 'uniform':
        bad = [(k, p['D_val']) for k, p in pts if p['empty_cells'] or p['min_frames'] < 50]
        ax.plot([b[0] for b in bad], [b[1] for b in bad], 'x', color='0.2', ms=5, mew=0.9,
                label='Uniform, constraint violated')
kbad = min(r['K'] for r in rows if r['dp'] == 'INFEASIBLE')
ax.axvspan(kbad, 200, color='0.92', lw=0)
ax.text(kbad * 1.12, 19, 'DP infeasible', fontsize=6.5, va='center', ha='center', rotation=90)
ax.axvline(60, color='0.5', lw=0.5, ls='-.')
ax.text(60 * 0.95, 14.5, 'used', fontsize=6.5, ha='right', va='center')
ax.plot([], [], ls='-', color='0.3', lw=1.0, label='held-out')
ax.plot([], [], ls=':', color='0.3', lw=0.8, label='training')
ax.set_xscale('log')
ax.set_xticks([5, 10, 20, 40, 60, 100, 150])
ax.set_xlim(4.5, 170)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylim(7, 30)
ax.set_xlabel('Number of lane-keeping cells $K$')
ax.set_ylabel('Distortion per frame (m$^2$)')
ax.grid(True, which='major', lw=0.3, color='0.85')
ax.legend(fontsize=6.5, frameon=False, loc='upper right')
fig.tight_layout(pad=0.3)
fig.savefig(sys.argv[2])
fig.savefig(sys.argv[2].replace('.pdf', '.png'), dpi=300)
print('wrote', sys.argv[2])
