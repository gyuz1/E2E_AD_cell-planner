"""Compare two cell layouts on one dataset, so a human can judge a re-split.

Draws, per command, what the config's committed edges and a fresh fit on the
given annotations do to the same frames. The question it is built to answer is
not "did the edges move" -- they always move a little -- but "is any cell
holding frames that disagree with each other, and would moving its edges help".

Per command, three panels:

  left    the target points, with both edge sets drawn over them. Committed
          edges solid, refit dashed. x is forward, y is lateral with LEFT on
          the left (plotted as -lateral), matching the car's own view.
  middle  within-cell 3s L2 per cell against cell position. This is the error
          a perfect per-cell mean would still leave -- the thing the fit
          minimizes -- so a tall bar is a cell whose frames disagree.
  right   frames per cell, with the --min-frames floor drawn. Short bars are
          cells that will be fit from very little.

The printed table carries the same numbers per cell, including how many
distinct scenes each holds: a cell with many frames from few scenes is one
clip repeated, not variety, and its head will overfit that clip.

Usage:
    python tools/plot_cell_layout_compare.py \
        --ann-file data/etri/annotations_10hz/vad_etri_10hz_infos_temporal_train.pkl \
        --config projects/configs/VAD/VADLAW_etri_tiny_clean_cellplanner_lk60.py \
        --out-dir reports/cell_layout_376
"""
import argparse
import os

import numpy as np

from tools.fit_cell_layout import (BOUNDS, COMMANDS, assign, fit, load,
                                   moving_frames, official_l2)


def cell_stats(tp, traj, scene, fwd, lat):
    """Per-cell frames, scenes and within-cell 3s L2 under one edge set."""
    n_fwd, n_lat = len(fwd) - 1, len(lat) - 1
    row, col = assign(tp, fwd, lat)
    g = row * n_lat + col
    K = n_fwd * n_lat
    counts = np.bincount(g, minlength=K)
    scenes = np.array([len(np.unique(scene[g == k])) if counts[k] else 0
                       for k in range(K)])
    mean = np.array([traj[g == k].mean(0) if counts[k] else traj.mean(0)
                     for k in range(K)])
    per_cell = np.array([official_l2(mean[k][None], traj[g == k]) if counts[k] else np.nan
                         for k in range(K)])
    return g, counts, scenes, per_cell, official_l2(mean[g], traj)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ann-file', required=True)
    p.add_argument('--config', required=True,
                   help='train config whose cell_layouts are the committed ones')
    p.add_argument('--out-dir', default='reports/cell_layout_compare')
    p.add_argument('--commands', default='0,1,2,3,4',
                   help='comma-separated command indices to draw')
    p.add_argument('--min-frames', type=int, default=50)
    p.add_argument('--min-scenes', type=int, default=3)
    p.add_argument('--min-width', type=float, default=1.0)
    p.add_argument('--stop-tp', type=float, default=1.0)
    p.add_argument('--stop-rule', default='forward', choices=('forward', 'norm'))
    p.add_argument('--grid-step', type=float, default=0.5)
    p.add_argument('--max-points', type=int, default=20000,
                   help='subsample the scatter so the figure stays readable')
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mmcv import Config

    layouts = Config.fromfile(args.config).model.pts_bbox_head['cell_layouts']
    tp, cmd, traj, scene, n_scenes = load(args.ann_file)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'{args.ann_file}: {len(tp)} frames, {n_scenes} scenes')
    print(f'committed layout: {args.config}\n')

    rng = np.random.default_rng(0)
    for c in [int(v) for v in args.commands.split(',')]:
        if layouts[c] is None:
            continue
        fe, le = ([float(v) for v in layouts[c][0]], [float(v) for v in layouts[c][1]])
        n_fwd, n_lat = len(fe) - 1, len(le) - 1
        mtp, mtraj, msc = moving_frames(tp, cmd, traj, scene, c,
                                        args.stop_rule, args.stop_tp)
        _, cnt_a, scn_a, cell_a, tot_a = cell_stats(mtp, mtraj, msc, fe, le)
        F, L = fit(mtp, mtraj, msc, n_fwd, n_lat, BOUNDS[c], args.grid_step,
                   args.min_frames, args.min_scenes, args.min_width)
        refit = F is not None
        if refit:
            _, cnt_b, scn_b, cell_b, tot_b = cell_stats(mtp, mtraj, msc, F, L)

        print(f'=== {COMMANDS[c]} {n_fwd}x{n_lat}, {len(mtp)} frames ===')
        print(f'  within-cell 3s L2: committed {tot_a:.4f}'
              + (f'   refit {tot_b:.4f}   ({100 * (tot_a - tot_b) / tot_a:+.2f}%)'
                 if refit else '   refit INFEASIBLE'))
        thin = np.where(cnt_a < args.min_frames)[0]
        few = np.where((scn_a < args.min_scenes) & (cnt_a > 0))[0]
        print(f'  cells under {args.min_frames} frames: {len(thin)}'
              f'   under {args.min_scenes} scenes: {len(few)}'
              f'   empty: {int((cnt_a == 0).sum())}')
        order = np.argsort(-np.nan_to_num(cell_a))
        print(f'  worst cells by within-cell L2 (committed edges):')
        print(f'    {"cell":>6} {"fwd range":>16} {"frames":>7} {"scenes":>7} {"L2":>7}')
        for k in order[:5]:
            r, cl = divmod(int(k), n_lat)
            print(f'    {f"({r},{cl})":>6} {f"{fe[r]:.1f}-{fe[r+1]:.1f}":>16}'
                  f' {cnt_a[k]:7d} {scn_a[k]:7d} {cell_a[k]:7.3f}')

        sel = rng.choice(len(mtp), min(args.max_points, len(mtp)), replace=False)
        fig, ax = plt.subplots(1, 3, figsize=(19, 5.2),
                               gridspec_kw={'width_ratios': [1.25, 1, 1]})
        fig.suptitle(f'{COMMANDS[c]}  {n_fwd}x{n_lat} cells   '
                     f'{len(mtp)} frames / {n_scenes} scenes   '
                     f'within-cell 3s L2 {tot_a:.4f}'
                     + (f' -> {tot_b:.4f}' if refit else ''), fontsize=12)

        ax[0].scatter(mtp[sel, 0], -mtp[sel, 1], s=1.5, c='0.72', linewidths=0)
        for v in fe:
            ax[0].axvline(v, color='#1f77b4', lw=0.7, alpha=0.9)
        for v in le:
            ax[0].axhline(-v, color='#1f77b4', lw=0.7, alpha=0.9)
        if refit:
            for v in F:
                ax[0].axvline(v, color='#d62728', lw=0.7, ls='--', alpha=0.9)
            for v in L:
                ax[0].axhline(-v, color='#d62728', lw=0.7, ls='--', alpha=0.9)
        ax[0].set_xlabel('forward (m)')
        ax[0].set_ylabel('left (m)')
        ax[0].set_title('target points   solid: committed   dashed: refit')

        centres = [(fe[k // n_lat] + fe[k // n_lat + 1]) / 2 for k in range(n_fwd * n_lat)]
        ax[1].bar(centres, np.nan_to_num(cell_a),
                  width=[fe[k // n_lat + 1] - fe[k // n_lat] for k in range(n_fwd * n_lat)],
                  color='#1f77b4', alpha=0.75, edgecolor='white', linewidth=0.3,
                  label='committed')
        if refit:
            cen_b = [(F[k // n_lat] + F[k // n_lat + 1]) / 2 for k in range(n_fwd * n_lat)]
            ax[1].step(cen_b, np.nan_to_num(cell_b), where='mid',
                       color='#d62728', lw=1.2, label='refit')
        ax[1].set_xlabel('cell centre, forward (m)')
        ax[1].set_ylabel('within-cell 3s L2 (m)')
        ax[1].set_title('per-cell disagreement (lower is better)')
        ax[1].legend(fontsize=8)

        ax[2].bar(range(len(cnt_a)), cnt_a, color='#1f77b4', alpha=0.75)
        ax[2].axhline(args.min_frames, color='#d62728', lw=1.0, ls='--',
                      label=f'min {args.min_frames}')
        ax[2].set_xlabel('cell id')
        ax[2].set_ylabel('frames')
        ax[2].set_title('frames per cell (committed edges)')
        ax[2].legend(fontsize=8)

        for a in ax:
            a.grid(alpha=0.25, lw=0.4)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f'{c}_{COMMANDS[c]}.png')
        fig.savefig(out, dpi=125)
        plt.close(fig)
        print(f'  -> {out}\n')


if __name__ == '__main__':
    main()
