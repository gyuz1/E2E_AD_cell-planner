"""Fit the cell planner's cell edges, and report what each cell holds.

The cell layouts in the configs were fitted by an ad-hoc script that never made
it into the repository, so nothing here could reproduce or re-derive them. This
is that script, generalized over the annotation file and the command, so the
same fit can be run on the 301-scene split and on all 376 scenes and the two
compared.

WHAT IS BEING MINIMIZED

Every moving command owns a grid of cells over the target point. All frames
whose target point falls in one cell share a waypoint head, distinguished only
by a fixed positional code, so a cell is only as useful as its frames are alike:
the fit minimizes the total within-cell squared spread of the 3 s trajectory,
which is exactly the error a per-cell mean would leave behind.

The forward and lateral edges are optimized by exact dynamic programming, one
axis at a time with the other held fixed, alternating to convergence (5 rounds,
or 1 when the command has a single lateral column). Each axis solve is exact
given the other, so this is coordinate descent on an exact sub-problem, not a
heuristic search. Cells must hold at least --min-frames frames from at least
--min-scenes distinct scenes and be at least --min-width metres wide; an
infeasible combination is reported rather than silently relaxed.

THE STOP RULE MATTERS TO THE FIT

Frames that route to the STOP head are excluded before fitting, so which rule
decides that changes which frames the cells see. The original fit used the
target point's FORWARD component; the router that actually runs
(cell_planner_utils.route_trajectory) uses its NORM. --stop-rule selects one,
and they disagree on frames whose target point is mostly lateral. Use
--stop-rule forward to reproduce the committed layouts and norm to fit the rule
the model really uses.

USAGE

  # reproduce a committed layout and check the edges come back identical
  python tools/fit_cell_layout.py --ann-file <split train pkl> \
      --command LANE_KEEP --cells 60x1 --stop-rule forward \
      --compare-config projects/configs/VAD/VADLAW_etri_tiny_clean_cellplanner_lk60.py

  # refit the same command on all 376 scenes
  python tools/fit_cell_layout.py --ann-file <fulldata train pkl> \
      --command LANE_KEEP --cells 60x1 --stop-rule forward
"""
import argparse
import json
import pickle

import numpy as np

# Command order is the one-hot order the dataset emits and VAD_head indexes
# cell_layouts by. U_TURN and STOP have no cells: they get plain heads.
COMMANDS = ['LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R',
            'TURN_LEFT', 'TURN_RIGHT', 'U_TURN', 'STOP']

# Outer extent of each command's grid, (forward_min, forward_max, lat_min,
# lat_max). These are the ranges the committed layouts span; a target point
# outside them routes to the outer cell and is flagged.
BOUNDS = {
    0: (1, 117, -20, 17),
    1: (1, 116, -16, 12),
    2: (1, 115, -12, 18),
    3: (1, 48, -2, 27),
    4: (1, 43, -26, 2),
}


def load(ann_file):
    with open(ann_file, 'rb') as fh:
        data = pickle.load(fh)
    infos = [i for i in data['infos'] if i.get('fut_valid_flag')]
    tp = np.stack([np.asarray(i['gt_ego_target_point'], float).reshape(-1)[:2]
                   for i in infos])
    cmd = np.array([int(np.asarray(i['gt_ego_fut_cmd']).argmax()) for i in infos])
    # cumsum: the annotations store per-step deltas, the metric scores the
    # cumulative positions, and the spread being minimized is over positions.
    traj = np.stack([np.asarray(i['gt_ego_fut_trajs'], float).cumsum(0)
                     for i in infos]).reshape(len(infos), -1)
    scene = np.unique([i['scene_token'] for i in infos], return_inverse=True)[1]
    return tp, cmd, traj, scene, len(set(i['scene_token'] for i in infos))


def official_l2(pred, gt):
    """Mean of the 1 s / 2 s / 3 s cumulative averages, as the metric defines it."""
    d = np.linalg.norm(pred.reshape(-1, 6, 2) - gt.reshape(-1, 6, 2), axis=-1)
    return float(np.mean([d[:, :2].mean(), d[:, :4].mean(), d[:, :6].mean()]))


def assign(tp, fwd, lat):
    return (np.clip(np.searchsorted(fwd[1:-1], tp[:, 0], side='right'), 0, len(fwd) - 2),
            np.clip(np.searchsorted(lat[1:-1], tp[:, 1], side='right'), 0, len(lat) - 2))


def dp_axis(bin_of, other, n_other, traj, scene, n_seg, grid,
            min_frames, min_scenes, min_width):
    """Exact DP over one axis, the other axis's assignment held fixed."""
    n_bins = len(grid) - 1
    n_scenes_total = scene.max() + 1
    cnt = np.zeros((n_bins, n_other))
    s1 = np.zeros((n_bins, n_other, traj.shape[1]))
    s2 = np.zeros((n_bins, n_other))
    sp = np.zeros((n_bins, n_other, n_scenes_total))
    np.add.at(cnt, (bin_of, other), 1)
    np.add.at(s1, (bin_of, other), traj)
    np.add.at(s2, (bin_of, other), (traj ** 2).sum(1))
    np.add.at(sp, (bin_of, other, scene), 1)
    pre = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), a.cumsum(0)])
    C, P1, P2, PS = pre(cnt), pre(s1), pre(s2), pre(sp)
    n = C[None] - C[:, None]
    d1 = P1[None] - P1[:, None]
    d2 = P2[None] - P2[:, None]
    n_sc = ((PS[None] - PS[:, None]) > 0).sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        # sum of squares minus the block mean's contribution = within-block spread
        cost = (d2 - (d1 ** 2).sum(-1) / np.maximum(n, 1)).sum(-1)
    g = np.asarray(grid)
    width = g[None, :] - g[:, None]
    bad = ((n < min_frames).any(-1) | (n_sc < min_scenes).any(-1)
           | (width < min_width - 1e-9))
    cost[bad] = np.inf
    cost[np.tril_indices(n_bins + 1)] = np.inf
    D = np.full((n_seg + 1, n_bins + 1), np.inf)
    D[0, 0] = 0
    arg = np.zeros((n_seg + 1, n_bins + 1), int)
    for k in range(1, n_seg + 1):
        t = D[k - 1][:, None] + cost
        arg[k] = t.argmin(0)
        D[k] = t.min(0)
    if not np.isfinite(D[n_seg, n_bins]):
        return None
    cuts, j = [n_bins], n_bins
    for k in range(n_seg, 0, -1):
        j = arg[k, j]
        cuts.append(j)
    return [float(grid[i]) for i in cuts[::-1]]


def moving_frames(tp, cmd, traj, scene, command, stop_rule, stop_tp):
    """Frames of one command that do NOT route to the STOP head."""
    if stop_rule == 'forward':
        not_stop = tp[:, 0] >= stop_tp
    else:
        not_stop = np.linalg.norm(tp, axis=-1) >= stop_tp
    m = (cmd == command) & not_stop
    return tp[m], traj[m], scene[m]


def fit(tp, traj, scene, n_fwd, n_lat, bounds, step,
        min_frames, min_scenes, min_width, rounds=5):
    x0, x1, y0, y1 = bounds
    fg = np.arange(x0, x1 + 1e-9, step); fg[-1] = x1
    lg = np.arange(y0, y1 + 1e-9, step); lg[-1] = y1
    fb = np.clip(np.searchsorted(fg[1:-1], tp[:, 0], side='right'), 0, len(fg) - 2)
    lb = np.clip(np.searchsorted(lg[1:-1], tp[:, 1], side='right'), 0, len(lg) - 2)
    # Start from quantiles: equal-occupancy edges, a feasible point to descend from.
    F = [x0, *np.quantile(tp[:, 0], np.arange(1, n_fwd) / n_fwd), x1]
    L = [y0, *np.quantile(tp[:, 1], np.arange(1, n_lat) / n_lat), y1]
    for _ in range(rounds):
        _, lc = assign(tp, F, L)
        Fn = dp_axis(fb, lc, n_lat, traj, scene, n_fwd, fg,
                     min_frames, min_scenes, min_width)
        if Fn is None:
            return None, None
        F = Fn
        if n_lat == 1:
            break
        fr, _ = assign(tp, F, L)
        Ln = dp_axis(lb, fr, n_fwd, traj, scene, n_lat, lg,
                     min_frames, min_scenes, min_width)
        if Ln is None:
            return None, None
        L = Ln
    return F, L


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ann-file', required=True)
    p.add_argument('--command', default='LANE_KEEP', choices=COMMANDS[:5])
    p.add_argument('--cells', default='60x1',
                   help='forward x lateral, e.g. 60x1 or 4x2')
    p.add_argument('--bounds', default=None,
                   help='x0,x1,y0,y1 -- defaults to this command\'s committed extent')
    p.add_argument('--grid-step', type=float, default=0.5,
                   help='candidate edge spacing the DP may cut at')
    p.add_argument('--min-frames', type=int, default=50)
    p.add_argument('--min-scenes', type=int, default=3)
    p.add_argument('--min-width', type=float, default=1.0)
    p.add_argument('--stop-tp', type=float, default=1.0)
    p.add_argument('--stop-rule', default='forward', choices=('forward', 'norm'),
                   help="forward reproduces the committed layouts; norm is what "
                        "route_trajectory actually applies")
    p.add_argument('--compare-config', default=None,
                   help='train config whose cell_layouts to diff against')
    p.add_argument('--out-json', default=None)
    args = p.parse_args()

    command = COMMANDS.index(args.command)
    n_fwd, n_lat = (int(v) for v in args.cells.lower().split('x'))
    bounds = (tuple(float(v) for v in args.bounds.split(','))
              if args.bounds else BOUNDS[command])

    tp, cmd, traj, scene, n_scenes = load(args.ann_file)
    print(f'annotations : {args.ann_file}')
    print(f'              {len(tp)} frames with a valid future, {n_scenes} scenes')
    mtp, mtraj, mscene = moving_frames(tp, cmd, traj, scene, command,
                                       args.stop_rule, args.stop_tp)
    print(f'command     : {args.command} ({n_fwd}x{n_lat} cells), '
          f'{len(mtp)} frames after the STOP rule')
    print(f'stop rule   : |TP{"_forward" if args.stop_rule == "forward" else ""}| '
          f'< {args.stop_tp} m routes to the STOP head')
    print(f'constraints : >= {args.min_frames} frames, >= {args.min_scenes} scenes, '
          f'>= {args.min_width} m wide; edges on a {args.grid_step} m grid')

    F, L = fit(mtp, mtraj, mscene, n_fwd, n_lat, bounds, args.grid_step,
               args.min_frames, args.min_scenes, args.min_width)
    if F is None:
        print('\nINFEASIBLE: no layout satisfies the constraints at this cell count.')
        return

    fr, lc = assign(mtp, F, L)
    g = fr * n_lat + lc
    K = n_fwd * n_lat
    means = np.array([mtraj[g == k].mean(0) for k in range(K)])
    counts = np.bincount(g, minlength=K)
    scenes_per = [len(np.unique(mscene[g == k])) for k in range(K)]
    flat = np.repeat(mtraj.mean(0)[None], len(mtraj), 0)

    print(f'\nforward edges: {[round(v, 1) for v in F]}')
    print(f'lateral edges: {[round(v, 1) for v in L]}')
    print(f'\nwithin-cell 3s L2 : {official_l2(means[g], mtraj):.4f}'
          f'   (one mean for the whole command: {official_l2(flat, mtraj):.4f})')
    print(f'min frames/cell   : {counts.min()}   min scenes/cell: {min(scenes_per)}')
    print(f'min forward width : {min(np.diff(F)):.1f} m', end='')
    if n_lat > 1:
        print(f'   min lateral width: {min(np.diff(L)):.1f} m', end='')
    print()

    if args.compare_config:
        from mmcv import Config
        layouts = Config.fromfile(args.compare_config).model.pts_bbox_head['cell_layouts']
        cf, cl = layouts[command]
        same_f = len(cf) == len(F) and np.allclose(np.asarray(cf, float), F)
        same_l = len(cl) == len(L) and np.allclose(np.asarray(cl, float), L)
        print(f'\nvs {args.compare_config}')
        print(f'  forward edges identical: {same_f}')
        print(f'  lateral edges identical: {same_l}')
        if not same_f:
            print(f'    config: {[round(float(v), 1) for v in cf]}')
        if not same_l:
            print(f'    config: {[round(float(v), 1) for v in cl]}')

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump({'ann_file': args.ann_file, 'command': args.command,
                       'cells': [n_fwd, n_lat], 'stop_rule': args.stop_rule,
                       'forward_edges': F, 'lateral_edges': L,
                       'frames_per_cell': counts.tolist(),
                       'scenes_per_cell': scenes_per,
                       'within_cell_l2': official_l2(means[g], mtraj)}, fh, indent=1)
        print(f'\nwrote {args.out_json}')


if __name__ == '__main__':
    main()
