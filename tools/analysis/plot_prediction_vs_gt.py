"""Draw predicted vs ground-truth 3s trajectories for individual val samples.

One panel per sample: the ground truth path, the submitted path, the 5s target
point that selected it, and -- for a cell planner -- every candidate the model
generated, so it is visible whether the selected one was the right choice and
how much the candidates differ at all.

Pick which samples to draw with --pick:
  worst   the highest-L2 samples (default; where the model actually fails)
  best    the lowest-L2 samples
  random  a random sample
and narrow with --command / --band (target-point distance in metres).

--grid draws a second figure instead of picking samples: one panel per
(command x target-point distance band), each overlaying up to --per-cell
ground-truth and submitted paths, so a whole command's failure mode is visible
at a glance.

Usage:
  python tools/plot_prediction_vs_gt.py <eval_config> <checkpoint> \
      --ann-file <val pkl> [--select-cell-by-tp | --stop-by-tp] \
      [--pick worst] [--n 16] [--command LANE_KEEP] [--band 9,1e9]
      [--out reports/pred_vs_gt]
Run on an idle GPU; it replays the same 3-frame stream the scoring uses.
"""
import argparse
import importlib
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.VAD.cell_planner_utils import (
    COMMANDS, ROUTE_CELL, ROUTE_STOP, ROUTE_U_TURN, route_trajectory)

HIS_FRAMES = 30
KIND = {ROUTE_CELL: 'cell', ROUTE_U_TURN: 'U_TURN head', ROUTE_STOP: 'STOP head'}


def official_l2(pred, gt):
    d = np.linalg.norm(pred - gt, axis=-1)
    return float(np.mean([d[:2].mean(), d[:4].mean(), d[:6].mean()]))


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'prev_bev2': None, 'prev_bev_pristine': None,
        'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--ann-file', required=True)
    p.add_argument('--frame-offsets', default='0,-5,-10')
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--select-cell-by-tp', action='store_true')
    p.add_argument('--stop-by-tp', action='store_true')
    p.add_argument('--test-commands', action='store_true')
    p.add_argument('--stop-tp-thresh', type=float, default=1.0)
    p.add_argument('--stop-speed-thresh', type=float, default=0.1)
    p.add_argument('--pick', default='worst', choices=('worst', 'best', 'random'))
    p.add_argument('--n', type=int, default=16, help='panels to draw')
    p.add_argument('--command', default=None, help='e.g. LANE_KEEP')
    p.add_argument('--band', default=None,
                   help='target-point distance range "lo,hi" in metres')
    p.add_argument('--max-windows', type=int, default=0, help='0 = the whole set')
    p.add_argument('--no-candidates', action='store_true')
    p.add_argument('--grid', action='store_true',
                   help='draw command x distance-band panels instead of picks')
    p.add_argument('--bands', default='1,9,30,60,90',
                   help='band edges in metres for --grid; the defaults split the val\n                        set into 219 / 129 / 582 / 1502 / 1215 / 403 samples')
    p.add_argument('--per-cell', type=int, default=25,
                   help='paths overlaid per (command, band) panel')
    p.add_argument('--out', default='reports/pred_vs_gt')
    p.add_argument('--device', type=int, default=0)
    return p.parse_args()


def draw_grid(samples, args, plt):
    """One panel per (command, target-point distance band)."""
    edges = [float(x) for x in args.bands.split(',')]
    names = ([f'|TP| < {edges[0]:g} m'] +
             [f'{a:g}-{b:g} m' for a, b in zip(edges[:-1], edges[1:])] +
             [f'>= {edges[-1]:g} m'])
    dist = np.array([np.linalg.norm(s['tp']) for s in samples])
    band = np.searchsorted(edges, dist, side='right')
    cmds = sorted({s['cmd'] for s in samples})
    rows, cols = len(cmds), len(names)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), dpi=130,
                             squeeze=False)
    rng = np.random.default_rng(0)
    for i, c in enumerate(cmds):
        for j in range(cols):
            ax = axes[i][j]
            sel = [s for k, s in enumerate(samples) if s['cmd'] == c and band[k] == j]
            if not sel:
                ax.set_title(f'{COMMANDS[c]} | {names[j]} | n=0', fontsize=8)
                ax.axis('off')
                continue
            take = sel if len(sel) <= args.per_cell else [
                sel[k] for k in rng.choice(len(sel), args.per_cell, replace=False)]
            for s in take:
                ax.plot(-s['gt'][:, 1], s['gt'][:, 0], color='#1f6fe0', lw=1,
                        alpha=0.5, zorder=2)
                ax.plot(-s['pred'][:, 1], s['pred'][:, 0], color='#e0242a', lw=1,
                        alpha=0.5, zorder=3)
            ax.scatter([0], [0], marker='^', s=40, c='black', zorder=5)
            ax.scatter([-s['tp'][1] for s in take], [s['tp'][0] for s in take],
                       marker='*', s=25, c='#f2a900', alpha=0.6, zorder=4)
            l2 = np.mean([s['l2'] for s in sel])
            ax.set_title(f'{COMMANDS[c]} | {names[j]}\nn={len(sel)}  L2 {l2:.3f}',
                         fontsize=8)
            ax.set_aspect('equal', 'datalim')
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
    fig.suptitle('blue = ground truth, red = submitted, star = 5s target point '
                 '(x: LEFT <- lateral -> RIGHT, y: forward [m])', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    path = f'{args.out}_grid.png'
    fig.savefig(path)
    print(f'-> {path}')


def main():
    args = parse_args()
    offsets = sorted(int(x) for x in args.frame_offsets.split(','))
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    wrap_fp16_model(model)
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    idx = list(cfg.model.pts_bbox_head.get('aux_bev_motion_idx') or [])
    speed_col = idx.index(7) if 7 in idx else None
    model = MMDataParallel(model.cuda(args.device), device_ids=[args.device]).eval()
    head = model.module.pts_bbox_head
    is_cell = getattr(head, 'cell_planner', False)
    edges = ({c: (f.cpu(), l.cpu()) for c, (f, l) in head.cell_edges().items()}
             if is_cell else None)
    band = [float(x) for x in args.band.split(',')] if args.band else None
    want_cmd = COMMANDS.index(args.command) if args.command else None

    scenes = {}
    for gi, info in enumerate(dataset.data_infos):
        scenes.setdefault(info['scene_token'], []).append(gi)

    samples = []
    for scene_token, ids in mmcv.track_iter_progress(list(scenes.items())):
        frame_to_gi = {dataset.data_infos[gi]['frame_idx']: gi for gi in ids}
        for gi in ids:
            info = dataset.data_infos[gi]
            if info['frame_idx'] < HIS_FRAMES or info['frame_idx'] % args.stride:
                continue
            if not info.get('fut_valid_flag', False):
                continue
            window = [info['frame_idx'] + off for off in offsets]
            if any(f not in frame_to_gi for f in window):
                continue
            cmd_gt = int(np.asarray(info['gt_ego_fut_cmd']).argmax())
            tp = np.asarray(info['gt_ego_target_point'], np.float64).reshape(-1)[:2]
            if want_cmd is not None and cmd_gt != want_cmd:
                continue
            if band and not (band[0] <= np.linalg.norm(tp) < band[1]):
                continue
            if args.max_windows and len(samples) >= args.max_windows:
                break
            reset_stream(model.module)
            for i, f in enumerate(window):
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                with torch.no_grad():
                    out = model(return_loss=False, rescale=True,
                                bev_only=(i != len(window) - 1), **collated)
            pts = out[0]['pts_bbox']
            gt = np.asarray(info['gt_ego_fut_trajs'], np.float64).cumsum(0)
            mode = cmd_gt
            cand = None
            kind = None
            if args.test_commands and mode == 6:
                mode = 0
                state = pts.get('ego_state_pred')
                if state is not None and speed_col is not None and float(
                        state.reshape(-1)[speed_col]) < args.stop_speed_thresh:
                    mode = 6
            if is_cell and args.select_cell_by_tp:
                sel, route = route_trajectory(
                    {c: t[None].float() for c, t in pts['cell_trajs'].items()},
                    pts['cell_u_turn'][None].float(), pts['cell_stop'][None].float(),
                    torch.as_tensor(tp[None], dtype=torch.float32),
                    torch.tensor([mode]), edges, head.cell_stop_tp_thresh)
                pred = sel[0].double().cumsum(0).numpy()
                kind = int(route['kind'][0])
                if not args.no_candidates and mode in pts['cell_trajs']:
                    cand = pts['cell_trajs'][mode].double().cumsum(-2).numpy()
                    cand = cand.reshape(-1, cand.shape[-2], 2)
            else:
                if args.stop_by_tp and np.linalg.norm(tp) < args.stop_tp_thresh:
                    mode = 6
                pred = pts['ego_fut_preds'][mode].double().cumsum(0).numpy()
            samples.append(dict(l2=official_l2(pred, gt), gt=gt, pred=pred, tp=tp,
                                cmd=cmd_gt, mode=mode, kind=kind, cand=cand,
                                scene=scene_token, frame=info['frame_idx']))

    if not samples:
        raise SystemExit('no sample matched the filters')
    order = np.argsort([s['l2'] for s in samples])
    if args.pick == 'worst':
        chosen = [samples[i] for i in order[::-1][:args.n]]
    elif args.pick == 'best':
        chosen = [samples[i] for i in order[:args.n]]
    else:
        rng = np.random.default_rng(0)
        chosen = [samples[i] for i in rng.choice(len(samples), min(args.n, len(samples)),
                                                 replace=False)]
    print(f'{len(samples)} samples matched, mean L2 {np.mean([s["l2"] for s in samples]):.4f}; '
          f'drawing the {args.pick} {len(chosen)}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if args.grid:
        draw_grid(samples, args, plt)
        return
    cols = int(np.ceil(np.sqrt(len(chosen))))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, chosen):
        if s['cand'] is not None:
            for c in s['cand']:
                ax.plot(-c[:, 1], c[:, 0], color='#bbbbbb', lw=0.8, zorder=1)
        ax.plot(-s['gt'][:, 1], s['gt'][:, 0], 'o-', color='#1f6fe0', lw=2,
                ms=3, label='ground truth', zorder=3)
        ax.plot(-s['pred'][:, 1], s['pred'][:, 0], 'o-', color='#e0242a', lw=2,
                ms=3, label='submitted', zorder=4)
        ax.scatter([0], [0], marker='^', s=60, c='black', zorder=5)
        ax.scatter([-s['tp'][1]], [s['tp'][0]], marker='*', s=140, c='#f2a900',
                   edgecolors='#6b4a00', zorder=6, label='target point (5s)')
        note = KIND.get(s['kind'], f'mode {COMMANDS[s["mode"]]}')
        ax.set_title(f'{COMMANDS[s["cmd"]]} | L2 {s["l2"]:.2f} m | {note}\n'
                     f'{s["scene"][:8]} frame {s["frame"]}', fontsize=8)
        ax.set_aspect('equal', 'datalim')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes[len(chosen):]:
        ax.axis('off')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'{args.pick} {len(chosen)} of {len(samples)} '
                 f'(x: LEFT <- lateral -> RIGHT, y: forward [m]; grey = candidates)',
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    path = f'{args.out}_{args.pick}{"_" + args.command if args.command else ""}.png'
    fig.savefig(path)
    print(f'-> {path}')


if __name__ == '__main__':
    main()
