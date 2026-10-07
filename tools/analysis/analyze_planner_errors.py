"""Where does the planner lose? Per distance band, per command, per cell.

Runs the scored-frame stream like eval_holdout_l2_and_tinfer.py, but keeps every
sample's prediction, ground truth, command, target point and (for a cell
planner) the selected cell and every candidate. Then reports:

  1. L2 by target-point distance band  (stationary clips are 26% of test)
  2. L2 by command
  3. L2 per cell, with the number of frames and scenes behind it
  4. cell usage: how far apart the candidates of one scene actually are
     (if the cells produce near-identical trajectories, selection cannot help)
  5. oracle cell: the L2 if the best candidate had been selected -- the
     headroom left in the selection rule itself
  6. figures: per-command cell heat map of L2, and predicted vs ground-truth
     3s endpoints coloured by cell

Everything per sample is saved to <out-dir>/records.npz for further work.

Usage:
  python tools/analyze_planner_errors.py <eval_config> <checkpoint> \
      --ann-file <val pkl> [--frame-offsets 0,-5,-10] [--select-cell-by-tp]
      [--stop-by-tp] [--test-commands] [--max-windows N] [--out-dir DIR]
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
    COMMANDS, ROUTE_CELL, route_trajectory)

HIS_FRAMES = 30


def official_l2(pred, gt):
    """[T, 2] cumulative positions -> the challenge's 1s/2s/3s average."""
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
    p.add_argument('--fp16', action='store_true', default=True)
    p.add_argument('--select-cell-by-tp', action='store_true')
    p.add_argument('--stop-by-tp', action='store_true')
    p.add_argument('--test-commands', action='store_true')
    p.add_argument('--stop-tp-thresh', type=float, default=1.0)
    p.add_argument('--stop-speed-thresh', type=float, default=0.1)
    p.add_argument('--max-windows', type=int, default=0, help='0 = all')
    p.add_argument('--bands', default='1,9,30,60,90',
                   help='target-point distance band edges in metres; the defaults\n                        split val into 219 / 129 / 582 / 1502 / 1215 / 403')
    p.add_argument('--out-dir', default='reports/planner_errors')
    p.add_argument('--device', type=int, default=0)
    return p.parse_args()


def collect(model, dataset, args, offsets):
    head = model.module.pts_bbox_head
    is_cell = getattr(head, 'cell_planner', False)
    edges = ({c: (f.cpu(), l.cpu()) for c, (f, l) in head.cell_edges().items()}
             if is_cell else None)
    scenes = {}
    for gi, info in enumerate(dataset.data_infos):
        scenes.setdefault(info['scene_token'], []).append(gi)
    rec = {k: [] for k in ('l2', 'cmd', 'row', 'col', 'kind', 'tp_dist', 'tp',
                           'gt_end', 'pred_end', 'spread', 'oracle_l2', 'scene')}
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
            if args.max_windows and len(rec['l2']) >= args.max_windows:
                return rec, is_cell, edges
            reset_stream(model.module)
            for i, f in enumerate(window):
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                with torch.no_grad():
                    out = model(return_loss=False, rescale=True,
                                bev_only=(i != len(window) - 1), **collated)
            pts = out[0]['pts_bbox']
            cmd_vec = np.asarray(collated['ego_fut_cmd'][0].data[0]).reshape(-1)
            mode = int(cmd_vec[:7].argmax())
            tp = np.asarray(info['gt_ego_target_point'], np.float64).reshape(-1)[:2]
            gt = np.asarray(info['gt_ego_fut_trajs'], np.float64).cumsum(0)
            row = col = -1
            kind = -1
            spread = np.nan
            oracle = np.nan
            if args.test_commands and mode == 6:
                mode = 0
                state = pts.get('ego_state_pred')
                if state is not None and args.speed_col is not None and float(
                        state.reshape(-1)[args.speed_col]) < args.stop_speed_thresh:
                    mode = 6
            if is_cell and args.select_cell_by_tp:
                sel, route = route_trajectory(
                    {c: t[None].float() for c, t in pts['cell_trajs'].items()},
                    pts['cell_u_turn'][None].float(), pts['cell_stop'][None].float(),
                    torch.as_tensor(tp[None], dtype=torch.float32),
                    torch.tensor([mode]), edges, head.cell_stop_tp_thresh)
                pred = sel[0].double().cumsum(0).numpy()
                kind, row, col = (int(route['kind'][0]), int(route['row'][0]),
                                  int(route['col'][0]))
                if kind == ROUTE_CELL:
                    cand = pts['cell_trajs'][mode].double().cumsum(-2).numpy()
                    ends = cand[..., -1, :].reshape(-1, 2)
                    spread = float(np.linalg.norm(ends - ends.mean(0), axis=-1).mean())
                    oracle = min(official_l2(c.reshape(-1, 2), gt)
                                 for c in cand.reshape(-1, cand.shape[-2], 2))
            else:
                if args.stop_by_tp and np.linalg.norm(tp) < args.stop_tp_thresh:
                    mode = 6
                pred = pts['ego_fut_preds'][mode].double().cumsum(0).numpy()
            rec['l2'].append(official_l2(pred, gt))
            rec['cmd'].append(mode)
            rec['row'].append(row)
            rec['col'].append(col)
            rec['kind'].append(kind)
            rec['tp_dist'].append(float(np.linalg.norm(tp)))
            rec['tp'].append(tp)
            rec['gt_end'].append(gt[-1])
            rec['pred_end'].append(pred[-1])
            rec['spread'].append(spread)
            rec['oracle_l2'].append(oracle)
            rec['scene'].append(scene_token)
    return rec, is_cell, edges


def report(rec, is_cell, edges, args):
    r = {k: np.asarray(v) for k, v in rec.items()}
    n = len(r['l2'])
    print(f'\nsamples: {n}   L2 overall: {r["l2"].mean():.4f}')

    band_edges = [float(x) for x in args.bands.split(',')]
    band = np.searchsorted(band_edges, r['tp_dist'], side='right')
    names = ([f'< {band_edges[0]:g} m'] +
             [f'{a:g} - {b:g} m' for a, b in zip(band_edges[:-1], band_edges[1:])] +
             [f'>= {band_edges[-1]:g} m'])
    print('\n1. by target-point distance')
    print(f'   {"band":<12}{"n":>7}{"L2":>10}{"share":>9}')
    for i, name in enumerate(names):
        m = band == i
        if m.any():
            print(f'   {name:<12}{int(m.sum()):>7}{r["l2"][m].mean():>10.4f}'
                  f'{m.mean() * 100:>8.1f}%')

    print('\n2. by command')
    print(f'   {"command":<16}{"n":>7}{"L2":>10}')
    for c in range(7):
        m = r['cmd'] == c
        if m.any():
            print(f'   {COMMANDS[c]:<16}{int(m.sum()):>7}{r["l2"][m].mean():>10.4f}')

    if not is_cell:
        return r
    print('\n3. by cell (selected cells only)')
    print(f'   {"command":<16}{"cell":>8}{"n":>7}{"scenes":>8}{"L2":>10}'
          f'{"oracle":>9}{"spread":>9}')
    for c in sorted(edges):
        m = (r['cmd'] == c) & (r['kind'] == ROUTE_CELL)
        if not m.any():
            continue
        for row in sorted(set(r['row'][m].tolist())):
            for col in sorted(set(r['col'][m & (r['row'] == row)].tolist())):
                s = m & (r['row'] == row) & (r['col'] == col)
                if not s.any():
                    continue
                print(f'   {COMMANDS[c]:<16}{f"({row},{col})":>8}{int(s.sum()):>7}'
                      f'{len(set(r["scene"][s].tolist())):>8}{r["l2"][s].mean():>10.4f}'
                      f'{np.nanmean(r["oracle_l2"][s]):>9.4f}'
                      f'{np.nanmean(r["spread"][s]):>9.2f}')
    cell = r['kind'] == ROUTE_CELL
    if cell.any():
        print(f'\n4. cell usage: candidates of one scene sit {np.nanmean(r["spread"][cell]):.2f} m '
              f'apart on average (3s endpoint). If this is far below the cell spacing, '
              f'the cell code is barely used.')
        print(f'5. oracle cell (best candidate per sample): {np.nanmean(r["oracle_l2"][cell]):.4f} '
              f'vs routed {r["l2"][cell].mean():.4f} -- the headroom in the selection rule.')
    return r


def figures(r, edges, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok=True)
    for c in sorted(edges):
        m = (r['cmd'] == c) & (r['kind'] == ROUTE_CELL)
        if not m.any():
            continue
        fe, le = [np.asarray(e, float) for e in edges[c]]
        nf, nl = len(fe) - 1, len(le) - 1
        grid = np.full((nf, nl), np.nan)
        cnt = np.zeros((nf, nl), int)
        for row in range(nf):
            for col in range(nl):
                s = m & (r['row'] == row) & (r['col'] == col)
                cnt[row, col] = int(s.sum())
                if s.any():
                    grid[row, col] = r['l2'][s].mean()
        fig, ax = plt.subplots(1, 2, figsize=(13, 7), dpi=140)
        im = ax[0].imshow(grid, origin='lower', aspect='auto', cmap='viridis')
        ax[0].set_title(f'{COMMANDS[c]}: L2 per cell')
        ax[0].set_xlabel('lateral cell'); ax[0].set_ylabel('forward cell')
        ax[0].set_xticks(range(nl)); ax[0].set_yticks(range(nf))
        for row in range(nf):
            for col in range(nl):
                if cnt[row, col]:
                    ax[0].text(col, row, f'{grid[row, col]:.2f}\nn={cnt[row, col]}',
                               ha='center', va='center', fontsize=7, color='w')
        fig.colorbar(im, ax=ax[0], label='L2 [m]')
        ax[1].scatter(-r['gt_end'][m][:, 1], r['gt_end'][m][:, 0], s=10, alpha=0.5,
                      label='ground truth 3s', c='#1f6fe0')
        ax[1].scatter(-r['pred_end'][m][:, 1], r['pred_end'][m][:, 0], s=10, alpha=0.5,
                      label='predicted 3s', c='#e0242a')
        ax[1].set_title(f'{COMMANDS[c]}: 3s endpoints')
        ax[1].set_xlabel('LEFT <- lateral -> RIGHT [m]'); ax[1].set_ylabel('forward [m]')
        ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(args.out_dir, f'{c}_{COMMANDS[c]}.png')
        fig.savefig(path); plt.close(fig)
        print(f'  -> {path}')


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
    if args.fp16:
        wrap_fp16_model(model)
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    idx = list(cfg.model.pts_bbox_head.get('aux_bev_motion_idx') or [])
    args.speed_col = idx.index(7) if 7 in idx else None
    model = MMDataParallel(model.cuda(args.device), device_ids=[args.device]).eval()

    rec, is_cell, edges = collect(model, dataset, args, offsets)
    r = report(rec, is_cell, edges, args)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, 'records.npz'),
             **{k: v for k, v in r.items() if k != 'scene'})
    print(f'\nper-sample records -> {os.path.join(args.out_dir, "records.npz")}')
    if is_cell and edges:
        figures(r, edges, args)


if __name__ == '__main__':
    main()
