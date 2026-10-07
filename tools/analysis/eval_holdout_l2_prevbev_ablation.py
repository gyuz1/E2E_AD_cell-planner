"""Hold-out planning L2, parametrized by which frame offsets get replayed
before each scored sample -- lets you compare "cold" (no prev_bev history)
against "warm" (prev_bev streamed across several frames) at matched L2.

Sibling of eval_holdout_l2.py (same isolated-window-per-sample methodology,
same STREAM_OFFSETS/HIS_FRAMES convention as
tools/data_converter/etri_test_converter.py), with one change:
--frame-offsets selects which of those offsets actually get replayed, instead
of always replaying the full 7-frame window.

Two headline conditions:
  --frame-offsets=0
      Single current-frame only. reset_stream() runs immediately before it,
      so prev_bev is guaranteed None -- no temporal context at all.
  --frame-offsets=-30,-25,-20,-15,-10,-5,0   (or any subset ending in 0)
      Cumulative streaming: reset once, then replay each offset in order,
      carrying prev_bev forward -- exactly what a real test clip does.

Reports L2 the same way as eval_holdout_l2.py (by command + overall), so the
two conditions' printed tables are directly comparable.
"""
import argparse
import importlib

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

HIS_FRAMES = 30
STREAM_STRIDE = 5

COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN', 'STOP',
)


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def parse_frame_offsets(spec):
    offsets = sorted(int(x) for x in spec.split(','))
    if offsets[-1] != 0:
        raise ValueError(
            f'--frame-offsets must include the current frame (0): {spec}')
    if offsets[0] < -HIS_FRAMES:
        raise ValueError(
            f'offset {offsets[0]} exceeds the provided '
            f'{HIS_FRAMES / 10:.1f}s history range: {spec}')
    for off in offsets:
        if off % STREAM_STRIDE != 0:
            raise ValueError(
                f'offset {off} is not a multiple of {STREAM_STRIDE} raw '
                f'frames ({STREAM_STRIDE / 10:.1f}s): {spec}')
    return offsets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--stride', type=int, default=5,
                        help='evaluation interval in frame_idx')
    parser.add_argument('--min-frame', type=int, default=HIS_FRAMES,
                        help='minimum frame_idx to have a full lookback window')
    parser.add_argument(
        '--frame-offsets', required=True,
        help='comma-separated raw-frame offsets (0.1s units) to replay '
             'before each scored sample, e.g. "0" for a single cold frame '
             'or "-30,-25,-20,-15,-10,-5,0" for the full streamed window. '
             'Must include 0 (the scored frame itself).')
    return parser.parse_args()


def main():
    args = parse_args()
    offsets = parse_frame_offsets(args.frame_offsets)
    print(f'--frame-offsets {args.frame_offsets} -> replaying {offsets} '
          f'({len(offsets)} frame(s)) before each scored sample')

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
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()

    scenes = {}
    for gi, info in enumerate(dataset.data_infos):
        scenes.setdefault(info['scene_token'], []).append(gi)

    l2_sums = np.zeros(3)
    n_eval = 0
    n_skipped_incomplete_window = 0

    n_cmd = len(COMMAND_VOCAB)
    cmd_all = np.zeros(n_cmd, dtype=int)
    cmd_valid = np.zeros(n_cmd, dtype=int)
    cmd_l2_sums = np.zeros((n_cmd, 3))

    for scene_token, sample_ids in mmcv.track_iter_progress(list(scenes.items())):
        frame_to_gi = {
            dataset.data_infos[gi]['frame_idx']: gi for gi in sample_ids}

        for gi in sample_ids:
            info = dataset.data_infos[gi]
            frame_idx = info['frame_idx']
            if frame_idx < args.min_frame or frame_idx % args.stride != 0:
                continue
            if not info.get('fut_valid_flag', False):
                continue

            gt_cmd_idx = int(np.asarray(info['gt_ego_fut_cmd']).argmax())
            cmd_all[gt_cmd_idx] += 1

            window_frames = [frame_idx + off for off in offsets]
            if any(f not in frame_to_gi for f in window_frames):
                n_skipped_incomplete_window += 1
                continue

            # Isolated clip: reset before every window, like a real test
            # clip. With offsets=[0] this makes prev_bev=None for the one
            # and only forward pass -- the "cold, no history" condition.
            reset_stream(model.module)
            result = None
            collated = None
            for f in window_frames:
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **collated)

            ego_fut_preds = result[0]['pts_bbox']['ego_fut_preds']
            cmd = np.array(collated['ego_fut_cmd'][0].data[0]).reshape(
                -1, ego_fut_preds.shape[0])[0]
            pred = ego_fut_preds[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
            gt = np.array(info['gt_ego_fut_trajs'], dtype=np.float64).cumsum(0)

            dist = np.linalg.norm(pred - gt, axis=-1)
            sample_l2 = np.array([dist[:2].mean(), dist[:4].mean(), dist[:6].mean()])
            l2_sums += sample_l2
            n_eval += 1

            cmd_valid[gt_cmd_idx] += 1
            cmd_l2_sums[gt_cmd_idx] += sample_l2

    l2 = l2_sums / max(n_eval, 1)
    print()
    print(f'evaluated samples: {n_eval}')
    print(f'skipped (incomplete lookback window): {n_skipped_incomplete_window}')
    print('Official cumulative ADE:')
    print(f'  L2@1s : {l2[0]:.6f} m')
    print(f'  L2@2s : {l2[1]:.6f} m')
    print(f'  L2@3s : {l2[2]:.6f} m')
    print(f'  Final Planning L2 avg: {l2.mean():.6f} m')

    print()
    print('----------- ETRI Planning L2 by Command -----------')
    header = (f"{'Command':<15} {'valid/all':>12} {'L2@1s':>10} "
              f"{'L2@2s':>10} {'L2@3s':>10} {'L2_avg':>10}")
    print(header)
    print('-' * len(header))
    for i, name in enumerate(COMMAND_VOCAB):
        n_v, n_a = cmd_valid[i], cmd_all[i]
        cmd_l2 = cmd_l2_sums[i] / max(n_v, 1)
        print(f"{name:<15} {f'{n_v}/{n_a}':>12} {cmd_l2[0]:>10.6f} "
              f"{cmd_l2[1]:>10.6f} {cmd_l2[2]:>10.6f} {cmd_l2.mean():>10.6f}")
    print('-' * len(header))


if __name__ == '__main__':
    main()
