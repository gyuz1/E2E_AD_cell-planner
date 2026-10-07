"""Hold-out planning L2 evaluation (challenge metric).

Mirrors etri_test_submit.py's actual inference conditions instead of
streaming a whole 30s/300-frame val scenario with one reset. Real test clips
are isolated, independent 3s windows: tools/data_converter/etri_test_converter.py
resets the model's temporal state per clip and feeds exactly 7 frames
(STREAM_FRAMES = range(-30, 1, 5), i.e. now and the prior 3.0s at 0.5s
spacing), then submits only the *last* frame's prediction. Streaming an
entire val scenario continuously -- as this script did before -- gives the
model far more accumulated prev_bev history than it will ever have at
submission time, so its L2 doesn't predict the real leaderboard score.

For every scoring point in a val scenario, this now replays that same
7-frame/3s reset-to-now window in isolation and scores only the final
frame's prediction, exactly like a real test clip.
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

# Must match tools/data_converter/etri_test_converter.py's HIS_FRAMES/
# STREAM_STRIDE exactly, since this replays that same window shape.
HIS_FRAMES = 30
STREAM_STRIDE = 5


def parse_frame_offsets(spec):
    """Raw-frame offsets (e.g. "-30,-15,0", or "0" for the current frame
    only -- no prev_bev history) -> sorted offset list. Mirrors
    etri_test_submit.py's parse_frame_offsets so holdout L2 can be measured
    under the exact same streaming condition actually used for submission
    (validated: --frame-offsets 0 costs ~0.8% L2 for up to ~11x lower
    cumulative T_infer, per Q&A.md section 2)."""
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

# Must match etri_vad_converter.py's COMMAND_VOCAB exactly -- index i here
# is mode i of the model's ego_fut_mode=7 trajectory decoder.
COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN', 'STOP',
)


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def parse_args():
    parser = argparse.ArgumentParser(description='hold-out L2 evaluation')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--stride', type=int, default=5,
                        help='evaluation interval in frame_idx')
    parser.add_argument('--min-frame', type=int, default=HIS_FRAMES,
                        help='minimum frame_idx to have a full lookback window')
    parser.add_argument(
        '--frame-offsets', default=None,
        help='comma-separated raw-frame offsets (0.1s units) to replay per '
             'window, e.g. "0" for the current frame only (no prev_bev '
             'history, matching etri_test_submit.py\'s default submission '
             'condition). Must include 0. Default: every frame (the full '
             '7-frame stream, -30,-25,...,0) -- NOT what gets submitted '
             'unless etri_test_submit.py is also run with its default.')
    return parser.parse_args()


def main():
    args = parse_args()
    stream_offsets = (
        parse_frame_offsets(args.frame_offsets) if args.frame_offsets
        else list(range(-HIS_FRAMES, 1, STREAM_STRIDE)))
    print(f'--frame-offsets {args.frame_offsets} -> replaying offsets '
          f'{stream_offsets} per window')
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

            window_frames = [frame_idx + off for off in stream_offsets]
            if any(f not in frame_to_gi for f in window_frames):
                n_skipped_incomplete_window += 1
                continue

            # Isolated clip: reset before every window, like a real test clip.
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
    print(f'skipped (incomplete 3s lookback window): {n_skipped_incomplete_window}')
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
