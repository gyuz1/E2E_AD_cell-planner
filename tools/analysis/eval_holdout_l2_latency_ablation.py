"""Hold-out planning L2 AND model-forward-only latency, together, for the
prev_bev ablation (condition 1: single cold frame vs condition 2: cumulative
streamed frames).

Combines eval_holdout_l2_prevbev_ablation.py's methodology (isolated window
per scored val sample, reset before each window -- NOT a continuous scene
stream) with measure_t_infer_fwd_only.py's timing discipline (only the
model(...) call is timed, per the 2026-08-25 organizer Q&A; dataset[gi]/
collate happen outside the clock). One pass over the val set gets both
metrics under identical conditions instead of two separate scripts
potentially drifting apart.

Why not reuse measure_t_infer_fwd_only.py directly against a val_split.pkl:
that script assumes its ann-file is already TEST-style, pre-chunked into
fixed 7-sample clips (data_infos grouped by scene_token, each group treated
as one clip's sequential 0..6 stream indices). A val_split.pkl scene is one
long continuous sequence instead (hundreds of frame_idx per scene_token), so
that grouping+indexing assumption doesn't hold -- it would silently time the
wrong frames. This script indexes by frame_idx + offset instead, exactly
like eval_holdout_l2_prevbev_ablation.py already does correctly.
"""
import argparse
import importlib
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
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
    parser.add_argument('--min-frame', type=int, default=HIS_FRAMES)
    parser.add_argument('--frame-offsets', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--max-samples', type=int, default=None,
                         help='cap for a quick smoke test')
    parser.add_argument('--scene-shard', default=None,
                         help='"i/n" -- process only every n-th scene '
                              '(index i, 0-based), for running multiple '
                              'shards in parallel across/within GPUs. Each '
                              'shard reports its own partial totals; merge '
                              'with --out-json + a separate aggregation '
                              'step.')
    parser.add_argument('--out-json', default=None,
                         help='write raw accumulator state here (for '
                              'merging shards) in addition to the printed '
                              'report')
    parser.add_argument('--warmup-samples', type=int, default=5,
                         help='scored samples run first and excluded from '
                              'the latency stats (still counted in L2), to '
                              'let cudnn autotuning/caching settle')
    return parser.parse_args()


def summarize(name, values):
    if not values:
        print(f'{name}: no samples')
        return
    arr = np.array(values)
    print(f'{name} (n={len(arr)}): '
          f'mean={arr.mean():.2f}ms  median={np.median(arr):.2f}ms  '
          f'p95={np.percentile(arr, 95):.2f}ms  max={arr.max():.2f}ms')


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
    if args.fp16:
        wrap_fp16_model(model)
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(args.device), device_ids=[args.device])
    model.eval()

    scenes = {}
    for gi, info in enumerate(dataset.data_infos):
        scenes.setdefault(info['scene_token'], []).append(gi)

    if args.scene_shard:
        shard_i, shard_n = (int(x) for x in args.scene_shard.split('/'))
        all_tokens = list(scenes.keys())
        keep_tokens = set(all_tokens[shard_i::shard_n])
        scenes = {k: v for k, v in scenes.items() if k in keep_tokens}
        print(f'--scene-shard {args.scene_shard} -> {len(scenes)}/'
              f'{len(all_tokens)} scenes')

    l2_sums = np.zeros(3)
    n_eval = 0
    n_skipped_incomplete_window = 0

    n_cmd = len(COMMAND_VOCAB)
    cmd_all = np.zeros(n_cmd, dtype=int)
    cmd_valid = np.zeros(n_cmd, dtype=int)
    cmd_l2_sums = np.zeros((n_cmd, 3))

    fwd_ms_all = []       # every timed forward call, across all samples
    sample_total_ms = []  # per-sample cumulative forward time (sum over offsets)

    n_done = 0
    for scene_token, sample_ids in mmcv.track_iter_progress(list(scenes.items())):
        frame_to_gi = {
            dataset.data_infos[gi]['frame_idx']: gi for gi in sample_ids}

        for gi in sample_ids:
            if args.max_samples and n_done >= args.max_samples:
                break
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

            warming_up = n_done < args.warmup_samples
            reset_stream(model.module)
            result = None
            collated = None
            sample_total = 0.0
            for f in window_frames:
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **collated)
                torch.cuda.synchronize()
                fwd_ms = (time.perf_counter() - t0) * 1000.0
                if not warming_up:
                    fwd_ms_all.append(fwd_ms)
                sample_total += fwd_ms
            if not warming_up:
                sample_total_ms.append(sample_total)
            n_done += 1

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
        if args.max_samples and n_done >= args.max_samples:
            break

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

    print()
    print('Latency (model.forward() only, dataset load/preprocess excluded '
          'per the 2026-08-25 organizer Q&A):')
    summarize('  single forward call', fwd_ms_all)
    summarize('  per-sample cumulative (sum over replayed offsets)', sample_total_ms)
    if sample_total_ms:
        t_infer = float(np.median(sample_total_ms))
        penalty = max(0.0, t_infer - 100.0) / 200.0
        print(f'  T_infer (median per-sample cumulative): {t_infer:.2f}ms -- '
              f'Error Score multiplier if real GPU: x{1.0 + penalty:.4f} '
              f'({"no penalty" if penalty == 0 else "penalized"})')

    if args.out_json:
        import json
        with open(args.out_json, 'w') as f:
            json.dump(dict(
                l2_sums=l2_sums.tolist(), n_eval=n_eval,
                n_skipped_incomplete_window=n_skipped_incomplete_window,
                cmd_all=cmd_all.tolist(), cmd_valid=cmd_valid.tolist(),
                cmd_l2_sums=cmd_l2_sums.tolist(),
                fwd_ms_all=fwd_ms_all, sample_total_ms=sample_total_ms,
            ), f)
        print(f'wrote raw accumulator state to {args.out_json}')


if __name__ == '__main__':
    main()
