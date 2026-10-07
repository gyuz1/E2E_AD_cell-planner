"""Measure T_infer for the Error Score formula:
Error Score = L2 x (1 + max(0, T_infer - 100) / 200), T_infer in ms on a
single RTX 4090.

T_infer is the PER-CLIP CUMULATIVE time of every model forward pass from
reset_stream (prev_bev=None) through the final frame's trajectory output --
NOT a single frame's latency. Confirmed by the competition organizers'
official Q&A (2026-08-24): "T_infer is measured as the cumulative time of
every model forward pass performed from initialization, through processing
past frames, up to producing the final trajectory output -- not just the
last frame." An earlier version of this script reported the median
single-warm-frame latency as "T_infer", which understated the real number
by roughly the clip's frame count (a clip's cumulative time is close to
frame_count x per-frame time, since cold and warm frames cost about the
same here).

The same Q&A also confirms frame count/sampling interval within the
provided 3s history window is free (image input for every used frame is
still mandatory) -- so unlike per-frame latency, cumulative T_infer can be
cut by using fewer history frames per clip, not just by making each
forward pass faster.

Mirrors etri_test_submit.py's exact inference pattern (reset_stream once per
clip, then replay each of that clip's frames through the model with
temporal state carried across them) so the measured cost matches what
actually runs at submission time -- not a synthetic single-frame benchmark.

Each timed step covers the full per-frame cost, not just the network
forward pass: dataset[gi] (live undistort/crop/resize -- this script must
be pointed at a non-cached config, since our geometry cache is a local
training-speed optimization that has no reason to exist in the grading
environment) through collation and the model call.

Still reports the cold/warm single-frame breakdown too (useful for finding
where time goes -- e.g. whether cost is dominated by BEV construction vs.
dataloading), but the headline T_infer figure is now the per-clip
cumulative one.

Caveat: this machine has RTX 3090s, not the RTX 4090 the competition
grades on. Treat these numbers as a same-model-different-GPU estimate, not
the official figure -- report them as such.
"""
import argparse
import importlib
import time
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


HIS_FRAMES = 30  # raw (0.1s) frames = 3.0s -- the provided past-info range
STREAM_STRIDE = 5  # raw frames between consecutive test-clip frames (0.5s)


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def parse_frame_offsets(spec):
    """Raw-frame offsets (e.g. "-30,-15,0") -> sequential test-clip indices
    (0=oldest=-30 ... 6=current=0), matching etri_test_converter.py's
    STREAM_FRAMES = range(-30, 1, 5) layout."""
    offsets = sorted(int(x) for x in spec.split(','))
    if offsets[-1] != 0:
        raise ValueError(
            f'--frame-offsets must include the current frame (0): {spec}')
    if offsets[0] < -HIS_FRAMES:
        raise ValueError(
            f'offset {offsets[0]} exceeds the provided '
            f'{HIS_FRAMES / 10:.1f}s history range: {spec}')
    indices = []
    for off in offsets:
        if off % STREAM_STRIDE != 0:
            raise ValueError(
                f'offset {off} is not a multiple of {STREAM_STRIDE} raw '
                f'frames ({STREAM_STRIDE / 10:.1f}s) -- the test ann-file '
                f'only has frames on that grid: {spec}')
        indices.append(off // STREAM_STRIDE + HIS_FRAMES // STREAM_STRIDE)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure T_infer (per-clip cumulative ms)')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--num-clips', type=int, default=50,
                         help='clips to measure (after warmup)')
    parser.add_argument('--warmup-clips', type=int, default=5,
                         help='clips run first and discarded, to let '
                              'cudnn autotuning/caching settle')
    parser.add_argument(
        '--frame-offsets', default=None,
        help='comma-separated raw-frame offsets (0.1s units, matching the '
             'test converter\'s STREAM_FRAMES convention) to replay per '
             'clip, e.g. "-30,-15,0" for 3 frames spanning the full 3.0s '
             'window. Must include 0 (current frame). Default: every frame '
             'the test ann-file provides (the current 7-frame baseline, '
             '-30,-25,...,0). Test clips store sequential indices 0..6 '
             '(0=oldest=-30, 6=current=0) rather than the offsets '
             'themselves -- converted internally.')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true',
                         help='run inference in fp16 via mmcv.wrap_fp16_model')
    parser.add_argument('--compile', action='store_true',
                         help='wrap the model with torch.compile')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config values, e.g. to match a checkpoint trained '
             'under an older config revision (xxx=yyy format)')
    return parser.parse_args()


def timed_step(model, dataset, gi):
    """Times the full per-frame cost a real deployment would pay, split into
    the two halves that scale with different hardware: CPU-bound data
    loading + preprocessing (dataset[gi] -- undistort/crop/resize, done
    live here since this must use a non-cached config; the geometry cache
    is a local training-speed optimization that won't exist in the grading
    environment -- plus collation) versus the GPU-bound forward pass. A
    faster GPU (e.g. the competition's RTX 4090 vs this machine's RTX
    3090) only speeds up the second half; the first half tracks this
    machine's CPU/disk instead, which the grading server's spec is
    unknown. torch.cuda.synchronize() brackets the GPU part since CUDA
    kernels launch asynchronously and a plain wall-clock read would
    undercount them."""
    start = time.perf_counter()
    sample = dataset[gi]
    collated = collate([sample], samples_per_gpu=1)
    torch.cuda.synchronize()
    data_ms = (time.perf_counter() - start) * 1000.0
    fwd_start = time.perf_counter()
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **collated)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - fwd_start) * 1000.0
    elapsed_ms = data_ms + fwd_ms
    return result, elapsed_ms, data_ms, fwd_ms


def run_clips(model, dataset, clips, clip_tokens):
    cold_ms, warm_ms, clip_total_ms, clip_avg_ms = [], [], [], []
    data_ms_all, fwd_ms_all = [], []
    for clip_token in clip_tokens:
        sample_ids = clips[clip_token]
        reset_stream(model.module)
        total = 0.0
        for i, gi in enumerate(sample_ids):
            _, elapsed_ms, data_ms, fwd_ms = timed_step(model, dataset, gi)
            (cold_ms if i == 0 else warm_ms).append(elapsed_ms)
            data_ms_all.append(data_ms)
            fwd_ms_all.append(fwd_ms)
            total += elapsed_ms
        clip_total_ms.append(total)
        clip_avg_ms.append(total / len(sample_ids))
    return cold_ms, warm_ms, clip_total_ms, clip_avg_ms, data_ms_all, fwd_ms_all


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
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
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
    model = model.cuda(args.device)
    model.eval()
    if args.compile:
        # forward_test's control flow depends on tensor shapes/None-ness
        # that change between the first ("cold", no prev_bev) and later
        # ("warm") calls, so let dynamo recompile per distinct shape
        # rather than erroring on the guard failure.
        model = torch.compile(model, dynamic=False)
    model = MMDataParallel(model, device_ids=[args.device])

    clips = OrderedDict()
    for gi, info in enumerate(dataset.data_infos):
        clips.setdefault(info['scene_token'], []).append(gi)

    if args.frame_offsets:
        keep_indices = set(parse_frame_offsets(args.frame_offsets))
        clips = OrderedDict(
            (token, [gi for i, gi in enumerate(gis) if i in keep_indices])
            for token, gis in clips.items())
        n_frames = len(keep_indices)
        print(f'--frame-offsets {args.frame_offsets} '
              f'({n_frames} frames/clip, indices {sorted(keep_indices)})')
    clip_tokens = list(clips.keys())

    warmup_tokens = clip_tokens[:args.warmup_clips]
    measure_tokens = clip_tokens[
        args.warmup_clips:args.warmup_clips + args.num_clips]
    if not measure_tokens:
        raise ValueError(
            f'ann-file only has {len(clip_tokens)} clips, not enough for '
            f'--warmup-clips {args.warmup_clips} + --num-clips {args.num_clips}')

    print(f'warming up on {len(warmup_tokens)} clips...')
    run_clips(model, dataset, clips, warmup_tokens)

    print(f'measuring on {len(measure_tokens)} clips...')
    cold_ms, warm_ms, clip_total_ms, clip_avg_ms, data_ms_all, fwd_ms_all = (
        run_clips(model, dataset, clips, measure_tokens))

    gpu_name = torch.cuda.get_device_name(args.device)
    print()
    print(f'GPU: {gpu_name} (competition grades on RTX 4090 -- '
          f'treat these as an estimate, not the official number)')
    summarize('cold frame (first frame of clip, no prev_bev)', cold_ms)
    summarize('warm frame (steady-state, prev_bev populated)', warm_ms)
    summarize('per-clip cumulative SUM (reset -> final output)', clip_total_ms)
    summarize('per-clip cumulative SUM / frame_count (avg per frame)',
               clip_avg_ms)
    print()
    print('CPU-bound vs GPU-bound split (per frame) -- only the GPU half '
          'scales with a faster GPU (e.g. this machine\'s RTX 3090 -> the '
          'competition\'s RTX 4090); the CPU half tracks this machine\'s '
          'CPU/disk, which the grading server\'s spec is unknown:')
    summarize('  data load + preprocess (CPU-bound)', data_ms_all)
    summarize('  model forward only (GPU-bound)', fwd_ms_all)
    if clip_total_ms:
        frames_per_clip = (len(cold_ms) + len(warm_ms)) / len(measure_tokens)
        t_infer_sum = float(np.median(clip_total_ms))
        t_infer_avg = float(np.median(clip_avg_ms))
        print()
        print(f'avg frames/clip: {frames_per_clip:.1f}')
        print(
            'Two candidate T_infer readings of the organizers\' official '
            'Q&A ("cumulative time of every model forward from '
            'initialization through the final trajectory output") -- '
            'confirmed NOT "last frame only", but whether "cumulative '
            'time" itself is the raw per-clip SUM or that sum averaged '
            'over the frame count is still ambiguous pending organizer '
            'follow-up. The baseline reference (T_infer=49.3ms) is only '
            'physically plausible under the AVERAGE reading -- a raw '
            '7-frame sum would require ~7ms/frame, implausible given '
            'FLOPs parity with our own ~479 GFLOPs/frame model. Reporting '
            'both; treat SUM as the conservative (worse-case) number until '
            'confirmed.')
        for label, t_infer in (
                ('SUM', t_infer_sum), ('SUM/frame_count (AVG)', t_infer_avg)):
            penalty = max(0.0, t_infer - 100.0) / 200.0
            print(f'  T_infer [{label}]: {t_infer:.2f}ms -- Error Score '
                  f'multiplier if real GPU: x{1.0 + penalty:.4f} '
                  f'({"no penalty" if penalty == 0 else "penalized"})')


if __name__ == '__main__':
    main()
