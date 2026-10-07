"""Hold-out planning L2 + T_infer, measured together in one pass over the
same replayed windows -- so a 1-frame vs 2-frame comparison needs one
checkpoint load and one sweep over the val set, not two separate scripts.

Merges eval_holdout_l2.py's windowing/L2 logic (isolated reset-per-window
replay of a real val scenario, scored like an actual test clip) with
measure_t_infer_fwd_only.py's timing methodology (clock only model.forward,
bracketed by torch.cuda.synchronize since CUDA kernels launch async -- see
that script's docstring for the full model-forward-only rationale from the
2026-08-25 organizer Q&A).

Use the FAST_EVAL config (not the cached one) so the timed forward pass
matches what a real test-time inference actually costs; L2 doesn't depend
on which pipeline loads the image, so this doesn't cost anything on the
accuracy side.

Each --frame-offsets config gets its own --warmup-windows (default 5)
untimed windows before recording starts, matching
measure_t_infer_fwd_only.py's --warmup-clips. This is not just about
excluding one-time cold-start cost (cudnn algorithm search, kernel JIT,
allocator growth) from the numbers -- it is required for a fair
multi-config comparison in the first place: configs run sequentially
against the same model instance, so without per-config warmup, whichever
config is listed first absorbs the entire cold-start cost while later
configs start already warm, making e.g. "1-frame vs 2-frame" comparisons
reflect list order as much as real cost.
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

from projects.mmdet3d_plugin.VAD.cell_planner_utils import route_trajectory
from projects.mmdet3d_plugin.VAD.goal_anchor_utils import nearest_valid_goal

HIS_FRAMES = 30
STREAM_STRIDE = 5

# Match etri_test_submit.py: three frames half a second apart, the window
# queue_length=3 / target_stride=5 trained on. An L2 measured on a different
# window is not the L2 the submission will have -- the BEV encoder attends to
# prev_bev recursively, so replaying the ann-file's full seven frames hands
# the scored frame six accumulated history steps against training's two.
# Pass --frame-offsets to compare windows deliberately; the default is the
# one that answers "what will the submission score".
SUBMIT_FRAME_OFFSETS = '-10,-5,0'
COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN', 'STOP',
)


def reset_stream(model):
    # Rebuild this dict from VAD.__init__'s keys, not from memory: VAD.py
    # reads every one of them unconditionally on the streaming path, so a
    # missing key is a KeyError the moment a second frame arrives. That has
    # already happened once, when prev_bev2 was added.
    #
    #   prev_bev2          the frame before prev_bev, read only by the
    #                      3-frame motion descriptor
    #   prev_bev_pristine  un-rotated copy of prev_bev -- the encoder
    #                      yaw-aligns prev_bev in place, so the rotated
    #                      tensor cannot become the next step's prev_bev2
    model.prev_frame_info = {
        'prev_bev': None, 'prev_bev2': None, 'prev_bev_pristine': None,
        'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


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
    parser = argparse.ArgumentParser(
        description='hold-out L2 + T_infer, one or more frame configs '
                    'in a single model load')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--stride', type=int, default=5,
                        help='evaluation interval in frame_idx')
    parser.add_argument('--min-frame', type=int, default=HIS_FRAMES,
                        help='minimum frame_idx to have a full lookback window')
    parser.add_argument(
        '--frame-offsets', nargs='+', default=None,
        help=f'One or more window configurations, e.g. --frame-offsets 0 '
             f'"-5,0" for 1-frame vs 2-frame side by side. Each is '
             f'comma-separated raw-frame offsets (0.1s units), must '
             f'include 0. Default: the submission window alone '
             f'({SUBMIT_FRAME_OFFSETS}), so the reported L2 is the one the '
             f'submission will have. The ann-file also offers the full '
             f'7-frame stream "-30,-25,-20,-15,-10,-5,0" for comparison.')
    parser.add_argument('--fp16', action='store_true',
                         help='run inference in fp16 via mmcv.wrap_fp16_model')
    parser.add_argument('--warmup-windows', type=int, default=5,
                         help='untimed windows run (per config) before '
                              'L2/T_infer recording starts')
    parser.add_argument(
        '--zero-ego-lcf', action='store_true',
        help='diagnostic for ego_lcf-ON builds (the distillation teachers): '
             'zero ego_lcf_feat before every forward. A teacher that really '
             'uses ego status must degrade sharply; little change means the '
             'privileged input is not actually reaching the decoder, which '
             'is exactly the silent failure that wasted a 22h run once '
             '(ego_target_point was accepted by forward_train and never '
             'forwarded). Compliant configs read this only as a loss '
             'target, so zeroing it is a no-op for them -- a useful '
             'control.')
    parser.add_argument(
        '--zero-target-point', action='store_true',
        help='diagnostic for target_point_shortcut builds: zero '
             'ego_target_point before every forward. A model that really '
             'uses the goal must degrade sharply (the 2026-08-25 '
             'measurement on this shortcut was 0.114m -> 5.45m); no change '
             'means the shortcut is dead and the checkpoint is not the '
             'teacher it claims to be. Compliant configs never read this '
             'input, so zeroing it is a no-op for them -- which is itself '
             'a useful control.')
    parser.add_argument(
        '--zero-can-bus-ego', action='store_true',
        help='compliance diagnostic: zero can_bus[7:16] (accel, '
             'rotation_rate, velocity) before every forward. That slice is '
             'raw ego status -- the same quantities ego_lcf carries -- and '
             'VAD_transformer embeds the whole can_bus vector through an '
             'MLP into the BEV queries (use_can_bus=True, the distributed '
             "baseline's own default), a path the ego_lcf gradient proof "
             'does not cover since can_bus arrives via img_metas. A large '
             'L2 change here means the model leans on dataset-provided ego '
             'status despite ego_lcf_feat_idx=None.')
    parser.add_argument(
        '--disable-bev-refine', action='store_true',
        help='skip refine_ego_trajs_with_bev on the scored frame. The module '
             'stays built and loaded, so this isolates its cost and its L2 '
             'contribution on one checkpoint instead of needing a second '
             'one trained without it. The score is L2 x a penalty that only '
             'starts above 100ms, so buying time back here can be worth more '
             'than the refine itself: it is 3 grid_sample + MLP stages on '
             'the critical path.')
    parser.add_argument(
        '--test-commands', action='store_true',
        help='score under test-time conditions: the val commands include a '
             'derived STOP label that test commands never carry, so without '
             'this every STOP sample is handed the right mode for free. With '
             'it, STOP is chosen from the model\'s own speed estimate, as in '
             'etri_test_submit.py. STOP is 6.4%% of val samples.')
    parser.add_argument(
        '--select-goal-by-tp', action='store_true',
        help='score the candidate whose 5s goal is closest to the given '
             'target point, instead of the one the network chose. The '
             'candidates come from the head built with '
             'goal_expose_candidates=True and are generated WITHOUT the '
             'target point; this only picks among them. Organizer answers '
             '2026-08-26 and 08-27 allow the target point for selection '
             'among model outputs (selection only) and forbid it '
             'for generating or correcting one. Off by default: the '
             'compliant-by-construction number is the one without it.')
    parser.add_argument(
        '--select-cell-by-tp', action='store_true',
        help='required for a cell planner: score the trajectory that '
             'cell_planner_utils.route_trajectory selects from everything the '
             'model generated (STOP when |TP| < --stop-tp-thresh or a STOP command, '
             'U_TURN head, else the command cell containing the target '
             'point). The same rule trains and submits the model.')
    parser.add_argument(
        '--stop-by-tp', action='store_true',
        help='for a 7-mode model: submit the STOP mode when the target point '
             'is within --stop-tp-thresh (straight line), instead of the speed estimate. '
             'Selection among generated modes only.')
    parser.add_argument('--stop-tp-thresh', type=float, default=1.0)
    parser.add_argument('--stop-speed-thresh', type=float, default=0.1)
    parser.add_argument('--bev-only-history', action='store_true',
                         help='run every non-scored frame of a window with '
                              'bev_only=True, skipping the decoders whose '
                              'output that frame discards anyway. Same L2 '
                              '(bev_embed is unchanged), lower T_infer.')
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


def run_config(model, dataset, scenes, stream_offsets, args):
    """One frame-offset config's full sweep over the val set. Returns L2
    aggregates (matching eval_holdout_l2.py) plus a per-window T_infer list
    (matching measure_t_infer_fwd_only.py's clip_total_ms).

    The first args.warmup_windows valid windows are run for real (so the
    model, cudnn algorithm cache, and CUDA allocator are actually warmed
    up) but excluded from every returned aggregate -- L2 sums, command
    counts, and window_total_ms all start counting only after warmup
    completes. This config's own warmup is independent of any other
    config's, so each config in a multi-config run is timed under the
    same cold-vs-warm conditions rather than whichever ran first eating
    the one-time startup cost.
    """
    l2_sums = np.zeros(3)
    n_eval = 0
    n_skipped = 0
    n_cmd = len(COMMAND_VOCAB)
    cmd_all = np.zeros(n_cmd, dtype=int)
    cmd_valid = np.zeros(n_cmd, dtype=int)
    cmd_l2_sums = np.zeros((n_cmd, 3))
    window_total_ms = []
    n_warmed = 0
    # L2 by how far away the target point is: stationary clips are ~9% of
    # val but ~26% of test, so the bands are reported separately.
    band_edges = (1.0, 9.0)
    band_sums = np.zeros((3, 3))
    band_n = np.zeros(3, dtype=int)
    head = model.module.pts_bbox_head
    if getattr(head, 'cell_planner', False):
        if not args.select_cell_by_tp:
            raise SystemExit('this is a cell planner: score it with --select-cell-by-tp')
        cell_edges = {c: (f.cpu(), l.cpu()) for c, (f, l) in head.cell_edges().items()}
    elif args.select_cell_by_tp:
        raise SystemExit('--select-cell-by-tp needs a cell planner config')

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

            window_frames = [frame_idx + off for off in stream_offsets]
            if any(f not in frame_to_gi for f in window_frames):
                # Only count skips against the real (post-warmup) tally --
                # a warmup-phase skip isn't evidence about the measured set.
                if n_warmed >= args.warmup_windows:
                    n_skipped += 1
                continue

            # Isolated window: reset before every one, like a real test clip.
            reset_stream(model.module)
            result = None
            collated = None
            total_ms = 0.0
            for i, f in enumerate(window_frames):
                # Every frame but the last is a history frame: its only
                # contribution is the bev_embed the next frame's temporal
                # fusion consumes, so with --bev-only-history its decoders
                # (55% of a forward) are skipped instead of computed and
                # thrown away. The scored frame always runs in full.
                is_scored = (i == len(window_frames) - 1)
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                if args.zero_can_bus_ego:
                    for meta in collated['img_metas'][0].data[0]:
                        meta['can_bus'][7:16] = 0.0
                if args.zero_ego_lcf:
                    lcf = collated.get('ego_lcf_feat')
                    if lcf is None:
                        raise KeyError(
                            'ego_lcf_feat is not in the collated batch, so '
                            '--zero-ego-lcf cannot prove anything. Check the '
                            'pipeline Collect keys.')
                    holder = lcf
                    while not torch.is_tensor(holder):
                        holder = (holder.data if hasattr(holder, 'data')
                                  else holder[0])
                    holder.zero_()
                if args.zero_target_point:
                    tp = collated.get('ego_target_point')
                    if tp is None:
                        raise KeyError(
                            'ego_target_point is not in the collated batch, '
                            'so --zero-target-point cannot prove anything. '
                            'Check the pipeline Collect keys.')
                    holder = tp
                    while not torch.is_tensor(holder):
                        holder = (holder.data if hasattr(holder, 'data')
                                  else holder[0])
                    holder.zero_()
                torch.cuda.synchronize()
                fwd_start = time.perf_counter()
                with torch.no_grad():
                    out = model(return_loss=False, rescale=True,
                                bev_only=(args.bev_only_history
                                          and not is_scored),
                                **collated)
                torch.cuda.synchronize()
                total_ms += (time.perf_counter() - fwd_start) * 1000.0
                if is_scored:
                    result = out

            if n_warmed < args.warmup_windows:
                n_warmed += 1
                continue  # ran for real (warms cudnn/allocator), not recorded

            window_total_ms.append(total_ms)
            cmd_all[gt_cmd_idx] += 1

            ego_fut_preds = result[0]['pts_bbox']['ego_fut_preds']
            cmd = np.array(collated['ego_fut_cmd'][0].data[0]).reshape(
                -1, ego_fut_preds.shape[0])[0]
            mode = int(cmd.argmax())
            tp_dist = float(np.linalg.norm(np.asarray(info['gt_ego_target_point'], dtype=np.float64).reshape(-1)[:2]))  # report grouping only
            band = int(np.searchsorted(band_edges, tp_dist, side='right'))
            if args.test_commands:
                # Reproduce what the submission can actually do. STOP is a
                # train-pkl-only label derived from the future, so a test
                # command never says STOP; the pre-override command is not
                # stored, so LANE_KEEP stands in for it (the context most stops
                # occur in). STOP is then chosen the way etri_test_submit.py
                # chooses it -- from the model's own speed estimate.
                if mode == 6:
                    mode = 0
                state = result[0]['pts_bbox'].get('ego_state_pred')
                if state is not None and args.speed_col is not None and \
                        float(state.reshape(-1)[args.speed_col]) < args.stop_speed_thresh:
                    mode = 6
            if args.select_goal_by_tp:
                tp = torch.as_tensor(np.asarray(info['gt_ego_target_point'],
                                                dtype=np.float32).reshape(-1)[:2])
                cand = result[0]['pts_bbox'].get('goal_cand_trajs')
                pts = result[0]['pts_bbox'].get('goal_cand_points')
                if cand is None or pts is None:
                    raise KeyError(
                        '--select-goal-by-tp needs goal_cand_trajs: set '
                        'goal_expose_candidates=True in the eval config')
                # Padded anchors are not candidates; without the mask a
                # padded (0,0) goal could win near a stationary target.
                valid = result[0]['pts_bbox'].get('goal_cand_mask')
                if valid is None:
                    raise KeyError('--select-goal-by-tp needs goal_cand_mask')
                pick = nearest_valid_goal(pts[mode].cpu(), tp, valid[mode].cpu())
                ego_fut_preds = cand[:, int(pick)]
            if args.stop_by_tp:
                tp = np.asarray(info['gt_ego_target_point'], dtype=np.float64)
                if float(np.linalg.norm(tp.reshape(-1)[:2])) < args.stop_tp_thresh:
                    mode = 6
            pred = ego_fut_preds[mode].cpu().double().cumsum(0).numpy()
            if args.select_cell_by_tp:
                tp = torch.as_tensor(np.asarray(info['gt_ego_target_point'],
                                                dtype=np.float32).reshape(1, -1))
                pb = result[0]['pts_bbox']
                sel, _ = route_trajectory(
                    {c: t[None].float() for c, t in pb['cell_trajs'].items()},
                    pb['cell_u_turn'][None].float(), pb['cell_stop'][None].float(),
                    tp, torch.tensor([mode]), cell_edges, head.cell_stop_tp_thresh)
                pred = sel[0].double().cumsum(0).numpy()
            gt = np.array(info['gt_ego_fut_trajs'], dtype=np.float64).cumsum(0)

            dist = np.linalg.norm(pred - gt, axis=-1)
            sample_l2 = np.array([dist[:2].mean(), dist[:4].mean(), dist[:6].mean()])
            l2_sums += sample_l2
            n_eval += 1

            cmd_valid[gt_cmd_idx] += 1
            cmd_l2_sums[gt_cmd_idx] += sample_l2
            band_sums[band] += sample_l2
            band_n[band] += 1

    return dict(l2_sums=l2_sums, n_eval=n_eval, n_skipped=n_skipped,
                cmd_all=cmd_all, cmd_valid=cmd_valid, cmd_l2_sums=cmd_l2_sums,
                window_total_ms=window_total_ms, band_sums=band_sums, band_n=band_n)


def main():
    args = parse_args()
    if sum([args.select_cell_by_tp, args.select_goal_by_tp, args.stop_by_tp]) > 1 or (
            args.test_commands and (args.select_cell_by_tp or args.stop_by_tp)):
        raise SystemExit('--select-cell-by-tp, --select-goal-by-tp and --stop-by-tp '
                         'are exclusive; the TP rules replace --test-commands')
    specs = args.frame_offsets or [SUBMIT_FRAME_OFFSETS]
    configs = [(f'{len(parse_frame_offsets(s))}frame', parse_frame_offsets(s))
               for s in specs]

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    frames_needed = cfg.model.pts_bbox_head.get('aux_bev_motion_frames') or 2
    for label, offsets in configs:
        if len(offsets) < frames_needed:
            print(f'WARNING {label}: the model was trained with a {frames_needed}-frame '
                  f'motion descriptor; with {len(offsets)} frames its acceleration block '
                  f'is zero at inference (train/inference mismatch). etri_test_submit.py '
                  f'refuses this window.')
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)
    dataset = build_dataset(cfg.data.test)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint is strict=False: a module whose shape differs between
    # the eval config and the checkpoint loads as random weights with only a
    # warning, and the L2 that follows looks like a real result. On
    # 2026-09-15 an eval config that silently lost aux_bev_motion_grid=8 would
    # have done exactly that to ego_status_est_net, which feeds the planner.
    # Evaluation refuses instead: every model weight must come from the
    # checkpoint with the right shape. Checkpoint-only keys (training-only
    # heads such as ego_status_decode_head) are expected and only listed.
    ck = torch.load(args.checkpoint, map_location='cpu')
    ck = ck.get('state_dict', ck)
    msd = model.state_dict()
    bad_shape = sorted(k for k in msd.keys() & ck.keys()
                       if tuple(msd[k].shape) != tuple(ck[k].shape))
    not_in_ckpt = sorted(k for k in msd.keys() - ck.keys()
                         if not k.endswith('num_batches_tracked'))
    ckpt_only = sorted(ck.keys() - msd.keys())
    print(f'checkpoint check: {len(bad_shape)} shape mismatches, '
          f'{len(not_in_ckpt)} model-only keys, {len(ckpt_only)} checkpoint-only keys')
    # ema_* are EMAHook's raw-weight copies (the ordinary slots already hold
    # the EMA weights), so they are counted, not listed.
    n_ema = sum(k.startswith('ema_') for k in ckpt_only)
    print(f'  checkpoint-only: {n_ema} ema_* keys (EMAHook raw copies)')
    for k in ckpt_only:
        if not k.startswith('ema_'):
            print(f'  checkpoint-only (unused at eval): {k}')
    if bad_shape or not_in_ckpt:
        for k in bad_shape:
            print(f'  shape mismatch: {k} model{tuple(msd[k].shape)} '
                  f'ckpt{tuple(ck[k].shape)}')
        for k in not_in_ckpt:
            print(f'  missing from the checkpoint (randomly initialized): {k}')
        raise SystemExit('stop: the eval config and the checkpoint differ structurally')
    del ck, msd
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fp16:
        wrap_fp16_model(model)
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    # Carried on args because run_config is where it is read.
    idx = list(cfg.model.pts_bbox_head.get('aux_bev_motion_idx') or [])
    args.speed_col = idx.index(7) if 7 in idx else None
    if args.disable_bev_refine:
        model.pts_bbox_head._debug_disable_bev_refine = True
        print('refine_ego_trajs_with_bev disabled (the module is still loaded)')
    model = MMDataParallel(model.cuda(args.device), device_ids=[args.device])
    model.eval()

    scenes = {}
    for gi, info in enumerate(dataset.data_infos):
        scenes.setdefault(info['scene_token'], []).append(gi)

    results = {}
    for label, stream_offsets in configs:
        print(f'\n=== {label} (offsets {stream_offsets}) ===')
        results[label] = run_config(model, dataset, scenes, stream_offsets, args)

    # --- L2 report, per config -- same format/wording as eval_holdout_l2.py
    # itself would print for that config alone, so results stay directly
    # comparable to every L2-only run already on record.
    for label, _ in configs:
        r = results[label]
        print(f'\n=== {label} ===')
        if r['n_eval'] == 0:
            print(f'0 windows recorded after warmup -- --warmup-windows '
                  f'({args.warmup_windows}) likely exceeds the val set for '
                  f'this config; skipping')
            continue

        l2 = r['l2_sums'] / max(r['n_eval'], 1)
        print()
        print(f'evaluated samples: {r["n_eval"]}')
        print(f'skipped (incomplete lookback window): {r["n_skipped"]}')
        print('Official cumulative ADE:')
        print(f'  L2@1s : {l2[0]:.6f} m')
        print(f'  L2@2s : {l2[1]:.6f} m')
        print(f'  L2@3s : {l2[2]:.6f} m')
        print(f'  Final Planning L2 avg: {l2.mean():.6f} m')

        print()
        print('----------- ETRI Planning L2 by Command -----------')
        cmd_header = (f"{'Command':<15} {'valid/all':>12} {'L2@1s':>10} "
                      f"{'L2@2s':>10} {'L2@3s':>10} {'L2_avg':>10}")
        print(cmd_header)
        print('-' * len(cmd_header))
        for i, name in enumerate(COMMAND_VOCAB):
            n_v, n_a = r['cmd_valid'][i], r['cmd_all'][i]
            cmd_l2 = r['cmd_l2_sums'][i] / max(n_v, 1)
            print(f"{name:<15} {f'{n_v}/{n_a}':>12} {cmd_l2[0]:>10.6f} "
                  f"{cmd_l2[1]:>10.6f} {cmd_l2[2]:>10.6f} "
                  f"{cmd_l2.mean():>10.6f}")
        print('-' * len(cmd_header))
        print('L2 by target-point distance |TP| (test share: <1m ~26%):')
        for i, name in enumerate(('< 1 m', '1 - 9 m', '>= 9 m')):
            n = r['band_n'][i]
            print(f"  {name:<8} n={n:<6} L2_avg {(r['band_sums'][i] / max(n, 1)).mean():.6f}")

    # --- T_infer, the part eval_holdout_l2.py alone never measured --------
    print()
    print(f'GPU: {torch.cuda.get_device_name(args.device)}')
    print(f'(each config warmed up on {args.warmup_windows} real, '
          f'unrecorded windows first)')
    print()
    t_header = (f'{"frames":<10}{"T_mean":>10}{"T_median":>10}'
                f'{"penalty":>20}')
    print(t_header)
    print('-' * len(t_header))
    for label, _ in configs:
        r = results[label]
        if r['n_eval'] == 0:
            continue
        arr = np.array(r['window_total_ms'])
        t_median = float(np.median(arr))
        penalty = max(0.0, t_median - 100.0) / 200.0
        status = ('OK (no penalty)' if penalty == 0
                  else f'PENALIZED x{1.0 + penalty:.4f}')
        print(f'{label:<10}{arr.mean():>8.2f}ms{np.median(arr):>8.2f}ms'
              f'{status:>20}')


if __name__ == '__main__':
    main()
