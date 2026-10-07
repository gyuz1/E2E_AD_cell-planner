import argparse
import importlib
import json
import os
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.VAD.cell_planner_utils import route_trajectory


HIS_FRAMES = 30
STREAM_STRIDE = 5

# The window the model was TRAINED on, and therefore the one it is
# submitted with: queue_length=3 at target_stride=5 means three frames
# half a second apart.
#
# The test ann-file offers seven (-30,-25,...,0). Replaying all of them
# hands the scored frame six accumulated history steps where training
# gave it two, because the BEV encoder attends to prev_bev recursively --
# different input, with no shape change and no warning. It also costs
# roughly twice the inference time.
#
# Pass --frame-offsets explicitly to replay anything else; that is for
# ablations, and whatever it prints is not the submission condition.
SUBMIT_FRAME_OFFSETS = '-10,-5,0'

# Must match etri_vad_converter_10hz.py's COMMAND_VOCAB exactly -- index i
# here is mode i of the model's ego_fut_mode=7 trajectory decoder. STOP is
# never a raw command value (it's a train-pkl-only derived label, since it
# needs GT future trajectory that doesn't exist at test time), so
# ego_fut_cmd here can only ever have one of the first 6 bits set --
# STOP_DISP_THRESH below is what recovers STOP-appropriate behavior at
# test time despite that.
COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN', 'STOP',
)
# Default: STOP is chosen from the model's own speed estimate. --stop-by-tp
# (7-mode models) and --select-cell-by-tp (cell planner) choose with the target
# point instead -- selection among trajectories the model already generated,
# never generation or correction.
STOP_SPEED_THRESH = 0.1
SPEED_COL = None


def reset_stream(model):
    # Rebuild this dict from VAD.__init__'s keys, not from memory: VAD.py
    # reads every one of them unconditionally on the streaming path, so a
    # missing key is a KeyError the moment a second frame arrives.
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
                f'frames ({STREAM_STRIDE / 10:.1f}s): {spec}')
        indices.append(off // STREAM_STRIDE + HIS_FRAMES // STREAM_STRIDE)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description='ETRI challenge submission')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--frame-offsets', default=SUBMIT_FRAME_OFFSETS,
        help=f'comma-separated raw-frame offsets (0.1s units) to replay per '
             f'clip. Must include 0. Default {SUBMIT_FRAME_OFFSETS} -- the '
             f'three-frame window the model was trained on. Override only '
             f'for ablations: "0" is the current frame alone (~0.8%% worse '
             f'L2, ~11x lower cumulative latency), and the ann-file\'s full '
             f'stream is "-30,-25,-20,-15,-10,-5,0", which gives the scored '
             f'frame six accumulated history steps where training gave it '
             f'two.')
    parser.add_argument(
        '--stop-speed-thresh', type=float, default=STOP_SPEED_THRESH,
        help='select the STOP trajectory when the model\'s own estimated '
             'speed (m/s) is below this. Never reads the target point.')
    parser.add_argument(
        '--fp16', action='store_true',
        help='run inference in fp16 (mmcv.wrap_fp16_model), as the holdout '
             'evaluation that chose the model measured it')
    parser.add_argument(
        '--select-cell-by-tp', action='store_true',
        help='required for a cell planner: submit the trajectory '
             'cell_planner_utils.route_trajectory selects from everything the '
             'model generated -- the rule the model was trained with. The '
             'target point only selects; it never enters the network.')
    parser.add_argument(
        '--stop-by-tp', action='store_true',
        help='for a 7-mode model: submit the STOP mode when the target point '
             'is within --stop-tp-thresh (straight line), instead of the speed estimate.')
    parser.add_argument('--stop-tp-thresh', type=float, default=1.0)
    parser.add_argument(
        '--bev-only-history', action='store_true',
        help='run every non-submitted frame of a clip with bev_only=True, '
             'skipping the decoders whose output that frame discards '
             'anyway. Identical submitted trajectory (bev_embed is '
             'unchanged), substantially lower T_infer.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)

    # A 3-frame motion descriptor needs two history frames in the window or
    # prev_bev2 is never populated and the acceleration third of the
    # descriptor is zero at submission time while training filled it. The
    # default window satisfies this by construction; the check stands for
    # the overrides, and it runs unconditionally now so a hand-shortened
    # window cannot reach the submission silently.
    _frames = cfg.model.pts_bbox_head.get('aux_bev_motion_frames') or 2
    if args.frame_offsets:
        _n = len(parse_frame_offsets(args.frame_offsets))
        if _n < _frames:
            raise SystemExit(
                f'the config wants aux_bev_motion_frames={_frames} but the window has '
                f'{_n} frames ({args.frame_offsets}). The submission would be '
                f'built with a zero acceleration block -- give at least '
                f'{_frames} frames.')


    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint is strict=False. For a submission that is the worst
    # possible default: a checkpoint that disagrees with the config loads
    # anyway, the mismatched modules stay randomly initialized, and the only
    # symptom is a worse score nobody can explain afterwards. Check it here,
    # where there is still time to do something about it.
    _raw = torch.load(args.checkpoint, map_location='cpu')
    _sd = _raw.get('state_dict', _raw)
    _msd = model.state_dict()
    _bad = [(k, tuple(_sd[k].shape), tuple(_msd[k].shape))
            for k in _sd if k in _msd and _sd[k].shape != _msd[k].shape]
    _missing = [k for k in _msd if k not in _sd]
    if _bad or _missing:
        for k, a_, b_ in _bad[:10]:
            print(f'  shape mismatch {k}: ckpt{a_} vs model{b_}')
        for k in _missing[:10]:
            print(f'  missing from the checkpoint: {k}')
        raise SystemExit(
            f'the checkpoint does not match the config ({len(_bad)} mismatched, '
            f'{len(_missing)} missing). Refusing to build a submission from '
            f'randomly initialized modules.')
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fp16:
        wrap_fp16_model(model)
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    # Which column of ego_state_pred is speed. ego_lcf layout puts speed at
    # index 7, and the head predicts the aux_bev_motion_idx subset in order.
    global SPEED_COL
    idx = list(cfg.model.pts_bbox_head.get('aux_bev_motion_idx') or [])
    SPEED_COL = idx.index(7) if 7 in idx else None
    if SPEED_COL is None:
        print('warning: aux_bev_motion_idx has no speed(7), disabling STOP selection')
    head = model.pts_bbox_head
    is_cell = getattr(head, 'cell_planner', False)
    if is_cell != args.select_cell_by_tp:
        raise SystemExit('a cell planner is submitted with --select-cell-by-tp, '
                         'and only a cell planner')
    if args.select_cell_by_tp and args.stop_by_tp:
        raise SystemExit('--stop-by-tp is for 7-mode models; the cell rule already selects STOP')
    cell_edges = ({c: (f.cpu(), l.cpu()) for c, (f, l) in head.cell_edges().items()}
                  if is_cell else None)
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()
    n_route = np.zeros(3, dtype=int)

    clips = OrderedDict()
    for gi, info in enumerate(dataset.data_infos):
        clips.setdefault(info['scene_token'], []).append(gi)

    if args.frame_offsets:
        keep_indices = set(parse_frame_offsets(args.frame_offsets))
        clips = OrderedDict(
            (token, [gi for i, gi in enumerate(gis) if i in keep_indices])
            for token, gis in clips.items())
        print(f'--frame-offsets {args.frame_offsets} -> '
              f'{len(keep_indices)} frame(s)/clip, indices {sorted(keep_indices)}')

    submission = {}
    for clip_token, sample_ids in mmcv.track_iter_progress(list(clips.items())):
        reset_stream(model.module)
        result = None
        collated = None
        for i, gi in enumerate(sample_ids):
            # Only the last frame's trajectory is submitted; earlier frames exist
            # solely to build up the bev_embed the next frame's temporal fusion
            # consumes. --bev-only-history skips those frames' decoders, which is
            # most of a forward pass and produces a bit-identical BEV, so the
            # submitted trajectory is unchanged while T_infer drops.
            is_scored = (i == len(sample_ids) - 1)
            collated = collate([dataset[gi]], samples_per_gpu=1)
            with torch.no_grad():
                out = model(return_loss=False, rescale=True,
                            bev_only=(args.bev_only_history and not is_scored),
                            **collated)
            if is_scored:
                result = out
        ego_fut_preds = result[0]['pts_bbox']['ego_fut_preds']
        cmd = np.array(collated['ego_fut_cmd'][0].data[0]).reshape(
            -1, ego_fut_preds.shape[0])[0]
        mode_idx = int(cmd.argmax())
        info = dataset.data_infos[sample_ids[-1]]
        if args.select_cell_by_tp:
            tp = torch.as_tensor(np.asarray(info['gt_ego_target_point'],
                                            dtype=np.float32).reshape(1, -1))
            pb = result[0]['pts_bbox']
            sel, route = route_trajectory(
                {c: t[None].float() for c, t in pb['cell_trajs'].items()},
                pb['cell_u_turn'][None].float(), pb['cell_stop'][None].float(),
                tp, torch.tensor([mode_idx]), cell_edges, head.cell_stop_tp_thresh)
            n_route[int(route['kind'][0])] += 1
            traj = sel[0].double().cumsum(0).numpy()
        else:
            if args.stop_by_tp:
                tp = np.asarray(info['gt_ego_target_point'], dtype=np.float64)
                if float(np.linalg.norm(tp.reshape(-1)[:2])) < args.stop_tp_thresh:
                    mode_idx = COMMAND_VOCAB.index('STOP')
            else:
                state = result[0]['pts_bbox'].get('ego_state_pred')
                if state is not None and SPEED_COL is not None:
                    if float(state.reshape(-1)[SPEED_COL]) < args.stop_speed_thresh:
                        mode_idx = COMMAND_VOCAB.index('STOP')
            traj = ego_fut_preds[mode_idx].cpu().double().cumsum(0).numpy()
        submission[clip_token] = traj.tolist()

    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    out_dict = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        if '__flops__' in existing:
            out_dict['__flops__'] = existing['__flops__']
    out_dict.update(submission)
    with open(args.out, 'w') as f:
        json.dump(out_dict, f)
    print(f'wrote {args.out} ({len(submission)} clips)')
    if is_cell:
        print(f'selection: cell {n_route[0]}, U_TURN {n_route[1]}, STOP {n_route[2]}')


if __name__ == '__main__':
    main()
