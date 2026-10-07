"""Where does a student's planning error come from: its speed estimate, or
what the planner does with it?

Streams real val windows exactly like eval_holdout_l2_and_tinfer.py (reset per
window, frames -10/-5/0) and, on each scored frame, reads four speeds against
the ground truth:

  aux     aux_bev_motion_head's estimate (outs['ego_state_pred'])
  slot    the status slot the planner actually consumes, read out through the
          trained ego_status_decode_head (built from the TRAIN config so the
          head exists; it is train-only in the eval network)
  plan    speed implied by the planned trajectory's first 0.5s step, in the
          ground-truth command's mode
  gt      gt_ego_lcf_feat[7], and the GT trajectory's first step

Reading it:
  aux/slot good, plan bad  -> the planner is not using the state it is given
  aux/slot bad             -> vision estimate fails at eval (compare
                              --no-bev-only-history: if that fixes it, the
                              history-frame path is broken)

Usage:
  python tools/diag_student_speed.py <train_cfg> <eval_cfg> <ckpt> [--n 300]
      [--no-bev-only-history] [--fp16] [--device 0]
"""
import argparse
import importlib

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

VAL_ANN = ('data/etri/.causal_regen_split_301_75_10hz/'
           'vad_etri_infos_temporal_val_split.pkl')
OFFSETS = (-10, -5, 0)


def reset_stream(m):
    m.prev_frame_info = {
        'prev_bev': None, 'prev_bev2': None, 'prev_bev_pristine': None,
        'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def stats(name, pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    err = pred - gt
    rmse = float(np.sqrt((err ** 2).mean()))
    r2 = 1 - (err ** 2).mean() / max(gt.var(), 1e-12)
    print(f'  {name:<28} RMSE {rmse:7.3f}  bias {err.mean():+7.3f}  '
          f'R2 {r2:6.3f}  (gt mean {gt.mean():6.2f} std {gt.std():5.2f})')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('eval_config')
    p.add_argument('ckpt')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--no-bev-only-history', action='store_true')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--ann-file', default=VAL_ANN,
                   help='point at the train pkl to run the same measurement on training data')
    p.add_argument('--cache-images', action='store_true',
                   help='replace the two JPEG loader stages with the LoadETRIGeometryCache '
                        'training used; everything after normalization is unchanged')
    p.add_argument('--raw-weights', action='store_true',
                   help='use the raw training weights EMAHook keeps in ema_*; the ordinary '
                        'checkpoint slots hold the EMA')
    p.add_argument('--encoder-dropout-on', action='store_true',
                   help='diagnostic: put only the BEV encoder dropout in train mode')
    p.add_argument('--dropout-pat', default='transformer.encoder',
                   help='with --encoder-dropout-on: only Dropout modules whose name contains this ("" = all)')
    args = p.parse_args()

    ecfg = Config.fromfile(args.eval_config)
    if hasattr(ecfg, 'plugin_dir'):
        importlib.import_module(ecfg.plugin_dir.replace('/', '.').rstrip('.'))
    ecfg.data.test.ann_file = args.ann_file
    ecfg.data.test.test_mode = True
    ecfg.data.test.pop('samples_per_gpu', None)
    ecfg.data.test.pop('map_ann_file', None)
    if args.cache_images:
        importlib.import_module('tools.cache.etri_geometry_cache')
        split = 'train' if 'train' in args.ann_file else 'val'
        cache = dict(
            type='LoadETRIGeometryCache',
            cache_root=('/workspace/VAD/work_dirs/etri_geometry_cache_v1'
                        if split == 'train' else
                        '/workspace/VAD/work_dirs/etri_geometry_cache_val_v1'),
            scale=0.4, strict=True, max_open_shards=8,
            require_complete_manifest=True,
            expected_scene_count=301 if split == 'train' else 75,
            expected_ann_file=('/workspace/VAD/data/etri/'
                               '.causal_regen_split_301_75/'
                               f'vad_etri_infos_temporal_{split}_split.pkl'),
            expected_frame_stride=5, expected_crop_size=(1920, 1080),
            expected_crop_keep_top=('camera_front_left', 'camera_front_right',
                                    'camera_rear_left', 'camera_rear_right',
                                    'camera_rear_wide'))
        pl = list(ecfg.data.test.pipeline)
        assert pl[0]['type'] == 'FastLoadMultiViewImageFromFiles', pl[0]
        assert pl[1]['type'] == 'FastUndistortCropScaleMultiViewImage', pl[1]
        ecfg.data.test.pipeline = [cache] + pl[2:]
        print('image source: geometry cache', [q['type'] for q in ecfg.data.test.pipeline])
    dataset = build_dataset(ecfg.data.test)

    tcfg = Config.fromfile(args.train_config)
    mc = tcfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    model = build_model(mc, test_cfg=ecfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt, map_location='cpu')
    if args.raw_weights:
        ck = torch.load(args.ckpt, map_location='cpu')
        ck = ck.get('state_dict', ck)
        sd = model.state_dict()
        n = 0
        for k in sd:
            ek = 'ema_' + k.replace('.', '_')
            if ek in ck and ck[ek].shape == sd[k].shape:
                sd[k] = ck[ek]
                n += 1
        model.load_state_dict(sd)
        print(f'swapped in raw weights: {n}/{len(sd)} tensors')
    if args.fp16:
        wrap_fp16_model(model)
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    head = model.pts_bbox_head
    idx = list(tcfg.model.pts_bbox_head.get('aux_bev_motion_idx') or [])
    sc = idx.index(7)
    captured = {}
    if head.ego_status_est_net is not None:
        head.ego_status_est_net.register_forward_hook(
            lambda m, i, o: captured.__setitem__('slot', o.detach().float()))
    model = MMDataParallel(model.cuda(args.device), device_ids=[args.device])
    model.eval()
    if args.encoder_dropout_on:
        k = 0
        for nm, m in model.module.named_modules():
            if isinstance(m, torch.nn.Dropout) and (args.dropout_pat in nm):
                m.train(); k += 1
        print(f'{k} encoder dropout modules in train mode')

    scenes = {}
    for gi, inf in enumerate(dataset.data_infos):
        scenes.setdefault(inf['scene_token'], {})[inf['frame_idx']] = gi
    cands = []
    for tok, f2g in scenes.items():
        for f, gi in f2g.items():
            inf = dataset.data_infos[gi]
            if (f >= 30 and f % 5 == 0 and inf.get('fut_valid_flag', False)
                    and all(f + o in f2g for o in OFFSETS)):
                cands.append((tok, f))
    stride = max(1, len(cands) // args.n)
    picked = cands[::stride][:args.n]
    print(f'{len(picked)} windows out of {len(cands)} candidates, bev_only_history='
          f'{not args.no_bev_only_history}, fp16={args.fp16}')

    rec = {k: [] for k in ('aux', 'slot', 'plan', 'gt', 'gt_step', 'l2')}
    for n, (tok, f) in enumerate(picked):
        f2g = scenes[tok]
        reset_stream(model.module)
        out = None
        for i, o in enumerate(OFFSETS):
            scored = i == len(OFFSETS) - 1
            batch = collate([dataset[f2g[f + o]]], samples_per_gpu=1)
            captured.clear()
            with torch.no_grad():
                r = model(return_loss=False, rescale=True,
                          bev_only=(not args.no_bev_only_history and not scored),
                          **batch)
            if scored:
                out, cb = r, batch
        info = dataset.data_infos[f2g[f]]
        pb = out[0]['pts_bbox']
        gt_speed = float(np.asarray(info['gt_ego_lcf_feat'])[7])
        gt_traj = np.asarray(info['gt_ego_fut_trajs'], dtype=np.float64)
        mode = int(np.asarray(info['gt_ego_fut_cmd']).argmax())
        preds = pb['ego_fut_preds'].cpu().double().numpy()
        rec['gt'].append(gt_speed)
        rec['gt_step'].append(np.linalg.norm(gt_traj[0]) / 0.5)
        rec['plan'].append(np.linalg.norm(preds[mode][0]) / 0.5)
        if pb.get('ego_state_pred') is not None:
            rec['aux'].append(float(pb['ego_state_pred'].reshape(-1)[sc]))
        if 'slot' in captured and head.ego_status_decode_head is not None:
            dec = head.ego_status_decode_head(
                captured['slot'].to(head.ego_status_decode_head.weight.dtype))
            rec['slot'].append(float(dec.reshape(-1)[sc]))
        d = np.linalg.norm(preds[mode].cumsum(0) - gt_traj.cumsum(0), axis=-1)
        rec['l2'].append(d[:6].mean() / 1.0)
        if (n + 1) % 50 == 0:
            print(f'  {n + 1}/{len(picked)}', flush=True)

    gt = rec['gt']
    g = np.asarray(gt)
    if rec['aux']:
        a = np.asarray(rec['aux'])
        print('\naux bias by speed band (prediction - GT)')
        for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 15), (15, 40)):
            m = (g >= lo) & (g < hi)
            if m.any():
                print(f'  {lo:>2}~{hi:<2} m/s  n={int(m.sum()):4d}  '
                      f'bias {float((a[m]-g[m]).mean()):+.3f}  '
                      f'ratio {float(a[m].mean()/max(g[m].mean(),1e-6)):.3f}')
    print('\nspeed (m/s), on the scored frame')
    if rec['aux']:
        stats('aux_bev_motion_head', rec['aux'], gt)
    if rec['slot']:
        stats('status slot -> decode', rec['slot'], gt)
    stats('plan, first 0.5s', rec['plan'], rec['gt_step'])
    stats('(reference) GT first 0.5s vs lcf speed', rec['gt_step'], gt)
    print(f'\nL2@3s over this subset (window mean): {np.mean(rec["l2"]):.4f}')


if __name__ == '__main__':
    main()
