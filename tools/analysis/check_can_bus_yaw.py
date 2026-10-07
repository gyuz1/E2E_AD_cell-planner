"""Check the yaw delta (can_bus[-1]) the BEV encoder receives, on the training
queue AND the inference stream, at frames whose heading crosses 0/360 deg.

Yaw is stored in [0, 360). Until 2026-09-15 the LAW path (training queue and
VADLAW.forward_test) subtracted without wrapping, so on 3.6% of train pairs
the encoder's can_bus_mlp read ~+-359 instead of ~+-1. This finds real
crossing samples and asserts, on both paths:
  |delta| <= 180 everywhere, and delta ~= gt yaw_rate * 0.5s (degrees)

Usage: python tools/check_can_bus_yaw.py <train_cfg> <eval_cfg> [--n 6]
Run with one visible GPU.
"""
import argparse
import importlib

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

ANN = 'data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_{}_split.pkl'


def yaw_deg(info):
    a = quaternion_yaw(Quaternion(info['ego2global_rotation'])) / np.pi * 180
    return a + 360 if a < 0 else a


def crossing(infos, n):
    idx = {(i['scene_token'], i['frame_idx']): k for k, i in enumerate(infos)}
    out = []
    for k, i in enumerate(infos):
        p = idx.get((i['scene_token'], i['frame_idx'] - 5))
        pp = idx.get((i['scene_token'], i['frame_idx'] - 10))
        if p is None or pp is None or i['frame_idx'] % 5:
            continue
        if abs(yaw_deg(i) - yaw_deg(infos[p])) > 180:
            out.append(k)
        if len(out) >= n:
            break
    return out, idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('train_config')
    ap.add_argument('eval_config')
    ap.add_argument('--n', type=int, default=6)
    args = ap.parse_args()
    ok = True

    # ---- training queue ----
    tcfg = Config.fromfile(args.train_config)
    importlib.import_module('tools.cache.etri_geometry_cache')
    importlib.import_module(tcfg.plugin_dir.replace('/', '.').rstrip('.'))
    tcfg.data.train.ann_file = ANN.format('train')
    tds = build_dataset(tcfg.data.train)
    tds._build_sample_indices()
    inv = {gi: j for j, gi in enumerate(tds._sample_indices)}
    ks, _ = crossing(tds.data_infos, 10 * args.n)
    ks = [k for k in ks if k in inv][:args.n]
    print(f'training queue: {len(ks)} samples crossing the boundary')
    for k in ks:
        item = tds[inv[k]]
        metas = item['img_metas'].data
        deltas = [float(metas[t]['can_bus'][-1]) for t in sorted(metas)]
        yr = float(np.asarray(tds.data_infos[k]['gt_ego_lcf_feat'])[4])
        exp = np.degrees(yr * 0.5)
        print(f'  idx {k}: queue yaw deltas {np.round(deltas, 2)}  (current frame expects ~{exp:+.2f})')
        if max(abs(d) for d in deltas) > 180 or abs(deltas[-1] - exp) > 3:
            print('  [FAIL]'); ok = False

    # ---- inference stream ----
    ecfg = Config.fromfile(args.eval_config)
    ecfg.data.test.ann_file = ANN.format('val')
    ecfg.data.test.test_mode = True
    ecfg.data.test.pop('samples_per_gpu', None)
    ecfg.data.test.pop('map_ann_file', None)
    vds = build_dataset(ecfg.data.test)
    mc = tcfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    model = build_model(mc, test_cfg=ecfg.get('test_cfg'))
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    seen = []
    tr = model.pts_bbox_head.transformer
    orig = tr.get_bev_features

    def spy(*a, **kw):
        seen.append(float(kw['img_metas'][0]['can_bus'][-1]))
        return orig(*a, **kw)
    tr.get_bev_features = spy
    model = MMDataParallel(model.cuda(0), device_ids=[0]).eval()
    vks, vidx = crossing(vds.data_infos, args.n)
    print(f'inference stream: {len(vks)} windows crossing the boundary')
    for k in vks:
        inf = vds.data_infos[k]
        model.module.prev_frame_info = {
            'prev_bev': None, 'prev_bev2': None, 'prev_bev_pristine': None,
            'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}
        seen.clear()
        for off in (-10, -5, 0):
            g = vidx[(inf['scene_token'], inf['frame_idx'] + off)]
            with torch.no_grad():
                model(return_loss=False, rescale=True, bev_only=(off != 0),
                      **collate([vds[g]], samples_per_gpu=1))
        exp = np.degrees(float(np.asarray(inf['gt_ego_lcf_feat'])[4]) * 0.5)
        print(f'  idx {k}: yaw deltas the encoder saw {np.round(seen, 2)}  (expects ~{exp:+.2f})')
        if max(abs(d) for d in seen) > 180 or abs(seen[-1] - exp) > 3:
            print('  [FAIL]'); ok = False
    print('verdict:', 'passed' if ok else 'failed')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
