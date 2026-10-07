"""Does batching mix samples? Compares sample A's outputs across batch shapes.

  noise    : batch [A] run twice              (run-to-run numeric noise)
  shape    : batch [A] vs [A, A]              (same content, different shape)
  mixing   : batch [A, A] vs [A, B], slot 0   (only the other sample differs)

If `mixing` is at the level of `noise`, samples do not leak into each other and
any batch-1/batch-2 gap is kernel numerics tied to the batch shape. GridMask is
off (its mask is random); everything else is the training forward.

Usage: python tools/diag_batch_equivalence.py <train_cfg> [--pairs 2]
"""
import argparse
import importlib

import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

TRAIN_ANN = ('data/etri/.causal_regen_split_301_75_10hz/'
             'vad_etri_infos_temporal_train_split.pkl')
KEYS = ('bev_embed', 'ego_feats', 'ego_fut_preds', 'all_traj_preds')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--pairs', type=int, default=2)
    args = p.parse_args()
    cfg = Config.fromfile(args.config)
    importlib.import_module('tools.cache.etri_geometry_cache')
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.train.ann_file = TRAIN_ANN
    ds = build_dataset(cfg.data.train)
    mc = cfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    model = build_model(mc, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, cfg.load_from, map_location='cpu', logger=None)
    model = model.cuda().train()
    model.use_grid_mask = False
    head = model.pts_bbox_head
    calls = []
    orig = head.forward

    def spy(*a, **k):
        out = orig(*a, **k)
        calls.append(out)
        return out
    head.forward = spy

    def run(items):
        calls.clear()
        _, kw = scatter_kwargs((), collate(items, samples_per_gpu=len(items)), [0])
        with torch.no_grad():
            model.forward_train(**kw[0])
        return list(calls)

    def slot0(t):
        # batch dim: 1 for [N, B, D] BEV / [L, B, ...] decoder stacks, else 0
        return t[:, 0] if t.dim() >= 3 and t.shape[0] in (3, 10000) else t[0]

    def gap(c1, c2):
        return {k: max(float((slot0(x[k]) - slot0(y[k])).abs().max())
                       for x, y in zip(c1, c2)) for k in KEYS}

    n = len(ds)
    worst = 0.0
    for j in range(args.pairs):
        a, b = ds[(j * 7919 + 11) % n], ds[(j * 104729 + n // 2) % n]
        a1, a1b = run([a]), run([a])
        aa, ab = run([a, a]), run([a, b])
        noise, shape, mixing = gap(a1, a1b), gap(a1, aa), gap(aa, ab)
        print(f'pair {j}')
        for k in KEYS:
            print(f'  {k:<16} noise {noise[k]:.2e}  shape {shape[k]:.2e}  mixing {mixing[k]:.2e}')
            worst = max(worst, mixing[k] / max(noise[k], shape[k], 1e-6))
    print(f'verdict: {"samples mix across the batch" if worst > 10 else "no mixing; batch gap is numeric"} '
          f'(max mixing / max(noise, shape) = {worst:.1f})')


if __name__ == '__main__':
    main()
