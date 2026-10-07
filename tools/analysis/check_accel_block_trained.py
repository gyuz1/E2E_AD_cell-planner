"""Prove from a trained checkpoint that the acceleration block got gradient.

check_accel_block_live.py reads the call path; this reads the outcome. If the
descriptor's third block is zero during training, the aux heads' input columns
4096:6144 multiply zeros, receive no gradient, and stay exactly at their
initial values. Comparing their statistics against the velocity block's is a
direct measurement of whether acceleration was learned or the config merely
claimed it.

What to expect when it works: the accel columns are SMALLER than the velocity
columns (second differences are noisier and smaller in magnitude) but the same
order, and clearly different from a fresh init. A ratio near zero, or columns
matching a freshly built model's init to several digits, means the block was
dead.

Usage:
    python tools/check_accel_block_trained.py <ckpt> --config <train config>
"""
import argparse

import mmcv
import torch
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401


def blocks(w, frames):
    """Split an aux head's first-layer weight into its per-frame blocks."""
    per = w.shape[1] // frames
    return [w[:, i * per:(i + 1) * per] for i in range(frames)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ckpt')
    p.add_argument('--config', required=True)
    args = p.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    frames = cfg.model.pts_bbox_head.get('aux_bev_motion_frames') or 2
    if frames < 3:
        print('not a 3-frame config -- nothing to check')
        return 0

    sd = torch.load(args.ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)

    # EMAHook runs at priority HIGH and swaps the EMA into the model at
    # after_train_epoch, before CheckpointHook (NORMAL) writes the file. So
    # the ordinary slots hold the EMA and the ema_* buffers hold the raw
    # training params. Verified on this run: between epoch 1 and 2 the
    # ordinary slots move 2.27x LESS than the ema_* ones.
    #
    # That matters here. With momentum 0.0002 the EMA lags by roughly 5000
    # iterations, so early checkpoints' ordinary slots sit near their init no
    # matter how well training is going -- reading them under-reports
    # movement and makes a healthy run look inert. The raw params are where
    # learning shows first, so prefer them when present.
    raw = {k[len('ema_'):]: v for k, v in sd.items() if k.startswith('ema_')}
    if raw:
        lookup = {}
        for k in sd:
            if not k.startswith('ema_'):
                flat = k.replace('.', '_')
                if flat in raw:
                    lookup[k] = raw[flat]
        if lookup:
            print(f'EMAHook detected: reading {len(lookup)} raw training parameters from '
                  'the ema_* buffers (the ordinary slots hold the EMA and stay near init)')
            sd = {**sd, **lookup}

    model_cfg = cfg.model.copy()
    model_cfg.pop('feature_distill_teacher_cfg', None)
    model_cfg.pop('feature_distill_teacher_ckpt', None)
    fresh = build_model(model_cfg, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg')).state_dict()

    names = [k for k in sd
             if k.endswith('.0.weight')
             and ('aux_bev_motion_head' in k
                  or 'aux_bev_future_motion_head' in k
                  or 'ego_status_est_net' in k)]
    if not names:
        print('  [FAIL] the aux/estimator head is not in the checkpoint')
        return 1

    failed = False
    for name in sorted(names):
        w = sd[name].float()
        f0 = fresh[name].float() if name in fresh else None
        bs = blocks(w, frames)
        labels = ['current', 'speed (1st diff)', 'acceleration (2nd diff)'][:frames]
        print(f'\n{name}  shape {tuple(w.shape)}')
        for lab, b in zip(labels, bs):
            print(f'   {lab:<14} mean|w| {b.abs().mean():.6f}   '
                  f'std {b.std():.6f}')
        accel, vel = bs[2], bs[1]
        ratio = (accel.abs().mean() / vel.abs().mean()).item()
        print(f'   acceleration/speed magnitude ratio = {ratio:.4f}')
        if f0 is not None:
            a0 = blocks(f0, frames)[2]
            drift = (accel - a0).abs().mean().item()
            init = a0.abs().mean().item()
            print(f'   drift from init = {drift:.6f}  (initial mean|w| {init:.6f})')
            # A block that never received gradient is bit-identical to its
            # init only if seeds match; they do not across processes, so
            # compare distributions instead. A dead block keeps the init's
            # std exactly, since nothing ever updated it.
            v0 = blocks(f0, frames)[1]
            vel_drift_std = abs(vel.std().item() - v0.std().item())
            acc_drift_std = abs(accel.std().item() - a0.std().item())
            print(f'   std change: speed {vel_drift_std:.6f} / '
                  f'acceleration {acc_drift_std:.6f}')
            if vel_drift_std > 0 and acc_drift_std < vel_drift_std * 0.02:
                print('   [FAIL] only the acceleration block still has its initial distribution '
                      '-- it never received gradient')
                failed = True
                continue
        if ratio < 0.01:
            print('   [FAIL] the acceleration block is effectively zero')
            failed = True
        else:
            print('   [OK]  the acceleration block was trained')

    print('\nverdict:', 'failed' if failed else 'passed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
