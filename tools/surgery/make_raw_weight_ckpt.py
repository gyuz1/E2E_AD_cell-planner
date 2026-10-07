"""Write a checkpoint whose ordinary slots hold the raw (non-EMA) weights.

EMAHook stores the averaged weights in the ordinary slots and keeps the raw
ones as ema_<name with dots replaced by underscores>. Evaluation therefore
scores the EMA weights; this rewrites the pairs so the same tool scores the raw
ones, and nothing else in the checkpoint changes.

Usage: python tools/make_raw_weight_ckpt.py <checkpoint> <output>
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('output')
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    swapped, missing = 0, []
    for name in [k for k in sd if not k.startswith('ema_')]:
        ema_key = 'ema_' + name.replace('.', '_')
        if ema_key in sd:
            if sd[ema_key].shape != sd[name].shape:
                raise RuntimeError(f'{ema_key} and {name} differ in shape')
            sd[name], sd[ema_key] = sd[ema_key], sd[name]
            swapped += 1
        elif sd[name].dtype.is_floating_point and sd[name].requires_grad is not None:
            missing.append(name)
    if not swapped:
        raise SystemExit('no ema_* keys: this checkpoint was trained without EMAHook')
    torch.save(ckpt, args.output)
    print(f'{swapped} tensors now hold raw weights, {len(missing)} had no ema_ twin '
          f'(buffers and training-only heads) -> {args.output}')


if __name__ == '__main__':
    main()
