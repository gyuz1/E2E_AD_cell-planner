#!/usr/bin/env python3
"""Merge ETRI stage-1 perception weights with the LAW world model.

Stage 1 (VAD_etri_tiny_stage1*.py) trains a plain VAD architecture -- it has
no bev_world_model submodule at all, so those weights never move during
stage 1. This script builds the stage-2 (VADLAW) init checkpoint by taking
everything from the stage-1 checkpoint (ETRI-domain-adapted perception,
detection, map, motion, ego decoder) and adding only the bev_world_model.*
keys from the nuScenes LAW checkpoint, which is the only place a trained
world model exists. No key exists in both sources, so this is a plain
union, not a conflict resolution.
"""
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', required=True,
                         help='ETRI stage-1 checkpoint (e.g. epoch_48.pth)')
    parser.add_argument('--world-model-source', required=True,
                         help='checkpoint containing bev_world_model.* '
                              '(e.g. ckpts/law_pretrained_nus.pth)')
    parser.add_argument(
        '--override-prefixes-from-world-model', default='',
        help='Comma-separated key prefixes to source from '
             '--world-model-source instead of --stage1, IN ADDITION to '
             'bev_world_model.*. For any key under one of these prefixes '
             'that exists in both checkpoints with a shape mismatch, the '
             'stage1 version is dropped entirely (left for the target '
             "model's own init) rather than copied incompatibly -- e.g. "
             'ego_fut_decoder.4 (the final, vocab-size-dependent layer) '
             'when stage1 and world-model-source used different '
             'ego_fut_mode. Use this when stage1 and stage2 disagree on '
             'ego_lcf_feat_idx (different ego_fut_decoder input width): '
             'stage1 then holds a shape stage2 cannot use for that module, '
             'while a same-lcf-setting world-model-source checkpoint (even '
             "if it's the plain nuScenes LAW checkpoint, not ETRI-domain-"
             'adapted) has the right shape for the layers that do match.')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    stage1_ckpt = torch.load(args.stage1, map_location='cpu')
    wm_ckpt = torch.load(args.world_model_source, map_location='cpu')

    stage1_sd = stage1_ckpt['state_dict']
    wm_sd = wm_ckpt['state_dict']

    wm_keys = [k for k in wm_sd if k.startswith('bev_world_model.')]
    if not wm_keys:
        raise ValueError(
            f'No bev_world_model.* keys found in {args.world_model_source}')

    overlap = set(wm_keys) & set(stage1_sd.keys())
    if overlap:
        raise ValueError(
            f'Unexpected key overlap between stage1 and world model '
            f'source: {sorted(overlap)}')

    merged_sd = dict(stage1_sd)
    for key in wm_keys:
        merged_sd[key] = wm_sd[key]

    # Unconditionally take these prefixes from world-model-source, replacing
    # whatever stage1 had. Some sub-keys under the prefix may still mismatch
    # the eventual TARGET model's shape (e.g. ego_fut_decoder's final layer,
    # sized by ego_fut_mode -- 36 here for nuScenes's 3 vs 84 for ETRI's 7) --
    # this script has no live model to check that against, so it doesn't try;
    # mmcv's load_checkpoint already skips (with a warning) any key whose
    # shape doesn't match the model actually being built, which is exactly
    # the right behavior for a layer that legitimately can't transfer.
    override_prefixes = [
        p for p in args.override_prefixes_from_world_model.split(',') if p]
    overridden = []
    for prefix in override_prefixes:
        for key, value in wm_sd.items():
            if not key.startswith(prefix):
                continue
            merged_sd[key] = value
            overridden.append(key)

    merged_ckpt = dict(stage1_ckpt)
    merged_ckpt['state_dict'] = merged_sd
    merged_ckpt['meta'] = dict(
        stage1_ckpt.get('meta', {}),
        merge_source_stage1=args.stage1,
        merge_source_world_model=args.world_model_source,
        merge_world_model_keys=len(wm_keys),
        merge_override_prefixes=override_prefixes,
        merge_overridden_keys=overridden)
    merged_ckpt.pop('optimizer', None)

    torch.save(merged_ckpt, args.output)
    print(f'Wrote {args.output}')
    print(f'  base keys (stage1):        {len(stage1_sd)}')
    print(f'  added world model keys:    {len(wm_keys)}')
    if override_prefixes:
        print(f'  overridden from world model: {len(overridden)} keys '
              f'({override_prefixes})')
        for k in overridden:
            was = tuple(stage1_sd[k].shape) if k in stage1_sd else None
            now = tuple(merged_sd[k].shape)
            flag = '' if was == now else f'  (stage1 was {was})'
            print(f'    + {k}  {now}{flag}')
    print(f'  total keys in merged ckpt: {len(merged_sd)}')


if __name__ == '__main__':
    main()
