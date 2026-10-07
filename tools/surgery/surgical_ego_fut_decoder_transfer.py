"""Surgically transfer stage1's (ego_lcf ON) ego_fut_decoder into a
stage2 (ego_lcf OFF) init checkpoint, with zero information loss anywhere
except an exact, well-defined removal of ego_lcf's own contribution.

ego_fut_decoder is Linear(in, hid) -> ReLU -> Linear(hid, hid) -> ReLU ->
Linear(hid, ego_fut_mode*fut_ts*2) (num_reg_fcs=2's convention). Only the
FIRST layer's INPUT side depends on ego_lcf_feat_idx (in = embed_dims*2
[+ len(ego_lcf_feat_idx)]); the hidden layer and final projection never
see ego_lcf at all -- their shape and trained weights are identical
regardless of ego_lcf_feat_idx. So:

  - layer 0 weight: stage1's [hid, in_ON] matrix, minus its last
    len(ego_lcf_feat_idx) INPUT columns. This is an exact removal of
    that additive term from the layer's output, not an approximation --
    a Linear layer's output is a sum over input columns of
    weight*input, and dropping a column exactly reproduces "that input
    were zero". Bias is untouched (doesn't depend on input width).
  - layers 2 and 4: copied byte-for-byte from stage1, unchanged --
    neither their shape nor their trained values depend on
    ego_lcf_feat_idx at all.

Requires the target stage2 config to build ego_fut_decoder with
ego_fut_dec_hidden_dim set to stage1's ON hidden width (520 for
ego_lcf_feat_idx=[0..7], embed_dims=256), so layers 2/4 need no
modification and only layer 0's input side is sliced. See
VAD_head.py's ego_fut_dec_hidden_dim constructor comment.
"""
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage1-on', required=True,
        help='ego_lcf-ON stage1 checkpoint (e.g. epoch_48.pth)')
    parser.add_argument(
        '--target', required=True,
        help='stage2-init checkpoint to patch in place (its own '
             'ego_fut_decoder.* keys, if any, are overwritten)')
    parser.add_argument(
        '--ego-lcf-n', type=int, default=8,
        help='len(ego_lcf_feat_idx) used by --stage1-on -- the last N '
             'input columns of layer 0 belong to ego_lcf')
    parser.add_argument(
        '--pad-input-cols', type=int, default=0,
        help='append N zero-initialized INPUT columns to layer 0 after the '
             'ego_lcf slice, for a target whose ego_feats is wider than the '
             'donor stage1 (e.g. aux_bev_motion_feedback concatenates the '
             "head's own motion estimate). Zero means the new channels "
             'start as an exact no-op, so the transferred decoder behaves '
             'identically at step 0 and only learns to use them if it '
             'helps -- same convention as the zero-init refine stages.')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    stage1_sd = torch.load(args.stage1_on, map_location='cpu')['state_dict']
    target = torch.load(args.target, map_location='cpu')
    target_sd = target['state_dict'] if 'state_dict' in target else target

    prefix = 'pts_bbox_head.ego_fut_decoder.'
    stage1_keys = {k: v for k, v in stage1_sd.items() if k.startswith(prefix)}
    print(f'stage1 ego_fut_decoder keys: {sorted(stage1_keys)}')
    assert len(stage1_keys) == 6, (
        f'expected 6 keys (layers 0/2/4 x weight/bias), got '
        f'{len(stage1_keys)}: {sorted(stage1_keys)}')

    w0_key = prefix + '0.weight'
    w0 = stage1_keys[w0_key]
    n = args.ego_lcf_n
    assert w0.dim() == 2 and w0.shape[1] > n >= 0, w0.shape
    if n:
        w0_sliced = w0[:, :-n].clone()
        print(f'layer 0 weight: {tuple(w0.shape)} -> {tuple(w0_sliced.shape)} '
              f'(dropped last {n} input columns = ego_lcf)')
    else:
        # A donor stage1 that was itself trained with ego_lcf_feat_idx=None
        # has no ego_lcf columns to remove -- only the padding below
        # applies. Slicing anything here would silently empty the matrix.
        w0_sliced = w0.clone()
        print(f'layer 0 weight: {tuple(w0.shape)} (no ego_lcf columns to '
              f'drop, --ego-lcf-n 0)')
    if args.pad_input_cols:
        pad = w0_sliced.new_zeros(w0_sliced.shape[0], args.pad_input_cols)
        w0_sliced = torch.cat([w0_sliced, pad], dim=1)
        print(f'layer 0 weight: padded with {args.pad_input_cols} zero '
              f'input columns -> {tuple(w0_sliced.shape)}')

    for k, v in stage1_keys.items():
        target_sd[k] = w0_sliced if k == w0_key else v.clone()

    if 'state_dict' in target:
        target['state_dict'] = target_sd
        torch.save(target, args.output)
    else:
        torch.save(target_sd, args.output)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
