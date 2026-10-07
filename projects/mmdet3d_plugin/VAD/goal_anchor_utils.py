"""Numerically safe ragged goal anchors (torch only; also CPU-testable).

Widths normalize regression targets, not candidate validity. Padding must be
masked explicitly both when making labels and when selecting predictions.
All distance/offset arithmetic is float32, including under fp16 inference.
"""
import math

import torch


def build_anchor_table(anchor_lists, num_modes):
    """Validate [mode][anchor][xc,xw,yc,yw], then pad with safe unit widths."""
    if len(anchor_lists) != num_modes:
        raise ValueError('goal_anchors must have one list per command mode')
    counts = [len(cells) for cells in anchor_lists]
    if not counts or min(counts) < 1:
        raise ValueError('every command mode needs at least one goal anchor')
    table = torch.zeros(num_modes, max(counts), 4, dtype=torch.float32)
    table[..., 1::2] = 1.0  # no zero division, even before a validity mask
    mask = torch.zeros(num_modes, max(counts), dtype=torch.bool)
    for mode, cells in enumerate(anchor_lists):
        for index, cell in enumerate(cells):
            if len(cell) != 4:
                raise ValueError('each goal anchor must be [xc, xw, yc, yw]')
            values = [float(value) for value in cell]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('goal anchor values must be finite')
            if values[1] <= 0 or values[3] <= 0:
                raise ValueError('goal anchor widths must be positive')
            table[mode, index] = torch.tensor(values, dtype=torch.float32)
            mask[mode, index] = True
    return table, mask, counts


def goal_points_from_offsets(offsets, table):
    """[B,M,K,2] normalized offsets -> goal coordinates in metres."""
    anchor = table.to(device=offsets.device, dtype=torch.float32)
    return anchor[None, ..., 0::2] + offsets.float() * anchor[None, ..., 1::2]


def nearest_valid_goal(points, target, valid_mask):
    """Choose a valid Euclidean-nearest point, without modifying any point.

    points: [..., K, 2], target: [..., 2], mask: [..., K]. Invalid points may
    contain any value (even NaN); valid points and the target must be finite.
    The caller owns the policy of whether TP selection is enabled.
    """
    points = points.float()
    target = torch.as_tensor(target, device=points.device, dtype=torch.float32)
    valid = torch.as_tensor(valid_mask, device=points.device, dtype=torch.bool)
    if tuple(valid.shape) != tuple(points.shape[:-1]):
        raise ValueError('candidate validity mask must match points[..., K]')
    if points.shape[-1] != 2 or tuple(target.shape) != tuple(points.shape[:-2]) + (2,):
        raise ValueError('goal points/target dimensions do not match')
    if not valid.any(dim=-1).all():
        raise ValueError('cannot select from an empty set of valid goals')
    if not torch.isfinite(target).all() or not torch.isfinite(points[valid]).all():
        raise ValueError('valid goal points and target must be finite')
    # Sanitize before arithmetic: adding a penalty after 0/0 does not fix NaN.
    safe_points = torch.where(valid[..., None], points, target[..., None, :])
    distance = (safe_points - target[..., None, :]).square().sum(dim=-1)
    distance = distance.masked_fill(~valid, float('inf'))
    return distance.argmin(dim=-1)


def goal_labels_from_targets(target, commands, table, valid_mask):
    """Normalized-nearest labels and offsets; no invalid/padded winner."""
    target = target.float()
    commands = commands.to(device=target.device, dtype=torch.long)
    table = table.to(device=target.device, dtype=torch.float32)[commands]
    valid = valid_mask.to(device=target.device, dtype=torch.bool)[commands]
    if target.ndim != 2 or target.shape[-1] != 2 or commands.shape != target.shape[:1]:
        raise ValueError('target must be [B,2] and commands [B]')
    if not torch.isfinite(target).all():
        raise ValueError('goal training targets must be finite')
    if not valid.any(dim=-1).all():
        raise ValueError('command has no valid goal anchors')
    widths = torch.where(valid[..., None], table[..., 1::2],
                         torch.ones_like(table[..., 1::2]))
    if (widths <= 0).any() or not torch.isfinite(widths).all():
        raise ValueError('valid goal anchor widths must be finite and positive')
    centres = torch.where(valid[..., None], table[..., 0::2], target[:, None, :])
    relative = (target[:, None, :] - centres) / widths
    distance = relative.square().sum(dim=-1).masked_fill(~valid, float('inf'))
    indices = distance.argmin(dim=-1)
    rows = torch.arange(target.shape[0], device=target.device)
    return indices, relative[rows, indices]
