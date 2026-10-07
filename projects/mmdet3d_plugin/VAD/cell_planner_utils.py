"""Command cell planner: fixed cell layouts, the cell positional encoding, and
the target-point selection rule shared by training, evaluation and submission.

The target point never reaches a feature. The head generates one trajectory
per cell of every command from scene features plus a fixed code of the cell's
centre; route_trajectory() then picks one of them, the way the navigation
command picks a mode. Selection is not learned.

Coordinates: ego frame, metres, x forward, y left positive.
"""
import math

import torch

COMMANDS = ('LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT',
            'TURN_RIGHT', 'U_TURN', 'STOP')
U_TURN = 5
STOP = 6
ROUTE_CELL, ROUTE_U_TURN, ROUTE_STOP = 0, 1, 2


def validate_layouts(layouts, num_modes):
    """[mode] -> None or (forward edges, lateral edges); returns float lists."""
    if layouts is None or len(layouts) != num_modes:
        raise ValueError(f'cell_layouts needs one entry per command ({num_modes})')
    out = []
    for c, layout in enumerate(layouts):
        if c in (U_TURN, STOP):
            if layout is not None:
                raise ValueError(f'{COMMANDS[c]} has a dedicated head and no cells')
            out.append(None)
            continue
        if layout is None or len(layout) != 2:
            raise ValueError(f'{COMMANDS[c]} needs (forward edges, lateral edges)')
        edges = []
        for axis in layout:
            e = [float(v) for v in axis]
            if len(e) < 2 or not all(math.isfinite(v) for v in e) or any(
                    b <= a for a, b in zip(e[:-1], e[1:])):
                raise ValueError(f'{COMMANDS[c]} edges must be finite and increasing: {axis}')
            edges.append(e)
        out.append(tuple(edges))
    return out


def cell_positional_encoding(fwd_edges, lat_edges, dim, wavelengths):
    """[Nf, Nl, dim] fixed sin/cos code of each cell centre (forward, left).

    dim/4 wavelengths per axis, log-spaced over `wavelengths` (metres): the
    shortest separates the narrowest neighbouring cells, the longest keeps
    every centre in the layout range unambiguous.
    """
    if dim % 4:
        raise ValueError('cell_pe_dim must be a multiple of 4')
    fe = torch.tensor(fwd_edges, dtype=torch.float64)
    le = torch.tensor(lat_edges, dtype=torch.float64)
    fc, lc = (fe[:-1] + fe[1:]) / 2, (le[:-1] + le[1:]) / 2
    centre = torch.stack(torch.meshgrid(fc, lc, indexing='ij'), dim=-1)
    lam = torch.logspace(math.log10(wavelengths[0]), math.log10(wavelengths[1]),
                         dim // 4, dtype=torch.float64)
    phase = centre[..., None] * (2 * math.pi / lam)            # [Nf, Nl, 2, n]
    pe = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
    return pe.reshape(len(fc), len(lc), dim).float()


def containing_cell(tp, fwd_edges, lat_edges):
    """Row/col of the cell containing each point [B, 2].

    An inner edge belongs to the next cell, the outermost edge to the last
    one, and a point outside the range to the nearest outer cell (flagged).
    """
    tp = tp.float()
    fe = fwd_edges.to(device=tp.device, dtype=torch.float32)
    le = lat_edges.to(device=tp.device, dtype=torch.float32)
    row = torch.searchsorted(fe[1:-1].contiguous(), tp[:, 0].contiguous(), right=True)
    col = torch.searchsorted(le[1:-1].contiguous(), tp[:, 1].contiguous(), right=True)
    outside = ((tp[:, 0] < fe[0]) | (tp[:, 0] > fe[-1])
               | (tp[:, 1] < le[0]) | (tp[:, 1] > le[-1]))
    return row, col, outside


def route_trajectory(cell_trajs, u_turn_traj, stop_traj, target_point, command,
                     edges, stop_tp_thresh):
    """Select one generated trajectory per sample. Not learned.

    cell_trajs  {command: [B, Nf, Nl, T, 2]}   u_turn_traj, stop_traj [B, T, 2]
    target_point [B, >=2] (x forward, y left)  command [B] (index, may be STOP)
    edges       {command: (forward edges, lateral edges) tensors}

    STOP when the target point is within stop_tp_thresh of the ego (straight
    line: a U-turn's target can be beside or behind the car) or the command
    is STOP; U_TURN's own head for U_TURN; otherwise the cell of the command
    that contains the target point. Returns ([B, T, 2], info).
    """
    b = stop_traj.shape[0]
    command = command.reshape(b).long().to(stop_traj.device)
    tp = target_point.reshape(b, -1)[:, :2].detach().float().to(stop_traj.device)
    needs_tp = command != STOP
    if not bool(torch.isfinite(tp[needs_tp]).all()):
        raise ValueError('non-finite target point for a moving command; '
                         'refusing to substitute one')
    safe_tp = torch.where(torch.isfinite(tp), tp, torch.zeros_like(tp))

    stop = (command == STOP) | (safe_tp.norm(dim=-1) < stop_tp_thresh)
    kind = torch.full((b,), ROUTE_CELL, dtype=torch.long, device=stop_traj.device)
    kind[(command == U_TURN) & ~stop] = ROUTE_U_TURN
    kind[stop] = ROUTE_STOP
    row = torch.full((b,), -1, dtype=torch.long, device=stop_traj.device)
    col = torch.full_like(row, -1)
    outside = torch.zeros(b, dtype=torch.bool, device=stop_traj.device)

    selected = stop_traj
    selected = torch.where((kind == ROUTE_U_TURN)[:, None, None], u_turn_traj, selected)
    ar = torch.arange(b, device=stop_traj.device)
    for c, trajs in cell_trajs.items():
        m = (kind == ROUTE_CELL) & (command == c)
        if not bool(m.any()):
            continue
        r, cc, out = containing_cell(safe_tp, *edges[c])
        cand = trajs[ar, r.clamp(max=trajs.shape[1] - 1), cc.clamp(max=trajs.shape[2] - 1)]
        selected = torch.where(m[:, None, None], cand, selected)
        row = torch.where(m, r, row)
        col = torch.where(m, cc, col)
        outside = outside | (m & out)
    if bool(((kind == ROUTE_CELL) & ~torch.isin(
            command, torch.tensor(list(cell_trajs), device=command.device))).any()):
        raise ValueError('a moving command has no cell layout')
    return selected, dict(kind=kind, row=row, col=col, outside=outside)
