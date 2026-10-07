"""Checks a cell planner config on real data before it trains. Every check
reads values, not config.

  1. init: every cell, U_TURN and STOP head reproduces the donor decoder's
     trajectory for its command (positional-code columns start at zero)
  2. the positional code and edges are buffers (never trained), each command's
     code is [Nf, Nl, dim] and no two cells share a code
  3. selection rule: each cell centre selects its own cell, an inner edge the
     next cell, the outer edge the last cell, out-of-range the outer cell
     (flagged); |target point| under the threshold or a STOP command -> STOP;
     U_TURN -> its head; a non-finite target point for a moving command raises
  4. real training batches: losses finite; every planned frame (history and
     current) is selected with its own target point and command
  5. eval replay of a real head call: changing the target point and command
     changes no generated trajectory
  6. backward on mixed / single-command / STOP-only batches: finite, reaches
     exactly the heads that were selected, never the target point

Usage: python tools/check_cell_planner_live.py <train_cfg> [--n 4] [--device 0]
Run with one visible GPU (GridMask builds its mask on cuda:0).
"""
import argparse
import copy
import importlib

import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.VAD.cell_planner_utils import (
    COMMANDS, ROUTE_CELL, ROUTE_STOP, ROUTE_U_TURN, STOP, U_TURN,
    route_trajectory)

vad_head_module = None  # set in main(), after the plugin is imported

# No module-level default any more. This used to name the 301-scene split and
# assign it over cfg.data.train.ann_file unconditionally, so a full-data config
# was checked against split frames and said nothing about it -- and on a machine
# without the split files it died with a FileNotFoundError naming a path its
# config never mentions. The config's own ann_file is what the run will train
# on, so that is what gets checked; --ann-file overrides it deliberately.


def clone_inputs(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_inputs(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_inputs(v) for v in value)
    return copy.deepcopy(value)


def report(ok, msg):
    print(f'  [{"OK" if ok else "FAIL"}] {msg}')
    return ok


def check_init(head, dev):
    feats = torch.randn(4, head.cell_heads[STOP][0].in_features, device=dev)
    with torch.no_grad():
        donor = head.ego_fut_decoder(feats).view(4, 7, head.fut_ts, 2)
        cells, u_turn, stop = head.cell_generate(feats)
    diff = max([float((cells[c] - donor[:, c, None, None]).abs().max()) for c in cells]
               + [float((u_turn - donor[:, U_TURN]).abs().max()),
                  float((stop - donor[:, STOP]).abs().max())])
    return report(diff < 1e-4, f'initial cell/U_TURN/STOP trajectories vs donor decoder: '
                               f'max diff {diff:.2e}')


def check_buffers(head):
    ok = True
    params = {n for n, _ in head.named_parameters()}
    for c, (fe, le) in head.cell_edges().items():
        pe = getattr(head, f'cell_pe_{c}')
        names = {f'cell_pe_{c}', f'cell_fwd_edges_{c}', f'cell_lat_edges_{c}'}
        shape_ok = tuple(pe.shape) == (len(fe) - 1, len(le) - 1, head.cell_pe_dim)
        flat = pe.reshape(-1, pe.shape[-1])
        d = torch.cdist(flat, flat) + torch.eye(len(flat), device=flat.device) * 1e9
        sep = float(d.min() / flat[0].norm()) if len(flat) > 1 else 1.0
        ok &= report(shape_ok and not (names & params) and not pe.requires_grad and sep > 0.1,
                     f'{COMMANDS[c]:<14} code {tuple(pe.shape)} buffer, closest two cells '
                     f'{sep * 100:.1f}% of |code| apart')
    return ok


def check_routing(head, dev):
    ok = True
    edges = head.cell_edges()
    th = head.cell_stop_tp_thresh
    fake = {c: torch.arange((len(f) - 1) * (len(l) - 1), device=dev, dtype=torch.float)
            .view(1, len(f) - 1, len(l) - 1, 1, 1).expand(1, -1, -1, head.fut_ts, 2)
            for c, (f, l) in edges.items()}
    ut = torch.full((1, head.fut_ts, 2), -1.0, device=dev)
    st = torch.full((1, head.fut_ts, 2), -2.0, device=dev)

    def pick(tp, cmd):
        sel, info = route_trajectory(fake, ut, st, torch.tensor([tp], device=dev),
                                     torch.tensor([cmd], device=dev), edges, th)
        return float(sel[0, 0, 0]), info

    for c, (fe, le) in edges.items():
        nl = len(le) - 1
        fc, lc = (fe[:-1] + fe[1:]) / 2, (le[:-1] + le[1:]) / 2
        own = all(pick([float(x), float(y)], c)[0] == r * nl + k
                  for r, x in enumerate(fc) for k, y in enumerate(lc))
        inner = pick([float(fe[1]), float(lc[0])], c)[0] == 1 * nl
        outer = pick([float(fe[-1]), float(lc[-1])], c)[0] == (len(fe) - 2) * nl + nl - 1
        v, info = pick([float(fe[-1]) + 50.0, float(le[0]) - 50.0], c)
        beyond = v == (len(fe) - 2) * nl and bool(info['outside'][0])
        ok &= report(own and inner and outer and beyond,
                     f'{COMMANDS[c]:<14} centres->own cell, inner edge->next, '
                     f'outer edge->last, out of range->outer (flagged)')
    stop_near = pick([th - 0.01, 0.0], 0)[1]['kind'][0] == ROUTE_STOP
    not_stop = pick([th, 0.0], 0)[1]['kind'][0] == ROUTE_CELL
    stop_cmd = pick([80.0, 0.0], STOP)[1]['kind'][0] == ROUTE_STOP
    uturn = pick([20.0, 5.0], U_TURN)[1]['kind'][0] == ROUTE_U_TURN
    # A U-turn ends beside or behind the car: forward < 1 m, but not stopped.
    uturn_behind = pick([-1.7, 13.6], U_TURN)[1]['kind'][0] == ROUTE_U_TURN
    ok &= report(stop_near and not_stop and stop_cmd and uturn and uturn_behind,
                 f'|TP| < {th} m -> STOP, |TP| = {th} m -> cell, STOP command -> STOP, '
                 f'U_TURN -> U_TURN head (also with its target behind the car)')
    try:
        pick([float('nan'), 0.0], 0)
        raised = False
    except ValueError:
        raised = True
    stop_nan = pick([float('nan'), 0.0], STOP)[1]['kind'][0] == ROUTE_STOP
    ok &= report(raised and stop_nan, 'non-finite TP raises for a moving command, '
                                      'is not needed for STOP')
    return ok


def check_train_batches(model, dataset, args):
    head = model.pts_bbox_head
    calls, captured = [], []
    real_route = vad_head_module.route_trajectory

    def spy(cells, ut, st, tp, cmd, edges, th):
        sel, info = real_route(cells, ut, st, tp, cmd, edges, th)
        calls.append((tp.reshape(-1)[:2].tolist(), int(cmd.reshape(-1)[0]),
                      int(info['kind'][0]), int(info['row'][0]), int(info['col'][0])))
        return sel, info

    vad_head_module.route_trajectory = spy
    original_forward = head.forward
    ok = True
    try:
        for j in range(args.n):
            item = dataset[(j * 1777 + 11) % len(dataset)]
            _, kw = scatter_kwargs((), collate([item], samples_per_gpu=1), [args.device])
            kw = kw[0]
            calls.clear()

            def capture(*a, **k):
                if j == 0 and k.get('ego_target_point') is not None and not k.get('only_bev'):
                    captured[:] = [clone_inputs(a), clone_inputs(k)]
                return original_forward(*a, **k)

            head.forward = capture
            model.train()
            with torch.no_grad():
                losses = model.forward_train(**kw)
            head.forward = original_forward
            metas = kw['img_metas'][0]
            frames = sorted(metas)
            expect = [(list(map(float, torch.as_tensor(metas[f]['ego_target_point'])
                                .reshape(-1)[:2])),
                       int(torch.as_tensor(metas[f]['ego_fut_cmd']).reshape(-1).argmax()))
                      for f in frames]
            got = [(c[0], c[1]) for c in calls]
            same = len(got) == len(expect) and all(
                g[1] == e[1] and max(abs(a - b) for a, b in zip(g[0], e[0])) < 1e-4
                for g, e in zip(got, expect))
            finite = all(torch.isfinite(v).all() for k, v in losses.items()
                         if torch.is_tensor(v) and ('plan' in k or 'waypoint' in k))
            kinds = ['cell', 'u_turn', 'stop']
            desc = ', '.join(f'{COMMANDS[c[1]]} TP({c[0][0]:.1f},{c[0][1]:.1f})->'
                             f'{kinds[c[2]]}' + (f'({c[3]},{c[4]})' if c[2] == ROUTE_CELL else '')
                             for c in calls)
            ok &= report(same and finite and 'loss_plan_reg' in losses,
                         f'batch {j}: {len(calls)} selections for {len(frames)} frames, '
                         f'each with its own TP/command; plan losses finite | {desc}')
    finally:
        vad_head_module.route_trajectory = real_route
        head.forward = original_forward
    return ok, captured


def check_eval_invariance(head, captured):
    if not captured:
        return report(False, 'no current-frame head call captured')
    a, k = captured
    head.eval()
    with torch.no_grad():
        first = head(*clone_inputs(a), **clone_inputs(k))
        changed = clone_inputs(k)
        changed['ego_target_point'] = torch.full_like(changed['ego_target_point'], 1234.5)
        changed['ego_fut_cmd'] = torch.roll(changed['ego_fut_cmd'], 1, dims=-1)
        second = head(*clone_inputs(a), **changed)
    same = all(torch.equal(first['cell_trajs'][c], second['cell_trajs'][c])
               for c in first['cell_trajs'])
    same &= torch.equal(first['cell_u_turn'], second['cell_u_turn'])
    same &= torch.equal(first['cell_stop'], second['cell_stop'])
    same &= torch.equal(first['ego_fut_preds'], second['ego_fut_preds'])
    shapes = {COMMANDS[c]: tuple(t.shape) for c, t in first['cell_trajs'].items()}
    return report(same, f'eval: TP and command changed, every generated trajectory '
                        f'bit-identical; shapes {shapes}, U_TURN/STOP '
                        f'{tuple(first["cell_stop"].shape)}')


def check_backward(head, dev):
    ok = True
    head.train()
    d = head.cell_heads[STOP][0].in_features
    edges = head.cell_edges()
    centre = {c: [float((f[0] + f[1]) / 2), float((l[0] + l[1]) / 2)]
              for c, (f, l) in edges.items()}
    cases = {
        'mixed (all 7 commands)': [(c, centre.get(c, [30.0, 0.0])) for c in range(7)],
        'single command (LANE_KEEP x4)': [(0, centre[0])] * 4,
        'STOP only (x3)': [(STOP, [0.0, 0.0])] * 3,
    }
    for name, rows in cases.items():
        head.zero_grad()
        feats = torch.randn(len(rows), d, device=dev, requires_grad=True)
        tp = torch.tensor([r[1] for r in rows], device=dev, requires_grad=True)
        cmd = torch.tensor([r[0] for r in rows], device=dev)
        cells, ut, st = head.cell_generate(feats)
        sel, info = route_trajectory(cells, ut, st, tp, cmd, edges, head.cell_stop_tp_thresh)
        sel.square().mean().backward()
        used = set()
        for r, kind in zip(rows, info['kind'].tolist()):
            used.add(STOP if kind == ROUTE_STOP else (U_TURN if kind == ROUTE_U_TURN else r[0]))
        graded = {c for c in range(7)
                  if head.cell_heads[c][-1].weight.grad is not None
                  and float(head.cell_heads[c][-1].weight.grad.abs().sum()) > 0}
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                     for p in head.cell_heads.parameters())
        ok &= report(graded == used and finite and tp.grad is None
                     and feats.grad is not None,
                     f'{name}: gradient reaches heads {sorted(COMMANDS[c] for c in graded)}, '
                     f'finite, none to the target point')
    head.zero_grad()
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--ann-file', default=None,
                   help="override the config's train ann_file; by default the "
                        'config is checked against the data it will train on')
    p.add_argument('--trained-init', action='store_true',
                   help='verify trained cell tensors, not equality to unused donor decoder')
    args = p.parse_args()
    cfg = Config.fromfile(args.train_config)
    importlib.import_module('tools.cache.etri_geometry_cache')
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    global vad_head_module
    vad_head_module = importlib.import_module('projects.mmdet3d_plugin.VAD.VAD_head')
    if args.ann_file:
        cfg.data.train.ann_file = args.ann_file
    print(f'   data: {cfg.data.train.ann_file}')
    dataset = build_dataset(cfg.data.train)
    mc = cfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    mc.pop('trajectory_distill_weight', None)
    model = build_model(mc, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    loaded = load_checkpoint(model, cfg.load_from, map_location='cpu')
    dev = f'cuda:{args.device}'
    for name in ('img_backbone', 'img_neck'):
        getattr(model, name).requires_grad_(False)
    model = model.to(dev)
    head = model.pts_bbox_head
    torch.manual_seed(0)

    print('1. initialisation from the donor')
    if args.trained_init:
        sd = loaded.get('state_dict', loaded)
        names = [k for k in model.state_dict() if k.startswith('pts_bbox_head.cell_')]
        ok = report(bool(names) and all(
            k in sd and torch.equal(model.state_dict()[k].cpu(), sd[k].cpu())
            for k in names), 'trained cell parameters and layout copied exactly')
    else:
        ok = check_init(head, dev)
    print('2. positional code and edges')
    ok &= check_buffers(head)
    print('3. selection rule')
    ok &= check_routing(head, dev)
    print('4. real training batches')
    ok_b, captured = check_train_batches(model, dataset, args)
    ok &= ok_b
    print('5. target point cannot change what is generated')
    ok &= check_eval_invariance(head, captured)
    print('6. backward')
    ok &= check_backward(head, dev)
    print('verdict:', 'passed' if ok else 'failed')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
