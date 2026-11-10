"""Does open-loop L2 reward a planner that cannot see the scene at all?

Scores closed-form kinematic extrapolations from GT ego state on *exactly* the
frames the model ablation scored, with *exactly* the same L2 aggregation and
the same STP3 collision path -- the occupancy builder and PlanningMetric are
imported from eval_holdout_l2_and_tinfer rather than reimplemented, so a
change there cannot silently desynchronise these numbers.

  zero         : p(t) = 0                      (stay put)
  const-vel    : p(t) = v t                    (perfect v, a assumed 0)
  const-accel  : p(t) = v t + a t^2 / 2        (perfect v AND a)

These extrapolators are blind by construction: no images, no agents, no map.
If one of them matches the model's L2 while colliding far more often, then
L2 in this range is measuring ego kinematics rather than driving quality,
and the gap between the two metrics is the size of that confusion.

No GPU and no network forward -- safe to run beside a training job.
"""
import argparse
import importlib.util
import os
import pickle

import numpy as np

FUT_TS = 6
DT = 0.5
HIS_FRAMES = 30


def _load_eval_helpers():
    """Import the scoring helpers from the model evaluator, by path."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        '_eval_holdout', os.path.join(here, 'eval_holdout_l2_and_tinfer.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann-file', required=True)
    p.add_argument('--stride', type=int, default=50,
                   help='must match the model run being compared against')
    p.add_argument('--min-frame', type=int, default=HIS_FRAMES)
    p.add_argument('--frame-offsets', default='-10,-5,0',
                   help='only used to reproduce which windows were complete, '
                        'so the frame set matches the model run exactly')
    p.add_argument('--warmup-windows', type=int, default=20,
                   help='the model run executes this many windows untimed and '
                        'unrecorded; drop the same leading frames here or the '
                        'two runs score different frame sets')
    p.add_argument('--no-collision', action='store_true')
    args = p.parse_args()

    ev = _load_eval_helpers()
    offsets = sorted(int(x) for x in args.frame_offsets.split(','))

    with open(args.ann_file, 'rb') as handle:
        infos = pickle.load(handle)['infos']

    # Same grouping the evaluator uses: scene -> {frame_idx: info}.
    scenes = {}
    for info in infos:
        scenes.setdefault(info['scene_token'], {})[info['frame_idx']] = info

    t = np.arange(1, FUT_TS + 1) * DT
    names = ('zero', 'const-vel', 'const-accel')
    l2 = {k: [] for k in names}
    coll = {k: np.zeros((2, 3)) for k in names}
    n_eval = n_coll = n_skipped = n_warmed = 0

    for by_frame in scenes.values():
        for frame_idx in sorted(by_frame):
            if frame_idx < args.min_frame or frame_idx % args.stride != 0:
                continue
            if any(frame_idx + off not in by_frame for off in offsets):
                n_skipped += 1
                continue
            info = by_frame[frame_idx]
            if not info.get('fut_valid_flag', False):
                continue
            gt = np.asarray(info['gt_ego_fut_trajs'], dtype=np.float64)
            if gt.shape != (FUT_TS, 2):
                continue
            if n_warmed < args.warmup_windows:
                n_warmed += 1      # mirrors the evaluator's warm-up windows
                continue
            gt_cum = gt.cumsum(0)

            lcf = np.asarray(info['gt_ego_lcf_feat'], dtype=np.float64)
            v, a = lcf[0:2], lcf[2:4]
            preds = {
                'zero': np.zeros_like(gt_cum),
                'const-vel': v[None, :] * t[:, None],
                'const-accel': v[None, :] * t[:, None]
                               + 0.5 * a[None, :] * (t[:, None] ** 2),
            }

            for k, pred in preds.items():
                l2[k].append(ev._official_l2(pred, gt_cum))
            n_eval += 1

            if not args.no_collision:
                occ = ev._occupancy_from_info(info, True)
                if occ is not None:
                    for k, pred in preds.items():
                        coll[k] += ev._collision_at(
                            ev._occupancy_from_info.metric, occ, pred, gt_cum)
                    n_coll += 1

    print(f'ann-file : {args.ann_file}')
    print(f'stride {args.stride}, min-frame {args.min_frame}, '
          f'offsets {args.frame_offsets}')
    print(f'evaluated {n_eval} frames (skipped {n_skipped} for an '
          f'incomplete window), collision on {n_coll}')
    print()
    print(f"{'oracle':<14}{'L2':>10}{'coll obj':>11}{'coll obj_box':>14}")
    print('-' * 49)
    for k in names:
        row = f'{k:<14}{np.mean(l2[k]):>10.4f}'
        if n_coll:
            c = coll[k] / n_coll
            row += f'{c[0].mean() * 100:>10.2f}%{c[1].mean() * 100:>13.2f}%'
        print(row)
    print()
    print('L2 = mean of the 1s/2s/3s window means, as the model report uses.')
    print('Collision = STP3 obj / obj_box, 3-horizon mean, percent.')
    if n_coll:
        print('obj is read off a mis-offset grid in the stock STP3 code; '
              'compare obj_box.')


if __name__ == '__main__':
    main()
