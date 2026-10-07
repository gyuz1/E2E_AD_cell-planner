"""Seed a distilled student from a trained teacher instead of from stage 1.

The student a distillation trains normally starts from the stage-1 donor, whose
planner is a single decoder with zero columns where the status slot will go and
no cell heads at all. The teacher that student is about to be aligned against
has already trained all of that -- and, since both sides now carry the same cell
planner with the same layout, most of it has exactly the same shape on the
student.

So copy it. The student then starts where the teacher ended on the parts they
share, and spends its training learning the one thing that genuinely differs:
filling the status slot from vision rather than from the real ego status.

WHAT IS COPIED, AND WHY EACH ONE IS SAFE

  pts_bbox_head.cell_heads.*          per-command waypoint heads. Identical
                                      architecture and identical cell layout on
                                      both sides (the layouts are compared here
                                      before anything is copied), so cell k means
                                      the same stretch of road in both.
  pts_bbox_head.ego_fut_decoder.*     520 wide on both: 512 scene + 8 status.
  pts_bbox_head.ego_status_decode_head.*
                                      (6, 8) on both. Sharing it means the two
                                      slots are read back into physical state by
                                      the same map, which is what makes a cosine
                                      between them well posed rather than a
                                      comparison of two arbitrary bases.
  everything else that matches        the BEV encoder, the detection and map
                                      heads, the world model: same shapes, and
                                      the teacher's are further trained.

WHAT IS NOT COPIED

  pts_bbox_head.ego_lcf_embed_net.*   the teacher's status producer. It reads
                                      the real ego status (8 numbers); the
                                      student's ego_status_est_net reads a 6144-d
                                      BEV descriptor. Different input, different
                                      shape, no correspondence.
  pts_bbox_head.ego_status_est_net.*  absent from the teacher by construction.
                                      Left at its own initialization.
  buffers whose shape disagrees       reported, never forced.

A NOTE ON THE 64-WIDE HIDDEN

It is tempting to make the student's ego_status_est_net hidden width match the
teacher's ego_lcf_embed_hidden so the second layer could be copied too
(ego_status_est_hidden exists for exactly that). Measure what it buys first: the
teacher's hidden is a ReLU of a Linear(8, 64), so it lives on a manifold of rank
at most 8 -- the same eight numbers the slot already carries. Matching it
transfers nothing extra while narrowing the student's vision path from
6144 -> 256 -> 8 to 6144 -> 64 -> 8.

USAGE

  python tools/make_kd_student_init.py \
      --teacher work_dirs/stage2_teacher8_lk60v3_fulldata_v1/epoch_12.pth \
      --student-config projects/configs/VAD/VADLAW_etri_tiny_base_nokd_fulldata.py \
      --teacher-config projects/configs/VAD/VADLAW_etri_tiny_teacher8_fulldata.py \
      --output work_dirs/fulldata_stage1_donors/kd_init_from_teacher8_v3.pth

The teacher checkpoint's ordinary slots hold the EMA weights (EMAHook writes the
average into them and keeps the raw ones as ema_* entries), so that is what gets
copied; --raw takes the ema_* side instead.
"""
import argparse
import json

import torch


def layouts_of(config_path):
    from mmcv import Config
    lay = Config.fromfile(config_path).model.pts_bbox_head.get('cell_layouts')
    if lay is None:
        return None
    return json.dumps([None if v is None
                       else [[float(x) for x in v[0]], [float(x) for x in v[1]]]
                       for v in lay])


def build_state_dict(config_path):
    from mmcv import Config
    from mmdet3d.models import build_model
    import projects.mmdet3d_plugin  # noqa: F401
    cfg = Config.fromfile(config_path)
    m = cfg.model.copy()
    for k in ('feature_distill_teacher_cfg', 'feature_distill_teacher_ckpt',
              'trajectory_distill_weight'):
        m.pop(k, None)
    model = build_model(m, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    return model.state_dict()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--teacher', required=True, help='trained teacher checkpoint')
    p.add_argument('--student-config', required=True)
    p.add_argument('--teacher-config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--raw', action='store_true',
                   help="take the teacher's ema_* (raw) weights instead of the "
                        'EMA ones that sit in the ordinary slots')
    p.add_argument('--allow-layout-mismatch', action='store_true',
                   help='copy even if the two configs disagree on cell_layouts. '
                        'Only for a deliberate experiment: with different '
                        'layouts, cell k is a different stretch of road on each '
                        'side and the copied heads are meaningless.')
    args = p.parse_args()

    ls, lt = layouts_of(args.student_config), layouts_of(args.teacher_config)
    same_layout = (ls == lt)
    print(f'cell_layouts identical: {same_layout}')
    if not same_layout and not args.allow_layout_mismatch:
        raise SystemExit(
            'REFUSING: the student and teacher configs have different cell '
            'layouts, so a copied cell head would be indexed by cells that mean '
            'different things. Pass --allow-layout-mismatch only if that is '
            'genuinely what you want.')

    tsd = torch.load(args.teacher, map_location='cpu')
    meta = tsd.get('meta', {})
    tsd = tsd.get('state_dict', tsd)
    if args.raw:
        raw = {k[len('ema_'):].replace('_', '.'): v
               for k, v in tsd.items() if k.startswith('ema_')}
        print(f'--raw: {len(raw)} ema_* entries recovered')
        tsd = {**{k: v for k, v in tsd.items() if not k.startswith('ema_')}, **raw}
    else:
        tsd = {k: v for k, v in tsd.items() if not k.startswith('ema_')}
    print(f'teacher: {args.teacher}')
    print(f'         epoch {meta.get("epoch")}, {len(tsd)} tensors '
          f'({"raw" if args.raw else "EMA"} weights)')

    ssd = build_state_dict(args.student_config)
    print(f'student: {len(ssd)} tensors built from {args.student_config}')

    copied, shape_bad, absent = {}, [], []
    for k, v in ssd.items():
        if k not in tsd:
            absent.append(k)
        elif tuple(tsd[k].shape) != tuple(v.shape):
            shape_bad.append((k, tuple(tsd[k].shape), tuple(v.shape)))
        else:
            copied[k] = tsd[k].clone()

    def count(prefix):
        return sum(1 for k in copied if k.startswith(prefix))

    print()
    print(f'copied {len(copied)} / {len(ssd)} tensors')
    for label, prefix in (('cell_heads', 'pts_bbox_head.cell_heads.'),
                          ('ego_fut_decoder', 'pts_bbox_head.ego_fut_decoder.'),
                          ('ego_status_decode_head',
                           'pts_bbox_head.ego_status_decode_head.'),
                          ('bev_world_model', 'bev_world_model.'),
                          ('img_backbone', 'img_backbone.')):
        print(f'   {label:<24} {count(prefix)}')

    print()
    print(f'not in the teacher: {len(absent)}')
    for k in absent:
        print(f'   {k}')
    print(f'shape mismatch (left alone): {len(shape_bad)}')
    for k, a, b in shape_bad:
        print(f'   {k}  teacher{a} vs student{b}')

    if not count('pts_bbox_head.cell_heads.'):
        raise SystemExit('REFUSING: no cell head was copied. The whole point of '
                         'this tool is that both sides are cell planners with '
                         'the same layout; check the two configs.')

    torch.save({'state_dict': copied,
                'meta': {'seeded_from': args.teacher,
                         'teacher_epoch': meta.get('epoch'),
                         'weights': 'raw' if args.raw else 'ema',
                         'student_config': args.student_config,
                         'teacher_config': args.teacher_config,
                         'copied': len(copied), 'student_total': len(ssd)}},
               args.output)
    print()
    print(f'wrote {args.output}')
    print('Point the distilled run\'s load_from at it. Everything it does not '
          'carry keeps its own initialization, which is what the missing list '
          'above is.')


if __name__ == '__main__':
    main()
