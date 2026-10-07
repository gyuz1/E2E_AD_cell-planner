"""One pass over a train config that checks the things this project keeps
getting silently wrong.

Every failure mode below has actually happened here, and none of them raised
at the time -- mmcv loads checkpoints with strict=False, so a mismatched or
missing module is a warning line in a log nobody reads, not a crash. The
22h-wasted incidents were all of this shape.

Checks, in order of how expensive the mistake was:

  1. DONOR SHAPES. Does load_from's ego_fut_decoder actually fit the model
     being built? A 576-wide donor in a 520-wide model silently trains from
     a random planner.
  2. TRAIN/EVAL PARITY. Does the paired eval config build the identical
     network? A setting that changes a module's width must be repeated there
     or evaluation scores a differently-initialized model.
  3. COMPLIANCE. For a submittable (student) config: ego_lcf must not reach
     the planner. Checks ego_lcf_feat_idx is None and that the status slot,
     if any, comes from the vision estimator.
  4. TEACHER/STUDENT SYMMETRY. The student distills ego_scene_feats from the
     teacher's BEV encoder, so both sides' motion descriptors should be
     built the same way -- otherwise the student is asked to reproduce
     structure the teacher was never pushed to encode.
  5. DATA LEAKAGE. train ann_file must not contain val scenes, and the KD
     teacher cache (when used) must not be read for val scenes.

Usage:
    python tools/audit_pipeline.py <train_config> [--eval-config <cfg>]
                                   [--ann-dir <dir>]
"""
import argparse
import hashlib
import json
import math
import re
import os
import pickle

import mmcv
import torch
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401


OK, WARN, FAIL = '  [OK]  ', '  [!!]  ', '  [FAIL]'


class Audit:
    def __init__(self):
        self.failed = False

    def ok(self, msg):
        print(OK + msg)

    def warn(self, msg):
        print(WARN + msg)

    def fail(self, msg):
        print(FAIL + ' ' + msg)
        self.failed = True


def build(path):
    cfg = mmcv.Config.fromfile(path)
    model_cfg = cfg.model.copy()
    # Training-only object whose checkpoint may not exist yet.
    model_cfg.pop('feature_distill_teacher_cfg', None)
    model_cfg.pop('feature_distill_teacher_ckpt', None)
    model = build_model(model_cfg, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    return cfg, model


def head_cfg(cfg):
    return cfg.model.pts_bbox_head


def goal_layout(h):
    """Canonical anchor values, including the legacy shared-bin format.

    Shapes cannot detect a changed centre/width. Keep all values in this
    representation so a train/eval comparison also catches that silent error.
    """
    if not h.get('goal_pred', False):
        return None
    modes = int(h.get('ego_fut_mode', 3))
    anchors = h.get('goal_anchors')
    if anchors is None:
        edges = h.get('goal_bin_edges')
        if edges is None or len(edges) < 2:
            raise ValueError('goal_pred needs goal_anchors or legacy goal_bin_edges')
        edges = [float(v) for v in edges]
        lat = float(h.get('goal_lat_scale', 25.0))
        if (not all(math.isfinite(v) for v in edges)
                or any(b <= a for a, b in zip(edges, edges[1:]))):
            raise ValueError('goal_bin_edges must be finite and strictly increasing')
        shared = [[(a + b) / 2, b - a, 0.0, lat]
                  for a, b in zip(edges, edges[1:])]
        anchors = [shared for _ in range(modes)]
    if len(anchors) != modes:
        raise ValueError(f'goal_anchors has {len(anchors)} modes, expected {modes}')
    result = []
    for mode, cells in enumerate(anchors):
        if not cells:
            raise ValueError(f'goal_anchors mode {mode} is empty')
        row = []
        for index, cell in enumerate(cells):
            if len(cell) != 4:
                raise ValueError(f'goal_anchors[{mode}][{index}] needs [xc,xw,yc,yw]')
            values = tuple(float(v) for v in cell)
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f'goal_anchors[{mode}][{index}] contains non-finite values')
            if values[1] <= 0 or values[3] <= 0:
                raise ValueError(f'goal_anchors[{mode}][{index}] widths must be positive')
            row.append(values)
        result.append(tuple(row))
    return tuple(result)


def goal_layout_hash(layout):
    payload = json.dumps(layout, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def check_goal_layout(a, cfg):
    if not head_cfg(cfg).get('goal_pred'):
        return None
    try:
        layout = goal_layout(head_cfg(cfg))
    except (TypeError, ValueError, OverflowError) as exc:
        a.fail(f'invalid goal anchor layout: {exc}')
        return None
    counts = [len(row) for row in layout]
    a.ok(f'goal anchor values valid: counts={counts}, sha256={goal_layout_hash(layout)}')
    return layout


def check_donor(a, cfg, model):
    print('\n1. donor checkpoint shapes')
    lf = cfg.get('load_from')
    if not lf:
        a.warn('no load_from (normal for training from scratch)')
        return
    if not os.path.exists(lf):
        a.warn(f'load_from file is not on this machine: {lf}')
        return
    sd = torch.load(lf, map_location='cpu')
    sd = sd.get('state_dict', sd)
    msd = model.state_dict()
    bad = [(k, tuple(sd[k].shape), tuple(msd[k].shape))
           for k in sd if k in msd and sd[k].shape != msd[k].shape]
    dec = [b for b in bad if 'ego_fut_decoder' in b[0]]
    # goal_pred heads lengthen the last decoder layer from fut_ts to
    # goal_long_ts steps on purpose, and VADHead._load_from_state_dict copies
    # the donor into the first fut_ts steps and repeats its last step. Don't
    # trust the shape arithmetic: run the real load and compare bit for bit.
    head = model.pts_bbox_head
    if dec and getattr(head, 'goal_pred', False):
        model.load_state_dict(dict(sd), strict=False)
        ext_ok = True
        m, t, tl = head.ego_fut_mode, head.fut_ts, head.ego_dec_ts
        for key, s_, _ in dec:
            got = model.state_dict()[key]
            src = sd[key].reshape(m, t, 2, *sd[key].shape[1:])
            new = got.reshape(m, tl, 2, *got.shape[1:])
            if not torch.equal(new[:, :t], src):
                ext_ok = False
                print(f'          {key}: first {t} steps differ from the donor')
            if not torch.equal(new[:, t:], src[:, -1:].expand_as(new[:, t:])):
                ext_ok = False
                print(f'          {key}: extra steps are not a copy of the donor last step')
        if ext_ok:
            a.ok(f'ego_fut_decoder last layer donor{dec[0][1]} -> model{dec[0][2]}: '
                 f'first {t} steps are the donor as-is, {tl - t} steps extended at constant velocity (verified by a real load)')
            dec = []
        else:
            a.fail('goal_pred decoder extension failed -> part of the planner is randomly initialized')
            return
    # A stage-1 config warm-starts from the nuScenes LAW checkpoint, whose
    # decoder cannot transfer by construction: it was trained with a
    # different ego_fut_mode, so the final layer's shape differs and the
    # hidden width follows from that. Reinitializing there is correct, and
    # is what every stage 1 in this repo has always done. Only a stage-2
    # donor (a stage2_init_merged_*.pth) is supposed to carry the decoder.
    stage1_warmstart = 'law_pretrained_nus' in os.path.basename(lf)
    if dec and stage1_warmstart:
        a.ok(f'ego_fut_decoder: {len(dec)} mismatches -- expected for a nuScenes '
             'warm start (the planner is trained fresh in stage 1)')
    elif dec:
        a.fail(f'ego_fut_decoder: {len(dec)} mismatches -> the planner is randomly initialized')
        for k, s, m in dec:
            print(f'          {k}: donor{s} vs model{m}')
    else:
        a.ok(f'ego_fut_decoder transfers cleanly ({os.path.basename(lf)})')
    other = [b for b in bad if 'ego_fut_decoder' not in b[0]]
    if other:
        a.warn(f'{len(other)} other mismatches (normal for new modules)')
        for k, s, m in other[:5]:
            print(f'          {k}: donor{s} vs model{m}')


def check_parity(a, train_path, eval_path):
    print('\n2. train/eval config parity')
    if not eval_path:
        a.warn('no eval config given -- skipped')
        return
    _, tm = build(train_path)
    _, em = build(eval_path)
    t, e = tm.state_dict(), em.state_dict()
    mism = sorted(k for k in t.keys() & e.keys() if t[k].shape != e[k].shape)
    eval_only = sorted(e.keys() - t.keys())
    if mism:
        a.fail(f'{len(mism)} shape mismatches -> silently random-initialized at eval')
        for k in mism[:5]:
            print(f'          {k}: train{tuple(t[k].shape)} vs eval{tuple(e[k].shape)}')
    else:
        a.ok('all shapes match')
    if eval_only:
        a.fail(f'{len(eval_only)} modules exist only in eval (no weights in the checkpoint)')
    else:
        a.ok('no eval-only modules')
    tc = mmcv.Config.fromfile(train_path)
    ec = mmcv.Config.fromfile(eval_path)
    try:
        ta, ea = goal_layout(head_cfg(tc)), goal_layout(head_cfg(ec))
        if ta != ea:
            a.fail('train/eval goal anchor VALUES differ -- matching head shapes are not enough')
        elif ta is not None:
            a.ok(f'train/eval goal anchors identical: sha256={goal_layout_hash(ta)}')
    except (TypeError, ValueError, OverflowError) as exc:
        a.fail(f'cannot compare goal anchor values: {exc}')
    check_behaviour_flags(a, train_path, eval_path)


# Settings that may differ between train and eval because they only shape a
# loss, a dropout, or a training-only module whose forward is gated on
# self.training. Everything NOT matched here must be identical.
TRAIN_ONLY_KEYS = re.compile(
    r'(^|\.)('
    r'loss_[a-z_]+\.loss_weight|[a-z_]*_weight|echo_cycle_weight|'
    r'prev_bev_dropout|ego_status_est_dropout|aux_bev_motion_norm|'
    r'aux_ego_motion(_idx)?|ego_status_decode|ego_status_distill_idx|'
    r'plan_reg_ts_weight_mode|privileged_distill(_idx)?|'
    r'remove_auxiliary_planning_losses|feature_distill_[a-z_]+|disable_dropout|'
    r'goal_expose_candidates|'
    r'train_cfg\..*'
    r')$')


def _flat(d, prefix=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f'{prefix}{k}.'))
    elif (isinstance(d, (list, tuple)) and d
          and all(isinstance(x, dict) for x in d)):
        for i, v in enumerate(d):
            out.update(_flat(v, f'{prefix}{i}.'))
    else:
        out[prefix[:-1]] = d
    return out


def check_behaviour_flags(a, train_path, eval_path):
    # The shape check above cannot see a flag that changes the forward pass
    # without changing a tensor. On 2026-09-15 two eval configs were wrong in
    # exactly that way and would both have produced numbers without an error:
    # a duplicated top-level `model = dict(...)` dropped grid 8 (that one did
    # change shapes), and the teacher's eval config lacked
    # ego_lcf_embed_residual=True (that one did not).
    tr = _flat(mmcv.Config.fromfile(train_path).model.to_dict())
    ev = _flat(mmcv.Config.fromfile(eval_path).model.to_dict())
    bad = []
    for k in sorted(set(tr) | set(ev)):
        if TRAIN_ONLY_KEYS.search(k):
            continue
        if tr.get(k, '<missing>') != ev.get(k, '<missing>'):
            bad.append((k, tr.get(k, '<missing>'), ev.get(k, '<missing>')))
    if bad:
        a.fail(f'{len(bad)} behaviour-changing settings differ between train and eval')
        for k, x, y in bad:
            print(f'          {k}: train={x!r} eval={y!r}')
    else:
        a.ok('all behaviour settings match (loss/dropout/train-only items excluded)')


def check_compliance(a, cfg):
    print('\n3. compliance (as a submitted model)')
    h = head_cfg(cfg)
    lcf_idx = h.get('ego_lcf_feat_idx')
    est = h.get('ego_status_est_dim')
    if lcf_idx is None:
        a.ok('ego_lcf_feat_idx=None -- real ego status never reaches the planner')
        if est:
            a.ok(f'the {est}-d status slot comes from ego_status_est_net (BEV) '
                 '-- a vision estimate, allowed by the 2026-09-04 Q&A')
    else:
        a.warn(f'ego_lcf_feat_idx={lcf_idx} -- not submittable (fine for a teacher)')
    if h.get('target_point_shortcut'):
        a.fail('target_point_shortcut is on -- the goal enters generation, not submittable')
    else:
        a.ok('target_point_shortcut is off')
    if h.get('aux_bev_motion_feedback'):
        a.warn('aux_bev_motion_feedback is on -- measured 0.5635 -> 0.6419 (worse)')

    # [user, 2026-09-16] TP may SELECT an independently generated candidate,
    # but may not generate or correct it (the specific, later Q&A ruling).
    # Ordinary generation must not read TP; only the explicit, reported
    # --select-goal-by-tp branch below is permitted to read it at inference.
    import re as _re
    for rel in ('tools/etri_test_submit.py',
                'tools/eval_holdout_l2_and_tinfer.py'):
        if not os.path.exists(rel):
            continue
        hits = []
        lines = open(rel).read().split('\n')
        for n, line in enumerate(lines, 1):
            code = line.split('#', 1)[0]
            if not _re.search(r'target_point', code):
                continue
            stripped = code.strip()
            # argparse help text and similar string-only lines
            if stripped[:1] in ('"', "'") or stripped.startswith('help='):
                continue
            if "'--zero-target-point'" in code:
                continue
            # --zero-target-point ERASES the target point to prove a model
            # does not depend on it; that removes information rather than
            # using it, so it is the opposite of what this check guards.
            window = '\n'.join(lines[max(0, n - 8):n])
            if 'args.zero_target_point' in window:
                continue
            if "'--select-goal-by-tp'" in code:
                continue
            # --select-goal-by-tp is the opt-in selection path: candidates are
            # generated without the target point and it only picks one, which
            # the organizers allow (2026-08-26, 08-27). It is off by default and
            # reported separately below, so it is not a silent leak.
            recent = '\n'.join(lines[max(0, n - 12):n])
            if any(f'args.{flag}' in recent for flag in
                   ('select_goal_by_tp', 'select_cell_by_tp', 'stop_by_tp')):
                continue
            # --select-cell-by-tp / --stop-by-tp: the same selection-only use.
            if line.rstrip().endswith('# report grouping only'):
                continue
            hits.append(n)
        if hits:
            a.fail(f'{rel} reads target_point outside explicit candidate selection '
                   f'(lines {hits[:5]}) -- it must not influence generation/correction')
        else:
            a.ok(f'{os.path.basename(rel)}: no unguarded target_point read outside selection')
    # eval_holdout_l2_and_tinfer.py's --select-goal-by-tp reads the target
    # point on purpose, to choose among candidates the model generated without
    # it (organizer answers 2026-08-26 / 08-27). It is opt-in and off by
    # default, so the check above must not see it as a silent leak: flag it
    # loudly instead.
    try:
        esrc = open('tools/eval_holdout_l2_and_tinfer.py').read()
        if '--select-goal-by-tp' in esrc:
            a.warn('the eval tool has --select-goal-by-tp (off by default). With it, the '
                   'given target point chooses among candidates -- the '
                   '"selection only" pattern the organizers allow, but be explicit '
                   'about which number is the submitted one')
    except OSError:
        pass
    if h.get('cell_planner'):
        import inspect as _insp
        from projects.mmdet3d_plugin.VAD.VAD_head import VADHead as _VH
        fsrc = _insp.getsource(_VH.forward)
        gen_mark = '# --- cell planner: generation (no target point) ---'
        sel_mark = '# --- cell planner: selection ---'
        ok = gen_mark in fsrc and sel_mark in fsrc
        if ok:
            gen = fsrc.split(gen_mark, 1)[1].split(sel_mark, 1)[0]
            sel = fsrc.split(sel_mark, 1)[1].split('elif self.goal_pred:', 1)[0]
            ok = ('target_point' not in gen
                  and 'target_point' not in _insp.getsource(_VH.cell_generate)
                  and fsrc.count('route_trajectory(') == 1
                  and sel.split('if self.training:', 1)[0].strip() == ''
                  and 'route_trajectory(' in sel.split('else:', 1)[0])
        if ok:
            layouts = [None if l is None else f'{len(l[0]) - 1}x{len(l[1]) - 1}'
                       for l in h.get('cell_layouts')]
            a.ok(f'cell_planner {layouts}: generation reads no target point; it only '
                 'selects, inside the training gate (callers select at inference). '
                 'Run tools/check_cell_planner_live.py for the dynamic proof.')
        else:
            a.fail('cell_planner: target point outside the training selection gate')
    if h.get('goal_pred'):
        import inspect as _insp
        from projects.mmdet3d_plugin.VAD.VAD_head import VADHead as _VH
        fsrc = _insp.getsource(_VH.forward)
        gate = 'if self.training and ego_target_point is not None:'
        # The target point may appear only to build labels, inside the gate.
        gated = fsrc.split(gate, 1)
        outside = gated[0] if len(gated) == 2 else fsrc
        if (len(gated) != 2 or fsrc.count('_goal_label_from_target(') != 1
                or '_goal_label_from_target(' in outside):
            a.fail('goal_pred reads target_point outside the self.training gate '
                   '-- using it at inference violates the rules')
        else:
            layout = check_goal_layout(a, cfg)
            if layout is not None:
                a.ok(f'goal_pred ({[len(row) for row in layout]} anchors per mode + '
                     'continuous offsets): source label gate verified; generation '
                     'uses predicted goals. Run the live checker for dynamic TP invariance.')


def check_train_eval_mismatch(a, cfg):
    """Settings that make training see something inference never does.

    Every one of these has cost this project results without raising: the
    network fits a regime it is not scored in. Measured 2026-09-15 for
    dropout alone: speed read +5% fast at inference, L2@3s 0.732 vs 0.553.
    A config that turns any of them on does not launch.
    """
    print('\n3c. train/inference mismatch settings (any one of them blocks training)')
    m = cfg.model
    h = m.pts_bbox_head
    bad = []
    if not m.get('disable_dropout', False):
        bad.append('disable_dropout is not True -- nn.Dropout is on in training only '
                   '(measured +5% speed bias)')
    if (m.get('prev_bev_dropout') or 0) > 0:
        bad.append(f"prev_bev_dropout={m.get('prev_bev_dropout')} -- inference always has a previous BEV")
    if (h.get('ego_status_est_dropout') or 0) > 0:
        bad.append(f"ego_status_est_dropout={h.get('ego_status_est_dropout')} -- inference always fills the slot")
    if h.get('prism_latent_supervision'):
        bad.append('prism_latent_supervision=True -- training uses a posterior that saw the GT future, inference the prior mean')
    if h.get('bev_residual_refine'):
        bad.append('bev_residual_refine=True -- measured 21-25% worse L2')
    if h.get('target_point_shortcut'):
        bad.append('target_point_shortcut=True -- a non-compliant diagnostic build')
    if h.get('privileged_distill'):
        bad.append('privileged_distill=True -- a train-only privileged path')
    # Yaw delta must be wrapped to [-180, 180) on the LAW path, in the training
    # queue and at inference (2026-09-15: unwrapped, 3.6% of frames fed ~359
    # into can_bus_mlp). Read the source, since both paths run without error
    # either way.
    try:
        law_ds = open('projects/mmdet3d_plugin/datasets/law_etri_dataset.py').read()
        law_md = open('projects/mmdet3d_plugin/LAW/VAD_LAW.py').read()
        wrap_ok = ('- previous_angle + 180) % 360 - 180' in law_ds
                   and '- self.prev_frame_info["prev_angle"] + 180) % 360 - 180' in law_md
                   and 'meta["can_bus"][-1] -= previous_angle' not in law_ds)
    except OSError:
        wrap_ok = False
    if not wrap_ok:
        bad.append('the can_bus yaw delta is not wrapped to +-180 in the LAW training queue or in VADLAW inference')
    if cfg.data.train.get('history_sampling', 'random') != 'fixed':
        bad.append("history_sampling is not 'fixed' -- frame gaps differ from inference")
    if bad:
        for x in bad:
            a.fail(x)
    else:
        a.ok('dropout, prev_bev_dropout, slot dropout, PRISM, refine and shortcut all off; fixed gaps; yaw wrapped')


def check_silent_noops(a, cfg):
    """Settings that build cleanly but do nothing unless a second flag is on.

    These are worse than a crash: the config reads as intended, the model
    builds, training runs to completion, and the feature was never active.
    """
    print('\n3b. silent no-ops')
    h = head_cfg(cfg)
    frames = h.get('aux_bev_motion_frames')
    grid = h.get('aux_bev_motion_grid')
    temporal = h.get('aux_bev_motion_temporal')
    future = h.get('aux_bev_future_motion')
    motion = h.get('aux_bev_motion')

    if (frames or grid) and not temporal:
        a.fail('aux_bev_motion_frames/grid are set but aux_bev_motion_temporal is off '
               '-- the descriptor falls back to a single-frame global mean and '
               'both settings are ignored')
    elif temporal:
        a.ok(f'temporal descriptor active (frames={frames or 2}, '
             f'grid={grid or 4})')
    if future and not motion:
        a.fail('aux_bev_future_motion reads aux_bev_motion\'s descriptor '
               '-- aux_bev_motion=True is required')
    # The 3-frame second difference is acceleration only when the two frame
    # gaps are equal. The dataset's default history sampling drops one of the
    # candidates at random, which makes them unequal in 67% of samples and
    # injects a v*dt term 46x the real a*dt^2 -- and evaluation always streams
    # equal gaps, so this is a train/eval mismatch on top of a wrong target.
    if frames and frames >= 3:
        sampling = cfg.data.train.get('history_sampling', 'random')
        if sampling != 'fixed':
            a.fail(f"aux_bev_motion_frames={frames} but "
                   f"data.train.history_sampling='{sampling}' -- 67% of training "
                   'samples get unequal frame gaps, so the second difference is '
                   "not acceleration. 'fixed' is required")
        else:
            a.ok("history_sampling='fixed' -- equal frame gaps (same as evaluation)")

    idx = h.get('aux_bev_motion_idx') or ()
    norm = h.get('aux_bev_motion_norm')
    if idx and not norm:
        a.warn(f'aux_bev_motion_idx={idx} with no normalization -- measured, vx/speed take '
               '98.7% of the L1 and yaw_rate 0.1%, so yaw is effectively unsupervised')
    elif norm and len(norm) != len(idx):
        a.fail(f'aux_bev_motion_norm has length {len(norm)} != idx length {len(idx)}')


def check_symmetry(a, cfg, train_path):
    print('\n4. teacher/student descriptor symmetry')
    tcfg_path = cfg.model.get('feature_distill_teacher_cfg')
    if not tcfg_path:
        a.warn('no distillation teacher -- skipped')
        return
    if not os.path.exists(tcfg_path):
        a.fail(f'teacher config file not found: {tcfg_path}')
        return
    tcfg = mmcv.Config.fromfile(tcfg_path)
    th, sh = head_cfg(tcfg), head_cfg(cfg)
    # Anything that shapes ego_feats belongs here, not just the descriptor:
    # ego_scene_feats and ego_status_feats ARE the two halves of ego_feats, so
    # a loss the teacher never had pulls the student's features away from the
    # target it is being aligned to. bev_residual_refine additionally decides
    # whether the teacher is any good -- it costs 21-25% L2 (measured).
    keys = ('aux_bev_motion_frames', 'aux_bev_motion_grid',
            'aux_bev_motion_proj_dim', 'aux_bev_motion_temporal',
            'aux_bev_motion_idx', 'ego_status_distill_idx',
            'bev_residual_refine', 'bev_refine_steps',
            'aux_long_horizon', 'aux_long_horizon_residual',
            'ego_status_decode')
    diff = [(k, th.get(k), sh.get(k)) for k in keys if th.get(k) != sh.get(k)]
    if diff:
        a.warn('teacher and student descriptor settings differ -- the student is asked '
               'to reproduce structure the teacher never encoded')
        for k, tv, sv in diff:
            print(f'          {k}: teacher={tv} student={sv}')
    else:
        a.ok('descriptor settings identical')
    # ego_lcf columns 5 and 6 are ego_length and ego_width: measured std
    # exactly 0 over the train split, one unique value each. Inside a cosine
    # target they are 21.5% of the squared norm on average and 99.7% of it on
    # stopped samples, so the student scores a near perfect cosine on the
    # frames where the teacher knew most. Narrowing the target costs nothing
    # -- ego_feats keeps all eight columns and the decoder stays 520 wide.
    if cfg.model.get('feature_distill_mode') == 'split':
        didx = sh.get('ego_status_distill_idx')
        tlcf = th.get('ego_lcf_feat_idx') or ()
        const_in_target = {5, 6} & set(tlcf)
        if const_in_target and (didx is None or const_in_target & set(didx)):
            a.warn('the status distillation target includes constant columns '
                   '(ego_length/ego_width) -- on stopped samples 99.7% of the '
                   'target is constant, so the cosine is satisfied for free. '
                   'ego_status_distill_idx=(0,1,2,3,4,7) is recommended')
        elif didx:
            a.ok(f'status distillation target = {didx} (constant columns excluded)')

    # The teacher is loaded with mmcv's load_checkpoint, which is
    # strict=False like everything else here. A teacher whose config and
    # checkpoint disagree loads anyway, with the mismatched modules randomly
    # initialized -- and then the student spends 12 epochs aligning itself to
    # a target produced by an untrained network. Nothing raises, and the
    # distillation losses look perfectly healthy while it happens.
    ckpt = cfg.model.get('feature_distill_teacher_ckpt')
    if ckpt and not os.path.exists(ckpt):
        a.warn(f'teacher checkpoint does not exist yet: {ckpt} (normal before training)')
    elif ckpt:
        tmodel = build_model(tcfg.model, train_cfg=tcfg.get('train_cfg'),
                             test_cfg=tcfg.get('test_cfg'))
        tsd = torch.load(ckpt, map_location='cpu')
        tsd = tsd.get('state_dict', tsd)
        msd = tmodel.state_dict()
        bad = [(k, tuple(tsd[k].shape), tuple(msd[k].shape))
               for k in tsd if k in msd and tsd[k].shape != msd[k].shape]
        missing = [k for k in msd if k not in tsd]
        if bad:
            a.fail(f'teacher checkpoint disagrees with the teacher config in {len(bad)} '
                   'places -> those modules build the distillation target from '
                   'random weights')
            for k, sh, mh in bad[:5]:
                print(f'          {k}: ckpt{sh} vs model{mh}')
        if missing:
            a.fail(f'{len(missing)} modules are missing from the teacher checkpoint '
                   '-> randomly initialized')
            for k in missing[:5]:
                print(f'          {k}')
        if not bad and not missing:
            a.ok(f'teacher checkpoint matches the config exactly '
                 f'({os.path.basename(ckpt)})')


def check_leakage(a, cfg, ann_dir):
    print('\n5. data leakage')
    # scripts/_common.sh launch_train overrides all three ann_file fields via
    # --cfg-options, so the path written in the config is NOT what trains.
    # Auditing the config's own path silently skipped this check entirely
    # (the configs still name a directory that does not exist on either
    # machine), which made the leakage check a no-op on every run so far.
    # Resolve the same way the launcher does.
    cfg_ann = cfg.data.train.get('ann_file')
    train_ann = os.path.join(ann_dir, 'vad_etri_infos_temporal_train_split.pkl')
    val_ann = os.path.join(ann_dir, 'vad_etri_infos_temporal_val_split.pkl')
    print(f'          train ann_file in the config : {cfg_ann}')
    print(f'          ann_dir training actually uses: {ann_dir}')
    print('          (launch_train overrides it via --cfg-options)')

    def scenes(p):
        with open(p, 'rb') as fh:
            return {i['scene_token'] for i in pickle.load(fh)['infos']}

    missing = [p for p in (train_ann, val_ann) if not os.path.exists(p)]
    if missing:
        # A fulldata config trains on every scene, so the 301/75 split files
        # this check compares do not exist next to it and never will. Say that
        # outright instead of failing on a check that cannot apply -- but never
        # report it as OK: nothing is held out, so no L2 from this lineage is a
        # hold-out number. A fulldata run is launched directly through
        # tools/train.py, not launch_train, so its own ann_file is what trains.
        if cfg_ann and os.path.exists(cfg_ann) and 'annotations_10hz' in cfg_ann:
            n = len(scenes(cfg_ann))
            a.warn(f'fulldata config -- trains on all {n} scenes, nothing held out. '
                   'Any L2 measured against the 301/75 split\'s val scenes is a '
                   'TRAINING-set number. Check 5 cannot apply.')
            print(f'          the 301/75 split files are absent here, as expected: '
                  f'{ann_dir}')
        else:
            a.fail('ann files not found -- check 5 cannot run. '
                   'Pass the real path with --ann-dir')
            for p in missing:
                print(f'          missing: {p}')
            return
    else:
        tr, va = scenes(train_ann), scenes(val_ann)
        overlap = tr & va
        if overlap:
            a.fail(f'train and val share {len(overlap)} scenes -- evaluation is meaningless')
        else:
            a.ok(f'train {len(tr)} / val {len(va)} scenes, 0 overlap')

    # KD teacher cache: only matters if the pipeline actually loads it.
    pipe = cfg.data.train.get('pipeline', [])
    tcache = next((t.get('cache_path') for t in pipe
                   if t.get('type') == 'LoadTeacherWaypoints'), None)
    if tcache is None:
        a.ok('no LoadTeacherWaypoints -- the Qwen KD cache is unused')
    else:
        a.warn(f'Qwen KD cache in use: {tcache}')
        print('          val entries in the cache are only ever looked up through the '
              'train ann_file, so this is not leakage -- but remember the '
              'memorization signal: hold-out 0.3511 vs train 0.1798')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('--eval-config')
    p.add_argument('--ann-dir',
                   default='data/etri/.causal_regen_split_301_75_10hz',
                   help='the ann directory training actually uses. Must match ANN_DIR in '
                        'scripts/_common.sh (not the path inside the config)')
    args = p.parse_args()

    print('=' * 70)
    print('auditing:', args.train_config)
    print('=' * 70)
    a = Audit()
    cfg, model = build(args.train_config)
    check_donor(a, cfg, model)
    check_parity(a, args.train_config, args.eval_config)
    check_compliance(a, cfg)
    check_train_eval_mismatch(a, cfg)
    check_silent_noops(a, cfg)
    check_symmetry(a, cfg, args.train_config)
    check_leakage(a, cfg, args.ann_dir)
    print('\n' + '=' * 70)
    print('verdict:', 'FAILED -- do not train until the [FAIL] items above are fixed'
          if a.failed else 'passed')
    return 1 if a.failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
