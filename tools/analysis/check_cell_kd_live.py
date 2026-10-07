"""One real full-KD forward/backward. No optimizer step, no checkpoint writes.

Verifies actual target tensors, independent temporal history, and gradients.
Run on an idle GPU, with CUDA_VISIBLE_DEVICES exposing that GPU as device 0.
"""
import argparse
import importlib
import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    importlib.import_module('tools.cache.etri_geometry_cache')
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    torch.manual_seed(0)
    dataset = build_dataset(cfg.data.train)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    loaded = load_checkpoint(model, cfg.load_from, map_location='cpu')
    sd = loaded['state_dict']
    matched = 0
    for key, value in model.state_dict().items():
        if key in sd and value.shape == sd[key].shape:
            assert torch.equal(value.cpu(), sd[key].cpu()), f'init mismatch: {key}'
            matched += 1
    assert matched > 0, 'no donor tensors loaded'
    print(f'[OK] {matched} compatible student tensors exactly match donor; '
          'new cell-head initialization is checked separately by check_cell_planner_live')
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
        print('[OK] student fp16 wrapping matches configured training')
    model.cuda().train()
    teacher = model._teacher_holder[0]
    assert teacher is not None and not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not any('teacher' in k for k in model.state_dict())
    _, kw = scatter_kwargs((), collate([dataset[11 % len(dataset)]], samples_per_gpu=1), [0])
    kw = kw[0]
    teacher_bevs, teacher_outputs, student_outputs = [], [], []
    student_history_ptrs = set()
    original_teacher = teacher.pts_bbox_head.forward
    original_student = model.pts_bbox_head.forward

    def student_spy(*a, **k):
        if k.get('prev_bev') is not None:
            student_history_ptrs.add(k['prev_bev'].data_ptr())
        out = original_student(*a, **k)
        if isinstance(out, dict):
            student_outputs.append(out)
        return out

    def teacher_spy(*a, **k):
        prev = k.get('prev_bev')
        if not teacher_bevs:
            assert prev is None, 'teacher must start with no history'
        else:
            assert prev is not None and torch.equal(prev, teacher_bevs[-1])
            assert prev.data_ptr() not in student_history_ptrs
        if not k.get('only_bev'):
            assert len(teacher_bevs) >= 2, 'three-frame teacher history missing'
            assert torch.equal(k['prev_bev2'], teacher_bevs[-2]), 'prev2 was rotated or reused'
        out = original_teacher(*a, **k)
        if k.get('only_bev'):
            teacher_bevs.append(out.detach().clone())
        else:
            teacher_outputs.append(out)
        return out

    teacher.pts_bbox_head.forward = teacher_spy
    model.pts_bbox_head.forward = student_spy
    try:
        losses = model.forward_train(**kw)
    finally:
        teacher.pts_bbox_head.forward = original_teacher
        model.pts_bbox_head.forward = original_student
    assert len(teacher_outputs) == 1
    tout, sout = teacher_outputs[0], student_outputs[-1]
    for key, width in [('ego_scene_feats', 512), ('ego_status_feats', 6)]:
        assert sout[key].shape == tout[key].shape and sout[key].shape[-1] == width
        assert not tout[key].requires_grad
    print('[OK] independent teacher BEVs, pristine prev2, detached 512/6 KD targets')
    for name in ('loss_plan_reg', 'loss_trajectory_distill', 'loss_scene_distill', 'loss_status_distill'):
        assert name in losses and torch.isfinite(losses[name]).all(), name
        print(f'{name}: {float(losses[name].detach()):.8f}')
    losses['loss_trajectory_distill'].backward(retain_graph=True)
    head_grad = sum(float(p.grad.abs().sum()) for p in model.pts_bbox_head.cell_heads.parameters() if p.grad is not None)
    status_grad = sum(float(p.grad.abs().sum()) for p in model.pts_bbox_head.ego_status_est_net.parameters() if p.grad is not None)
    print(f'[diagnostic] trajectory-only gradients: cell={head_grad:.6g}, status={status_grad:.6g}')
    assert head_grad > 0, 'trajectory KD does not reach cell heads'
    if status_grad == 0:
        # The 512-d stage1 donor is padded to 520 with zero status columns.
        # dL/dstatus is zero initially, but dL/dW_status must open that gate.
        first_layers = [h[0] for h in model.pts_bbox_head.cell_heads]
        assert all(torch.count_nonzero(h.weight[:, 512:520]).item() == 0
                   for h in first_layers), 'status gradient vanished with a nonzero gate'
        gate_grad = sum(float(h.weight.grad[:, 512:520].abs().sum())
                        for h in first_layers if h.weight.grad is not None)
        assert gate_grad > 0, 'trajectory KD cannot learn the zero-padded status gate'
        assert any(p.grad is not None for p in model.pts_bbox_head.ego_status_est_net.parameters()), \
            'status path is detached, not merely zero-gated'
        print(f'[OK] stage1 zero status gate; gate gradient={gate_grad:.6g}')
    assert all(p.grad is None for p in teacher.parameters())
    print(f'[OK] trajectory-only gradients: cell={head_grad:.6g}, status={status_grad:.6g}; teacher none')
    model.zero_grad(set_to_none=True)
    sum(v.mean() if torch.is_tensor(v) else sum(x.mean() for x in v)
        for k, v in losses.items() if 'loss' in k).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    full_status_grad = sum(float(p.grad.abs().sum())
                           for p in model.pts_bbox_head.ego_status_est_net.parameters()
                           if p.grad is not None)
    assert full_status_grad > 0, 'full objective does not train the status estimator'
    print(f'[OK] full objective status gradient={full_status_grad:.6g}')
    assert all(p.grad is None for p in teacher.parameters())
    print('[OK] full loss backward finite; no optimizer step taken')
    print(f'peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
