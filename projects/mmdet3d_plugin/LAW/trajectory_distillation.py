"""Selected-trajectory KD in the same delta/weight/reduction convention as GT.

No GT target point enters candidate generation. The target point is used only
to mirror the existing router's STOP override when selecting a teacher mode.
"""
import torch


def selected_trajectory_kd(student, teacher, command, target_point, masks,
                           loss_fn, stop_threshold, time_mode='cumulative',
                           command_weights=None):
    if student.ndim != 4 or student.shape != teacher.shape:
        raise ValueError('KD requires matching [B, modes, time, 2] trajectories')
    b, modes, steps, xy = student.shape
    if modes != 7 or xy != 2 or steps < 3 or steps % 3:
        raise ValueError('Cell KD requires seven commands and a 3-window horizon')
    cmd = command.reshape(b, -1).detach()
    if cmd.shape[1] != modes or not bool(((cmd == 0) | (cmd == 1)).all()) or not bool((cmd.sum(-1) == 1).all()):
        raise ValueError('KD command must be one-hot over seven modes')
    mode = cmd.argmax(-1)
    tp = target_point.detach().reshape(b, -1)[:, :2]
    if not bool(torch.isfinite(tp[mode != 6]).all()):
        raise ValueError('Non-finite target point for moving KD sample')
    effective_mode = torch.where(tp.norm(dim=-1) < stop_threshold,
                                 torch.full_like(mode, 6), mode)
    ar = torch.arange(b, device=student.device)
    # Cell forward repeats the selected trajectory across seven modes. Index
    # ONCE, rather than comparing these seven copies with seven teacher modes.
    prediction = student[ar, mode].float()
    target = teacher.detach()[ar, effective_mode].float()
    mask = masks.reshape(b, steps).detach().float()
    if not bool(torch.isfinite(prediction).all() & torch.isfinite(target).all() & torch.isfinite(mask).all()):
        raise ValueError('Non-finite trajectory KD inputs')
    width = steps // 3
    pos = [sum(1.0 / (k * width) for k in (1, 2, 3) if i < k * width) / 3
           for i in range(steps)]
    if time_mode == 'cumulative':
        weights = [sum(pos[i:]) for i in range(steps)]
    elif time_mode == 'position':
        weights = pos
    else:
        raise ValueError('Unknown planning time weight mode')
    weights = mask.new_tensor(weights) * (steps / sum(weights))
    weight = (mask * weights)[..., None].expand_as(prediction)
    if command_weights is not None:
        weight = weight * cmd.new_tensor(command_weights)[mode, None, None]
    # GT plan_reg averages the masked [B,7,T,2] tensor, including six zero
    # modes. Keep that denominator, so a KD weight of .1 really means .1
    # times the equivalent GT regression objective, not seven times larger.
    return loss_fn(prediction, target, weight) / modes
