"""VADHead subclass adding a challenge-metric-aligned teacher distillation loss.

Does not modify VAD_head.py, VAD.py, or VAD_LAW.py at all. Teacher waypoints
reach this class entirely through img_metas -- the same channel VADLAW
already uses for ego_fut_trajs/ego_fut_masks/ego_fut_cmd/ego_lcf_feat (see
VAD_LAW.py's _stack_meta_tensor and forward_train's current_metas, which is
exactly `[sample_meta[queue_length - 1] for sample_meta in img_metas]` --
the current/last frame's meta dict per batch sample). Registering
'teacher_ego_waypoints' and 'teacher_valid' in the pipeline's CustomCollect3D
meta_keys is the only wiring needed; union2one() already copies the base
per-frame meta dict forward untouched.

Works for both plain VAD (stage 1) and VADLAW (stage 2): both eventually
call self.pts_bbox_head.loss(..., img_metas=<list of per-sample current-frame
meta dicts>), which is all this class depends on.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmdet.models import HEADS

from projects.mmdet3d_plugin.VAD.VAD_head import VADHead


def _stack_meta_field(img_metas: Sequence[Dict], key: str,
                       device: torch.device) -> torch.Tensor:
    values = []
    for meta in img_metas:
        if key not in meta:
            raise KeyError(
                f"'{key}' missing from img_metas -- add it to "
                "CustomCollect3D's meta_keys in the KD pipeline config.")
        value = meta[key]
        tensor = value if torch.is_tensor(value) else torch.as_tensor(
            np.asarray(value))
        values.append(tensor)
    return torch.stack(values, dim=0).to(device=device)


def challenge_l2(student_waypoints: torch.Tensor,
                  target_waypoints: torch.Tensor) -> torch.Tensor:
    """Per-sample L2, matching eval_holdout_l2.py / etri_test_submit.py.

    Args:
        student_waypoints, target_waypoints: [B, 6, 2] cumulative positions.
    Returns:
        [B] challenge-metric distance (mean of 1s/2s/3s windowed ADE).
    """
    dist = torch.linalg.norm(student_waypoints - target_waypoints, dim=-1)
    l2_1s = dist[:, :2].mean(dim=1)
    l2_2s = dist[:, :4].mean(dim=1)
    l2_3s = dist[:, :6].mean(dim=1)
    return (l2_1s + l2_2s + l2_3s) / 3.0


@HEADS.register_module()
class VADHeadKD(VADHead):
    """VADHead + a teacher-distillation planning loss (train-time only).

    The teacher is never part of the forward graph -- only its cached
    trajectory (read via img_metas) is used, so inference-time FLOPs/latency
    are identical to plain VADHead.
    """

    def __init__(self, kd_weight: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.kd_weight = kd_weight

    def _select_command_trajectory(self, ego_fut_preds: torch.Tensor,
                                    ego_fut_cmd: torch.Tensor) -> torch.Tensor:
        """[B, ego_fut_mode, fut_ts, 2] -> [B, fut_ts, 2] via GT command."""
        batch_size, num_modes = ego_fut_preds.shape[:2]
        command = ego_fut_cmd.reshape(batch_size, -1)
        if command.shape[1] != num_modes:
            raise ValueError(
                f"Expected {num_modes} command values per sample, got "
                f"{tuple(command.shape)}.")
        mode_index = command.argmax(dim=-1)
        batch_index = torch.arange(batch_size, device=ego_fut_preds.device)
        return ego_fut_preds[batch_index, mode_index]

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        loss_dict = super().loss(
            gt_bboxes_list,
            gt_labels_list,
            map_gt_bboxes_list,
            map_gt_labels_list,
            preds_dicts,
            ego_fut_gt,
            ego_fut_masks,
            ego_fut_cmd,
            gt_attr_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            map_gt_bboxes_ignore=map_gt_bboxes_ignore,
            img_metas=img_metas,
        )

        if img_metas is None or self.kd_weight == 0:
            return loss_dict

        ego_fut_preds = preds_dicts['ego_fut_preds']
        device = ego_fut_preds.device

        teacher_valid = _stack_meta_field(
            img_metas, 'teacher_valid', device=device).reshape(-1).bool()
        if not torch.any(teacher_valid):
            # No teacher target in this batch (e.g. cache not built yet for
            # these samples) -- contribute a real zero, not a skipped key,
            # so loss_plan_kd stays visible in the logs at 0.
            loss_dict['loss_plan_kd'] = ego_fut_preds.sum() * 0.0
            return loss_dict

        teacher_waypoints = _stack_meta_field(
            img_metas, 'teacher_ego_waypoints', device=device
        ).float().reshape(-1, self.fut_ts, 2)

        # ego_fut_cmd arrives as a proper tensor argument to this very
        # method (VADHead.loss's signature), so use it directly. The
        # previous version re-read it out of img_metas, which needed it
        # duplicated into CustomCollect3D's meta_keys and then failed
        # anyway: as a meta value it survives the pipeline as an
        # object-dtype numpy array that torch.as_tensor() cannot convert.
        # Only teacher_valid/teacher_ego_waypoints genuinely have to come
        # through the meta channel, since nothing else carries them.
        command = ego_fut_cmd.to(device)
        student_delta = self._select_command_trajectory(ego_fut_preds, command)
        student_waypoints = student_delta.cumsum(dim=-2)

        per_sample_l2 = challenge_l2(student_waypoints, teacher_waypoints)
        per_sample_l2 = torch.nan_to_num(per_sample_l2)
        loss_plan_kd = (per_sample_l2 * teacher_valid.float()).sum() / \
            teacher_valid.float().sum().clamp(min=1.0)

        loss_dict['loss_plan_kd'] = self.kd_weight * loss_plan_kd
        return loss_dict
