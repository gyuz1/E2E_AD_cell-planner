"""LAW temporal queue with per-frame VAD ego LCF metadata, for the ETRI
challenge dataset.

Ported from LAWVADCustomNuScenesDataset (LAW_p-based project) with one
change: the base class is VADCustomETRIDataset instead of
VADCustomNuScenesDataset, so ETRI-specific adaptations (union2one's caller,
crop/undistort pipeline, causal ego motion, etc.) are preserved. The
union2one override itself is unchanged -- LAW applies its previous-frame
waypoint/world-model objective to history frames as well, so every temporal
img_metas entry needs to carry ego_lcf_feat (the default VAD union2one only
keeps it on the current/last frame).
"""

import copy

import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS

from .etri_vad_dataset import VADCustomETRIDataset


@DATASETS.register_module()
class LAWVADCustomETRIDataset(VADCustomETRIDataset):
    """Original LAW temporal format plus previous-frame ego LCF metadata."""

    @staticmethod
    def _container_data(value):
        if isinstance(value, list):
            value = value[0]
        return value.data if isinstance(value, DC) else value

    def union2one(self, queue):
        imgs_list = []
        semantic_imgs_list = []
        metas_map = {}
        previous_position = None
        previous_angle = None

        for index, sample in enumerate(queue):
            imgs_list.append(self._container_data(sample["img"]))
            if "semantic_img" in sample:
                semantic_imgs_list.append(
                    self._container_data(sample["semantic_img"])
                )

            meta = copy.deepcopy(self._container_data(sample["img_metas"]))
            meta.update({
                "ego_fut_trajs": self._container_data(
                    sample["ego_fut_trajs"]
                ),
                "ego_fut_masks": self._container_data(
                    sample["ego_fut_masks"]
                ),
                "ego_fut_cmd": self._container_data(
                    sample["ego_fut_cmd"]
                ),
                "ego_his_trajs": self._container_data(
                    sample["ego_his_trajs"]
                ),
                "ego_lcf_feat": self._container_data(
                    sample["ego_lcf_feat"]
                ),
            })
            if "ego_target_point" in sample:
                meta["ego_target_point"] = self._container_data(
                    sample["ego_target_point"]
                )
            metas_map[index] = meta

            # Preserve original BEVFormer temporal ego-motion metadata.
            if index == 0:
                meta["prev_bev"] = False
                previous_position = copy.deepcopy(meta["can_bus"][:3])
                previous_angle = copy.deepcopy(meta["can_bus"][-1])
                meta["can_bus"][:3] = 0
                meta["can_bus"][-1] = 0
            else:
                meta["prev_bev"] = True
                current_position = copy.deepcopy(meta["can_bus"][:3])
                current_angle = copy.deepcopy(meta["can_bus"][-1])
                meta["can_bus"][:3] -= previous_position
                # Wrap to [-180, 180). Yaw is stored in [0, 360), so a turn
                # through north read +-359 deg instead of +-1 deg on 3.6% of
                # 0.5s train pairs (3.9% val) -- harmless to the prev_bev
                # rotation (359 == -1) but fed raw into can_bus_mlp, which
                # adds it to every BEV query. Stage 1's dataset
                # (nuscenes_vad_dataset.union2one) and VAD.forward_test already
                # wrap; the LAW path did not, in training or at inference.
                meta["can_bus"][-1] = (
                    (meta["can_bus"][-1] - previous_angle + 180) % 360 - 180)
                previous_position = current_position
                previous_angle = current_angle

        # Same as LAW: only image/meta are temporal; all top-level GT fields
        # are taken from the final/current queue frame.
        output = queue[-1]
        output["img"] = DC(
            torch.stack(imgs_list),
            cpu_only=False,
            stack=True,
        )
        if semantic_imgs_list:
            output["semantic_img"] = DC(
                torch.stack(semantic_imgs_list),
                cpu_only=False,
                stack=True,
            )
        output["img_metas"] = DC(metas_map, cpu_only=True)
        return output
