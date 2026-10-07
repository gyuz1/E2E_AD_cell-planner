"""Pipeline transform that attaches a cached teacher trajectory per sample.

Reads a JSON cache built by generate_teacher_cache.py, keyed by
"{scene_token}::{frame_idx}" -> {"waypoints": [[x,y]*6], "valid": bool,
"teacher_l2": float}. Sets results['teacher_ego_waypoints'] ([6,2] float32,
cumulative current-frame-relative positions, zeros if missing) and
results['teacher_valid'] (bool) unconditionally, so every sample in a batch
has the same shape/keys regardless of whether that specific frame has a
cached teacher prediction -- VADHeadKD masks out invalid ones at loss time.

Add 'teacher_ego_waypoints' and 'teacher_valid' to CustomCollect3D's
meta_keys in the KD config's train pipeline; no other file needs to change.
"""
import json

import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadTeacherWaypoints:

    def __init__(self, cache_path, fut_ts=6, strict_shape=True):
        self.cache_path = cache_path
        self.fut_ts = fut_ts
        self.strict_shape = strict_shape
        with open(cache_path) as handle:
            self.cache = json.load(handle)

    def __call__(self, results):
        key = f"{results['scene_token']}::{results['frame_idx']}"
        entry = self.cache.get(key)

        if entry is not None and entry.get('valid', False):
            waypoints = np.asarray(entry['waypoints'], dtype=np.float32)
            if self.strict_shape and waypoints.shape != (self.fut_ts, 2):
                raise ValueError(
                    f"Teacher cache entry {key} has shape "
                    f"{waypoints.shape}, expected ({self.fut_ts}, 2).")
            results['teacher_ego_waypoints'] = waypoints
            results['teacher_valid'] = True
        else:
            results['teacher_ego_waypoints'] = np.zeros(
                (self.fut_ts, 2), dtype=np.float32)
            results['teacher_valid'] = False

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(cache_path={self.cache_path})"
