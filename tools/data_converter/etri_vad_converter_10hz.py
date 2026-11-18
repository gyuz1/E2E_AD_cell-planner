import argparse
import os
from os import path as osp

import cv2
import mmcv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

CAM_NAMES = [
    'camera_front',
    'camera_front_right',
    'camera_front_left',
    'camera_rear_wide',
    'camera_rear_left',
    'camera_rear_right',
]
CLASS_NAMES = ('Car', 'Pedestrian', 'Cyclist')
# Fixed order for the 7-way ego_fut_cmd one-hot. Must match
# etri_test_converter.py's non-STOP 6 exactly -- index i here is mode i of
# the model's ego_fut_mode=7 trajectory decoder. STOP is a label WE derive
# (raw command.parquet never contains it -- confirmed empirically, all 376
# train scenes only ever have the other 6 values) from the GT future
# trajectory, so it can only be assigned at train-pkl build time, never at
# test time (no future GT then). See STOP_DISP_THRESH below.
COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN', 'STOP',
)
# A sample is labeled STOP (overriding whatever raw_command says) when the
# ego's straight-line displacement from now to FUT_TS*TRAJ_STEP rows ahead
# (3.0s @ 10Hz) is below this. Calibrated against real train data: genuine
# stopped/parked frames measured ~0.002-0.004m over that window, genuine
# moving frames measured 17m+ -- huge separation, no ambiguous middle
# ground observed. Real-data audit: 7792/112800 (6.9%) of train frames
# qualify, spread across 87/376 scenes; among them raw_command was
# LANE_KEEP 71% of the time but TURN_LEFT 22% (unprotected-left-waiting-
# for-oncoming-traffic scenes) -- both were being mislabeled as "moving"
# commands before this fix.
STOP_DISP_THRESH = 0.5
FRAME_OFFSET = 50
MAIN_FRAMES = 300
MIN_FRAME = -FRAME_OFFSET
MAX_FRAME = FRAME_OFFSET + MAIN_FRAMES - 1
TRAJ_STEP = 5
FUT_TS = 6
# 5.0s at TRAJ_STEP(0.5s) spacing -- matches TARGET_POINT_FRAMES's known-
# available window. Privileged (training-only) extended future trajectory
# for PRISM-style latent supervision (arxiv 2608.01201, applied to VAD-
# Tiny): a posterior encoder sees this during training only and
# regularizes the vision-only prior via KL, discarded entirely at
# inference. Unlike target_point, never a model input at test time.
LONG_FUT_TS = 10
HIS_TS = 2
# 5.0s ahead at the raw 10Hz rate -- the "5-second target point" goal
# point ETRI documents providing per README.md's test clip structure
# (ego_pose.parquet's frame=50 row), used for goal conditioning (distinct
# from the 3.0s/FUT_TS trajectory). Every main-window frame_id has this
# available without clipping since the 400-frame scenario layout (50 past
# + 300 main + 50 future) was sized exactly for it.
TARGET_POINT_FRAMES = 50


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def euler_to_matrix(euler, degrees=False):
    return Rotation.from_euler('xyz', euler, degrees=degrees).as_matrix()


def matrix_to_quat_wxyz(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return [w, x, y, z]


def undistorted_intrinsic(intrinsic, distortion, image_size, is_fisheye):
    if is_fisheye:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            intrinsic, distortion[:4], image_size, np.eye(3), balance=0.0)
    new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
        intrinsic, distortion, image_size, alpha=0)
    return new_intrinsic


def load_camera_infos(scenario_dir):
    calib = pd.read_parquet(
        osp.join(scenario_dir, 'calibration', 'calibration.parquet'))
    calib = calib.set_index('camera_name')
    cam_infos = {}
    for cam_name in CAM_NAMES:
        row = calib.loc[cam_name]
        intrinsic = np.array(row['K'], dtype=np.float64).reshape(3, 3)
        distortion = np.array(row['distortion'], dtype=np.float64)
        is_fisheye = bool(row['is_fisheye'])
        image_size = (int(row['image_width']), int(row['image_height']))
        cam2ego_rotation = euler_to_matrix(
            np.array(row['euler'], dtype=np.float64), degrees=True)
        cam2ego_translation = np.array(row['translation'], dtype=np.float64)
        cam_infos[cam_name] = dict(
            type=cam_name,
            sensor2ego_translation=cam2ego_translation.tolist(),
            sensor2ego_rotation=matrix_to_quat_wxyz(cam2ego_rotation),
            sensor2lidar_rotation=cam2ego_rotation,
            sensor2lidar_translation=cam2ego_translation,
            cam_intrinsic=undistorted_intrinsic(
                intrinsic, distortion, image_size, is_fisheye),
            cam_intrinsic_raw=intrinsic,
            distortion=distortion,
            is_fisheye=is_fisheye,
            image_width=image_size[0],
            image_height=image_size[1],
        )
    return cam_infos


def load_scenario(scenario_dir):
    frames = pd.read_parquet(
        osp.join(scenario_dir, 'meta', 'timestamps.parquet'))
    frames = frames[['timestamp', 'frame_id']]
    ego_pose = pd.read_parquet(
        osp.join(scenario_dir, 'annotation', 'ego_pose.parquet'))
    ego_pose = ego_pose.merge(frames, on='timestamp').sort_values('frame_id')
    objects = pd.read_parquet(
        osp.join(scenario_dir, 'annotation', 'object.parquet'))
    objects = objects.merge(frames, on='timestamp').sort_values('frame_id')
    if 'num_points' not in objects.columns:
        raise ValueError(
            f'{scenario_dir}: object.parquet has no num_points column')
    command_df = pd.read_parquet(
        osp.join(scenario_dir, 'meta', 'command.parquet'))
    command_df = command_df.merge(frames, on='timestamp').sort_values('frame_id')

    timestamps = np.full(MAX_FRAME - MIN_FRAME + 1, np.nan)
    timestamps[frames['frame_id'].to_numpy() + FRAME_OFFSET] = \
        frames['timestamp'].to_numpy()

    commands = np.full(len(timestamps), '', dtype=object)
    commands[command_df['frame_id'].to_numpy() + FRAME_OFFSET] = \
        command_df['command'].to_numpy()

    ego_pose_xyz = np.full((len(timestamps), 3), np.nan)
    ego_pose_rpy = np.full((len(timestamps), 3), np.nan)
    pose_index = ego_pose['frame_id'].to_numpy() + FRAME_OFFSET
    ego_pose_xyz[pose_index] = ego_pose[['x', 'y', 'z']].to_numpy()
    ego_pose_rpy[pose_index] = ego_pose[['roll', 'pitch', 'yaw']].to_numpy()

    # hd_map.parquet's lane geometry is surveyed in the HD-map localization
    # frame (hd_ego_pose.parquet), which is NOT the same as the regular
    # ego_pose.parquet frame -- the two differ by several meters (measured
    # p95 ~2.9m, max ~6m across scenarios). Lanes must be cropped using
    # hd_ego_pose's translation/yaw, not the regular ego2global.
    hd_ego_pose = pd.read_parquet(
        osp.join(scenario_dir, 'annotation', 'hd_ego_pose.parquet'))
    hd_ego_pose = hd_ego_pose.merge(
        frames, on='timestamp').sort_values('frame_id')
    hd_ego_pose_xy = np.full((len(timestamps), 2), np.nan)
    hd_ego_pose_yaw = np.full(len(timestamps), np.nan)
    hd_pose_index = hd_ego_pose['frame_id'].to_numpy() + FRAME_OFFSET
    hd_ego_pose_xy[hd_pose_index] = hd_ego_pose[['x', 'y']].to_numpy()
    hd_ego_pose_yaw[hd_pose_index] = hd_ego_pose['yaw'].to_numpy()

    ego_rows = objects[objects['class'] == 'ego']
    ego_anno = np.full((len(timestamps), 4), np.nan)
    ego_index = ego_rows['frame_id'].to_numpy() + FRAME_OFFSET
    ego_anno[ego_index] = ego_rows[
        ['x[m]', 'y[m]', 'z[m]', 'heading[rad]']].to_numpy()

    object_rows = objects[objects['class'] != 'ego'].copy()
    object_rows['obj_id'] = object_rows['obj_id'].astype(np.int64)
    frame_objects = {
        frame_id: group
        for frame_id, group in object_rows.groupby('frame_id')
    }
    tracks = {
        obj_id: dict(
            zip(group['frame_id'],
                group[['x[m]', 'y[m]', 'heading[rad]']].to_numpy()))
        for obj_id, group in object_rows.groupby('obj_id')
    }
    return dict(
        timestamps=timestamps,
        ego_pose_xyz=ego_pose_xyz,
        ego_pose_rpy=ego_pose_rpy,
        hd_ego_pose_xy=hd_ego_pose_xy,
        hd_ego_pose_yaw=hd_ego_pose_yaw,
        ego_anno=ego_anno,
        commands=commands,
        frame_objects=frame_objects,
        tracks=tracks,
    )


def yaw_rotation_2d(yaw):
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array([[cos, -sin], [sin, cos]])


def track_velocity(data, obj_id, frame_id):
    track = data['tracks'][obj_id]
    prev_id = frame_id - 1 if frame_id - 1 in track else frame_id
    next_id = frame_id + 1 if frame_id + 1 in track else frame_id
    if prev_id == next_id:
        return np.zeros(2)
    dt = (data['timestamps'][next_id + FRAME_OFFSET] -
          data['timestamps'][prev_id + FRAME_OFFSET]) / 1e3
    return (track[next_id][:2] - track[prev_id][:2]) / dt


def agent_annotations(data, frame_id):
    group = data['frame_objects'].get(frame_id)
    num_box = 0 if group is None else len(group)
    anns = dict(
        gt_boxes=np.zeros((num_box, 7)),
        gt_names=np.zeros(num_box, dtype='<U16'),
        gt_velocity=np.zeros((num_box, 2)),
        num_lidar_pts=np.zeros(num_box, dtype=np.int64),
        num_radar_pts=np.zeros(num_box, dtype=np.int64),
        valid_flag=np.zeros(num_box, dtype=bool),
        gt_agent_fut_trajs=np.zeros((num_box, FUT_TS * 2), dtype=np.float32),
        gt_agent_fut_masks=np.zeros((num_box, FUT_TS), dtype=np.float32),
        gt_agent_lcf_feat=np.zeros((num_box, 9), dtype=np.float32),
        gt_agent_fut_yaw=np.zeros((num_box, FUT_TS), dtype=np.float32),
        gt_agent_fut_goal=np.zeros(num_box, dtype=np.float32),
    )
    if num_box == 0:
        return anns

    ego_x, ego_y, ego_z, ego_heading = data['ego_anno'][frame_id +
                                                        FRAME_OFFSET]
    to_ego = yaw_rotation_2d(-ego_heading)

    xyz = group[['x[m]', 'y[m]', 'z[m]']].to_numpy()
    heading = group['heading[rad]'].to_numpy()
    wlh = group[['length[m]', 'width[m]', 'height[m]']].to_numpy()
    obj_ids = group['obj_id'].to_numpy()
    names = group['class'].to_numpy().astype(str)
    num_points = group['num_points'].to_numpy().astype(np.int64)

    centers = np.column_stack([
        (xyz[:, :2] - [ego_x, ego_y]) @ to_ego.T,
        xyz[:, 2] + wlh[:, 2] / 2 - ego_z,
    ])
    yaw_ego = wrap_angle(heading - ego_heading)
    anns['gt_boxes'] = np.column_stack(
        [centers, wlh, wrap_angle(-yaw_ego - np.pi / 2)])
    anns['gt_names'] = names
    anns['num_lidar_pts'] = num_points
    anns['valid_flag'] = num_points > 0

    fut_trajs = np.zeros((num_box, FUT_TS, 2), dtype=np.float32)
    for i, obj_id in enumerate(obj_ids):
        track = data['tracks'][obj_id]
        velocity = to_ego @ track_velocity(data, obj_id, frame_id)
        anns['gt_velocity'][i] = velocity

        label = CLASS_NAMES.index(names[i]) if names[i] in CLASS_NAMES else -1
        anns['gt_agent_lcf_feat'][i] = [
            centers[i, 0], centers[i, 1], yaw_ego[i], velocity[0],
            velocity[1], wlh[i, 0], wlh[i, 1], wlh[i, 2], label
        ]

        prev_xy, prev_heading = xyz[i, :2], heading[i]
        for j in range(FUT_TS):
            fut_id = frame_id + (j + 1) * TRAJ_STEP
            if fut_id not in track:
                break
            fut_x, fut_y, fut_heading = track[fut_id]
            fut_trajs[i, j] = to_ego @ ([fut_x, fut_y] - prev_xy)
            anns['gt_agent_fut_yaw'][i, j] = wrap_angle(fut_heading -
                                                        prev_heading)
            anns['gt_agent_fut_masks'][i, j] = 1
            prev_xy, prev_heading = np.array([fut_x, fut_y]), fut_heading

        fut_coords = np.cumsum(fut_trajs[i], axis=0)
        coord_diff = fut_coords[-1] - fut_coords[0]
        if coord_diff.max() < 1.0:
            anns['gt_agent_fut_goal'][i] = 9
        else:
            motion_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
            anns['gt_agent_fut_goal'][i] = motion_yaw // (np.pi / 4)

    anns['gt_agent_fut_trajs'] = fut_trajs.reshape(num_box, FUT_TS * 2)
    return anns


def ego_positions_in_frame(data, frame_id, frame_ids):
    rotation = euler_to_matrix(data['ego_pose_rpy'][frame_id + FRAME_OFFSET])
    origin = data['ego_pose_xyz'][frame_id + FRAME_OFFSET]
    clipped = np.clip(np.asarray(frame_ids), MIN_FRAME, MAX_FRAME)
    positions = data['ego_pose_xyz'][clipped + FRAME_OFFSET]
    return (positions - origin) @ rotation, clipped == np.asarray(frame_ids)


def _vander_eval(coeffs, t):
    """Evaluate per-column polynomials (numpy.polyfit coefficient layout,
    highest power first) at sample points `t`. coeffs: (degree+1, D)."""
    degree = coeffs.shape[0] - 1
    return np.vander(t, degree + 1) @ coeffs


def _causal_poly_fit(t, pos, degree, tau=0.3, huber_c=1.5, iters=3):
    """Causal (t <= 0, 'now' at t=0) weighted polynomial fit.

    Combines exponential recency weighting (`tau`, seconds) with Huber-IRLS
    outlier down-weighting, refit `iters` times. Returns None if there
    aren't enough samples for the requested degree.
    """
    if len(t) < degree + 1:
        return None
    w = np.exp(t / tau)
    coeffs = np.polyfit(t, pos, degree, w=w)
    for _ in range(iters - 1):
        resid_norm = np.linalg.norm(pos - _vander_eval(coeffs, t), axis=1)
        sigma = 1.4826 * np.median(resid_norm) + 1e-6
        huber_w = np.minimum(1.0, huber_c * sigma / np.maximum(resid_norm, 1e-9))
        w2 = w * huber_w
        if w2.sum() < 1e-9:
            break
        coeffs = np.polyfit(t, pos, degree, w=w2)
    return coeffs


def robust_motion(times, positions, vel_window=20, accel_window=25):
    """Causal, robust velocity/acceleration at the last sample.

    `times` (seconds, ascending) and `positions` (Nx3) must only contain
    samples up to and including 'now' (the last row) -- no future frame is
    ever referenced, so this is safe to use identically at train and test
    time.

    10Hz variant: callers now pass RAW-frame-spaced samples (every 0.1s),
    not TRAJ_STEP(0.5s)-spaced ones. Past pose may be used to derive the
    current ego status without feeding the corresponding images -- pose used
    for this fit does NOT need a corresponding fed image, unlike the
    sibling (non-10hz) script's assumption. vel_window/accel_window are
    now raw-frame counts (20/25 = 2.0s/2.5s of history, same total time
    span the sibling script covered with 4/5 TRAJ_STEP-spaced samples --
    only the sampling density changes, not the lookback horizon), fixing
    the degenerate case where TEST's earliest replayed frames (-30/-25)
    had too few TRAJ_STEP-spaced points for the degree-2 accel fit
    (len(t) < 3) and silently zero-fell-back. Velocity comes from a
    degree-1 fit over the last `vel_window` samples; acceleration from a
    degree-2 fit over the last `accel_window` samples (one more sample
    than velocity, since the 2nd derivative amplifies noise more).
    Falls back to zeros where too little history is available (e.g. the
    first couple of frames of a clip/scenario).
    """
    now_t = times[-1]

    start_v = max(0, len(times) - vel_window)
    coeffs_v = _causal_poly_fit(
        times[start_v:] - now_t, positions[start_v:], degree=1)
    velocity = coeffs_v[0] if coeffs_v is not None else np.zeros(3)

    start_a = max(0, len(times) - accel_window)
    coeffs_a = _causal_poly_fit(
        times[start_a:] - now_t, positions[start_a:], degree=2)
    accel = 2 * coeffs_a[0] if coeffs_a is not None else np.zeros(3)

    return velocity, accel


def local_motion(data, index, rotation, accel_hops=4):
    """Robust causal velocity/accel/rotation_rate at `index`.

    10Hz variant: builds a DENSE window (every raw 0.1s frame, not just
    every TRAJ_STEP-th) spanning the same `accel_hops * TRAJ_STEP` raw
    frames (2.0s) of lookback the sibling (non-10hz) script covers with
    sparse TRAJ_STEP-spaced samples -- same horizon, denser sampling
    within it. Ending at `index` (never using index+1 or later). This does
    not need every one of these timestamps' images fed to the model (only the images actually fed
    matter for that rule; pose used purely for this derivative calc does
    not). Delegates to `robust_motion` for velocity/acceleration.
    rotation_rate stays a backward difference over one TRAJ_STEP hop,
    unchanged from the sibling script. Mirrors
    etri_test_converter_10hz.py's `local_motion`.
    """
    offsets = np.arange(accel_hops * TRAJ_STEP, -1, -1)
    window = index - offsets
    window = window[window >= 0]
    valid = ~np.isnan(data['timestamps'][window])
    window = window[valid]

    if len(window) < 2:
        velocity = np.zeros(3)
        accel = np.zeros(3)
    else:
        times = data['timestamps'][window] / 1e3
        positions = data['ego_pose_xyz'][window] @ rotation
        velocity, accel = robust_motion(times, positions)

    prev_index = index - TRAJ_STEP
    has_prev = prev_index >= 0 and not np.isnan(data['timestamps'][prev_index])
    if has_prev:
        dt = (data['timestamps'][index] -
              data['timestamps'][prev_index]) / 1e3
        rotation_rate = wrap_angle(data['ego_pose_rpy'][index] -
                                   data['ego_pose_rpy'][prev_index]) / dt
    else:
        rotation_rate = np.zeros(3)

    return velocity, accel, rotation_rate


def ego_annotations(data, frame_id, ego_length, ego_width):
    his_ids = frame_id + TRAJ_STEP * np.arange(-HIS_TS, 1)
    his_trajs, _ = ego_positions_in_frame(data, frame_id, his_ids)
    his_trajs = np.diff(his_trajs[:, :2], axis=0)

    fut_ids = frame_id + TRAJ_STEP * np.arange(FUT_TS + 1)
    fut_trajs, fut_in_range = ego_positions_in_frame(data, frame_id, fut_ids)
    fut_trajs = np.diff(fut_trajs[:, :2], axis=0)
    fut_masks = fut_in_range[1:]
    fut_valid_flag = bool(fut_in_range.all())

    # Extended (0-5s) version of the same trajectory, train-pkl-only, for
    # PRISM-style privileged supervision. See LONG_FUT_TS's comment above.
    long_fut_ids = frame_id + TRAJ_STEP * np.arange(LONG_FUT_TS + 1)
    long_fut_trajs, long_fut_in_range = ego_positions_in_frame(
        data, frame_id, long_fut_ids)
    long_fut_trajs = np.diff(long_fut_trajs[:, :2], axis=0)
    long_fut_masks = long_fut_in_range[1:]
    long_fut_valid_flag = bool(long_fut_in_range.all())

    index = frame_id + FRAME_OFFSET
    raw_command = data['commands'][index]
    command = np.zeros(len(COMMAND_VOCAB))
    # STOP overrides raw_command whenever the full 3.0s future window is
    # available and the ego barely moves in it -- raw_command reflects
    # intended maneuver, not motion state, so a car stopped at a red
    # light mid-LANE_KEEP (or waiting to unprotected-left-turn mid-
    # TURN_LEFT) still gets that non-STOP label from the source data. See
    # STOP_DISP_THRESH's comment above for calibration against real data.
    fut_total_disp = (
        np.linalg.norm(fut_trajs.sum(axis=0)) if fut_valid_flag else np.inf)
    if fut_total_disp < STOP_DISP_THRESH:
        command[COMMAND_VOCAB.index('STOP')] = 1
    elif raw_command in COMMAND_VOCAB:
        command[COMMAND_VOCAB.index(raw_command)] = 1
    else:
        command[COMMAND_VOCAB.index('LANE_KEEP')] = 1
    rotation = euler_to_matrix(data['ego_pose_rpy'][index])
    velocity, accel, rotation_rate = local_motion(data, index, rotation)

    can_bus = np.zeros(18)
    can_bus[:3] = data['ego_pose_xyz'][index]
    can_bus[3:7] = matrix_to_quat_wxyz(rotation)
    can_bus[7:10] = accel
    can_bus[10:13] = rotation_rate
    can_bus[13:16] = velocity

    target_pos, target_in_range = ego_positions_in_frame(
        data, frame_id, [frame_id + TARGET_POINT_FRAMES])
    target_point = target_pos[0, :2] if target_in_range[0] else np.zeros(2)

    ego_lcf_feat = np.array([
        velocity[0], velocity[1], accel[0], accel[1], rotation_rate[2],
        ego_length, ego_width, np.linalg.norm(velocity[:2]), 0.0
    ])
    return dict(
        gt_ego_his_trajs=his_trajs.astype(np.float32),
        gt_ego_fut_trajs=fut_trajs.astype(np.float32),
        gt_ego_fut_masks=fut_masks.astype(np.float32),
        gt_ego_fut_cmd=command.astype(np.float32),
        gt_ego_lcf_feat=ego_lcf_feat.astype(np.float32),
        # Train-pkl-only privileged signal for PRISM-style latent
        # supervision (see LONG_FUT_TS above) -- NOT a model input at any
        # time, unlike gt_ego_target_point below. long_fut_valid_flag is
        # stored (not used to drop the sample) so the training loop can
        # mask out the auxiliary KL loss per-sample near scene ends,
        # exactly how fut_valid_flag/ego_fut_masks already work.
        gt_ego_long_fut_trajs=long_fut_trajs.astype(np.float32),
        gt_ego_long_fut_masks=long_fut_masks.astype(np.float32),
        gt_ego_long_fut_valid_flag=long_fut_valid_flag,
        # target_point is a genuine model input, used only for selecting
        # among already-generated candidates. Not fed into generation (VAD_head.py no
        # longer has a residual/attention path for it at all).
        gt_ego_target_point=target_point.astype(np.float32),
    ), can_bus, fut_valid_flag


# Maps the 10 raw HD-map line classes onto the 3 classes VAD's map head was
# designed for (divider / ped_crossing / boundary). centerline, stop_line
# and white_virtual have no equivalent in that taxonomy and are dropped.
CLASS2VAD = {
    'white_dashed': 0,
    'white_solid': 0,
    'yellow_dashed': 0,
    'yellow_solid': 0,
    'yellow_double': 0,
    'crosswalk': 1,
    'boundary': 2,
    'centerline': -1,
    'stop_line': -1,
    'white_virtual': -1,
}


def load_lanes(scenario_dir):
    """Load map lines with their VAD map class.

    Returns a list of (class_id, points) pairs, class_id in {0, 1, 2}
    (divider / ped_crossing / boundary). `points` is Nx3 float64 (global
    xyz, z padded with 0 -- hd_map.parquet only has xy and the map
    pipeline only ever consumes the first `map_pts_dim`=2 columns).
    """
    hd_map_path = osp.join(scenario_dir, 'annotation', 'hd_map.parquet')
    hd_map = pd.read_parquet(hd_map_path)
    lanes = []
    for cls, points in zip(hd_map['class'], hd_map['points']):
        class_id = CLASS2VAD.get(cls, -1)
        if class_id == -1:
            continue
        pts_xy = np.asarray([np.asarray(pt, dtype=np.float64)
                             for pt in points])
        if len(pts_xy) < 2:
            continue
        pts_xyz = np.zeros((len(pts_xy), 3), dtype=np.float64)
        pts_xyz[:, :2] = pts_xy[:, :2]
        lanes.append((class_id, pts_xyz))
    return lanes


def convert_scenario(scenario_dir, ego_length, ego_width):
    scenario = osp.basename(osp.normpath(scenario_dir))
    data = load_scenario(scenario_dir)
    cam_infos = load_camera_infos(scenario_dir)
    lanes = load_lanes(scenario_dir)

    infos = []
    for frame_id in range(MAIN_FRAMES):
        index = frame_id + FRAME_OFFSET
        timestamp = data['timestamps'][index]
        ego2global_rotation = matrix_to_quat_wxyz(
            euler_to_matrix(data['ego_pose_rpy'][index]))
        ego2global_translation = data['ego_pose_xyz'][index].tolist()
        # HD-map localization frame (yaw-only; z/roll/pitch borrowed from
        # the regular pose since hd_ego_pose doesn't have them and lane
        # geometry is only ever used in 2D).
        hd_ego2global_rotation = matrix_to_quat_wxyz(
            euler_to_matrix(
                np.array([0.0, 0.0, data['hd_ego_pose_yaw'][index]])))
        hd_ego2global_translation = [
            data['hd_ego_pose_xy'][index][0],
            data['hd_ego_pose_xy'][index][1],
            data['ego_pose_xyz'][index][2],
        ]

        cams = {}
        for cam_name in CAM_NAMES:
            cam_info = dict(cam_infos[cam_name])
            cam_info.update(
                data_path=osp.join(scenario_dir, cam_name,
                                   f'{frame_id:08d}.jpg'),
                sample_data_token=f'{scenario}_{frame_id:08d}_{cam_name}',
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
                timestamp=timestamp,
            )
            cams[cam_name] = cam_info

        ego_anns, can_bus, fut_valid_flag = ego_annotations(
            data, frame_id, ego_length, ego_width)

        info = dict(
            lidar_path='',
            token=f'{scenario}_{frame_id:08d}',
            prev=f'{scenario}_{frame_id - 1:08d}' if frame_id > 0 else '',
            next=f'{scenario}_{frame_id + 1:08d}'
            if frame_id < MAIN_FRAMES - 1 else '',
            can_bus=can_bus,
            frame_idx=frame_id,
            sweeps=[],
            cams=cams,
            scene_token=scenario,
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            hd_ego2global_translation=hd_ego2global_translation,
            hd_ego2global_rotation=hd_ego2global_rotation,
            timestamp=timestamp,
            fut_valid_flag=fut_valid_flag,
            map_location=scenario,
        )
        info.update(agent_annotations(data, frame_id))
        info.update(ego_anns)
        infos.append(info)
    return infos, lanes


def create_etri_infos(root_path, out_dir, info_prefix, ego_length, ego_width):
    scenarios = sorted(
        name for name in os.listdir(root_path)
        if osp.isfile(
            osp.join(root_path, name, 'annotation', 'object.parquet')))
    if not scenarios:
        raise FileNotFoundError(f'no scenario found under {root_path}')
    print(f'{len(scenarios)} scenarios found')

    infos = []
    map_lanes = {}
    for scenario in mmcv.track_iter_progress(scenarios):
        scenario_infos, lanes = convert_scenario(
            osp.join(root_path, scenario), ego_length, ego_width)
        infos.extend(scenario_infos)
        map_lanes[scenario] = lanes

    print(f'train sample: {len(infos)}')
    mmcv.mkdir_or_exist(out_dir)
    info_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_train.pkl')
    mmcv.dump(
        dict(infos=infos,
             metadata=dict(version='etri-v1.0', map_lanes=map_lanes)),
        info_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='ETRI E2E Driving Dataset converter')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/etri',
        help='root path of the scenario directories')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/etri',
        help='output directory of the info pkl')
    parser.add_argument('--extra-tag', type=str, default='vad_etri')
    parser.add_argument('--ego-length', type=float, default=4.635)
    parser.add_argument('--ego-width', type=float, default=1.890)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_etri_infos(args.root_path, args.out_dir, args.extra_tag,
                      args.ego_length, args.ego_width)

