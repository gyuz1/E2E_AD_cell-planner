"""Verify cached ETRI images and projection metadata against the old path."""

import argparse
import copy
import json
import os

import mmcv
import numpy as np
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles
from nuscenes.eval.common.utils import Quaternion

from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import (
    CropMultiViewImage,
    NormalizeMultiviewImage,
    PhotoMetricDistortionMultiViewImage,
    RandomScaleImageMultiViewImage,
    UndistortMultiViewImage,
)
from tools.cache.etri_geometry_cache import (
    DEFAULT_ANN_FILE,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CROP_KEEP_TOP,
    LoadETRIGeometryCache,
)


NORM_CONFIG = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)


def make_results(info, crop_keep_top):
    image_paths = []
    lidar2img = []
    cam_intrinsic = []
    crop_boxes = []
    for camera_name, camera in info['cams'].items():
        image_paths.append(camera['data_path'])
        lidar2cam_r = np.linalg.inv(camera['sensor2lidar_rotation'])
        lidar2cam_t = camera['sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = np.asarray(camera['cam_intrinsic'])
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img.append(viewpad @ lidar2cam_rt.T)
        cam_intrinsic.append(viewpad)

        offset_x = (int(camera['image_width']) - 1920) // 2
        offset_y = (
            0 if camera_name in crop_keep_top
            else int(camera['image_height']) - 1080)
        crop_boxes.append((offset_x, offset_y, 1920, 1080))

    return dict(
        scene_token=str(info['scene_token']),
        frame_idx=int(info['frame_idx']),
        img_filename=image_paths,
        lidar2img=lidar2img,
        cam_intrinsic=cam_intrinsic,
        crop_box=crop_boxes,
        cam_intrinsic_raw=[
            camera['cam_intrinsic_raw'] for camera in info['cams'].values()
        ],
        cam_intrinsic_undist=[
            camera['cam_intrinsic'] for camera in info['cams'].values()
        ],
        distortion=[
            camera['distortion'] for camera in info['cams'].values()
        ],
        is_fisheye=[
            camera['is_fisheye'] for camera in info['cams'].values()
        ],
        # Included to catch accidental differences in future metadata code.
        camera2ego=[
            np.block([
                [Quaternion(camera['sensor2ego_rotation']).rotation_matrix,
                 np.asarray(camera['sensor2ego_translation'])[:, None]],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ])
            for camera in info['cams'].values()
        ])


def clone_results(results):
    cloned = {}
    for key, value in results.items():
        if key == 'img':
            cloned[key] = [image.copy() for image in value]
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def image_error(left, right):
    absolute = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return dict(
        mae=float(absolute.mean()),
        p99=float(np.quantile(absolute, 0.99)),
        max=float(absolute.max()))


def choose_infos(ann_file, cache_root, sample_count):
    with open(os.path.join(cache_root, 'cache_manifest.json')) as handle:
        manifest = json.load(handle)
    cached_scenes = {item['scene_token'] for item in manifest['scenes']}
    payload = mmcv.load(ann_file)
    candidates = [
        info for info in payload['infos']
        if str(info['scene_token']) in cached_scenes
        and int(info['frame_idx']) % manifest['frame_stride'] == 0
    ]
    if not candidates:
        raise RuntimeError('No annotation frames match the cache manifest')
    if sample_count >= len(candidates):
        selected = candidates
    else:
        positions = np.linspace(
            0, len(candidates) - 1, sample_count, dtype=np.int64)
        selected = [candidates[int(position)] for position in positions]
    # Deep-copy the small selection before releasing the ~1 GB annotation.
    selected = copy.deepcopy(selected)
    del candidates
    del payload
    return selected


def verify(args):
    infos = choose_infos(args.ann_file, args.cache_root, args.samples)
    load_raw = LoadMultiViewImageFromFiles(to_float32=True)
    undistort = UndistortMultiViewImage()
    crop = CropMultiViewImage()
    scale = RandomScaleImageMultiViewImage(scales=[0.4])
    cached_loader = LoadETRIGeometryCache(
        cache_root=args.cache_root, scale=0.4, strict=True)
    photo = PhotoMetricDistortionMultiViewImage()
    normalize = NormalizeMultiviewImage(**NORM_CONFIG)

    reports = []
    failed = False
    for info in infos:
        original = make_results(info, set(args.crop_keep_top))
        high = crop(undistort(load_raw(clone_results(original))))
        baseline_geometry = scale(clone_results(high))
        cached = cached_loader(clone_results(original))

        geometry_errors = [
            image_error(old, new)
            for old, new in zip(baseline_geometry['img'], cached['img'])
        ]
        lidar_error = max(
            float(np.max(np.abs(old - new)))
            for old, new in zip(
                baseline_geometry['lidar2img'], cached['lidar2img']))
        intrinsic_error = max(
            float(np.max(np.abs(old - new)))
            for old, new in zip(
                baseline_geometry['cam_intrinsic'], cached['cam_intrinsic']))

        augmented_errors = []
        for seed in args.seeds:
            np.random.seed(seed)
            old = scale(normalize(photo(clone_results(high))))
            np.random.seed(seed)
            new = normalize(photo(clone_results(cached)))
            augmented_errors.extend(
                image_error(left, right)
                for left, right in zip(old['img'], new['img']))

        report = dict(
            scene_token=str(info['scene_token']),
            frame_idx=int(info['frame_idx']),
            geometry_mae=max(item['mae'] for item in geometry_errors),
            geometry_p99=max(item['p99'] for item in geometry_errors),
            geometry_max=max(item['max'] for item in geometry_errors),
            augmented_mae=max(item['mae'] for item in augmented_errors),
            augmented_p99=max(item['p99'] for item in augmented_errors),
            augmented_max=max(item['max'] for item in augmented_errors),
            lidar2img_max_abs_error=lidar_error,
            cam_intrinsic_max_abs_error=intrinsic_error)
        report['passed'] = bool(
            report['geometry_mae'] <= args.max_geometry_mae
            and report['geometry_max'] <= args.max_geometry_error
            and report['augmented_mae'] <= args.max_augmented_mae
            and report['augmented_p99'] <= args.max_augmented_p99
            and lidar_error <= args.max_matrix_error
            and intrinsic_error <= args.max_matrix_error)
        failed = failed or not report['passed']
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    summary = dict(
        passed=not failed,
        samples=len(reports),
        max_geometry_mae=max(item['geometry_mae'] for item in reports),
        max_augmented_mae=max(item['augmented_mae'] for item in reports),
        max_augmented_p99=max(item['augmented_p99'] for item in reports),
        max_lidar2img_error=max(
            item['lidar2img_max_abs_error'] for item in reports))
    print(json.dumps(summary, sort_keys=True))
    if failed:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', default=DEFAULT_ANN_FILE)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE_ROOT)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--crop-keep-top', nargs='+',
                        default=DEFAULT_CROP_KEEP_TOP)
    parser.add_argument('--max-geometry-mae', type=float, default=0.51)
    parser.add_argument('--max-geometry-error', type=float, default=0.51)
    parser.add_argument('--max-augmented-mae', type=float, default=0.02)
    parser.add_argument('--max-augmented-p99', type=float, default=0.10)
    parser.add_argument('--max-matrix-error', type=float, default=1e-9)
    return parser.parse_args()


if __name__ == '__main__':
    verify(parse_args())
