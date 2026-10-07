"""Build and load deterministic low-resolution ETRI camera geometry caches.

The cache contains only deterministic image geometry:

    JPEG decode -> float32 undistort -> crop -> resize -> rounded uint8

Photometric distortion, normalization, annotation loading, and padding remain
online.  Each scene is one mmap-friendly ``.npy`` shard so training avoids
both HDD JPEG seeks and repeated full-resolution remapping.
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES


CACHE_VERSION = 1
DEFAULT_ANN_FILE = (
    '/workspace/VAD/data/etri/.causal_regen/'
    'vad_etri_infos_temporal_train_split.pkl')
DEFAULT_CACHE_ROOT = '/tmp/vad_etri_geometry_cache_v1'
DEFAULT_REPO_ROOT = '/workspace/VAD'
DEFAULT_CROP_SIZE = (1920, 1080)
DEFAULT_SCALE = 0.4
DEFAULT_CROP_KEEP_TOP = (
    'camera_front_left',
    'camera_front_right',
    'camera_rear_left',
    'camera_rear_right',
    'camera_rear_wide',
)

_WORKER_MAP_CACHE = {}


def scene_stem(scene_token):
    """Return a stable filesystem-safe identifier for a scene token."""
    return hashlib.sha1(str(scene_token).encode('utf-8')).hexdigest()


def scene_paths(cache_root, scene_token):
    stem = scene_stem(scene_token)
    return (
        os.path.join(cache_root, stem + '.npy'),
        os.path.join(cache_root, stem + '.json'),
    )


def _atomic_json_dump(payload, path):
    temp_path = path + '.tmp'
    with open(temp_path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def _resolve_image_path(path, repo_root):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(repo_root, path))


def _crop_box(camera_name, camera, crop_size, crop_keep_top):
    crop_w, crop_h = crop_size
    image_w = int(camera['image_width'])
    image_h = int(camera['image_height'])
    offset_x = (image_w - crop_w) // 2
    offset_y = 0 if camera_name in crop_keep_top else image_h - crop_h
    if offset_x < 0 or offset_y < 0:
        raise ValueError(
            f'Crop {crop_size} does not fit {camera_name} image '
            f'{(image_w, image_h)}')
    return offset_x, offset_y, crop_w, crop_h


def _geometry_fingerprint(camera_name, camera, crop_size, crop_keep_top):
    """Hash every calibration/config field that changes cached pixels."""
    digest = hashlib.sha256()
    digest.update(camera_name.encode('utf-8'))
    for key in ('cam_intrinsic_raw', 'distortion', 'cam_intrinsic'):
        value = np.ascontiguousarray(camera[key], dtype=np.float64)
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    digest.update(str(bool(camera['is_fisheye'])).encode('ascii'))
    digest.update(str(int(camera['image_width'])).encode('ascii'))
    digest.update(str(int(camera['image_height'])).encode('ascii'))
    digest.update(str(tuple(crop_size)).encode('ascii'))
    digest.update(str(tuple(sorted(crop_keep_top))).encode('utf-8'))
    return digest.hexdigest()


def _undistort_maps(camera, image_size):
    raw_k = np.asarray(camera['cam_intrinsic_raw'], dtype=np.float64)
    distortion = np.asarray(camera['distortion'], dtype=np.float64)
    new_k = np.asarray(camera['cam_intrinsic'], dtype=np.float64)
    is_fisheye = bool(camera['is_fisheye'])
    key = (
        raw_k.tobytes(), distortion.tobytes(), new_k.tobytes(),
        tuple(image_size), is_fisheye)
    maps = _WORKER_MAP_CACHE.get(key)
    if maps is not None:
        return maps
    if is_fisheye:
        maps = cv2.fisheye.initUndistortRectifyMap(
            raw_k, distortion[:4], np.eye(3), new_k, image_size,
            cv2.CV_32FC1)
    else:
        maps = cv2.initUndistortRectifyMap(
            raw_k, distortion, None, new_k, image_size, cv2.CV_32FC1)
    _WORKER_MAP_CACHE[key] = maps
    return maps


def _cache_one_image(camera_name, camera, repo_root, crop_size, scale,
                     crop_keep_top):
    source_path = _resolve_image_path(camera['data_path'], repo_root)
    image = mmcv.imread(source_path, 'unchanged')
    if image is None:
        raise FileNotFoundError(source_path)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected HxWx3 image at {source_path}: {image.shape}')
    image = image.astype(np.float32)
    height, width = image.shape[:2]
    map1, map2 = _undistort_maps(camera, (width, height))
    image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    offset_x, offset_y, crop_w, crop_h = _crop_box(
        camera_name, camera, crop_size, crop_keep_top)
    image = image[offset_y:offset_y + crop_h,
                  offset_x:offset_x + crop_w]
    output_size = (int(crop_w * scale), int(crop_h * scale))
    image = mmcv.imresize(image, output_size, return_scale=False)
    # The cache is intentionally uint8.  Rounding gives a <=0.5 raw-pixel
    # quantization error before online augmentation/normalization.
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _validate_existing(shard_path, meta_path, task):
    if not os.path.isfile(shard_path) or not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path) as handle:
            meta = json.load(handle)
        if (meta.get('cache_version') != CACHE_VERSION
                or meta.get('scene_token') != task['scene_token']
                or meta.get('frame_indices') != task['frame_indices']
                or meta.get('camera_names') != task['camera_names']
                or meta.get('source_paths') != task['source_paths']
                or meta.get('geometry_fingerprints')
                != task['geometry_fingerprints']
                or meta.get('crop_size') != list(task['crop_size'])
                or meta.get('crop_keep_top') != list(task['crop_keep_top'])
                or meta.get('scale') != task['scale']
                or not meta.get('complete', False)):
            return False
        shard = np.load(shard_path, mmap_mode='r')
        expected = tuple(meta['shape'])
        valid = shard.dtype == np.uint8 and shard.shape == expected
        mmap = getattr(shard, '_mmap', None)
        if mmap is not None:
            mmap.close()
        return bool(valid)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _build_scene(task):
    cv2.setNumThreads(1)
    shard_path, meta_path = scene_paths(
        task['cache_root'], task['scene_token'])
    if not task['overwrite'] and _validate_existing(
            shard_path, meta_path, task):
        return dict(
            scene_token=task['scene_token'], status='skipped',
            frames=len(task['frames']), bytes=os.path.getsize(shard_path))

    frame_count = len(task['frames'])
    camera_count = len(task['camera_names'])
    crop_w, crop_h = task['crop_size']
    output_w = int(crop_w * task['scale'])
    output_h = int(crop_h * task['scale'])
    shape = (frame_count, camera_count, output_h, output_w, 3)
    temp_shard = shard_path + '.tmp'
    shard = np.lib.format.open_memmap(
        temp_shard, mode='w+', dtype=np.uint8, shape=shape)
    try:
        # Camera-major source traversal keeps successive JPEG reads in the
        # same on-disk camera directory.  The output remains frame-major so a
        # training queue reads all six views from one contiguous row.
        for camera_position, camera_name in enumerate(task['camera_names']):
            for frame_position, frame in enumerate(task['frames']):
                shard[frame_position, camera_position] = _cache_one_image(
                    camera_name,
                    frame['cameras'][camera_name],
                    task['repo_root'],
                    tuple(task['crop_size']),
                    task['scale'],
                    set(task['crop_keep_top']))
        shard.flush()
    finally:
        mmap = getattr(shard, '_mmap', None)
        if mmap is not None:
            mmap.close()
        del shard
    os.replace(temp_shard, shard_path)

    meta = dict(
        cache_version=CACHE_VERSION,
        complete=True,
        scene_token=task['scene_token'],
        frame_indices=task['frame_indices'],
        camera_names=task['camera_names'],
        source_paths=task['source_paths'],
        geometry_fingerprints=task['geometry_fingerprints'],
        shape=list(shape),
        dtype='uint8',
        crop_size=list(task['crop_size']),
        crop_keep_top=list(task['crop_keep_top']),
        scale=task['scale'])
    _atomic_json_dump(meta, meta_path)
    return dict(
        scene_token=task['scene_token'], status='built',
        frames=frame_count, bytes=os.path.getsize(shard_path))


def _compact_scene_tasks(infos, cache_root, repo_root, frame_stride,
                         crop_size, scale, crop_keep_top, overwrite):
    scenes = OrderedDict()
    for info in infos:
        frame_index = int(info['frame_idx'])
        if frame_index % frame_stride != 0:
            continue
        scene_token = str(info['scene_token'])
        camera_names = list(info['cams'].keys())
        cameras = OrderedDict()
        source_paths = []
        geometry_fingerprints = []
        for camera_name in camera_names:
            source = info['cams'][camera_name]
            cameras[camera_name] = dict(
                data_path=source['data_path'],
                cam_intrinsic_raw=source['cam_intrinsic_raw'],
                distortion=source['distortion'],
                cam_intrinsic=source['cam_intrinsic'],
                is_fisheye=source['is_fisheye'],
                image_width=source['image_width'],
                image_height=source['image_height'])
            source_paths.append(os.path.normpath(source['data_path']))
            geometry_fingerprints.append(_geometry_fingerprint(
                camera_name, source, crop_size, crop_keep_top))
        scenes.setdefault(scene_token, []).append(dict(
            frame_idx=frame_index,
            camera_names=camera_names,
            cameras=cameras,
            source_paths=source_paths,
            geometry_fingerprints=geometry_fingerprints))

    tasks = []
    for scene_token, frames in scenes.items():
        frames.sort(key=lambda item: item['frame_idx'])
        camera_names = frames[0]['camera_names']
        if any(frame['camera_names'] != camera_names for frame in frames):
            raise ValueError(f'Camera order changed within scene {scene_token}')
        frame_indices = [frame['frame_idx'] for frame in frames]
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError(f'Duplicate frame_idx in scene {scene_token}')
        tasks.append(dict(
            cache_root=cache_root,
            repo_root=repo_root,
            scene_token=scene_token,
            frames=frames,
            frame_indices=frame_indices,
            camera_names=camera_names,
            source_paths=[frame['source_paths'] for frame in frames],
            geometry_fingerprints=[
                frame['geometry_fingerprints'] for frame in frames
            ],
            crop_size=tuple(crop_size),
            scale=float(scale),
            crop_keep_top=tuple(crop_keep_top),
            overwrite=bool(overwrite)))
    return tasks


class _ShardLRU:

    def __init__(self, max_open):
        self.max_open = int(max_open)
        self.items = OrderedDict()

    def get(self, path):
        shard = self.items.pop(path, None)
        if shard is None:
            shard = np.load(path, mmap_mode='r')
        self.items[path] = shard
        while len(self.items) > self.max_open:
            _, old = self.items.popitem(last=False)
            mmap = getattr(old, '_mmap', None)
            if mmap is not None:
                mmap.close()
        return shard


@PIPELINES.register_module()
class LoadETRIGeometryCache:
    """Load a cached scene/frame and reproduce crop+scale metadata updates."""

    def __init__(self, cache_root=DEFAULT_CACHE_ROOT, scale=DEFAULT_SCALE,
                 strict=True, max_open_shards=8,
                 require_complete_manifest=False,
                 expected_scene_count=None, expected_ann_file=None,
                 expected_frame_stride=None, expected_crop_size=None,
                 expected_crop_keep_top=None):
        self.cache_root = cache_root
        self.scale = float(scale)
        self.strict = bool(strict)
        self.expected_crop_size = (
            tuple(expected_crop_size)
            if expected_crop_size is not None else None)
        self.expected_crop_keep_top = (
            tuple(expected_crop_keep_top)
            if expected_crop_keep_top is not None else None)
        self.shards = _ShardLRU(max_open_shards)
        self.metadata = {}
        if require_complete_manifest:
            self._validate_complete_manifest(
                expected_scene_count=expected_scene_count,
                expected_ann_file=expected_ann_file,
                expected_frame_stride=expected_frame_stride,
                expected_crop_size=expected_crop_size,
                expected_crop_keep_top=expected_crop_keep_top)

    def _validate_complete_manifest(
            self, expected_scene_count, expected_ann_file,
            expected_frame_stride, expected_crop_size,
            expected_crop_keep_top):
        """Fail before training if the detached cache build is incomplete."""
        manifest_path = os.path.join(
            self.cache_root, 'cache_manifest.json')
        try:
            with open(manifest_path) as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f'Complete cache manifest is required: {manifest_path}') \
                from error

        requested = manifest.get('requested_scene_count')
        completed = manifest.get('completed_scene_count')
        scenes = manifest.get('scenes', [])
        scene_tokens = [item.get('scene_token') for item in scenes]
        if (manifest.get('cache_version') != CACHE_VERSION
                or requested != completed
                or completed != len(scenes)
                or len(set(scene_tokens)) != len(scene_tokens)):
            raise RuntimeError(
                f'Incomplete cache manifest at {manifest_path}: '
                f'requested={requested}, completed={completed}, '
                f'listed={len(scenes)}')
        if (expected_scene_count is not None
                and completed != int(expected_scene_count)):
            raise RuntimeError(
                f'Unexpected cache coverage at {manifest_path}: '
                f'expected {int(expected_scene_count)} scenes, got '
                f'{completed}')
        expected_fields = {
            'frame_stride': expected_frame_stride,
            'crop_size': (
                list(expected_crop_size)
                if expected_crop_size is not None else None),
            'crop_keep_top': (
                list(expected_crop_keep_top)
                if expected_crop_keep_top is not None else None),
            'scale': self.scale,
        }
        for key, expected in expected_fields.items():
            if expected is not None and manifest.get(key) != expected:
                raise RuntimeError(
                    f'Cache manifest {key} mismatch at {manifest_path}: '
                    f'expected {expected!r}, got {manifest.get(key)!r}')
        if expected_ann_file is not None:
            actual_ann_file = manifest.get('ann_file')
            if (actual_ann_file is None
                    or os.path.realpath(actual_ann_file)
                    != os.path.realpath(expected_ann_file)):
                raise RuntimeError(
                    f'Cache annotation mismatch at {manifest_path}: '
                    f'expected {expected_ann_file!r}, got '
                    f'{actual_ann_file!r}')

    def _get_meta(self, scene_token):
        meta = self.metadata.get(scene_token)
        if meta is not None:
            return meta
        shard_path, meta_path = scene_paths(self.cache_root, scene_token)
        with open(meta_path) as handle:
            meta = json.load(handle)
        if (meta.get('cache_version') != CACHE_VERSION
                or not meta.get('complete', False)
                or meta.get('scene_token') != scene_token
                or meta.get('dtype') != 'uint8'
                or meta.get('scale') != self.scale):
            raise ValueError(f'Invalid cache metadata: {meta_path}')
        shape = meta.get('shape', [])
        crop_size = meta.get('crop_size', [])
        frame_indices = meta.get('frame_indices', [])
        camera_names = meta.get('camera_names', [])
        source_paths = meta.get('source_paths', [])
        expected_height = (
            int(crop_size[1] * self.scale) if len(crop_size) == 2 else None)
        expected_width = (
            int(crop_size[0] * self.scale) if len(crop_size) == 2 else None)
        if (len(shape) != 5
                or shape[0] != len(frame_indices)
                or shape[1] != len(camera_names)
                or shape[2:] != [expected_height, expected_width, 3]
                or len(set(frame_indices)) != len(frame_indices)
                or len(set(camera_names)) != len(camera_names)
                or len(source_paths) != shape[0]
                or any(len(row) != shape[1] for row in source_paths)
                or (self.expected_crop_size is not None
                    and tuple(crop_size) != self.expected_crop_size)
                or (self.expected_crop_keep_top is not None
                    and tuple(meta.get('crop_keep_top', []))
                    != self.expected_crop_keep_top)):
            raise ValueError(f'Invalid cache shape metadata: {meta_path}')
        try:
            shard = np.load(shard_path, mmap_mode='r')
            actual_shape = shard.shape
            actual_dtype = shard.dtype
            mmap = getattr(shard, '_mmap', None)
            if mmap is not None:
                mmap.close()
        except (OSError, ValueError) as error:
            raise ValueError(f'Unreadable cache shard: {shard_path}') from error
        if actual_shape != tuple(shape) or actual_dtype != np.uint8:
            raise ValueError(
                f'Cache shard mismatch at {shard_path}: expected '
                f'{tuple(shape)}/uint8, got {actual_shape}/{actual_dtype}')
        meta['_frame_positions'] = {
            int(frame): position
            for position, frame in enumerate(meta['frame_indices'])
        }
        meta['_shard_path'] = shard_path
        self.metadata[scene_token] = meta
        return meta

    def __call__(self, results):
        scene_token = str(results['scene_token'])
        frame_index = int(results['frame_idx'])
        meta = self._get_meta(scene_token)
        if frame_index not in meta['_frame_positions']:
            raise KeyError(
                f'Frame {frame_index} missing from cached scene {scene_token}')
        position = meta['_frame_positions'][frame_index]
        filenames = [os.path.normpath(path) for path in results['img_filename']]
        expected_sources = meta['source_paths'][position]
        if self.strict and filenames != expected_sources:
            raise ValueError(
                f'Camera source/order mismatch for {scene_token} frame '
                f'{frame_index}: {filenames} != {expected_sources}')
        shard = self.shards.get(meta['_shard_path'])
        frame = shard[position]
        if self.strict and frame.shape[0] != len(filenames):
            raise ValueError(
                f'Cached camera count mismatch for {scene_token} frame '
                f'{frame_index}')

        images = [
            frame[index].astype(np.float32, copy=True)
            for index in range(frame.shape[0])
        ]
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = self.scale
        scale_matrix[1, 1] = self.scale
        for index, crop in enumerate(results['crop_box']):
            offset_x, offset_y, _, _ = crop
            crop_matrix = np.eye(4)
            crop_matrix[0, 2] = -offset_x
            crop_matrix[1, 2] = -offset_y
            if 'lidar2img' in results:
                results['lidar2img'][index] = (
                    scale_matrix @ crop_matrix @ results['lidar2img'][index])
            # Match the current pipeline exactly: Crop updates cam_intrinsic,
            # while RandomScale updates only lidar2img.
            if 'cam_intrinsic' in results:
                results['cam_intrinsic'][index] = (
                    crop_matrix @ results['cam_intrinsic'][index])

        shapes = [image.shape for image in images]
        results['filename'] = results['img_filename']
        results['img'] = images
        results['img_shape'] = shapes
        results['ori_shape'] = shapes
        results['pad_shape'] = shapes
        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
            to_rgb=False)
        results['geometry_cache'] = dict(
            version=CACHE_VERSION,
            scene_token=scene_token,
            frame_idx=frame_index)
        return results


def build_cache(args):
    cache_root = os.path.abspath(args.cache_root)
    repo_root = os.path.abspath(args.repo_root)
    os.makedirs(cache_root, exist_ok=True)
    payload = mmcv.load(args.ann_file)
    infos = payload['infos']
    tasks = _compact_scene_tasks(
        infos, cache_root, repo_root, args.frame_stride,
        tuple(args.crop_size), args.scale, tuple(args.crop_keep_top),
        args.overwrite)
    del payload
    del infos

    if args.scene_token:
        tasks = [
            task for task in tasks
            if task['scene_token'] == args.scene_token
        ]
        if not tasks:
            raise ValueError(f'Unknown scene token: {args.scene_token}')
    if args.max_scenes is not None:
        tasks = tasks[:args.max_scenes]

    expected_bytes = sum(
        len(task['frames']) * len(task['camera_names'])
        * int(task['crop_size'][0] * task['scale'])
        * int(task['crop_size'][1] * task['scale']) * 3
        for task in tasks)
    free_bytes = shutil.disk_usage(cache_root).free
    if free_bytes < int(expected_bytes * 1.10):
        raise OSError(
            f'Insufficient cache space: need about {expected_bytes:,} bytes '
            f'plus margin, have {free_bytes:,}')

    results = []
    if args.workers == 1:
        for task in tasks:
            result = _build_scene(task)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_build_scene, task): task for task in tasks}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)
            except BaseException:
                # Do not start hundreds of additional scene jobs after the
                # first corrupt source/calibration or an operator interrupt.
                for future in futures:
                    future.cancel()
                raise

    manifest = dict(
        cache_version=CACHE_VERSION,
        ann_file=os.path.abspath(args.ann_file),
        frame_stride=args.frame_stride,
        crop_size=list(args.crop_size),
        crop_keep_top=list(args.crop_keep_top),
        scale=args.scale,
        requested_scene_count=len(tasks),
        completed_scene_count=len(results),
        expected_bytes=expected_bytes,
        scenes=sorted(results, key=lambda item: item['scene_token']))
    _atomic_json_dump(
        manifest, os.path.join(cache_root, 'cache_manifest.json'))
    print(json.dumps(dict(
        cache_root=cache_root,
        scenes=len(results),
        bytes=sum(item['bytes'] for item in results)), sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', default=DEFAULT_ANN_FILE)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE_ROOT)
    parser.add_argument('--repo-root', default=DEFAULT_REPO_ROOT)
    parser.add_argument('--frame-stride', type=int, default=5)
    parser.add_argument('--crop-size', type=int, nargs=2,
                        default=DEFAULT_CROP_SIZE, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE)
    parser.add_argument('--crop-keep-top', nargs='+',
                        default=DEFAULT_CROP_KEEP_TOP)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-scenes', type=int)
    parser.add_argument('--scene-token')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    build_cache(parse_args())
