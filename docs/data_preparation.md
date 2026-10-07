# Data preparation

The ETRI 2026 E2E Driving dataset, plus a precomputed geometry cache that training reads
instead of the raw images.

## Layout

```
data/
├── train/                          376 scenes
│   └── <scene_token>/
│       ├── camera_front/           frame_-30.jpg ... frame_0.jpg
│       ├── camera_front_left/
│       ├── camera_front_right/
│       ├── camera_rear_left/
│       ├── camera_rear_right/
│       ├── camera_rear_wide/
│       ├── calibration.parquet
│       ├── command.parquet
│       └── ego_pose.parquet
├── test/                           1125 scenes, same structure
└── etri/
    └── annotations_10hz/
        ├── vad_etri_10hz_infos_temporal_train.pkl
        ├── vad_etri_10hz_infos_temporal_test.pkl
        └── vad_etri_10hz_infos_temporal_val_split.pkl    (hold-out, optional)
```

Frames are named by offset in 0.1 s units: `frame_0.jpg` is the scored frame,
`frame_-30.jpg` is 3 seconds earlier. The models here use a three-frame window —
`-10, -5, 0`, half a second apart.

Six cameras. Five of them are cropped from the top (`crop_keep_top`): the front-left,
front-right, rear-left, rear-right and rear-wide. `camera_front` is not.

## Geometry cache

Training does **not** read JPEGs. Its pipeline begins and ends with
`LoadETRIGeometryCache`; there is no image loader in it at all. The cache holds the
deterministic half of preprocessing:

```
JPEG decode → float32 undistort → crop → resize → rounded uint8
```

Photometric distortion, normalisation, annotation loading and padding stay online, so
augmentation is unaffected. One `.npy` shard per scene, mmap-friendly, which avoids both
HDD seeks and repeated full-resolution remapping every epoch.

### Build it

```bash
python tools/cache/etri_geometry_cache.py \
    --ann-file data/etri/annotations_10hz/vad_etri_10hz_infos_temporal_train.pkl \
    --cache-root work_dirs/etri_geometry_cache_fulldata_376_10hz \
    --repo-root $(pwd) \
    --frame-stride 5 --crop-size 1920 1080 --scale 0.4
```

For 376 scenes at `scale=0.4` this produces **about 126 GB** — 376 shards plus
`cache_manifest.json`, 753 files.

The `cache_root` in the configs is an absolute container path
(`/workspace/VAD/work_dirs/etri_geometry_cache_fulldata_376_10hz`). Change it to wherever
you built the cache, in both `cached_train_pipeline` and `cached_history_pipeline`.

### Verify it

```bash
python tools/cache/verify_etri_geometry_cache.py \
    --ann-file data/etri/annotations_10hz/vad_etri_10hz_infos_temporal_train.pkl \
    --cache-root work_dirs/etri_geometry_cache_fulldata_376_10hz \
    --samples 3 --seeds 0 1 2
```

This replays sampled frames through the original path and compares them against the cache,
bounding both the geometry error and the augmented-image error.

### The loader checks itself

Every cached pipeline entry carries assertions, and they are worth leaving on:

```python
dict(type='LoadETRIGeometryCache',
     cache_root='.../etri_geometry_cache_fulldata_376_10hz',
     scale=0.4, strict=True,
     require_complete_manifest=True,
     expected_scene_count=376,
     expected_ann_file='.../vad_etri_10hz_infos_temporal_train.pkl',
     expected_frame_stride=5,
     expected_crop_size=(1920, 1080),
     expected_crop_keep_top=(...))
```

`expected_scene_count` and `expected_ann_file` exist because a cache directory named for
one dataset once held another. The names matched; the contents did not. A mismatched cache
produces a perfectly valid-looking run on the wrong data, so the loader refuses to start
instead of trusting the path.

Confirm what a cache actually holds before a long run:

```bash
python -c "
import json
d = json.load(open('work_dirs/etri_geometry_cache_fulldata_376_10hz/cache_manifest.json'))
print(d['completed_scene_count'], d['ann_file'], d['frame_stride'], d['crop_size'])
"
```

## Inference reads raw images

The evaluation and submission pipelines do **not** use the cache. They start with
`LoadMultiViewImageFromFiles → UndistortMultiViewImage → CropMultiViewImage`, so that the
timed forward pass costs what real inference costs. Accuracy is unaffected — the same
geometry either way — but it means a machine that only holds the cache can train and
cannot evaluate.

## Next

[Training and evaluation](train_eval.md)
