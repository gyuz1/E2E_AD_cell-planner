# Data Preparation

All experiments use the ETRI E2E Driving Dataset (376 scenes at 10 Hz, six cameras),
split by scene into 301 training and 75 held-out scenes. The held-out scenes are listed in
[`data/splits/val_scenes.txt`](../data/splits/val_scenes.txt).

## Layout

```
data/
├── train/                       376 scenes
│   └── <scene>/
│       ├── camera_front/ ... camera_rear_wide/
│       ├── calibration.parquet
│       ├── command.parquet
│       └── ego_pose.parquet
├── test/
├── splits/val_scenes.txt
└── etri/split_301_75/           created below
```

## Annotations

Builds the annotation files and the 301/75 split. Ego velocity and acceleration are derived
from the 10 Hz ego pose.

```bash
python tools/data_converter/regenerate_causal_infos_10hz.py \
    --train-root data/train --test-root data/test \
    --output-dir data/etri/split_301_75 \
    --val-scenes data/splits/val_scenes.txt --prefix vad_etri
```

Output: `vad_etri_infos_temporal_train_split.pkl` (301 scenes) and
`vad_etri_infos_temporal_val_split.pkl` (75 scenes).

## Geometry cache (training only)

Training reads a precomputed image cache instead of the JPEG files
(decode → undistort → crop → resize). Inference reads the raw images.

```bash
python tools/cache/etri_geometry_cache.py \
    --ann-file data/etri/split_301_75/vad_etri_infos_temporal_train_split.pkl \
    --cache-root data/etri_geometry_cache --repo-root . \
    --frame-stride 5 --crop-size 1920 1080 --scale 0.4 --workers 8
```

The training configs check that the cache was built from the same annotation file
(`expected_ann_file`) with 301 scenes and refuse to start otherwise.
