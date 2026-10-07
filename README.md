<div align="center">

# Cell Planner

**Command-Conditioned Cell Planning for End-to-End Driving without Privileged Ego State**

[![Weights](https://img.shields.io/badge/🤗%20HuggingFace-Weights-yellow)](https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

ETRI 2026 Autonomous Driving Challenge · E2E Driving track

</div>

## Table of Contents

- [Introduction](#introduction)
- [Results](#results)
- [Model Zoo](#model-zoo)
- [Getting Started](#getting-started)
- [Repository](#repository)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Introduction

A planner head is swept over a fixed set of sinusoidal **cell codes** — one partition of
the 5-second goal space per driving command — producing one trajectory per cell. The
target point never enters generation; it selects among candidates afterwards. The ego
vehicle's own state, which competition rules forbid the planner from reading, is estimated
from a BEV motion descriptor instead, and a teacher that *is* permitted to read it transfers
its features to the compliant student through a split cosine objective.

```
                      ┌─ cell PE (fixed, 256) ─┐
  images ─► BEV ─► ego_feats (520) ─►  concat  ─► per-command head ─► 99 candidates
                         ▲                                                  │
            vision-estimated ego status (8)                    target point ─┘ selects
```

Three things define it:

1. **One head per command, not one per cell.** Sixty LANE_KEEP candidates are a single
   smooth function sampled at sixty positional codes, not sixty independent predictors.
2. **`ego_lcf_feat_idx=None` on every submittable config.** The 8 status columns are read
   out of vision and supervised by decoding them back to physical ego state.
3. **Cell edges follow data density, not distance.** Cells holding fewer than 20 training
   scenes score 0.79–0.97 val L2 against 0.20 for cells holding 30 or more, while the
   oracle stays flat across that range.

## Results

Held-out 75 of 376 scenes, three-frame window. These are the ablations that shaped the
design; the released checkpoints train on all 376 scenes.

| Method | val L2 (3s avg) |
|---|---|
| No distillation, train/inference mismatches present | 0.5018 |
| ├─ mismatches removed | 0.3772 |
| Per-command anchors + target-point selection | 0.3115 |
| **Cell planner** | **0.2668** |
| Teacher (reads real ego state, not submittable) | 0.2182 |

Frame count against inference cost — fp16 with `--bev-only-history`, RTX 3090:

| Frames | T_infer | Penalty |
|---|---|---|
| 2 | 89.6 ms | ×1.00 |
| **3** | **117.3 ms** | **×1.087** |
| 4 | 145.0 ms | ×1.225 |

Full record of measurements, abandoned directions, and the silent-failure bugs:
[`docs/experiments.md`](docs/experiments.md).

## Model Zoo

All models train on 376 scenes at 10 Hz, 12 epochs, two 24 GB GPUs.
50.17 M parameters, 479.9 GFLOPs per frame.

| Model | Config | Distillation | Planning loss | Weights |
|---|---|---|---|---|
| No-KD | [config](configs/nokd.py) | — | per-step delta | [ckpt](https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge/blob/main/stage2_fulldata376_10hz_nokd_epoch12.pth) |
| No-KD (extended) | [config](configs/nokd_continue20.py) | — | per-step delta | [ckpt](https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge/blob/main/stage2_fulldata376_10hz_nokd_epoch17.pth) |
| KD (delta) | [config](configs/kd_delta.py) | scene 0.3 + status 0.5 | per-step delta | [ckpt](https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge/blob/main/stage2_fulldata376_10hz_kd_deltaloss_epoch12.pth) |
| KD (cumulative) | [config](configs/kd_cumloss.py) | scene 0.3 + status 0.5 | cumulative position | [ckpt](https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge/blob/main/stage2_fulldata376_10hz_kd_cumloss_epoch12.pth) |
| Teacher ⚠️ | [config](configs/teacher_NOSUBMIT.py) | is the teacher | per-step delta | [ckpt](https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge/blob/main/stage2_fulldata376_10hz_teacher8_NOSUBMIT_epoch12.pth) |
| Stage 1 donor | — | — | — | [ckpt](https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge/blob/main/stage1_fulldata376_10hz_nolcf_epoch48.pth) |

> ⚠️ The teacher reads the real ego state. It is **not submittable** and exists only to be
> distilled from.

Submitted trajectory files:
[`submissions/`](https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge/tree/main/submissions).

Saved weights are EMA weights — `EMAHook` swaps the average into the model before writing
and parks the raw weights in `ema_*` buffers, so `unexpected key: ema_*` on load is
expected, not a mismatch.

## Getting Started

- [Installation](docs/install.md) — environment, Docker notes, build check
- [Data preparation](docs/data_preparation.md) — dataset layout and the geometry cache
- [Training and evaluation](docs/train_eval.md) — training, evaluation, submission

```bash
# train
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py \
    configs/nokd.py --launcher pytorch --work-dir work_dirs/nokd

# evaluate
python tools/eval_holdout_l2_and_tinfer.py \
    configs/eval/nokd.py work_dirs/nokd/epoch_12.pth \
    --ann-file <val pkl> --fp16 --bev-only-history --select-cell-by-tp \
    --stride 50 --warmup-windows 20

# submission
bash tools/make_submission.sh epoch_12.pth configs/eval/nokd.py nokd
```

Run the audits before any long training — this is where several silent failures were
caught:

```bash
python tools/audit_pipeline.py configs/nokd.py --eval-config configs/eval/nokd.py
python tools/check_accel_block_live.py configs/nokd.py
```

## Repository

```
configs/                one flat file per model; the research repo had a 19-deep chain
  eval/                 evaluation counterpart of each
projects/mmdet3d_plugin/
tools/                  train, evaluate, submit, audit
  analysis/             the measurements in docs/experiments.md
  surgery/              stage-1 → stage-2 checkpoint transfer
  cache/                geometry cache builder and loader
docs/
```

Configs here are resolved output, written flat — re-reading one gives a config identical
to what the inheritance chain produced, and each is readable on its own.

## Limitations

- The released models train on all 376 scenes, so any held-out split sits inside their
  training set. Design choices were compared on a 301/75 split; this lineage is judged
  only by the challenge score.
- T_infer was never measured on the scoring GPU. The score is
  `L2 × (1 + max(0, T−100)/200)` with T on a 4090; ours come from a 3090 and an A5000.
- Whether distillation improves the final score is not established — no-KD against KD on
  full data can only be separated by the leaderboard.
- Several settings are inherited rather than tuned, and are listed with their status in
  [`docs/experiments.md`](docs/experiments.md).

## Citation

```bibtex
@misc{cellplanner2026,
  title  = {Command-Conditioned Cell Planning for End-to-End Driving without Privileged Ego State},
  year   = {2026},
  note   = {ETRI 2026 Autonomous Driving Challenge, E2E Driving track}
}
```

## Acknowledgement

Built on [VAD](https://github.com/hustvl/VAD) and
[LAW](https://github.com/BraveGroup/LAW) — the perception stack, BEV encoder and much of
the training scaffolding are theirs. Released under Apache 2.0, as they are; see
[NOTICE](NOTICE) for attribution.
