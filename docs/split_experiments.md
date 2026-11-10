# Split lineage (301 / 75)

This branch holds the lineage every design decision was actually made on.

`main` trains on all 376 scenes, so any held-out split sits inside its training set and
an ablation measured there means nothing. This branch splits the scenes 301 / 75 and
keeps the 75 out, so the numbers below are measured on data the model never saw.

## Configs

| Config | Moving cells | Distillation | Role |
|---|---|---|---|
| [`stage1.py`](../configs/split/stage1.py) | — | — | perception / world model, 48 epochs |
| [`teacher_NOSUBMIT.py`](../configs/split/teacher_NOSUBMIT.py) ⚠️ | 105 | is the teacher | reads the real ego state |
| [`cellplanner_base.py`](../configs/split/cellplanner_base.py) | 60 | — | first cell layout |
| [`cellplanner_lk60.py`](../configs/split/cellplanner_lk60.py) | 105 | — | LANE_KEEP widened to 60 |
| [`cellplanner_lk60_kd.py`](../configs/split/cellplanner_lk60_kd.py) | 105 | scene 0.1 + status 0.1 + traj 0.1 | **final** |
| [`cellplanner_lk60_scenefloor.py`](../configs/split/cellplanner_lk60_scenefloor.py) | 99 | — | per-cell scene floor |

> ⚠️ The teacher reads the real ego state. It is **not submittable**; it exists only to be
> distilled from, and it never enters the inference path.

Cell counts are moving cells. `U_TURN` and `STOP` have no cells — they are separated by
target-point distance. The layout lists are edges, so a command contributes
`(len(fwd) - 1) x (len(lat) - 1)` cells: LANE_KEEP 60x1, LANE_CHANGE_L 11x1,
LANE_CHANGE_R 14x1, TURN_LEFT 4x2, TURN_RIGHT 6x2.

## Measured

Held-out 75 scenes, three-frame window unless noted. 350 agent-bearing frames carry the
collision denominator.

| Model | L2 (3s avg) | Collision |
|---|---|---|
| Student, no cell planner, no distillation | 0.3181 | 0.58% |
| Student + cell planner | 0.2573 | 0.63% |
| Student + distillation, no cell planner | 0.4491 | 0.57% |
| **Student + cell planner + distillation** | **0.2577** | **0.41%** |
| ├─ same checkpoint, two-frame window | 0.2603 | 0.43% |
| Teacher (reads real ego state) ⚠️ | 0.2052 | 0.60% |
| Constant-acceleration extrapolation, no images | 0.2725 | 0.91% |

Three things in that table are worth stating plainly.

**Distillation does not move L2; it moves collision.** 0.2573 to 0.2577 is nothing.
0.63% to 0.41% is the whole effect. Reporting this model as a displacement-error
improvement would misdescribe what it does.

**Distillation alone makes it worse.** Without the cell planner, the same distillation
takes 0.3181 to 0.4491. The two are not independent contributions.

**A blind kinematic predictor beats the vision baseline on L2.** Extrapolating the
measured ego state at constant acceleration, with no access to any image, reaches 0.2725
against the vision baseline's 0.3181 — while colliding 1.6x as often. Displacement error
alone does not separate a planner from an extrapolator;
[`tools/analysis/kinematic_oracle_collision.py`](../tools/analysis/kinematic_oracle_collision.py)
is there so that reference is always available.

## Reading the collision metric

Two numbers come out of the STP-3 metric, and only one is usable.

- **`obj_box`** rasterizes the ego footprint, 4.084 x 1.85 m, centred 0.5 m forward. Use this.
- **`obj`** treats the ego as a single pixel, and maps it with `-bx/2` where `obj_box`
  uses `-bx`. That is a constant 50-cell offset — **25 m in both axes** — so the column is
  not a stricter variant of `obj_box`, it is wrong. Do not report it.

The metric also ships nuScenes' 23-class vocabulary. ETRI has three classes, and without
overriding `category_index` to `{'vehicle': [0, 2], 'human': [1]}` the occupancy grid
comes out empty on every frame — no crash, just a collision rate of zero.

Both columns mask timesteps where the ground-truth trajectory itself collides, and frames
with no valid agent are excluded from the denominator rather than counted as collision-free.

## The two-frame window

`aux_bev_motion_frames = 3`. With two frames the second difference of displacement is
identically zero, so the acceleration term carries no information. The damage is not
spread evenly: stationary segments collapse while moving segments are untouched. That is
why `tools/etri_test_submit.py` refuses a two-frame window, and why the three-frame
default is in the configs rather than left to a flag.

## Per-cell scene floor

Cells below roughly 20 training scenes behave differently from cells above it. The oracle
— the best candidate the layout can offer — stays flat across them, while the error
actually achieved swings wide. The candidates are close enough; what is missing is data to
support them. So `cellplanner_lk60_scenefloor.py` merges thin cells instead of splitting
further, taking LANE_CHANGE_L from 11 to 10 and LANE_CHANGE_R from 14 to 9.

The layout was fitted on the 301 training scenes. Refitting on all 376 moves LANE_KEEP by
0.02%, so the split fit is kept — it is the one comparable to this lineage's earlier
measurements.

## Reproducing

```bash
# train
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py \
    configs/split/cellplanner_lk60_kd.py --launcher pytorch \
    --work-dir work_dirs/split_cellplanner_lk60_kd

# evaluate on the held-out 75
python tools/eval_holdout_l2_and_tinfer.py \
    configs/split/eval/cellplanner_lk60_kd.py \
    work_dirs/split_cellplanner_lk60_kd/epoch_10.pth \
    --ann-file data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl \
    --fp16 --bev-only-history --select-cell-by-tp --stride 50 --warmup-windows 20

# the kinematic reference, no GPU
python tools/analysis/kinematic_oracle_collision.py \
    --ann-file data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl \
    --stride 50 --warmup-windows 20
```

The evaluator skips 20 warm-up windows without recording them. Anything scored outside it
has to skip the same 20, or the frame sets do not match — scoring 375/370 frames against
the evaluator's 355/350 silently shifts every number.
