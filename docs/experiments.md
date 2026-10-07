# Experiments

What was varied, and what it measured. Unless stated otherwise, numbers are planning L2
(3 s average, metres) on a held-out 75 of 376 scenes, three-frame window, EMA weights.

- [1. Ego status](#1-ego-status)
- [2. Distillation](#2-distillation)
- [3. How the target point is used](#3-how-the-target-point-is-used)
- [4. Cell layout](#4-cell-layout)
- [5. Frame window](#5-frame-window)
- [6. Train/inference mismatches](#6-traininference-mismatches)
- [7. Training schedule](#7-training-schedule)
- [8. Final models](#8-final-models)

---

## 1. Ego status

The planner may not read the ego vehicle's own state (`ego_lcf`). Everything here is about
what replaces it.

### Can vision see ego motion at all?

Precondition for the whole approach. BEV is 100×100 over `pc_range` x −30…30, so one cell
is **0.600 m**. Mean speed 10.57 m/s × 0.5 s = 5.29 m = **9 cells** between consecutive
frames in the window.

→ Ego motion is resolvable at this BEV resolution. `check_descriptor_sensitivity.py`.

### Does the acceleration term actually train?

A three-frame descriptor whose second difference is identically zero trains, converges, and
means nothing. Checked on real gradients rather than assumed.

| | result |
|---|---|
| acceleration block, epoch 1 | **receives gradient** |
| acceleration block \|mean\| on live data | 1.9533 (non-zero) |

→ `check_accel_block_live.py` / `check_accel_block_trained.py` are run before every long
training.

### Status slot width

| width | outcome |
|---|---|
| 8 | works |
| 64 | **`ValueError` by construction** |

`ego_lcf_embed_residual` adds the raw ego columns onto the teacher's embedding, which
forces the slot to be exactly as wide as the ego status selection. Only at 8 does
dimension *i* mean the same thing on both sides, which is what makes a cosine between
student and teacher well posed.

→ 8 chosen. Not a tuning result; a well-posedness constraint.

### Future speed target

Whether a future-motion head has anything to learn beyond kinematic extrapolation.

| | value |
|---|---|
| mean current speed | 10.57 m/s |
| mean speed 3 s later | 10.58 m/s |
| **per-sample absolute difference** | **0.842 m/s** |
| valid flag coverage | 100% |

→ The means match, so extrapolation looks adequate in aggregate and is not. 0.842 m/s is
what the future head targets.

---

## 2. Distillation

### Trajectory KD vs feature KD

Two different things were tried under the name "KD". Only the second survived.

| | teacher | what transfers | result |
|---|---|---|---|
| Trajectory KD | Qwen / EvoDriveVLA VLA | predicted waypoints | teacher **collapses on hold-out** — largely memorisation. Dropped |
| **Feature KD** | our own `teacher8` | `ego_feats`, two halves | **used in the final models** |

### Does feature distillation transfer at all?

| | L2 |
|---|---|
| student v1, feature distillation | **0.4218** |

Predicted beforehand at 0.44–0.46, so it came out better than expected.

→ Feature distillation transfers. This is what justified building a privileged teacher.

### Fused vs split objective

| mode | target | why |
|---|---|---|
| `fused` | `ego_plan_hidden`, the post-fusion activation | status contribution is smeared across every hidden unit and **cannot be weighted separately** |
| **`split`** | `ego_scene_feats` and `ego_status_feats` | two independent alignments, each weightable |

Final objective:

```
loss_scene_distill  = 0.3 · (1 − cos(student.ego_scene_feats,  teacher.ego_scene_feats))
loss_status_distill = 0.5 · (1 − cos(student.ego_status_feats, teacher.ego_status_feats))
```

The second term is the ego_lcf transfer: the student's **vision-estimated** status is
pulled toward the teacher's embedding of the **real `ego_lcf`**.

`feature_distill_weight=0.0` in the configs is the *fused* weight, unused in split mode. It
does not mean distillation is off.

### Weights

`scene 0.3` / `status 0.5`, and `feature_distill_weight=0.3` in the fused form. The fused
value was set so a cosine loss that never reaches zero would not dominate the trajectory
loss late in training — **reasoning only, never tuned**.

---

## 3. How the target point is used

### Conditioning vs selection

| approach | TP enters | L2 |
|---|---|---|
| Goal-grid conditioning (5×5) | generation | cutting the goal space into a coarse grid **destroys most of the TP information** |
| Goal prediction head | generation (predicted TP) | superseded |
| Per-command anchors + TP selection | selection only | 0.3115 |
| **Cell planner** | **selection only** | **0.2668** |

The target-point prediction path — `goal_cls`, `goal_off`, `goal_embed`, `goal_follow`,
`goal_select`, the argmax trajectory loss, the 10-step decoder extension — was removed
entirely. Generation never sees the target point; selection happens afterwards and is not
learned.

### Is a cell partition worth anything?

Measured with a linear surrogate before training anything (per-cell ridge, train→val).

| | command only | cell partition | continuous TP |
|---|---|---|---|
| **overall** | 0.2234 | **0.1928 (−13.7%)** | 0.1191 |
| LANE_KEEP | 0.2226 | 0.1877 | 0.1156 |
| LANE_CHANGE_L | 0.3106 | 0.3187 (**no gain**) | 0.1449 |
| LANE_CHANGE_R | 0.2461 | 0.2465 (**no gain**) | 0.1328 |
| TURN_LEFT | 0.3648 | 0.2935 | 0.2105 |
| TURN_RIGHT | 0.3245 | 0.2944 | 0.1780 |
| U_TURN | 0.6508 | 0.6957 (**worse**) | 0.7339 |

Three conclusions, all of which went into the final layout:

1. **Lane changes want lateral splits, not forward ones.** Forward partition gives nothing;
   three lateral cells give LC_L 0.2876 / **LC_R 0.2102** (two lateral cells: 0.2837 / 0.2334).
2. **U_TURN should not be partitioned at all** (134 frames). It gets a cell-free head, as
   does STOP.
3. **The gap to continuous TP is large** (0.1928 vs 0.1191) — discretisation costs real
   information, and the cell count is a floor on how much.

### Forward vs lateral, per command

| command | forward only | lateral only | forward × lateral |
|---|---|---|---|
| LANE_KEEP | 15×1 **−15.7%** (quantile edges −18.4%) | ×3 −3.0% | 15×3 −19.7%, but min 21 frames/cell |

→ LANE_KEEP forward-only. 15×3 scores better and was rejected: at 21 frames per cell the
support is too thin, which is the same effect that later showed up as the scene-count cliff.

### STOP selection threshold

Rule is `|TP| < t` → STOP head.

| t | STOP head train frames (of which moving) | frames excluded from loss | val L2 overall | test clips routed to STOP |
|---|---|---|---|---|
| **1 m** | 5672 (252) | 660 | 0.1932 | 295 |
| 3 m | 6767 (946) | 259 | 0.1925 | 322 |
| 6 m | 7977 (1915) | **18** | **0.1923** | 334 |

6 m scores marginally best and **1 m was chosen**:

- the surrogate takes GT acceleration as input, so it overestimates "about to move" intent
  and is biased toward the larger threshold
- TP 1–6 m frames are 28% stopped / 72% moving over 3 s — a distinction only the TP
  carries, and 6 m discards it
- the whole disagreement covers 39 test clips (3.5%)

An earlier form of the rule used forward distance rather than `|TP|` and misclassified
U-turns.

---

## 4. Cell layout

### Scene-count cliff

The most load-bearing measurement in this work.

| training scenes in a cell | val L2 | oracle |
|---|---|---|
| under 20 | **0.79 – 0.97** | 0.11 – 0.24 |
| 30 or more | **0.20** | 0.11 – 0.24 |

The oracle is flat across the range. A thin cell is therefore not bad because its
candidates are bad — it is bad because **one head barely saw that region of the code**.

→ Cell edges are fitted under a floor on *scenes* per cell, not on width or frame count.
Consecutive frames of one scene carry nearly the same information, so frame counts
overestimate support.

A counter-example worth recording: LANE_CHANGE_R has all 14 cells under 20 scenes and still
scores 0.3187, better than TURN_LEFT's 0.4545 with 5 of 8 under. Scene count explains much
of the variance, not all of it.

### The oracle is not headroom

| | value |
|---|---|
| measured oracle | 0.1411 |
| goal spread | 5.69 m |
| cells | 60 |
| nearest-point expectation, 5.69/60 | ~0.17 m |

The oracle mostly measures candidate spacing. Doubling the cell count halves it and changes
nothing real.

### Three layouts

| | LANE_KEEP | constraint | result |
|---|---|---|---|
| v1 | 15×1 | ≥50 frames, ≥3 scenes, ≥4 m per cell | per-command val below |
| v2 | 60×1 | ≥1 m | **0.2668** |
| **v3** | 60×1 | **≥25 scenes (LK), ≥10 (others)** | shipped |

v1 per-command val: LANE_KEEP 0.807 · LANE_CHANGE_L 1.155 · LANE_CHANGE_R 0.934 ·
TURN_LEFT 0.995 · TURN_RIGHT 0.790.

TURN_LEFT 7×2 violated the width and scene constraints, so 4×2 was used (5×2 val 0.998,
7×1 val 1.261).

v2 → v3 keeps the cell count and redraws the edges under the scene floor.

Measured at the v1→v2 step (3-frame, EMA):

| | LK60 (v2) | LK15 (v1) |
|---|---|---|
| overall | **0.267105** | 0.273261 |
| LANE_KEEP | **0.254363** | 0.267317 |
| LANE_CHANGE_L | 0.606292 | **0.494504** |
| TURN_LEFT | 0.454473 | **0.416920** |

LANE_KEEP improves, which is what the split targeted; the two thin-support commands get
worse, consistent with the cliff.

### Final layout (v3)

99 moving cells. U_TURN and STOP have no cells.

| command | cells | forward range (m) | lateral edges (m) |
|---|---|---|---|
| LANE_KEEP | 60 × 1 | 1 – 117 | -20, 17 |
| LANE_CHANGE_L | 10 × 1 | 1 – 116 | -16, 12 |
| LANE_CHANGE_R | 9 × 1 | 1 – 115 | -12, 18 |
| TURN_LEFT | 4 × 2 | 1 – 48 | -2, 8.5, 27 |
| TURN_RIGHT | 6 × 2 | 1 – 43 | -26, -11.5, 2 |

LANE_KEEP forward edges:

```
  1,    2,    3.5,  6.5,  9,    12,   15,   17,   19,   21,
 22.5, 23.5, 25.5, 27.5, 29.5, 31,   32.5, 34,   35,   36.5,
 38,   39.5, 40.5, 41.5, 43.5, 45,   46,   47.5, 49,   50.5,
 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5,
 61.5, 62.5, 63.5, 64.5, 65.5, 67,   68,   69,   70.5, 72,
 73.5, 82,   95,   97.5, 99,  100.5, 102,  103,  104.5, 105.5,
117
```

```
LANE_CHANGE_L  forward  1, 29.5, 33.5, 40.5, 44.5, 48.5, 53, 57, 62.5, 71.5, 116
LANE_CHANGE_R  forward  1, 30, 38.5, 42.5, 46.5, 50.5, 55.5, 63, 75, 115
TURN_LEFT      forward  1, 20.5, 24.5, 30, 48
TURN_RIGHT     forward  1, 13.5, 17.5, 22.5, 27.5, 33, 43
```

Spacing is deliberately uneven — 2–3 m apart below 21 m, 1 m apart between 50 and 73 m,
then 9–13 m jumps past 73 m. The partition follows where the data is, not where the metres
are.

Fitted by exact alternating DP, minimising within-cell 3 s trajectory variation subject to
the floor. `tools/analysis/fit_cell_layout.py`.

Note: lateral splits for lane changes were measured as valuable (LC_R 0.2461 → 0.2102) and
**are not in the final layout**, which splits forward only. Open.

### Positional code

Non-learned buffer. Cell centre [forward, lateral], 64 frequencies per axis × sin/cos =
**256 dims**, wavelengths `geomspace(1, 300) m`. Planner input is
`cat(ego_feats 520, cell PE 256) = 776`.

Head initialisation copies the donor's `ego_fut_decoder` per command, with the PE columns
of the first layer zeroed, so the **initial output equals the donor's**.

---

## 5. Frame window

### Cost per frame

fp16 with `--bev-only-history`, RTX 3090. Score penalty is `1 + max(0, T−100)/200`.

| frames | T_infer | penalty |
|---|---|---|
| 2 | 89.6 ms | ×1.00 |
| **3** | **117.3 ms** | **×1.087** |
| 4 | 145.0 ms | ×1.225 |

Decomposition: a history frame costs **27.7 ms**, the scored frame **61.9 ms**.
`--bev-only-history` skips the decoders whose output a history frame discards; `bev_embed`
is unchanged, so the submitted trajectory is **bit-identical**. Without the flag the
three-frame cost is 135 ms (×1.176).

Same measurement on an A5000: 3-frame 140.92 ms (×1.2046), 2-frame 102.90 ms (×1.0145).
Never measured on the scoring 4090.

→ The third frame costs 8.7%. The acceleration term it enables is worth more than that.

### Window at inference must match training

Training uses `queue_length=3`, `target_stride=5` — three frames 0.5 s apart. Replaying the
annotation file's full seven frames instead gives the scored frame **six accumulated
history steps where training gave it two**, because the BEV encoder attends to `prev_bev`
recursively. Same tensor shapes, no warning, different input distribution.

→ `SUBMIT_FRAME_OFFSETS = '-10,-5,0'` is the default in both the submission and evaluation
tools.

---

## 6. Train/inference mismatches

Settings that behave differently at train and test time. Removing them was worth more than
any architecture change in this project.

| setting | effect | measured |
|---|---|---|
| `nn.Dropout` | train/eval feature distribution differs | **+5% speed bias** |
| `prev_bev_dropout=0.5` | same | not measured individually |
| `ego_status_est_dropout=0.3` | same | not measured individually |
| `prism_latent_supervision` | trains on GT future posterior, infers from prior mean | not measured individually |

| | L2 |
|---|---|
| no distillation, mismatches present | 0.5018 |
| **mismatches removed** | **0.3772 (−25%)** |

The last three were removed on the same principle without separate measurement.
`audit_pipeline.py` now fails if any of the four is enabled.

### `bev_residual_refine`

Enabled by default and quietly harmful.

| stage-1 epoch | refine on | refine off | change |
|---|---|---|---|
| 24 | 0.6395 | **0.4799** | −25.0% |
| 36 | — | 0.4886 | — |
| 48 | 0.6089 | **0.4807** | −21.1% |

Latency difference: 196.2 ms → 194.9 ms, i.e. **1.3 ms**. Not a speed/accuracy trade; it
was simply making trajectories worse.

The refine-off numbers are flat from epoch 24 to 48 (0.4799 / 0.4886 / 0.4807) while the
refine-on numbers improve (0.6395 → 0.6089): the coarse trajectory had converged by epoch
24, and the remaining 24 epochs went into undoing the refinement.

→ Disabled for training, evaluation and submission.

### Other settings that changed numbers silently

| | effect |
|---|---|
| `can_bus` yaw delta wrapping at the 0/360 boundary | fed ±359° into the BEV encoder |
| zero-init embedding residual | widths matched, values did not flow — transfer broke silently |
| submission script reading GT `target_point` at inference | made the measured number non-compliant |
| evaluation giving STOP samples (6.4%) the correct mode for free | inflated STOP performance |

---

## 7. Training schedule

### Batch size

| batch/GPU | lr | warmup | EMA momentum |
|---|---|---|---|
| 1 | 5e-5 | 500 | 2e-4 |
| **2** | **1e-4** | **250** | **4e-4** |

Same learning rate per sample, same warmup length in samples, same averaging window. The
four move together; changing one alone makes it a different experiment.

Batch 1 and batch 2 have been observed to give **different results with the package
applied**. Cause not established.

### Epochs

12-epoch cosine (1e-4 → 1e-7). Per-epoch training planning loss:

| epoch | lr | `loss_plan_reg` | change |
|---|---|---|---|
| 10 | 1.47e-5 | 0.005808 | −6.2% |
| 11 | 6.79e-6 | 0.005570 | −4.1% |
| 12 | 1.80e-6 | 0.005430 | −2.5% |

The loss is still falling 6% per epoch at epoch 10, with the learning rate already down to
15% of peak. It flattens only in epochs 11–12, where the step size collapses.

→ The run stops because the schedule ends, not because it converged.

### Extending a finished run

Resumed epoch 12 with the learning rate reset to the tail of a 20-epoch cosine (3.46e-5 at
that point — a 19× increase over where it stopped).

| epoch | lr | `loss_plan_reg` | vs epoch 12 |
|---|---|---|---|
| 13 | 3.46e-5 | 0.006167 | **+13.6%** |
| 14 | 2.74e-5 | 0.005946 | +9.5% |
| 15 | 2.07e-5 | 0.005679 | +4.6% |
| 16 | 1.47e-5 | 0.005482 | +1.0% |
| 17 | 9.64e-6 | 0.005332 | **−1.8%** |
| 18 | 5.54e-6 | 0.005221 | **−3.8%** |

Raising the learning rate knocks the model out of the minimum it had settled into, and it
takes **five epochs to return to break-even**.

→ One or two extra epochs is strictly worse than stopping. Training loss only — full data
has no hold-out, so whether this is generalisation or memorisation is not separable here.

---

## 8. Final models

Four submitted models, all trained on 376 scenes at 10 Hz, 12 epochs.

### Shared

| | |
|---|---|
| window | `queue_length=3`, `target_stride=5`, `aux_bev_motion_frames=3` |
| optimizer | AdamW, lr 1e-4, weight decay 0.01, `img_backbone` lr_mult 0.1 |
| schedule | CosineAnnealing, `min_lr_ratio=0.001` → 1e-4 to 1e-7 |
| warmup | linear, 250 iters, ratio 1/3 |
| batch | 2 per GPU × 2 GPUs, 4 workers |
| EMA | momentum 4e-4 |
| clipping | `max_norm=35`; fp16 `loss_scale=512.0` |
| planner | `ego_lcf_feat_idx=None`, `ego_status_est_dim=8`, decoder hidden 512 |
| cells | v3 layout, `cell_pe_dim=256`, `cell_stop_tp_thresh=1.0` |
| time weighting | `plan_reg_ts_weight_mode='cumulative'` |

### Differences

| model | initialised from | distillation | planning loss |
|---|---|---|---|
| nokd | stage-1 donor, planner first layer widened by zero columns | — | per-step delta |
| kd_delta | **the trained teacher** (857 of 861 tensors) | scene 0.3 + status 0.5 | per-step delta |
| kd_cumloss | the trained teacher | scene 0.3 + status 0.5 | **cumulative position** |
| nokd_continue20 | resumes `nokd` epoch 12 | — | per-step delta |
| teacher (not submittable) | stage-1 donor | — | per-step delta |

The four tensors not copied into the KD initialisation are `ego_status_est_net`: the
teacher reads the real ego state and has no such network.

### Delta vs cumulative planning loss

The metric measures positions; the default loss measures per-step deltas. Since
`|Σe| ≤ Σ|e|`, delta L1 is an upper bound that is loose exactly where errors cancel.
`plan_reg_cumulative=True` applies the chain rule with signs intact.

When it is on, time weighting falls back to `position` regardless of
`plan_reg_ts_weight_mode` — the `cumulative` mode is the same chain rule with signs
discarded, and applying both would count it twice.

On a synthetic probe with independent per-step errors the cumulative form came out 1.3×
the delta form (0.0693 vs 0.0526). Real trajectory errors are sign-correlated, so
cancellation is rarer and the factor should be larger. **Effect on the final score not
established.**

### Cost

| | |
|---|---|
| parameters | 50.17 M (teacher 48.60 M) |
| FLOPs, single frame | 479.9 G — architecture only, identical across the four |
| T_infer | 117.3 ms (3-frame, fp16, `--bev-only-history`, 3090) |

All four produce the same selection distribution — 828 cell / 2 U_TURN / 295 STOP over
1125 test clips — because selection is target-point driven. Only the trajectory inside the
chosen cell differs.

---

## Settings in use that were never measured

| setting | basis |
|---|---|
| stage-1 `loss_plan_reg=0.0` | inherited from the original VAD recipe. Its premise (don't train a planner on an immature BEV) is already broken once KD trains the planner in stage 1 |
| `feature_distill_weight=0.3` | reasoning only, never tuned |
| `plan_reg_ts_weight_mode='cumulative'` | the A/B that supported it (0.2328 vs 0.2542) was **confounded** — the two teachers also differed in embedding width |
| `aux_bev_motion_temporal=True` | never measured in loss-only mode |
| batch 2 differing from batch 1 | observed, cause not established |
