# Training and evaluation

Two stages. Stage 1 trains the world model with the planner's trajectory loss off; stage 2
trains the planner on top of it. All four released models share one stage-1 donor.

```
stage 1  (48 epochs, ego_lcf OFF)
   │
   ├─ merge + widen the planner's first layer by the status slot   → donor
   │
   ├─ configs/nokd.py              ──► nokd_continue20.py (resume at 13)
   └─ configs/teacher_NOSUBMIT.py  ──► kd_delta.py / kd_cumloss.py
```

## Before any long run

Not optional. Several wrong-but-silent runs were caught here, and each one of those cost a
day.

```bash
python tools/audit_pipeline.py configs/nokd.py --eval-config configs/eval/nokd.py
python tools/check_accel_block_live.py configs/nokd.py
python tools/check_cell_planner_live.py configs/nokd.py
```

`audit_pipeline.py` checks compliance (no privileged input reaching the planner), the
train/inference settings that must match, silent no-ops, and data leakage.
`check_accel_block_live.py` asserts the three-frame descriptor's acceleration block is
non-zero on real data — a descriptor that is identically zero trains and converges and
means nothing. `check_cell_planner_live.py` verifies the cell machinery before it trains:
that perturbing the target point leaves every candidate unchanged, that no gradient flows
to it, and that each cell's centre selects its own cell.

To compare a training config against its evaluation counterpart:

```bash
python tools/diff_train_eval_config.py configs/nokd.py configs/eval/nokd.py
```

## Training

```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py \
    configs/nokd.py --launcher pytorch \
    --work-dir work_dirs/nokd --deterministic
```

Two 24 GB GPUs at `samples_per_gpu=2` is what these were trained on, at roughly
1.25 s/iteration and 5358 iterations per epoch — about 22 hours for 12 epochs. The
distilled runs add the teacher's forward pass and cost ~4% more.

### Hyperparameters

Identical across the four models, and already in the configs:

| | |
|---|---|
| epochs | 12 |
| optimizer | AdamW, lr 1e-4, weight decay 0.01, `img_backbone` lr_mult 0.1 |
| schedule | CosineAnnealing, `min_lr_ratio=0.001` — 1e-4 down to 1e-7 |
| warmup | linear, 250 iters, ratio 1/3 |
| batch | 2 per GPU × 2 GPUs, 4 workers |
| EMA | momentum 4e-4 |
| gradient clipping | `max_norm=35` |
| fp16 | `loss_scale=512.0` |

**Batch size does not travel alone.** At one sample per GPU the package is lr 5e-5 /
warmup 500 / EMA 2e-4; at two it is 1e-4 / 250 / 4e-4. Same learning rate per sample, same
warmup length in samples, same averaging window. Changing the batch without the other three
makes it a different experiment, and the two have been observed to give different results.

### The distilled models

They instantiate the teacher at construction and need its weights first:

```bash
# either train it
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py \
    configs/teacher_NOSUBMIT.py --launcher pytorch --work-dir work_dirs/teacher

# or download it
R=https://huggingface.co/gyuz/2026_ETRI_Autonomous_Driving_Challenge/resolve/main
mkdir -p work_dirs/teacher
wget -O work_dirs/teacher/epoch_12.pth "$R/stage2_fulldata376_10hz_teacher8_NOSUBMIT_epoch12.pth"
```

Then seed the student from the teacher rather than from stage 1:

```bash
python tools/surgery/make_kd_student_init.py \
    --teacher work_dirs/teacher/epoch_12.pth \
    --out work_dirs/donors/kd_init_from_teacher8_v3.pth
```

It copies 857 of 861 tensors. The four it does not copy are `ego_status_est_net` — the
teacher reads the real ego state and has no such network, and the student has to learn one.
Any other missing tensor is a bug, and the script reports the count so you can check.

```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py \
    configs/kd_delta.py --launcher pytorch --work-dir work_dirs/kd_delta
```

The teacher is a training-time object only. It never appears on the inference path.

### Extending a finished run

`configs/nokd_continue20.py` resumes `work_dirs/nokd/epoch_12.pth` and lays a 20-epoch
cosine over it, so epoch 13 starts at 3.46e-5 — the real tail of a 20-epoch schedule rather
than an approximation of one.

`resume_from`, not `load_from`: the checkpoint carries `meta.epoch`, `meta.iter`, the AdamW
moments and the EMA buffers, so the run continues instead of restarting. Warmup does not
re-fire, because it is gated on an iteration count already long past.

Expect the loss to **rise** first. A 12-epoch cosine ends at 1.8e-6; restarting at 3.46e-5
is a 19× increase and it knocks the model out of the minimum it had settled into. On our
run it took five epochs to get back to where it started, and only then improved. One or two
extra epochs is strictly worse than stopping.

## Evaluation

```bash
python tools/eval_holdout_l2_and_tinfer.py \
    configs/eval/nokd.py work_dirs/nokd/epoch_12.pth \
    --ann-file data/etri/annotations_10hz/vad_etri_10hz_infos_temporal_val_split.pkl \
    --fp16 --bev-only-history --select-cell-by-tp \
    --stride 50 --warmup-windows 20
```

L2 and T_infer come from the same replayed windows, so a frame-count comparison needs one
checkpoint load rather than two runs. Each `--frame-offsets` configuration gets its own
warmup, otherwise whichever is listed first absorbs the cold-start cost.

T_infer is only meaningful on an idle GPU. Numbers measured while anything else was running
have been discarded more than once.

To compare windows deliberately:

```bash
--frame-offsets "0,-5,-10" "0,-5"      # three-frame against two-frame
```

### Flags that silently change the answer

| flag | why |
|---|---|
| `--select-cell-by-tp` | required for a cell planner. Off by default, and without it nothing selects among the generated candidates |
| `--fp16` | evaluation and submission must agree. One submission ran fp32 against an fp16 evaluation and the trajectories differed by up to 0.040 m |
| `--bev-only-history` | skips the decoders whose output a history frame discards. `bev_embed` is unchanged, so the submitted trajectory is bit-identical; T_infer drops from 185 ms to 117 ms at three frames |
| matching eval config | each model has a counterpart in `configs/eval/`. A mismatch scores a **different network**, with no warning |

The default window is `-10,-5,0`, matching `queue_length=3` / `target_stride=5`. It used to
default to the annotation file's full seven frames, which handed the scored frame six
accumulated history steps where training gave it two — same shapes, no error, different
input distribution.

## Submission

```bash
bash tools/make_submission.sh epoch_12.pth configs/eval/nokd.py nokd
```

Three steps, kept together because separating them is how one submission went out without
the second:

1. `tools/etri_test_submit.py` — trajectories for all 1125 test clips
2. `tools/measure_flops.py` — single-frame FLOPs, merged into the same JSON as `__flops__`
3. `tools/_pack_submission.py` — format check, then zip

The scorer derives its cutoff from `__flops__`, and a file without it is rejected after
upload with `CUTOFF_RESULT_MISSING` — which costs an attempt. The packer asserts the field
exists, that there are exactly 1125 scenes, and that every clip has six two-dimensional
waypoints, before it will write the archive.

The resulting JSON has 1126 keys and the zip contains it as `submission.json`.

## Reproducing the measurements

`tools/analysis/` holds what produced the numbers in
[experiments.md](experiments.md):

| | |
|---|---|
| `fit_cell_layout.py` | fit cell edges under a per-command floor on training scenes |
| `plot_cell_layout_compare.py` | compare two layouts on one dataset |
| `kinematic_oracle_ceiling.py` | how far perfect ego velocity/acceleration estimation alone reaches |
| `goal_grid_value.py` | what knowing the 5 s goal cell is worth, per grid size |
| `analyze_planner_errors.py` | error decomposition by distance band, command, cell |
| `eval_holdout_l2_latency_ablation.py` | one setting on and off against the same windows |
| `check_descriptor_sensitivity.py` | whether the BEV motion descriptor can see ego motion at all |
