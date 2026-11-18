# Training and Inference

```
stage 1  configs/stage1.py   48 epochs, from ckpts/law_pretrained_nus.pth
   │
   └─ merge world model + widen the planner input by the 8 state columns → ckpts/stage2_init.pth
        │
        ├─ configs/cell105.py   105-cell planner   12 epochs
        ├─ configs/cell15.py    15-cell planner    12 epochs
        └─ configs/nocell.py    no-cell baseline   12 epochs
```

All runs use two 24 GB GPUs, one sample per GPU.

## Stage 1

```bash
bash tools/dist_train.sh configs/stage1.py 2 --work-dir work_dirs/stage1 --no-validate
```

## Stage-2 initialization

```bash
python tools/merge_stage1_world_model.py \
    --stage1 work_dirs/stage1/epoch_48.pth \
    --world-model-source ckpts/law_pretrained_nus.pth \
    --output work_dirs/stage1/merged.pth
python tools/surgical_ego_fut_decoder_transfer.py \
    --stage1-on work_dirs/stage1/merged.pth --target work_dirs/stage1/merged.pth \
    --ego-lcf-n 0 --pad-input-cols 8 --output ckpts/stage2_init.pth
```

## Stage 2

```bash
bash tools/dist_train.sh configs/cell105.py 2 --work-dir work_dirs/cell105 --no-validate
bash tools/dist_train.sh configs/cell15.py  2 --work-dir work_dirs/cell15  --no-validate
bash tools/dist_train.sh configs/nocell.py  2 --work-dir work_dirs/nocell  --no-validate
```

The cell layouts are stored in the configs. `tools/fit_cell_layout.py` refits them from the
training annotations.

## Inference

`tools/evaluate.py` replays each held-out window (three frames, 0.5 s apart), reports the
planning L2 and, with `--dump-npz`, saves every window's plan, recorded trajectory, measured
state and estimated state. These files are the input of `analysis/`.

```bash
python tools/evaluate.py configs/eval/cell105.py ckpts/cell105.pth \
    --ann-file data/etri/split_301_75/vad_etri_infos_temporal_val_split.pkl \
    --fp16 --bev-only-history --select-cell-by-tp --stride 5 \
    --dump-npz outputs/cell105.npz
```

Use `configs/eval/cell15.py` with `--select-cell-by-tp` for the 15-cell planner and
`configs/eval/nocell.py` without it for the no-cell baseline.
