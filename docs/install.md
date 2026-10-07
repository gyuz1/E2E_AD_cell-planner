# Installation

This is the VAD / LAW stack. If you already have a working VAD or LAW environment, it will
run this repository unchanged — the versions below are what we trained and evaluated with.

## Environment

| | version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| CUDA (build) | 12.8 |
| cuDNN | 9.7.1 |
| mmcv-full | 1.4.0 |
| mmdet | 2.14.0 |
| mmsegmentation | 0.14.1 |
| mmdet3d | 0.17.1 |

## Setup

```bash
conda create -n cellplanner python=3.10 -y
conda activate cellplanner

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# mmdet3d 0.17.1 is installed from source, as VAD and LAW do
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout v0.17.1 && pip install -v -e . && cd ..

pip install nuscenes-devkit pyquaternion shapely similaritymeasures \
            einops timm numba yapf==0.40.1
```

Then clone this repository and run from its root — `plugin_dir` in every config is
`projects/mmdet3d_plugin/`, resolved relative to the working directory.

```bash
git clone <this repo>
cd E2E_AD_cell-planner
```

### ImageNet backbone

Training initialises the image backbone from ResNet-50:

```bash
mkdir -p ckpts
wget -P ckpts https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## Verify

```bash
python -c "import torch, mmcv, mmdet, mmdet3d; \
print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmdet.__version__, mmdet3d.__version__)"
```

A config should resolve and build a model without any data present:

```bash
python -c "
import importlib, mmcv
from mmdet3d.models import build_model
cfg = mmcv.Config.fromfile('configs/nokd.py')
importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
m = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
print('params %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
"
# params 50.17M
```

`configs/nokd.py` and `configs/teacher_NOSUBMIT.py` build standalone. The two distilled
configs instantiate the teacher at construction time and need its weights at
`work_dirs/teacher/epoch_12.pth` first — see [train_eval.md](train_eval.md).

## Docker

We trained inside a container (`etri-vad:cu128`) rather than a conda environment. If you
build your own image on the versions above, two things are worth knowing.

**`--shm-size` matters.** The dataloader uses 4 workers per GPU over multi-camera tensors;
the Docker default of 64 MB will deadlock it. We ran with `--shm-size=64g` for training and
`16g` for inference.

**CUDA forward-compat libraries break on GeForce cards.** NVIDIA's CUDA images ship
`/usr/local/cuda/compat`, which only works on datacenter GPUs. On a GeForce host whose
driver is older than the image's CUDA version, `torch.cuda.is_available()` returns `False`
with `Error 804: forward compatibility was attempted on non supported HW`. Mask the
directory and let the host driver serve:

```bash
docker run -d --gpus all --ipc=host --shm-size=64g \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    --mount type=tmpfs,destination=/usr/local/cuda/compat \
    -v $(pwd):/workspace/VAD -w /workspace/VAD \
    etri-vad:cu128 sleep infinity
```

With that, a CUDA 12.8 image runs on a 535 driver (CUDA 12.2) through minor-version
compatibility. A driver newer than the image needs neither flag.

## Next

[Data preparation](data_preparation.md) → [Training and evaluation](train_eval.md)
