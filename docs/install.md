# Installation

The planners are built on the [VAD](https://github.com/hustvl/VAD) /
[LAW](https://github.com/BraveGroup/LAW) stack. The versions below are the ones used for
the paper.

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.7.1 (CUDA 12.8) |
| torchvision | 0.22.1 |
| mmcv-full | 1.4.0 |
| mmdet | 2.14.0 |
| mmsegmentation | 0.14.1 |
| mmdet3d | 0.17.1 |

```bash
conda create -n kinproj python=3.10 -y
conda activate kinproj

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install mmcv-full==1.4.0 mmdet==2.14.0 mmsegmentation==0.14.1

# mmdet3d 0.17.1 from source, as in VAD and LAW
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout v0.17.1 && pip install -v -e . && cd ..

pip install nuscenes-devkit pyquaternion shapely similaritymeasures einops timm \
            scikit-image pandas pyarrow yapf==0.40.1
pip install -r requirements.txt     # projection and analysis
```

Run every command from the repository root; the configs load the model code from
`projects/mmdet3d_plugin/` relative to the working directory.

```bash
export PYTHONPATH=.
```

## Pretrained weights

```bash
mkdir -p ckpts
# ImageNet ResNet-50 (image backbone)
wget -P ckpts https://download.pytorch.org/models/resnet50-19c8e357.pth
```

Stage 1 starts from the LAW checkpoint pretrained on nuScenes, released by the
[LAW](https://github.com/BraveGroup/LAW) authors; save it as `ckpts/law_pretrained_nus.pth`.
The weights of the three planners evaluated in the paper are linked in the README.

## Check

```bash
python -c "
import importlib, mmcv
from mmdet3d.models import build_model
cfg = mmcv.Config.fromfile('configs/cell105.py')
importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
m = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
print('params %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
"
# params 50.17M
```
