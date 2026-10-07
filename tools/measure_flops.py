import argparse
import importlib
import warnings

warnings.filterwarnings('ignore')
import torch
import torch.utils.module_tracker as _mt
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.flop_counter import FlopCounterMode


class _NoHandle:
    def remove(self):
        pass


_mt.register_multi_grad_hook = lambda *a, **k: _NoHandle()


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)

    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda().eval()
    model.compute_planner_metric_stp3 = lambda *a, **k: {}

    reset_stream(model)
    with torch.no_grad():
        model(return_loss=False, rescale=True,
              **scatter(collate([dataset[0]]), [0])[0])
    data = scatter(collate([dataset[0]]), [0])[0]
    img_shape = tuple(data['img'][0].shape)

    counter = FlopCounterMode(display=False, depth=3)
    with torch.no_grad(), counter:
        model(return_loss=False, rescale=True, **data)
    glob = counter.get_flop_counts().get('Global', {})
    total = sum(glob.values())

    params = sum(p.numel() for p in model.parameters())
    g = total / 1e9
    print('=' * 60)
    print(f'input img : {img_shape}  bev {cfg.bev_h_}x{cfg.bev_w_}')
    for op, v in sorted(glob.items(), key=lambda x: -x[1]):
        print(f'  {str(op):42s} {v / 1e9:10.2f} G')
    print(f'FLOPs (single frame, prev_bev present) : {g:.1f} GFLOPs')
    print(f'Params : {params / 1e6:.2f} M')
    print('=' * 60)

    if args.out:
        import json
        import os
        out_dict = {'__flops__': int(total)}
        if os.path.exists(args.out):
            with open(args.out) as f:
                existing = json.load(f)
            existing.pop('__flops__', None)
            out_dict.update(existing)
        with open(args.out, 'w') as f:
            json.dump(out_dict, f)
        print(f'wrote __flops__={int(total)} to {args.out}')


if __name__ == '__main__':
    main()
