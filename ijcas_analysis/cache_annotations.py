"""Extract the arrays the partition and feasibility analyses need into one npz.

Loading the 887 MB training pickle costs several GB of RAM; doing it once and
keeping only small arrays lets every later analysis run in seconds.

Saved per split (train / val):
  tp      (N,2)  target point in the ego frame (x forward, y left)
  cmd     (N,)   command index (argmax of gt_ego_fut_cmd)
  traj    (N,12) cumulative 3-s future positions, 6 steps x (x,y), from per-step deltas
  mask    (N,6)  gt_ego_fut_masks
  scene   (N,)   scene index (shared across splits by name)
  frame   (N,)   frame_idx
  canbus  (N,18) can_bus vector (for the measured speed at t0)
Only frames with fut_valid_flag are kept, as in tools/fit_cell_layout.py.
"""
import pickle
import sys

import numpy as np


def extract(path):
    with open(path, 'rb') as fh:
        infos = pickle.load(fh)['infos']
    infos = [i for i in infos if i.get('fut_valid_flag')]
    out = dict(
        tp=np.stack([np.asarray(i['gt_ego_target_point'], float).reshape(-1)[:2] for i in infos]),
        cmd=np.array([int(np.asarray(i['gt_ego_fut_cmd']).argmax()) for i in infos]),
        traj=np.stack([np.asarray(i['gt_ego_fut_trajs'], float).cumsum(0) for i in infos]).reshape(len(infos), -1),
        mask=np.stack([np.asarray(i['gt_ego_fut_masks'], float).reshape(-1)[:6] for i in infos]),
        scene_name=np.array([i['scene_token'] for i in infos]),
        frame=np.array([int(i.get('frame_idx', -1)) for i in infos]),
        canbus=np.stack([np.asarray(i['can_bus'], float).reshape(-1) for i in infos]),
    )
    del infos
    return out


def main(train_pkl, val_pkl, out):
    tr, va = extract(train_pkl), extract(val_pkl)
    names = np.unique(np.concatenate([tr['scene_name'], va['scene_name']]))
    assert not set(tr['scene_name']) & set(va['scene_name']), 'train/val scenes overlap'
    save = {}
    for tag, d in (('train', tr), ('val', va)):
        d['scene'] = np.searchsorted(names, d.pop('scene_name'))
        for k, v in d.items():
            save[f'{tag}_{k}'] = v
        print(f'{tag}: {len(d["tp"])} frames, {len(np.unique(d["scene"]))} scenes, '
              f'traj {d["traj"].shape}, canbus {d["canbus"].shape}')
    np.savez_compressed(out, **save)
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:4])
