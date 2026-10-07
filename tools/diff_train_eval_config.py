"""Every resolved model setting that differs between a train and eval config.

audit_pipeline.py's parity check compares state_dict SHAPES, which catches a
wrong descriptor width but not a flag that changes behaviour without changing
a tensor (bev_residual_refine, aux_long_horizon_residual, history frames).
This prints the full resolved difference so the only lines left are the ones
that are supposed to differ (pipelines, test-time switches, distillation).

Usage: python tools/diff_train_eval_config.py <train_cfg> <eval_cfg>
"""
import sys

import mmcv


def flat(d, prefix=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f'{prefix}{k}.'))
    elif isinstance(d, (list, tuple)) and d and all(isinstance(x, dict) for x in d):
        for i, v in enumerate(d):
            out.update(flat(v, f'{prefix}{i}.'))
    else:
        out[prefix[:-1]] = d
    return out


def main():
    tr = flat(mmcv.Config.fromfile(sys.argv[1]).model.to_dict())
    ev = flat(mmcv.Config.fromfile(sys.argv[2]).model.to_dict())
    n = 0
    for k in sorted(set(tr) | set(ev)):
        a, b = tr.get(k, '<missing>'), ev.get(k, '<missing>')
        if a != b:
            n += 1
            print(f'  {k}\n      train: {a}\n      eval : {b}')
    print(f'{n} differences')


if __name__ == '__main__':
    main()
