"""Plot each command's goal cells over the train / val / test 5s target points.

One figure per command: cell grid, the outer range (dashed), train TPs (dots),
val TPs (circles), test TPs (stars), and per cell its (row, col) and train
count. Cells with no train frame are red, cells under --min-train yellow.
Plot x is -lateral, so LEFT is drawn on the left.

Also prints, per cell, train/val/test counts and how similar the 3s trajectory
is inside the cell: the official L2 of each frame's 3s waypoints against the
train mean of its cell (lower = one cell, one trajectory). Test has no future
trajectory, so it is counted only.

Cells: forward edges x lateral edges per command, in metres, ego frame
(x forward, y left positive); cell_id = row * n_lateral + col; a TP belongs to
the cell containing it, out-of-range TPs to the nearest outer cell.

Usage:
    python tools/plot_command_cells.py [--layout chosen|request]
        [--out-dir reports/cell_plots] [--min-train 30] [--frame-stride 1]
"""
import argparse
import os
import pickle

import numpy as np

ANN = 'data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_{}.pkl'
COMMANDS = ['LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT',
            'TURN_RIGHT', 'U_TURN', 'STOP']

# command id -> (forward edges, lateral edges); None = no cells.
LAYOUTS = {
    # Edges minimising the within-cell 3s trajectory spread on train
    # (exact DP, >= 50 train frames per cell), 2026-09-17.
    'chosen': {
        0: ([-6, 9, 17, 24, 31, 38, 45, 51, 56, 61, 66, 72, 82, 94, 102, 117],
            [-20, 17]),
        1: ([-6, 20, 25, 36, 44, 51, 58, 66, 78, 94, 103, 116], [-16, 12]),
        2: ([-6, 23, 30, 35, 43, 47, 52, 59, 64, 74, 81, 87, 93, 100, 115],
            [-12, 18]),
        3: ([-3, 16, 19, 21, 24, 27, 30, 48], [-2, 10, 27]),
        4: ([-3, 12, 16, 20, 26, 33, 43], [-26, -11.5, 2]),
        5: None,
        6: None,
    },
    # stage2_planner_request.md
    'request': {
        0: ([-6.0, 3.5276535580608908, 11.427310899107322, 21.51901184776763,
             29.2876652667183, 34.45271744437977, 43.092417157405364,
             51.236155176672845, 57.161575043879026, 62.784429701578276,
             69.31846174381577, 76.65878140047634, 85.1672331167891,
             93.6601397285273, 102.63177906313388, 117.0], [-20, 17]),
        1: ([-6.0, 19.730564428956665, 30.340758890902038, 42.14008774977505,
             47.97590789741159, 53.2005037179496, 60.01605355011668,
             65.9392982598299, 82.93152938625995, 95.89110946545412,
             102.87971666857808, 116.0], [-16, 12]),
        2: ([-6.0, 13.486152993510313, 26.545606854111977, 29.88276819976049,
             37.481771737323584, 42.3243082364258, 45.90096288317217,
             49.062432929258165, 53.48461092091346, 57.38027644474694,
             63.453761130482874, 73.68793447656628, 80.17607609383604,
             92.40993330669595, 115.0], [-12, 18]),
        3: (list(np.linspace(-3, 48, 8)), [-2, 12.5, 27]),
        4: (list(np.linspace(-3, 43, 7)), [-26, 2]),
        5: ([-3, 6, 15, 24, 33], [-4, 23]),
        6: None,
    },
}


def load(split, stride):
    d = pickle.load(open(ANN.format(split), 'rb'))
    infos = d['infos'] if isinstance(d, dict) else d
    if split == 'test':
        # 7 frames per clip share one TP/command; the scored frame is "_0".
        infos = [i for i in infos if str(i['token']).endswith('_0')]
    else:
        infos = [i for i in infos if i.get('fut_valid_flag')
                 and int(i['frame_idx']) % stride == 0]
    tp = np.stack([np.asarray(i['gt_ego_target_point'], np.float64)
                   .reshape(-1)[:2] for i in infos])
    cmd = np.array([int(np.asarray(i['gt_ego_fut_cmd']).argmax()) for i in infos])
    traj = np.stack([np.asarray(i['gt_ego_fut_trajs'], np.float64).cumsum(0)
                     for i in infos])
    return tp, cmd, traj


def cell_of(tp, fe, le):
    row = np.clip(np.searchsorted(fe[1:-1], tp[:, 0], side='right'), 0, len(fe) - 2)
    col = np.clip(np.searchsorted(le[1:-1], tp[:, 1], side='right'), 0, len(le) - 2)
    outside = ((tp[:, 0] < fe[0]) | (tp[:, 0] > fe[-1])
               | (tp[:, 1] < le[0]) | (tp[:, 1] > le[-1]))
    return row, col, outside


def official_l2(pred, gt):
    d = np.linalg.norm(pred - gt, axis=-1)
    return float(np.mean([d[:, :2].mean(), d[:, :4].mean(), d[:, :6].mean()]))


def plot_command(c, layout, name, data, args, plt):
    tr, va, te = [(tp[cm == c], traj[cm == c]) for tp, cm, traj in data]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    if layout is None:
        title = f'{COMMANDS[c]} | no cells | {name}'
    else:
        fe, le = np.asarray(layout[0], float), np.asarray(layout[1], float)
        nf, nl = len(fe) - 1, len(le) - 1
        title = f'{COMMANDS[c]} | {nf}×{nl}={nf * nl} cells | {name}'
        r, co, out_tr = cell_of(tr[0], fe, le)
        ids = r * nl + co
        cnt = np.bincount(ids, minlength=nf * nl)
        means = np.array([tr[1][ids == k].mean(0) if cnt[k] else tr[1].mean(0)
                          for k in range(nf * nl)])
        for row in range(nf):
            for col in range(nl):
                k = row * nl + col
                n = int(cnt[k])
                face = '#ffe9e9' if n == 0 else ('#fff5dc' if n < args.min_train else '#f4f4f4')
                tcol = '#c0001a' if n == 0 else ('#9a6500' if n < args.min_train else '#333333')
                ax.add_patch(plt.Rectangle((-le[col + 1], fe[row]), le[col + 1] - le[col],
                                           fe[row + 1] - fe[row], facecolor=face,
                                           edgecolor='none', zorder=0))
                ax.text(-(le[col] + le[col + 1]) / 2, (fe[row] + fe[row + 1]) / 2,
                        f'({row},{col})\nn={n}', ha='center', va='center', fontsize=7,
                        color=tcol, zorder=4,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        for v in fe:
            ax.axhline(v, color='#444444', lw=1.2, zorder=1)
        for v in le:
            ax.axvline(-v, color='#444444', lw=1.2, zorder=1)
        ax.plot([-le[0], -le[-1], -le[-1], -le[0], -le[0]],
                [fe[0], fe[0], fe[-1], fe[-1], fe[0]], 'k--', lw=2.2, zorder=3)

        # Table: counts and within-cell trajectory similarity.
        rv, cv, out_va = cell_of(va[0], fe, le)
        rt, ct, out_te = cell_of(te[0], fe, le)
        idv, idt = rv * nl + cv, rt * nl + ct
        print(f'\n{title}')
        print(f'  outside range: train {out_tr.mean() * 100:.2f}%  '
              f'val {out_va.mean() * 100:.2f}%  test {out_te.mean() * 100 if len(te[0]) else 0:.2f}%')
        if len(va[0]):
            print(f'  within-cell 3s L2 vs train cell mean: train '
                  f'{official_l2(means[ids], tr[1]):.4f}  val {official_l2(means[idv], va[1]):.4f}  '
                  f'(no cells: val {official_l2(np.repeat(tr[1].mean(0)[None], len(va[1]), 0), va[1]):.4f})')
        print(f'  {"cell":<8}{"forward [m]":>16}{"lateral [m]":>16}{"train":>7}{"val":>6}{"test":>6}'
              f'{"L2 train":>10}{"L2 val":>9}')
        for row in range(nf):
            for col in range(nl):
                k = row * nl + col
                mt, mv = ids == k, idv == k
                l2t = official_l2(means[ids[mt]], tr[1][mt]) if mt.any() else float('nan')
                l2v = official_l2(means[idv[mv]], va[1][mv]) if mv.any() else float('nan')
                print(f'  ({row},{col}){"":<3}{fe[row]:>7.1f}~{fe[row + 1]:<7.1f}'
                      f'{le[col]:>8.1f}~{le[col + 1]:<6.1f}{int(mt.sum()):>7}{int(mv.sum()):>6}'
                      f'{int((idt == k).sum()):>6}{l2t:>10.3f}{l2v:>9.3f}')

    ax.scatter(-tr[0][:, 1], tr[0][:, 0], s=9, c='#6aa37a', alpha=0.35, lw=0,
               label=f'TRAIN (n={len(tr[0])})', zorder=2)
    ax.scatter(-va[0][:, 1], va[0][:, 0], s=36, facecolors='none', edgecolors='#1f6fe0',
               lw=1.1, label=f'VAL (n={len(va[0])})', zorder=5)
    ax.scatter(-te[0][:, 1], te[0][:, 0], s=110, marker='*', c='#e0242a',
               edgecolors='#5a0000', lw=0.5, label=f'TEST (n={len(te[0])})', zorder=6)
    ax.scatter([0], [0], s=160, marker='^', c='black', label='Ego', zorder=7)
    ax.set_title(title + '\nFULL DISTRIBUTION', fontsize=13, fontweight='bold')
    ax.set_xlabel('LEFT ←  lateral  → RIGHT  [m]', fontsize=12)
    ax.set_ylabel('forward [m]', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    pts = np.concatenate([tr[0], va[0], te[0], np.zeros((1, 2))])
    xs, ys = -pts[:, 1], pts[:, 0]
    if layout is not None:
        xs = np.concatenate([xs, -le]); ys = np.concatenate([ys, fe])
    ax.set_xlim(xs.min() - 5, xs.max() + 2)
    ax.set_ylim(ys.min() - 2.5, ys.max() + 2.5)
    fig.tight_layout()
    path = os.path.join(args.out_dir, name, f'{c}_{COMMANDS[c]}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layout', default='chosen', choices=sorted(LAYOUTS))
    p.add_argument('--out-dir', default='reports/cell_plots')
    p.add_argument('--min-train', type=int, default=30,
                   help='cells with fewer train frames are drawn yellow')
    p.add_argument('--frame-stride', type=int, default=1,
                   help='keep train/val frames with frame_idx %% stride == 0')
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(args.out_dir, args.layout), exist_ok=True)
    data = [load('train_split', args.frame_stride), load('val_split', args.frame_stride),
            load('test', 1)]
    for c in range(7):
        path = plot_command(c, LAYOUTS[args.layout][c], args.layout, data, args, plt)
        print(f'  -> {path}')


if __name__ == '__main__':
    main()
