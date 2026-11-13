"""Select the projection weights (rho_a, rho_kappa) on TRAINING scenes, then report them on the
held-out scenes (open loop and controller in the loop). Measured state; inference-only dumps.
usage: weight_select_train.py <train_dump.npz> <heldout_dump.npz> <tag>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
from smoothing_baseline import off
import closedloop_sim as cl
R = __file__.rsplit('/', 1)[0] + '/results/'
GA = (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
GK = (0.2, 1.0, 5.0, 20.0, 100.0)

def load(p):
    d = np.load(p); idx = np.where(d['mode'] <= 4)[0]
    return d['pred'][idx].astype(float), d['gt'][idx].astype(float), d['ego_lcf'][idx].astype(float), d['scene'][idx]

def E_grid(pred, gt, lcf):
    return {(a, k): np.array([off(project_state(pred[i], lcf[i], 'meas', wa=a, wk=k), gt[i]) for i in range(len(gt))]) for a in GA for k in GK}

def main(trp, hop, tag):
    pt, gtt, lt, st = load(trp); ph, gth, lh, sh = load(hop)
    Et = E_grid(pt, gtt, lt); base_t = np.array([off(pt[i], gtt[i]) for i in range(len(gtt))]).mean()
    best = min(Et, key=lambda g: Et[g].mean())
    Eh = E_grid(ph, gth, lh); base_h = np.array([off(ph[i], gth[i]) for i in range(len(gth))]).mean()
    rep = dict(tag=tag, n_train=len(gtt), n_train_scenes=int(len(np.unique(st))), n_heldout=len(gth),
               train_none=float(base_t), heldout_none=float(base_h), pick=list(best),
               pick_on_grid_edge=bool(best[0] in (GA[0], GA[-1]) or best[1] in (GK[0], GK[-1])),
               train_grid={f'{a},{k}': float(v.mean()) for (a, k), v in Et.items()},
               heldout_grid={f'{a},{k}': float(v.mean()) for (a, k), v in Eh.items()},
               heldout_pick=float(Eh[best].mean()), heldout_oracle=float(min(v.mean() for v in Eh.values())))
    # bootstrap CI on held-out scenes
    Pp = np.stack([project_state(ph[i], lh[i], 'meas', wa=best[0], wk=best[1]) for i in range(len(gth))])
    e0 = np.array([off(ph[i], gth[i]) for i in range(len(gth))]); e1 = Eh[best]; diff = e1 - e0
    us = np.unique(sh); per = {s: diff[sh == s] for s in us}; rng = np.random.default_rng(0)
    bs = [np.concatenate([per[s] for s in rng.choice(us, len(us))]).mean() for _ in range(2000)]
    rep['heldout_delta'] = dict(mean=float(diff.mean()), rel=float(diff.mean() / e0.mean()),
                                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    # controller in the loop, same controller/vehicle as closedloop_sim.py
    g = json.load(open(R + 'closedloop_gains.json'))['gains']; x0 = cl.initial(lh)
    m0 = cl.metrics(ph, gth, x0, g)[0]; m1 = cl.metrics(Pp, gth, x0, g)[0]
    d2 = m1['E_exec'] - m0['E_exec']; per = {s: d2[sh == s] for s in us}
    bs = [np.concatenate([per[s] for s in rng.choice(us, len(us))]).mean() for _ in range(2000)]
    rep['closedloop'] = dict(plan={q: float(v.mean()) for q, v in m0.items()}, proj={q: float(v.mean()) for q, v in m1.items()},
                             delta=dict(mean=float(d2.mean()), rel=float(d2.mean() / m0['E_exec'].mean()),
                                        ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
    np.save(R + f'proj_trainsel_{tag}.npy', Pp)
    json.dump(rep, open(R + f'weight_select_train_{tag}.json', 'w'), indent=1)
    print(tag, 'pick', best, 'edge', rep['pick_on_grid_edge'], 'train none', round(base_t, 4), 'train pick', round(Et[best].mean(), 4))
    print(' held-out none', round(base_h, 4), 'pick', round(rep['heldout_pick'], 4), 'oracle', round(rep['heldout_oracle'], 4), rep['heldout_delta'])
    print(' closed loop plan', round(m0['E_exec'].mean(), 4), 'proj', round(m1['E_exec'].mean(), 4), rep['closedloop']['delta'],
          'track', round(m0['track_rms'].mean(), 3), round(m1['track_rms'].mean(), 3), 'jerk', round(m0['jerk_rms'].mean(), 2), round(m1['jerk_rms'].mean(), 2))

if __name__ == '__main__':
    main(*sys.argv[1:4])
