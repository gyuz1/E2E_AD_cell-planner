"""Kinematic projection of the selected plan, evaluated on a dump.
For each moving-command window: selected plan -> best trajectory of the kinematic
vehicle model (trackability_precheck.track) from the vision-estimated or measured
state.  Also the smoothing controls.  Per command, plus scene-level bootstrap CI
of the improvement.  Writes results/projection_eval_<tag>.json and a per-window npz."""
import sys, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from trackability_precheck import track
from smoothing_baseline import fit, off

def job(a):
    P, v0, a0, k0 = a
    return track(P, v0, a0, k0)[1]

def main(path, tag, workers=6):
    d = np.load(path); idx = np.where(d['mode'] <= 4)[0]
    sp, lcf = d['state_pred'], d['ego_lcf']
    mk = lambda i, s: (d['pred'][i].astype(float), s[i, -1 if s is sp else 7], s[i, 2], s[i, 4] / max(s[i, -1 if s is sp else 7], 2.0))
    with ProcessPoolExecutor(workers) as ex:
        Pobs = list(ex.map(job, [mk(i, sp) for i in idx], chunksize=16))
        Pmeas = list(ex.map(job, [mk(i, lcf) for i in idx], chunksize=16))
    gt = d['gt'][idx]
    E = {'none': [off(d['pred'][i], d['gt'][i]) for i in idx],
         'poly4': [off(fit(d['pred'][i], 4), d['gt'][i]) for i in idx],
         'poly4_v0_obs': [off(fit(d['pred'][i], 4, sp[i, :2]), d['gt'][i]) for i in idx],
         'poly4_v0_meas': [off(fit(d['pred'][i], 4, lcf[i, :2]), d['gt'][i]) for i in idx],
         'proj_obs': [off(P, g) for P, g in zip(Pobs, gt)],
         'proj_meas': [off(P, g) for P, g in zip(Pmeas, gt)]}
    E = {k: np.array(v) for k, v in E.items()}
    cmd = d['mode'][idx]; scene = d['scene'][idx]
    names = ['LK', 'LCL', 'LCR', 'TL', 'TR']
    rep = {'n': int(len(idx)), 'overall': {k: float(v.mean()) for k, v in E.items()},
           'by_cmd': {names[c]: dict(n=int((cmd == c).sum()), **{k: float(v[cmd == c].mean()) for k, v in E.items()}) for c in range(5)}}
    rng = np.random.default_rng(0); us = np.unique(scene)
    for k in ('proj_obs', 'proj_meas', 'poly4_v0_obs'):
        diff = E[k] - E['none']; per = {s: diff[scene == s] for s in us}; bs = []
        for _ in range(2000):
            ss = rng.choice(us, len(us)); bs.append(np.concatenate([per[s] for s in ss]).mean())
        rep[f'ci95_delta_{k}'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        rep[f'frac_windows_improved_{k}'] = float((diff < 0).mean())
    print(json.dumps(rep, indent=1))
    base = __file__.rsplit('/', 1)[0] + '/results/'
    json.dump(rep, open(base + f'projection_eval_{tag}.json', 'w'), indent=1)
    np.savez_compressed(base + f'projection_{tag}.npz', idx=idx, proj_obs=np.array(Pobs), proj_meas=np.array(Pmeas), **{f'E_{k}': v for k, v in E.items()})

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
