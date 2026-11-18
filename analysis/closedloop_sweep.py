"""Dependence of the controller-in-the-loop result on controller stiffness and actuator lag.
usage: closedloop_sweep.py <dump.npz> <tag>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import closedloop_sim as cl
from fast_projection import project_state
R = cl.R

def main(dump, tag):
    d = np.load(dump); f = np.load(R + f'final_{tag}.npz'); rep0 = json.load(open(R + f'final_{tag}.json'))
    idx = f['idx']; gt = d['gt'][idx].astype(float); pred = d['pred'][idx].astype(float)
    lcf = d['ego_lcf'][idx].astype(float); scene = d['scene'][idx]; x0 = cl.initial(lcf)
    us0 = np.unique(scene); folds = np.array_split(np.random.default_rng(0).permutation(us0), 5)
    Pcv = np.empty_like(pred)
    for fo, (wa, wk) in zip(folds, rep0['cv_picks']['meas']):
        for i in np.where(np.isin(scene, fo))[0]:
            Pcv[i] = project_state(pred[i], lcf[i], 'meas', wa=wa, wk=wk)
    refs = dict(plan=pred, proj_meas=f['proj_meas'], proj_meas_cv=Pcv, gt=gt)
    g0 = np.array(json.load(open(R + 'closedloop_gains.json'))['gains'], float)
    out = dict(tag=tag, base_gains=g0.tolist(), stiffness={}, tau_a={})
    for s in (0.25, 0.5, 1.0, 2.0, 4.0):
        out['stiffness'][s] = {k: {q: float(v.mean()) for q, v in cl.metrics(W, gt, x0, g0 * s)[0].items() if q in ('E_exec', 'track_rms', 'jerk_rms', 'sat_frac')} for k, W in refs.items()}
        print('stiff', s, {k: round(v['E_exec'], 4) for k, v in out['stiffness'][s].items()}, {k: round(v['track_rms'], 3) for k, v in out['stiffness'][s].items()}, flush=True)
    tau0 = cl.P['tau_a']
    for ta in (0.1, 0.3, 0.6):
        cl.P['tau_a'] = ta
        out['tau_a'][ta] = {k: {q: float(v.mean()) for q, v in cl.metrics(W, gt, x0, g0)[0].items() if q in ('E_exec', 'track_rms', 'jerk_rms', 'sat_frac')} for k, W in refs.items()}
        print('tau_a', ta, {k: round(v['E_exec'], 4) for k, v in out['tau_a'][ta].items()}, flush=True)
    cl.P['tau_a'] = tau0
    json.dump(out, open(R + f'closedloop_sweep_{tag}.json', 'w'), indent=1)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
