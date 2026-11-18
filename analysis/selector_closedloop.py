"""Executed error of the selector vs the fixed CV weight (same controller, measured initial state).
usage: selector_closedloop.py <heldout_dump> <tag>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import closedloop_sim as cl
from deep_analysis import boot
R = __file__.rsplit('/', 1)[0] + '/results/'
dump, tag = sys.argv[1], sys.argv[2]
d = np.load(dump); z = np.load(R + f'selector_{tag}_meas.npz'); idx = z['idx']
gt = d['gt'][idx].astype(float); lcf = d['ego_lcf'][idx].astype(float); scene = d['scene'][idx]
g = json.load(open(R + 'closedloop_gains.json'))['gains']; x0 = cl.initial(lcf); out = dict(tag=tag)
M = {}
for k in ('Q_fixed', 'Q_sel'):
    m, *_ = cl.metrics(z[k], gt, x0, g); M[k] = m
    out[k] = {q: float(np.mean(m[q])) for q in ('E_exec', 'E_ref', 'track_rms', 'jerk_rms', 'sat_frac')}
dd = M['Q_sel']['E_exec'] - M['Q_fixed']['E_exec']; out['delta_exec'] = float(dd.mean()); out['ci'] = boot(dd, scene)
json.dump(out, open(R + f'selector_closedloop_{tag}.json', 'w'), indent=1); print(out)
