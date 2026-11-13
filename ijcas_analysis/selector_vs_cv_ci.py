"""Paired scene-bootstrap CIs of selector minus fixed-CV projection in the closed loop
(executed error, tracking RMS, jerk RMS, share of windows with saturation). Measured state.
usage: selector_vs_cv_ci.py <heldout_dump> <tag>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import closedloop_sim as cl
from deep_analysis import boot
R = __file__.rsplit('/', 1)[0] + '/results/'
dump, tag = sys.argv[1:3]
d = np.load(dump); z = np.load(R + f'selector_{tag}_meas.npz'); idx = z['idx']
gt = d['gt'][idx].astype(float); L = d['ego_lcf'][idx].astype(float); sc = d['scene'][idx]
g = json.load(open(R + 'closedloop_gains.json'))['gains']; x0 = cl.initial(L)
cv = np.load(R + f'deep_{tag}.npz')['P_meas_cv']
M = {}
for k, W in (('cv', cv), ('sel', z['Q_sel'])):
    m, *_ = cl.metrics(W, gt, x0, g); M[k] = dict(E_exec=m['E_exec'], track=m['track_rms'], jerk=m['jerk_rms'], sat=(m['sat_frac'] > 0).astype(float))
out = {q: dict(cv=float(M['cv'][q].mean()), sel=float(M['sel'][q].mean()), diff=float((M['sel'][q] - M['cv'][q]).mean()),
               ci=boot(M['sel'][q] - M['cv'][q], sc)) for q in M['cv']}
json.dump(out, open(R + f'selector_vs_cv_ci_{tag}.json', 'w'), indent=1); print(tag, json.dumps(out))
