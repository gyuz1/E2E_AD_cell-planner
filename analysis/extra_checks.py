"""(a) Main planning-error result replicated on every frame of the 10 Hz dump (105-cell, measured and
estimated state, weights (5,5) and (2,1) as selected by CV on the 5-frame windows).
(b) Controller-in-the-loop evaluation of the estimated-state CV projection (vehicle starts from the
measured state). usage: extra_checks.py <s1_dump> <s5_dump>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
from smoothing_baseline import off
from deep_analysis import boot
import closedloop_sim as cl
R = __file__.rsplit('/', 1)[0] + '/results/'
s1, s5 = sys.argv[1:3]; rep = {}
d = np.load(s1); i = np.where(d['mode'] <= 4)[0]
P = d['pred'][i].astype(float); Y = d['gt'][i].astype(float); L = d['ego_lcf'][i].astype(float); S = d['state_pred'][i].astype(float); sc = d['scene'][i]
EP = np.array([off(P[k], Y[k]) for k in range(len(P))])
Em = np.array([off(project_state(P[k], L[k], 'meas', wa=5.0, wk=5.0), Y[k]) for k in range(len(P))])
Ee = np.array([off(project_state(P[k], S[k], 'est', wa=2.0, wk=1.0), Y[k]) for k in range(len(P))])
rep['all_frames_10hz'] = dict(n=int(len(P)), E_none=float(EP.mean()), E_meas=float(Em.mean()), ci_meas=boot(Em - EP, sc),
                              E_est=float(Ee.mean()), ci_est=boot(Ee - EP, sc))
print(rep, flush=True)
d = np.load(s5); f = np.load(R + 'final_cell105.npz'); dp = np.load(R + 'deep_cell105.npz'); idx = f['idx']
Y = d['gt'][idx].astype(float); L = d['ego_lcf'][idx].astype(float); sc = d['scene'][idx]; P = d['pred'][idx].astype(float)
g = json.load(open(R + 'closedloop_gains.json'))['gains']; x0 = cl.initial(L); M = {}
for k, W in (('plan', P), ('est_cv', dp['P_est_cv'])):
    m, *_ = cl.metrics(W, Y, x0, g); M[k] = m
rep['closedloop_est_cv'] = dict(E_exec_plan=float(M['plan']['E_exec'].mean()), E_exec_est_cv=float(M['est_cv']['E_exec'].mean()),
                                ci=boot(M['est_cv']['E_exec'] - M['plan']['E_exec'], sc), track=float(M['est_cv']['track_rms'].mean()),
                                jerk=float(M['est_cv']['jerk_rms'].mean()))
json.dump(rep, open(R + 'extra_checks_cell105.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
