"""Planning-error replication on every 10 Hz frame for one planner.
usage: rep10hz.py <s1_dump> <tag> <wa_meas> <wk_meas> <wa_est> <wk_est>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
from smoothing_baseline import off
from deep_analysis import boot
R = __file__.rsplit('/', 1)[0] + '/results/'
dump, tag = sys.argv[1:3]; wm, km, we, ke = map(float, sys.argv[3:7])
d = np.load(dump); i = np.where(d['mode'] <= 4)[0]
P = d['pred'][i].astype(float); Y = d['gt'][i].astype(float); L = d['ego_lcf'][i].astype(float); S = d['state_pred'][i].astype(float); sc = d['scene'][i]
EP = np.array([off(P[k], Y[k]) for k in range(len(P))])
Em = np.array([off(project_state(P[k], L[k], 'meas', wa=wm, wk=km), Y[k]) for k in range(len(P))])
Ee = np.array([off(project_state(P[k], S[k], 'est', wa=we, wk=ke), Y[k]) for k in range(len(P))])
rep = dict(tag=tag, n=int(len(P)), weights_meas=[wm, km], weights_est=[we, ke], E_none=float(EP.mean()), E_meas=float(Em.mean()),
           ci_meas=boot(Em - EP, sc), E_est=float(Ee.mean()), ci_est=boot(Ee - EP, sc))
json.dump(rep, open(R + f'rep10hz_{tag}.json', 'w'), indent=1); print(rep)
