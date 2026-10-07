"""How fast the target point's value decays with prediction error (linear proxy, see goal_grid_value.py)."""
import sys; sys.path.insert(0,'/workspace/VAD')
import numpy as np
from tools.goal_grid_value import load, grouped, official_l2, kin_tp
Xtr,Ytr,ctr,ttr,ltr = load('train'); Xte,Yte,cte,tte,lte = load('val')
base = official_l2(grouped(Xtr,Ytr,ctr,Xte,cte),Yte)
rng = np.random.default_rng(0)
res_tr = ttr - kin_tp(ltr); res_te = tte - kin_tp(lte)
print('command only', round(base,4))
print('TP residual over const-accel: |res| mean val %.2f m, std x %.2f y %.2f' % (np.linalg.norm(res_te,axis=1).mean(), res_te[:,0].std(), res_te[:,1].std()))
print('kinematic TP as input (no new info): %.4f' % official_l2(grouped(np.c_[Xtr,kin_tp(ltr)],Ytr,ctr,np.c_[Xte,kin_tp(lte)],cte),Yte))
print(f'{"TP error sigma (m, per axis)":<30}{"L2":>8}{"gain":>8}  (same noise in fit and eval)')
for s in [0,1,2,3,4,6,8,12]:
    ntr = ttr + rng.normal(0,s,ttr.shape); nte = tte + rng.normal(0,s,tte.shape)
    l = official_l2(grouped(np.c_[Xtr,ntr],Ytr,ctr,np.c_[Xte,nte],cte),Yte)
    print(f'{s:<26}{l:>8.4f}{base-l:>+8.4f}')
# x only (forward distance) vs y only
for nm,cols in [('TP x only',[0]),('TP y only',[1])]:
    l = official_l2(grouped(np.c_[Xtr,ttr[:,cols]],Ytr,ctr,np.c_[Xte,tte[:,cols]],cte),Yte)
    print(f'{nm:<26}{l:>8.4f}{base-l:>+8.4f}')
