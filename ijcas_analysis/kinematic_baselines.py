"""Image-free kinematic extrapolations on the same moving windows:
constant acceleration along the heading (CA) and constant turn rate and
acceleration (CTRA), from the measured or the network-estimated state."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from smoothing_baseline import off

def ctra(v0, a, w, turn=True, T=3.0, dt=0.01):
    x = y = psi = 0.0; v = v0; out = []; t = 0.0; nxt = 0.5
    for k in range(int(round(T / dt))):
        x += dt * v * np.cos(psi); y += dt * v * np.sin(psi)
        if turn: psi += dt * w
        v = max(v + dt * a, 0.0); t += dt
        if t >= nxt - 1e-9: out.append((x, y)); nxt += 0.5
    return np.array(out)

d = np.load(sys.argv[1]); idx = np.where(d['mode'] <= 4)[0]; gt = d['gt'][idx]
sp, lcf = d['state_pred'][idx], d['ego_lcf'][idx]
rep = {}
for src, st, vc in (('meas', lcf, 7), ('est', sp, 5)):
    for name, turn in (('CA', False), ('CTRA', True)):
        rep[f'{name}_{src}'] = float(np.mean([off(ctra(st[i, vc], st[i, 2], st[i, 4], turn), gt[i]) for i in range(len(idx))]))
print(json.dumps(rep, indent=1)); json.dump(rep, open(__file__.rsplit('/', 1)[0] + f'/results/kinematic_baselines_{sys.argv[2]}.json', 'w'), indent=1)
