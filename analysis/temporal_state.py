"""Temporal structure of the network's state-estimation error at 10 Hz (stride-1 dump).
Per scene: error series of speed, longitudinal acceleration and yaw rate; bias, RMS, and the
autocorrelation of the error at lags 0.1-2 s (pooled over scenes, mean removed per scene).
Also the autocorrelation of the plan's first-interval speed mismatch e_v.
usage: temporal_state.py <stride1_dump> <tag>"""
import sys, json, numpy as np
R = __file__.rsplit('/', 1)[0] + '/results/'
TAU = 0.5
def acf(series, lags):
    num = {l: 0.0 for l in lags}; den = {l: 0 for l in lags}; var = 0.0; nv = 0
    for x in series:
        x = x - x.mean(); var += (x ** 2).sum(); nv += len(x)
        for l in lags:
            if len(x) > l: num[l] += (x[:-l] * x[l:]).sum(); den[l] += len(x) - l
    v = var / nv
    return {f'{l/10:.1f}s': float(num[l] / den[l] / v) for l in lags if den[l]}
def main(dump, tag):
    d = np.load(dump, allow_pickle=True)
    m = d['mode'] <= 4; sc = d['scene']; fr = d['frame']
    L = d['ego_lcf'].astype(float); S = d['state_pred'].astype(float); P = d['pred'].astype(float)
    err = dict(speed=S[:, 5] - L[:, 7], accel=S[:, 2] - L[:, 2], yaw_rate=S[:, 4] - L[:, 4])
    ev = np.linalg.norm(P[:, 0], axis=-1) / TAU - (L[:, 7] + L[:, 2] * TAU / 2)
    lags = [1, 2, 3, 5, 10, 20]
    rep = dict(tag=tag, n_frames=int(len(sc)), n_moving=int(m.sum()), n_scenes=int(len(np.unique(sc))))
    def per_scene(x):
        out = []
        for s in np.unique(sc):
            k = np.where((sc == s) & m)[0]; k = k[np.argsort(fr[k])]
            if len(k) > 25: out.append(x[k])
        return out
    for name, e in list(err.items()) + [('plan_speed_mismatch', ev)]:
        ser = per_scene(e); allv = np.concatenate(ser)
        bias_per_scene = np.array([x.mean() for x in ser])
        rep[name] = dict(rms=float(np.sqrt((allv ** 2).mean())), bias=float(allv.mean()),
                         sd_of_scene_bias=float(bias_per_scene.std()), share_var_from_scene_bias=float(bias_per_scene.var() / allv.var()),
                         acf=acf(ser, lags))
    # does the estimation error explain the plan mismatch at 10 Hz?
    rep['corr_ev_speed_err'] = float(np.corrcoef(ev[m], err['speed'][m])[0, 1])
    json.dump(rep, open(R + f'temporal_state_{tag}.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
