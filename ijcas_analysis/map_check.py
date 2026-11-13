"""Quantitative check of the MAP reading rho_a^2 ~ sigma_plan^2 / sigma_da^2 (and rho_k likewise)
on training and held-out scenes. sigma_plan: waypoint residual of the learned plan (per coordinate,
per waypoint, RMS). sigma_da / sigma_dk: changes of acceleration / curvature of the recorded
driving between 0.5 s intervals, from finite differences of the recorded waypoints, starting from
the measured state (same discretization as the rate terms of the projection).
usage: map_check.py <train_dump> <heldout_dump> <tag>"""
import sys, json, numpy as np
R = __file__.rsplit('/', 1)[0] + '/results/'
TAU = 0.5

def stats(dump):
    d = np.load(dump); i = np.where(d['mode'] <= 4)[0]
    P, Y, L = d['pred'][i].astype(float), d['gt'][i].astype(float), d['ego_lcf'][i].astype(float)
    e = P - Y                                            # (n,6,2)
    sig_t = np.sqrt((e ** 2).mean((0, 2)))               # per waypoint, per coordinate RMS
    sig_plan = float(np.sqrt((e ** 2).mean()))           # pooled
    E = e.reshape(len(e), 6, 2).transpose(0, 2, 1).reshape(-1, 6)   # rows: (window, coord)
    C = np.corrcoef(E.T)
    Yz = np.concatenate([np.zeros((len(Y), 1, 2)), Y], 1)
    seg = np.diff(Yz, axis=1); v = np.linalg.norm(seg, axis=-1) / TAU                 # mean speed per interval
    v0 = L[:, 7]; a0 = L[:, 2]
    a = np.empty_like(v); a[:, 0] = (v[:, 0] - v0) / (TAU / 2); a[:, 1:] = np.diff(v, axis=1) / TAU
    da = np.diff(np.concatenate([a0[:, None], a], 1), axis=1)
    hd = np.arctan2(seg[..., 1], seg[..., 0]); ok = v > 2.0
    k = np.full_like(v, np.nan); k[:, 1:] = np.diff(np.unwrap(hd, axis=1), axis=1) / (TAU * np.maximum(v[:, 1:], 2.0))
    k0 = L[:, 4] / np.maximum(v0, 2.0); k[:, 0] = np.nan
    dk = np.diff(np.concatenate([k0[:, None], k], 1), axis=1)
    m = ok[:, 1:] & ok[:, :-1]
    return dict(n=int(len(i)), sigma_plan=sig_plan, sigma_plan_by_t=sig_t.round(4).tolist(),
                corr_adjacent=float(np.mean([C[j, j + 1] for j in range(5)])), corr_1_6=float(C[0, 5]),
                sigma_da=float(np.sqrt(np.nanmean(da ** 2))), sigma_da_by_t=np.sqrt(np.nanmean(da ** 2, 0)).round(3).tolist(),
                sigma_dk=float(np.sqrt(np.nanmean(dk[:, 1:][m] ** 2))))

def main(td, hd, tag):
    rep = dict(tag=tag)
    for name, dump in (('train', td), ('heldout', hd)):
        s = stats(dump); s['rho_a_pred'] = s['sigma_plan'] / s['sigma_da']; s['rho_k_pred'] = s['sigma_plan'] / s['sigma_dk']
        rep[name] = s
    ws = json.load(open(R + f'weight_select_train_{tag}.json')); fj = json.load(open(R + f'final_{tag}.json'))
    rep['best_train'] = ws['pick']; rep['best_heldout_cv'] = fj['cv_picks']['meas'][0]
    rep['ratio_pred_rho_a_heldout_over_train'] = rep['heldout']['rho_a_pred'] / rep['train']['rho_a_pred']
    rep['ratio_best_rho_a_heldout_over_train'] = rep['best_heldout_cv'][0] / rep['best_train'][0]
    rep['ratio_variance_plan'] = (rep['heldout']['sigma_plan'] / rep['train']['sigma_plan']) ** 2
    json.dump(rep, open(R + f'map_check_{tag}.json', 'w'), indent=1); print(json.dumps(rep, indent=1))

if __name__ == '__main__':
    main(*sys.argv[1:4])
