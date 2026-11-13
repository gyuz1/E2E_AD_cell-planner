"""Check of Proposition (nonexpansive projection): with the box-constrained,
linear-in-inputs approximation, projecting onto the reachable set C(z0) of the
TRUE state never increases ||P - gt|| when gt in C(z0); with a state error the
increase is at most dist(gt, C(z0_hat)) <= ||s_v|| |dv| (+ linearization).
Here: pure projection (no rate terms, hard input box only)."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fast_projection import project_state
d = np.load(sys.argv[1]); idx = np.where(d['mode'] <= 4)[0]
sp, lcf = d['state_pred'][idx], d['ego_lcf'][idx]; pred = d['pred'][idx].astype(float); gt = d['gt'][idx]
n = lambda X: np.sqrt((X ** 2).sum(-1).sum(-1))
out = {}
for src, st, vc in (('meas', lcf, 7), ('est', sp, 5)):
    P = np.stack([project_state(pred[i], st[i], src, wa=1e-6, wk=1e-6, ws=0.0) for i in range(len(idx))])
    G = np.stack([project_state(gt[i], lcf[i], 'meas', wa=1e-6, wk=1e-6, ws=0.0) for i in range(len(idx))])
    dist_gt = n(G - gt)                       # distance of gt to C(true state)
    Gu = np.stack([project_state(gt[i], st[i], src, wa=1e-6, wk=1e-6, ws=0.0) for i in range(len(idx))])
    dist_used = n(Gu - gt)                    # distance of gt to C(state actually used)
    e0, e1 = n(pred - gt), n(P - gt)
    dv = np.abs(st[:, vc] - lcf[:, 7]); tj = np.arange(1, 7) * 0.5; sv = np.sqrt((tj ** 2).sum())
    bound = e0 + dist_used                    # Proposition 1 (convex case)
    bound_lin = e0 + dist_gt + sv * dv        # Corollary (first order in dv)
    out[src] = dict(frac_increase=float((e1 > e0 + 1e-6).mean()), mean_e0=float(e0.mean()), mean_e1=float(e1.mean()),
                    frac_increase_gt_reachable=float((e1 > e0 + 1e-6)[dist_gt < 0.05].mean()), n_gt_reachable=int((dist_gt < 0.05).sum()),
                    frac_bound_violated=float((e1 > bound + 1e-6).mean()), frac_bound_lin_violated=float((e1 > bound_lin + 1e-6).mean()),
                    mean_excess_when_violated=float(np.mean((e1 - bound)[e1 > bound + 1e-6])) if (e1 > bound + 1e-6).any() else 0.0,
                    dist_used_mean=float(dist_used.mean()), sv_norm=float(sv), gt_dist_mean=float(dist_gt.mean()), gt_dist_p95=float(np.percentile(dist_gt, 95)))
print(json.dumps(out, indent=1)); json.dump(out, open(__file__.rsplit('/', 1)[0] + f'/results/projection_theory_check_{sys.argv[2]}.json', 'w'), indent=1)
