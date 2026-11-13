"""Finite-difference kinematic quantities of a 6-step, 0.5-s planned trajectory.

Used for the data envelope U (eq. 3) AND, later, for checking candidates, so
both go through exactly the same estimator.

Input: cumulative positions P (N,6,2) in the ego frame at t (x forward, y left);
the trajectory starts at the origin with heading 0.
  segment s_j = p_{j+1} - p_j, j = 0..5 (p_0 = origin)       -> 6 segments
  speed   v_j = |s_j| / tau                                   -> 6
  heading th_j = atan2(s_j), th_{-1} = 0 (ego heading at t)
  curvature k_j = wrap(th_j - th_{j-1}) / |s_j|                -> 6 (j=0 uses th_{-1}=0;
                  a half-step approximation, see paper)
  lateral acc  a_lat_j = v_j^2 k_j                             -> 6
  longitudinal acc a_j = (v_{j+1} - v_j)/tau                  -> 5
  jerk j_j = (a_{j+1} - a_j)/tau                               -> 4
Curvature (and so lateral acceleration) is undefined when a segment is nearly
zero length; those entries are NaN when v_j < V_MIN.
"""
import numpy as np

TAU = 0.5
V_MIN = 2.0  # m/s; below this one 0.5-s segment is < 1 m and heading is noise


def quantities(P, tau=TAU, v_min=V_MIN):
    P = np.asarray(P, float).reshape(-1, 6, 2)
    pts = np.concatenate([np.zeros((len(P), 1, 2)), P], 1)
    seg = np.diff(pts, axis=1)                       # (N,6,2)
    ln = np.linalg.norm(seg, axis=-1)                # (N,6)
    v = ln / tau
    th = np.arctan2(seg[..., 1], seg[..., 0])
    th_prev = np.concatenate([np.zeros((len(P), 1)), th[:, :-1]], 1)
    dth = (th - th_prev + np.pi) % (2 * np.pi) - np.pi
    with np.errstate(divide='ignore', invalid='ignore'):
        k = dth / ln
    low = v < v_min
    # heading of a near-zero segment is meaningless, and so is the turn INTO the next one
    low_prev = np.concatenate([np.zeros((len(P), 1), bool), low[:, :-1]], 1)
    k = np.where(low | low_prev, np.nan, k)
    a = np.diff(v, axis=1) / tau
    j = np.diff(a, axis=1) / tau
    return dict(v=v, kappa=k, a_lat=v ** 2 * k, a_lon=a, jerk=j)


def violations(q, U):
    """Per-trajectory boolean: any step outside the envelope U (dict of bounds)."""
    out = {}
    with np.errstate(invalid='ignore'):
        out['kappa'] = np.nan_to_num(np.abs(q['kappa']) > U['kappa'], nan=0).any(1)
        out['a_lat'] = np.nan_to_num(np.abs(q['a_lat']) > U['a_lat'], nan=0).any(1)
        out['a_lon'] = ((q['a_lon'] > U['a_max']) | (q['a_lon'] < U['a_min'])).any(1)
        out['jerk'] = (np.abs(q['jerk']) > U['jerk']).any(1)
    out['any'] = out['kappa'] | out['a_lat'] | out['a_lon'] | out['jerk']
    return out
