<div align="center">

# Kinematic Projection

**Kinematic Projection of Learned Trajectory Plans: Conditions for Improving Planning and Execution Accuracy**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official code for the IJCAS paper

</div>

## Table of Contents

- [Introduction](#introduction)
- [Results](#results)
- [Getting Started](#getting-started)
- [Paper and Scripts](#paper-and-scripts)
- [Repository](#repository)
- [Limitations](#limitations)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Introduction

A frozen learned end-to-end planner outputs six waypoints over 3 s. Its plans often
start at a speed the vehicle cannot reach from its current state. The interface studied
here projects each plan onto the reachable set of a kinematic vehicle model, starting
from the current speed and course angle, and passes the projected plan to a tracking
controller. The planner is not retrained.

<p align="center">
  <img src="assets/fig_architecture.png" width="100%" alt="System overview">
</p>

The code answers when this projection helps:

1. **Diagnosis.** The error of the learned plans is mainly along the path and comes from
   an inconsistent initial speed; the projection removes this part and slightly increases
   the path-shape error.
2. **Conditions.** The gain depends on the accuracy of the state, on how well the
   recorded motion fits the vehicle model, and on the rate weights. A run-time selector
   picks the weight per window from signals available at run time.
3. **Execution.** Plans are driven by the same controller on a dynamic vehicle model, so
   the projection is judged on the executed motion, not only on the plan.

## Results

75 held-out scenes of the ETRI E2E Driving Dataset (3777 moving windows), three planners
of one architecture. Planning error E_plan (m), measured state, weights chosen by
five-fold scene cross-validation (CV):

| Planner | Learned plan | Projected (CV) | Change |
|---|---|---|---|
| No-cell (baseline) | 0.3299 | 0.2476 | −25% |
| 15-cell | 0.2857 | 0.2216 | −22% |
| **105-cell** | **0.2793** | **0.2062** | **−26%** |

Controller-in-the-loop, 105-cell planner, measured state:

| Reference | Executed error (m) | Tracking RMS (m) | Jerk RMS (m/s³) |
|---|---|---|---|
| Learned plan | 0.227 | 0.086 | 0.48 |
| Projected, fixed weights | 0.251 | 0.054 | 0.79 |
| Projected, CV | 0.204 | 0.034 | 0.16 |
| **Projected, selector** | **0.190** | 0.038 | 0.23 |
| Constant acceleration (no images) | 0.280 | 0.031 | 0.05 |

From the network's own state estimate the planning gain shrinks to 4–10%, and the
executed-error gain is not significant. Intervals, per-command results and the separate
scene split are in the paper and in [`analysis/results/`](analysis/results).

## Getting Started

```bash
pip install -r requirements.txt     # CPU only: numpy, scipy, numba, scikit-learn, matplotlib, pillow
```

The scripts read planner-output dumps (`.npz`), one per planner, with per-window fields
`pred` (learned plan, 6×2), `gt` (recorded future positions), `ego_lcf` (measured
state), `state_pred` (network-estimated state), `mode` (command), `scene` and `frame`.
The dumps are derived from the dataset and are not redistributed; every number in the
paper is stored in [`analysis/results/`](analysis/results).

```bash
# planning error and CV weights (Table 1); writes the projected plans used later
python analysis/final_tables.py <heldout.npz> cell105

# decomposition, transmission, training vs held-out
python analysis/deep_analysis.py <heldout.npz> <train.npz> cell105

# controller-in-the-loop (Table 6); --tune fits the controller gains on recorded driving
python analysis/closedloop_sim.py <heldout.npz> cell105

# figures (written to figures/)
python analysis/plot_closedloop.py <heldout.npz> cell105
```

Run `final_tables.py` and `deep_analysis.py` first; the other scripts read their
intermediate `.npz` files. Tags: `cell105`, `cell15`, `nocell`.

## Paper and Scripts

| Paper | Script | Result |
|---|---|---|
| Sec. 3.2 input bounds | `cache_annotations.py`, `feasibility_envelope.py` | `feasibility_envelope.json` |
| Sec. 4.4 solver check | `projection_eval.py`, `fast_projection.py` | `projection_eval_s5.json`, `fast_projection_s5.json` |
| Table 1, Sec. 5.2 | `final_tables.py`, `weight_cv.py` | `final_*.json`, `weight_cv_cell105.json` |
| Sec. 5.2 mismatch, training vs held-out | `deep_analysis.py`, `stats_extra.py` | `deep_*.json`, `stats_extra_*.json` |
| Sec. 5.2 temporal structure, 10 Hz | `temporal_state.py`, `rep10hz.py`, `extra_checks.py` | `temporal_state_*.json`, `rep10hz_*.json`, `extra_checks_cell105.json` |
| Table 2 | `deep_analysis.py`, `split_terms.py` | `deep_*.json`, `split_*.json` |
| Table 3, Sec. 5.4 | `split_terms.py`, `stats_extra.py` | `split_*.json`, `stats_extra_*.json` |
| Table 4 | `weight_vs_noise.py`, `stats_extra.py` | `weight_vs_noise_cell105.json` |
| Sec. 5.5 transmission | `deep_analysis.py`, `stats_extra.py` | `deep_*.json`, `stats_extra_*.json` |
| Sec. 5.6 weights | `weight_cv.py`, `weight_select_train.py` | `weight_cv_cell105.json`, `weight_select_train_*.json` |
| Table 5 | `selector_cv.py` | `selector_*.json` |
| Table 6, Sec. 5.7 | `closedloop_sim.py`, `stats_extra.py`, `selector_closedloop.py`, `selector_vs_cv_ci.py`, `closedloop_sweep.py`, `extra_checks.py` | `closedloop_*.json`, `selector_closedloop_*.json`, `selector_vs_cv_ci_*.json` |
| Table 7 | `locked_test.py`, `locked_repeat.py` | `locked_test_*.json`, `locked_repeat_*.json` |
| Fig. 4 | `plot_consistency_b.py` | `correction_quintiles_cell105.json` |
| Fig. 5 | `plot_qualitative.py` | `fig_qualitative_cell105.json` |
| Fig. 6 | `plot_tradeoff.py` | from `stats_extra_*.json` |
| Fig. 7 | `plot_closedloop.py` | `fig_example_cell105.json` |

## Repository

```
analysis/
  fast_projection.py        projection solver: projected Levenberg–Marquardt,
                            Jacobian propagated through the Euler rollout (numba)
  trackability_precheck.py  input box U and the same problem with SciPy (reference solver)
  kinematics.py             finite-difference speed, curvature, acceleration, jerk
  closedloop_sim.py         dynamic single-track vehicle, 50 Hz tracking controller
  smoothing_baseline.py     planning error E_plan; polynomial smoothing baseline
  deep_analysis.py          error decomposition, transmission, scene bootstrap
  ...                       one entry point per table or figure (see above)
  results/                  measured results (JSON) reported in the paper
figures/                    output of the plotting scripts
```

## Limitations

- The controller-in-the-loop evaluation simulates only the ego vehicle with a generic,
  not identified, vehicle model; other road users are not simulated. It evaluates the
  planning–control interface, not driving performance or safety.
- The input bounds are the envelope of recorded human driving, not the limits of the
  test vehicle.
- The held-out scenes were used to choose the weights and the selector by
  cross-validation; a separate tuning/evaluation split confirms the main gains, but its
  evaluation scenes are not an independent test set.
- All results come from one dataset and three planners of one architecture.

## Citation

```bibtex
@article{kinematicprojection2026,
  title   = {Kinematic Projection of Learned Trajectory Plans: Conditions for Improving Planning and Execution Accuracy},
  journal = {International Journal of Control, Automation, and Systems},
  year    = {2026},
  note    = {under review}
}
```

## Acknowledgement

The learned planners build on [VAD](https://github.com/hustvl/VAD) and
[LAW](https://github.com/BraveGroup/LAW). Data: ETRI E2E Driving Dataset.
