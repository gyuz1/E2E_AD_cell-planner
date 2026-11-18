<div align="center">
<h1>Kinematic Projection</h1>
<h3>Kinematic Projection of Learned Trajectory Plans: Conditions for Improving Planning and Execution Accuracy</h3>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)&nbsp;
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)&nbsp;
![CPU only](https://img.shields.io/badge/Hardware-CPU%20only-lightgrey.svg)

</div>

## News
* **` Nov. 2026`:** We release the projection solver, the analysis scripts and all measured results of the paper.

## Table of Contents
- [Introduction](#introduction)
- [Main Results](#main-results)
- [Demo](#demo)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Acknowledgement](#acknowledgement)

## Introduction
Learned end-to-end planners predict future waypoints directly from sensor data, but their
plans often start at a speed that the vehicle cannot reach from its current state. We place
a kinematic projection between a frozen learned planner and the tracking controller: each
plan is mapped onto the set of trajectories that a kinematic vehicle model can realize from
the current speed and course angle, under bounds on acceleration and curvature. Instead of
proposing a new planner, we study **when this projection helps and when it hurts**. With the
measured vehicle state, the projection reduces the planning error of three learned planners
by 22–26% and the error of the executed motion by 10–16%; under scene cross-validation, a
run-time weight selector lowers both further. Most of the gain comes from correcting the
speed along the path, and it shrinks when the planner's own state estimate is used instead
of the measured state.

<div align="center">
<img src="assets/fig_architecture.png" />
</div>

## Main Results

> Planning error E<sub>plan</sub> (m) on 75 held-out scenes, projection from the measured state with cross-validated weights

| Planner | Learned plan | Projected | Change |
| :---: | :---: | :---: | :---: |
| No-cell (baseline) | 0.330 | 0.248 | −25% |
| 15-cell | 0.286 | 0.222 | −22% |
| 105-cell | 0.279 | **0.206** | **−26%** |

> Controller-in-the-loop evaluation, 105-cell planner

| Reference | Executed error (m) | Tracking RMS (m) | Jerk RMS (m/s³) |
| :---: | :---: | :---: | :---: |
| Learned plan | 0.227 | 0.086 | 0.48 |
| Projected | 0.204 | **0.034** | **0.16** |
| Projected + selector | **0.190** | 0.038 | 0.23 |


## Demo
The three held-out windows shown in the paper (lane keeping, lane change and turn) are
included in [`demo/`](demo) with their front-camera images. The demo projects the learned
plan of each window from the measured state and draws the result.

```bash
pip install -r requirements.txt
python demo/demo.py
```

```
Lane keeping  v0 =  15.1 m/s   E_plan: learned 0.416 m -> projected 0.168 m
Lane change   v0 =   9.0 m/s   E_plan: learned 0.547 m -> projected 0.339 m
Turn          v0 =   5.1 m/s   E_plan: learned 0.524 m -> projected 0.381 m
```

## Repository Structure
```
analysis/
├── fast_projection.py    # kinematic projection solver
├── closedloop_sim.py     # controller-in-the-loop evaluation
├── selector_cv.py        # run-time weight selector
├── ...                   # analysis and plotting scripts
└── results/              # measured results reported in the paper
demo/
├── demo.py               # projection of the three example windows
├── demo_windows.npz      # learned plans, recorded trajectories and vehicle states
├── cameras.json          # front-camera calibration
└── images/               # front-camera images
```

## Dataset
All experiments use the **ETRI E2E Driving Dataset** provided by the
[Electronics and Telecommunications Research Institute (ETRI)](https://www.etri.re.kr).
It contains 376 driving scenes recorded at 10 Hz with six surround-view cameras, recorded
vehicle motion and future trajectories. We split it by scene into 301 training and 75
held-out scenes. The dataset is not redistributed in this repository; only the three example windows used in
the paper are included for the demo.

## Acknowledgement
The learned planners in this work are built on [VAD](https://github.com/hustvl/VAD) and
[LAW](https://github.com/BraveGroup/LAW). We thank the authors for releasing their code, and
ETRI for providing the dataset.
