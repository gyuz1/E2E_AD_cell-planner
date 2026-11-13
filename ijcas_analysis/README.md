# 이 압축 파일 안내 (다른 AI용)

목적: 학습 기반 end-to-end 주행 planner의 출력 궤적을 차량 운동학 모델의 도달 가능 집합으로 투영(projection)하는 연구의
분석 코드와 측정 결과. 논문 주제(분석 중심 재구성)를 함께 검토하기 위한 자료.

먼저 읽을 것
1. `ijcas_analysis/RESULTS_DIGEST.md` — 설정, 측정 결과 요약표, 아직 안 한 것, 원자료 JSON 전체.
2. 원고(LaTeX)는 이 저장소에 포함하지 않았다.

코드 (ijcas_analysis/)
- `fast_projection.py` — 투영 solver(projected Levenberg–Marquardt, numba, 정확한 Jacobian). `resjac`이 잔차·Jacobian,
  `project_state`가 상태 행에서 투영. rollout 초기조건은 v0, psi0뿐이고 a0, omega0는 첫 rate 항 기준값.
- `deep_analysis.py` — 오차 종·횡 분해, 초기 속도 불일치 상관, 학습/검증 비교, 투영 정리 확인, oracle 가중치, 상태 전달량.
- `closedloop_sim.py`, `closedloop_sweep.py` — 제어기 포함 평가(동역학 차량 모델, 50 Hz 추종 제어기, 비반응 재생).
- `weight_select_train.py` — 학습 scene으로 가중치 선택(부정 결과).
- `final_tables.py`, `weight_cv.py`, `weight_vs_noise.py`, `sens_weights.py` — 원고 표 수치, scene 교차검증, 잡음 민감도.
- 나머지는 이전 단계 분석.

실행에 필요한 것 (압축에 없음)
- 입력 덤프 `*_s5_3frame.npz`(planner 출력, GT, 상태)는 데이터셋 파생물이라 포함하지 않았다. 코드는 읽기용.
- 결과는 `ijcas_analysis/results/*.json`에 모두 있다.

표기: 수치는 모두 측정값. 가설은 RESULTS_DIGEST.md에서 [가설]로 표시.
