#!/usr/bin/env bash
# 제출물 한 벌을 만든다. 세 단계를 한 곳에 묶어둔 이유는, 2026-09-23 에
# FLOPs 병합을 빠뜨린 채 제출해서 CUTOFF_RESULT_MISSING 으로 거절당했기 때문이다.
#
#   사용법: make_submission.sh <ckpt 파일명> <eval config 파일명> <태그>
set -euo pipefail
CKPT=$1; CFG=$2; TAG=$3
cd /workspace/VAD
OUT=submission_${TAG}.json

echo "### 1/3 궤적"
python tools/etri_test_submit.py \
    "projects/configs/VAD/${CFG}" "work_dirs/${CKPT}" \
    --ann-file data/etri/annotations_10hz/vad_etri_10hz_infos_temporal_test.pkl \
    --out "$OUT" \
    --fp16 --select-cell-by-tp --bev-only-history

echo "### 2/3 FLOPs 병합"
python tools/measure_flops.py \
    "projects/configs/VAD/${CFG}" \
    --ann-file data/etri/annotations_10hz/vad_etri_10hz_infos_temporal_test.pkl \
    --out "$OUT"

echo "### 3/3 zip + 검증"
# 컨테이너에 zip 실행파일이 없다. zipfile 로 만든다. 안쪽 이름은
# submission.json 으로 고정한다 -- 통과한 제출물이 그 이름이었다.
OUT="$OUT" TAG="$TAG" python tools/_pack_submission.py
