#!/bin/bash
# ------------------------------------------------------------------------
# YOLOv5 "NPU측 후처리(nms_core)" 단독 타이밍 벤치마크 실행 스크립트.
# infer_yolov5_hailo8.cpp 상단 주석 참고 — HEF는 hailo_model_zoo 공식 배포본
# (yolov5xs_wo_spp_nms_core.hef, engine=nn_core/auto)을 사용한다.
#
# 파라미터: batch=1, threshold=1, timeout=0ms, priority=0 (infer_yolov5_hailo8.cpp에 고정값,
# 사용자 지정 기본값과 동일 — 필요시 스크립트 상단이 아니라 .cpp의 #define 블록을 고칠 것).
#
# 실행 위치: 보드(rpi4, ~/hailo_cpp_test/), infer_yolov5_hailo8.cpp와 같은 디렉토리.
# 사용법: chmod +x run_yolov5_nms_core.sh && ./run_yolov5_nms_core.sh
# ------------------------------------------------------------------------
set -e
cd "$(dirname "$0")/.."   # hailo_8/ (infer_yolov5_hailo8.cpp가 있는 위치) 기준으로 이동

SRC=infer_yolov5_hailo8.cpp
BIN=infer_yolov5_hailo8
# [2026-08-07 추가] rpi4 보드에서 hailort.hpp가 표준 경로(/usr/include)가 아니라 이 경로에
# 있는 게 실기에서 확인됨 -I 없이 빌드하면 첫 include부터 못 찾아서 대량 연쇄 에러가 남.
# .so 자체는 /usr/lib/libhailort.so(ldconfig 캐시에 등록됨)라 -L은 필요 없음.
HAILORT_INCLUDE="/home/rpi4/hm/monitor/hailort/hailort/libhailort/include"
HEF_DIR="${HOME}/hailo_cpp_test/resources"
HEF_NAME=yolov5xs_wo_spp_nms_core.hef
HEF_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/${HEF_NAME}"
OUTDIR="experiments/$(date +%Y-%m-%d)_yolov5_nms_core_exp1"
CSV="${OUTDIR}/csv/results_yolov5_nms_core.csv"
REPEATS=3
START_RUN_ID=1

if [ ! -f "$SRC" ]; then
    echo "[오류] $SRC 를 찾을 수 없음. hailo_8/ 디렉토리에서 실행할 것."
    exit 1
fi

mkdir -p "$HEF_DIR" "${OUTDIR}/csv"

if [ ! -f "${HEF_DIR}/${HEF_NAME}" ]; then
    echo "[HEF 없음] ${HEF_DIR}/${HEF_NAME} 를 공식 hailo_model_zoo S3에서 받는다..."
    wget -q --show-progress -O "${HEF_DIR}/${HEF_NAME}" "$HEF_URL"
fi

echo "[HEF 확인] hailortcli parse-hef 결과:"
hailortcli parse-hef "${HEF_DIR}/${HEF_NAME}" | head -20
echo ""
echo "[중요] 위 출력의 Architecture가 HAILO8인지, 그리고 postprocess/NMS 관련 라인이"
echo "       있는지 육안으로 확인할 것 (nms_core 계열은 engine=nn_core 또는 engine=auto로"
echo "       컴파일되어 있어야 함 — infer_yolov5_hailo8.cpp 상단 주석 참고)."
read -p "계속 진행하려면 Enter, 중단하려면 Ctrl+C: " _

echo "빌드 중..."
g++ "$SRC" -o "$BIN" -I"$HAILORT_INCLUDE" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17

run_id=$START_RUN_ID
for rep in $(seq 1 $REPEATS); do
    echo ""
    echo "--- 반복 $rep/$REPEATS (run_id=$run_id) ---"
    ./"$BIN" "$run_id" "$CSV"
    run_id=$((run_id + 1))

    # HRTT 트레이스가 켜져 있다면(HAILO_TRACE=scheduler) run별로 파일명이 겹치지 않게
    # 잠깐 대기 — 기존 auto_experiment_*.sh 관례와 동일.
    sleep 1
done

echo ""
echo "[완료] 결과: $CSV (run_id ${START_RUN_ID}~$((run_id-1)))"
echo "[참고] engine=cpu 기준선(yolov8s)과 비교하려면 infer_yolov5_hailo8.cpp의"
echo "       USE_CPU_BASELINE_INSTEAD를 1로 바꿔 재빌드 후 같은 방식으로 실행할 것."
