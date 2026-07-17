#!/bin/bash
# ===== priority × threshold 실험: CSV에서 빠진 11개 조건만 강제 재실행 =====
# 기존 prithr 폴더에 이어붙임. 해당 조건의 기존 HRTT는 삭제 후 새로 생성(중복 방지).
cd ~/hailo_cpp_test

EXP_DIR=$(ls -dt experiments/*_prithr_exp* 2>/dev/null | head -1)
if [ -z "$EXP_DIR" ]; then echo "prithr 폴더 없음"; exit 1; fi
CSV_FILE="$HOME/hailo_cpp_test/${EXP_DIR}/results_all.csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"
echo "이어붙일 폴더: $EXP_DIR"
echo ""

compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local PRI_D=$4 PRI_S=$5 PRI_P=$6
    local THR_D=$7 THR_S=$8 THR_P=$9
    local RUN_ID=${10}
    local sig="${USE_D}D-${USE_S}S-${USE_P}P_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_${THR_D}TD-${THR_S}TS-${THR_P}TP_run${RUN_ID}"

    echo "=== D=$USE_D(p$PRI_D,t$THR_D) S=$USE_S(p$PRI_S,t$THR_S) P=$USE_P(p$PRI_P,t$THR_P) run$RUN_ID ==="
    # 해당 조건 기존 HRTT 삭제 (중복 방지)
    rm -f "$TRACES_DIR"/${sig}_*.hrtt

    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      1/" infer_scheduler.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      0/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_PER_MODEL.*/#define THRESHOLD_PER_MODEL 1/" infer_scheduler.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    $USE_D/" infer_scheduler.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    $USE_S/" infer_scheduler.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   $USE_P/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    $PRI_D/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    $PRI_S/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   $PRI_P/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_DET.*/#define THRESHOLD_DET   $THR_D/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_SEG.*/#define THRESHOLD_SEG   $THR_S/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_POSE.*/#define THRESHOLD_POSE  $THR_P/" infer_scheduler.cpp

    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread
    if [ $? -ne 0 ]; then echo "컴파일 실패!"; return; fi

    > npu_log.txt
    rm -f /tmp/hmon_files/*
    rm -f "$TRACES_DIR"/hailort_*.hrtt

    source ~/hailo_platform_venv/bin/activate
    python3 hailo_utilization.py &
    NPU_PID=$!
    sleep 2

    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1

    ./infer_scheduler $RUN_ID "$CSV_FILE"

    kill $NPU_PID; sleep 1; rm -f /tmp/hmon_files/*

    LATEST_HRTT=""
    for i in $(seq 1 35); do
        LATEST_HRTT=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST_HRTT" ] && break; sleep 1
    done
    if [ -n "$LATEST_HRTT" ]; then
        TS=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
        mv "$LATEST_HRTT" "${TRACES_DIR}/${sig}_${TS}.hrtt"
        echo "HRTT: ${sig}_${TS}.hrtt"
    else
        echo "경고: HRTT 미생성"
    fi
    python3 parse_npu_log.py "$CSV_FILE"
}

echo "===== 빠진 11개 조건 재실행 ====="
compile_and_run 1 1 1 15 31 0 15 15 1 1
compile_and_run 1 1 1 15 31 0 15 15 15 1
compile_and_run 1 1 1 15 31 0 15 31 1 1
compile_and_run 1 1 1 15 31 0 15 31 15 1
compile_and_run 1 1 1 15 31 15 15 15 1 1
compile_and_run 1 1 1 15 31 15 15 15 15 1
compile_and_run 1 1 1 15 31 15 15 15 31 1
compile_and_run 1 1 1 15 31 31 15 1 31 1
compile_and_run 1 1 1 15 31 31 15 15 1 1
compile_and_run 1 1 1 15 31 31 15 15 15 1
compile_and_run 1 1 1 15 31 31 15 15 31 1
echo "===== 완료! (11개) ====="
