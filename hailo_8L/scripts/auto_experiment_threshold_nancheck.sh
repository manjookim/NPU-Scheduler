#!/bin/bash
# ===== threshold=priority 실험: NaN(starvation) 조건 14개 재실행 (3회씩) =====
# 모드: THRESHOLD_EQ_PRIORITY=1 (threshold=priority, priority=0이면 1), PER_MODEL=0
cd ~/hailo_cpp_test

REPEAT=3
EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_threshold_nancheck_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_threshold_nancheck_exp${EXP_NUM}"
done
mkdir -p "$EXP_DIR/traces"
CSV_FILE="$HOME/hailo_cpp_test/${EXP_DIR}/results_all.csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"
echo "실험 폴더: $EXP_DIR (NaN 조건 재실행, threshold=priority)"
echo ""

compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local PRI_D=$4 PRI_S=$5 PRI_P=$6
    local RUN_ID=$7
    echo "=== D=$USE_D(p$PRI_D) S=$USE_S(p$PRI_S) P=$USE_P(p$PRI_P) run$RUN_ID ==="
    mkdir -p "$TRACES_DIR"

    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      1/" infer_scheduler.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      0/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_PER_MODEL.*/#define THRESHOLD_PER_MODEL 0/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_EQ_PRIORITY.*/#define THRESHOLD_EQ_PRIORITY 1/" infer_scheduler.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    $USE_D/" infer_scheduler.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    $USE_S/" infer_scheduler.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   $USE_P/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    $PRI_D/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    $PRI_S/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   $PRI_P/" infer_scheduler.cpp

    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread
    if [ $? -ne 0 ]; then echo "컴파일 실패!"; return; fi

    > npu_log.txt; rm -f /tmp/hmon_files/*; rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate
    python3 hailo_utilization.py & NPU_PID=$!; sleep 2
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
        mv "$LATEST_HRTT" "${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_run${RUN_ID}_${TS}.hrtt"
        echo "HRTT 저장됨"
    else
        echo "경고: HRTT 미생성"
    fi
    python3 parse_npu_log.py "$CSV_FILE"
}

# NaN(starvation) 14개 조건
conds="0 15 15
0 15 31
0 31 15
0 31 31
15 0 15
15 0 31
15 15 0
15 31 0
15 31 31
31 0 15
31 0 31
31 15 0
31 15 31
31 31 15"

echo "$conds" | while read pd ps pp; do
    for run in $(seq 1 $REPEAT); do
        compile_and_run 1 1 1 $pd $ps $pp $run
    done
done
echo "===== 완료! (14조건 × ${REPEAT}회) ====="
