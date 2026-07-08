#!/bin/bash
# ===== threshold 효과 데모: 같은 케이스를 두 설정으로 비교 =====
# 케이스 고정: 3모델, priority 15-15-15, threshold 31-31-31 (THRESHOLD_PER_MODEL=1)
#  A) 수정: batch=8, timeout=1000ms  -> threshold가 효과를 낼 수 있는 조건
#  B) 원래: batch=1, timeout=0       -> threshold 무효(기존 실험과 동일)
cd ~/hailo_cpp_test

EXP_DATE=$(date +%Y-%m-%d)
EXP_DIR="experiments/${EXP_DATE}_demo_thr"
N=1; while [ -d "${EXP_DIR}${N}" ]; do N=$((N+1)); done
EXP_DIR="${EXP_DIR}${N}"
mkdir -p "$EXP_DIR/traces"
CSV_FILE="$HOME/hailo_cpp_test/${EXP_DIR}/results_all.csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"
echo "데모 폴더: $EXP_DIR"
echo ""

run_case() {
    local TAG=$1 BATCH=$2 TIMEOUT=$3
    echo "======================================"
    echo "[$TAG] batch=$BATCH, timeout=$TIMEOUT | pri 15-15-15, thr 31-31-31"
    echo "======================================"

    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      $BATCH/" infer_scheduler.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      $TIMEOUT/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_PER_MODEL.*/#define THRESHOLD_PER_MODEL 1/" infer_scheduler.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    1/" infer_scheduler.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    1/" infer_scheduler.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   1/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    15/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    15/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   15/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_DET.*/#define THRESHOLD_DET   31/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_SEG.*/#define THRESHOLD_SEG   31/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_POSE.*/#define THRESHOLD_POSE  31/" infer_scheduler.cpp

    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread
    if [ $? -ne 0 ]; then echo "컴파일 실패!"; return; fi

    > npu_log.txt; rm -f /tmp/hmon_files/*; rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate
    python3 hailo_utilization.py & NPU_PID=$!; sleep 2
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1
    ./infer_scheduler 1 "$CSV_FILE"
    kill $NPU_PID; sleep 1; rm -f /tmp/hmon_files/*

    LATEST=""
    for i in $(seq 1 35); do
        LATEST=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST" ] && break; sleep 1
    done
    if [ -n "$LATEST" ]; then
        TS=$(basename "$LATEST" .hrtt | sed 's/hailort_//')
        mv "$LATEST" "${TRACES_DIR}/${TAG}_batch${BATCH}_timeout${TIMEOUT}_${TS}.hrtt"
        echo "HRTT: ${TAG}_batch${BATCH}_timeout${TIMEOUT}_${TS}.hrtt"
    fi
    python3 parse_npu_log.py "$CSV_FILE"
    echo ""
}

run_case A_modified 8 1000
run_case B_original 1 0

# 원래 설정으로 복구
sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      1/" infer_scheduler.cpp
sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      0/" infer_scheduler.cpp
echo "===== 데모 완료! (A_modified, B_original) ====="
