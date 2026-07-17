#!/bin/bash
# ===== threshold 효과 데모 (입력 속도 제한 O) =====
# 공통: 3모델, priority 15-15-15, batch=32(큐 여유), timeout=2000ms, INPUT_FPS=20
#   -> 입력을 20fps로 제한해 큐가 항상 꽉 차지 않게 함 (threshold가 실제로 작동하는 조건)
# threshold만 다르게: A=1(오는 대로 처리) vs B=16(16장 모아서 처리)
# 기대: B가 활성화당 프레임 많고(≈16), 전환 적고, latency 높음 / A는 반대
cd ~/hailo_cpp_test

EXP_DATE=$(date +%Y-%m-%d)
EXP_DIR="experiments/${EXP_DATE}_thr_effect"
N=1; while [ -d "${EXP_DIR}${N}" ]; do N=$((N+1)); done
EXP_DIR="${EXP_DIR}${N}"
mkdir -p "$EXP_DIR/traces"
CSV_FILE="$HOME/hailo_cpp_test/${EXP_DIR}/results_all.csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"
echo "데모 폴더: $EXP_DIR"
echo "공통: batch=16, timeout=1000, priority 15-15-15 | threshold만 A=1 vs B=16"
echo ""

run_case() {
    local TAG=$1 THR=$2
    echo "======================================"
    echo "[$TAG] threshold=$THR-$THR-$THR (batch=16, timeout=1000, pri 15-15-15)"
    echo "======================================"

    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      32/" infer_scheduler.cpp
    sed -i "s/#define INPUT_FPS.*/#define INPUT_FPS       20/" infer_scheduler.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      2000/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_PER_MODEL.*/#define THRESHOLD_PER_MODEL 1/" infer_scheduler.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    1/" infer_scheduler.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    1/" infer_scheduler.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   1/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    15/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    15/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   15/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_DET.*/#define THRESHOLD_DET   $THR/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_SEG.*/#define THRESHOLD_SEG   $THR/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_POSE.*/#define THRESHOLD_POSE  $THR/" infer_scheduler.cpp

    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread
    if [ $? -ne 0 ]; then echo "컴파일 실패!"; return; fi

    > npu_log.txt; rm -f /tmp/hmon_files/*; rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate
    python3 hailo_utilization.py & NPU_PID=$!; sleep 2
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1
    ./infer_scheduler 1 "$CSV_FILE"     # <- 여기서 "적용 실패!" 경고 뜨는지 확인
    kill $NPU_PID; sleep 1; rm -f /tmp/hmon_files/*

    LATEST=""
    for i in $(seq 1 35); do
        LATEST=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST" ] && break; sleep 1
    done
    if [ -n "$LATEST" ]; then
        TS=$(basename "$LATEST" .hrtt | sed 's/hailort_//')
        mv "$LATEST" "${TRACES_DIR}/${TAG}_thr${THR}_batch32_to2000_fps20_${TS}.hrtt"
        echo "HRTT: ${TAG}_thr${THR}_batch32_to2000_fps20_${TS}.hrtt"
    fi
    python3 parse_npu_log.py "$CSV_FILE"
    echo ""
}

run_case A_lowTH  1
run_case B_highTH 16

# 원래 설정 복구
sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      1/" infer_scheduler.cpp
sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      0/" infer_scheduler.cpp
sed -i "s/#define INPUT_FPS.*/#define INPUT_FPS       0/" infer_scheduler.cpp
echo "===== 데모 완료! (A_lowTH, B_highTH) ====="
