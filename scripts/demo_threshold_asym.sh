#!/bin/bash
# ===== threshold 비대칭 대조 실험 (효과 검증용) =====
# 공통: priority 15-15-15(동일), timeout=1000, INPUT_FPS=30, batch=32
#  ctrl: Det/Seg/Pose threshold 전부 1  (모두 즉시 활성화)
#  test: Det만 threshold=16, Seg/Pose=1 (Det만 16장 모아 처리 → Det 큐대기 ↑)
# 기대: test의 Det latency가 ctrl보다 뚜렷이 상승. Seg/Pose는 두 경우 비슷.
cd ~/hailo_cpp_test

EXP_DATE=$(date +%Y-%m-%d)
EXP_DIR="experiments/${EXP_DATE}_thr_asym"
N=1; while [ -d "${EXP_DIR}${N}" ]; do N=$((N+1)); done
EXP_DIR="${EXP_DIR}${N}"
mkdir -p "$EXP_DIR/traces"
CSV_FILE="$HOME/hailo_cpp_test/${EXP_DIR}/results_all.csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"
echo "데모 폴더: $EXP_DIR (threshold 비대칭 대조)"
echo ""

run_case() {
    local TAG=$1 THR_D=$2 THR_S=$3 THR_P=$4
    echo "======================================"
    echo "[$TAG] Det thr=$THR_D / Seg thr=$THR_S / Pose thr=$THR_P (pri 15-15-15, timeout=1000, fps=30, batch=32)"
    echo "======================================"

    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      32/" infer_scheduler.cpp
    sed -i "s/#define INPUT_FPS.*/#define INPUT_FPS       30/" infer_scheduler.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      1000/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_PER_MODEL.*/#define THRESHOLD_PER_MODEL 1/" infer_scheduler.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    1/" infer_scheduler.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    1/" infer_scheduler.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   1/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    15/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    15/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   15/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_DET.*/#define THRESHOLD_DET   $THR_D/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_SEG.*/#define THRESHOLD_SEG   $THR_S/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_POSE.*/#define THRESHOLD_POSE  $THR_P/" infer_scheduler.cpp

    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread
    if [ $? -ne 0 ]; then echo "컴파일 실패!"; return; fi

    > npu_log.txt; rm -f /tmp/hmon_files/*; rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate
    python3 hailo_utilization.py & NPU_PID=$!; sleep 2
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1
    ./infer_scheduler 1 "$CSV_FILE"     # 로그에 [적용확인] threshold OK 뜨는지 확인
    kill $NPU_PID; sleep 1; rm -f /tmp/hmon_files/*

    LATEST=""
    for i in $(seq 1 35); do
        LATEST=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST" ] && break; sleep 1
    done
    if [ -n "$LATEST" ]; then
        TS=$(basename "$LATEST" .hrtt | sed 's/hailort_//')
        mv "$LATEST" "${TRACES_DIR}/${TAG}_Dthr${THR_D}_Sthr${THR_S}_Pthr${THR_P}_${TS}.hrtt"
        echo "HRTT: ${TAG}_Dthr${THR_D}_Sthr${THR_S}_Pthr${THR_P}_${TS}.hrtt"
    fi
    python3 parse_npu_log.py "$CSV_FILE"
    echo ""
}

run_case ctrl 1  1 1     # 전부 threshold 1
run_case test 16 1 1     # Det만 threshold 16

# 원래 설정 복구
sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      1/" infer_scheduler.cpp
sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      0/" infer_scheduler.cpp
sed -i "s/#define INPUT_FPS.*/#define INPUT_FPS       0/" infer_scheduler.cpp
echo "===== 데모 완료! (ctrl vs test) — Det latency 비교하세요 ====="
