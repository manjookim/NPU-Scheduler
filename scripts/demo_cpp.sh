#!/bin/bash
# ===== C++ API 버전 검증 데모 (2케이스) =====
# 공통: 3모델, priority 15-15-15, timeout=1000, batch=32
#  ctrl: Det/Seg/Pose threshold 전부 1
#  test: Det만 threshold=16, Seg/Pose=1
# 목적: C++ 재작성 빌드/실행 확인 + [적용확인] 로그로 파라미터 적용 검증
cd ~/hailo_cpp_test

EXP_DATE=$(date +%Y-%m-%d)
EXP_DIR="experiments/${EXP_DATE}_cpp_test"
N=1; while [ -d "${EXP_DIR}${N}" ]; do N=$((N+1)); done
EXP_DIR="${EXP_DIR}${N}"
mkdir -p "$EXP_DIR/traces"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"
echo "데모 폴더: $EXP_DIR"
echo ""

run_case() {
    local TAG=$1 THR_D=$2 THR_S=$3 THR_P=$4
    echo "======================================"
    echo "[$TAG] Det thr=$THR_D / Seg thr=$THR_S / Pose thr=$THR_P (pri 15-15-15, timeout=1000, batch=32)"
    echo "======================================"

    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      32/" infer_scheduler.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      1000/" infer_scheduler.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    1/" infer_scheduler.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    1/" infer_scheduler.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   1/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    15/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    15/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   15/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_DET.*/#define THRESHOLD_DET   $THR_D/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_SEG.*/#define THRESHOLD_SEG   $THR_S/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_POSE.*/#define THRESHOLD_POSE  $THR_P/" infer_scheduler.cpp

    # C++ 빌드 (-std=c++17, OpenCV 미사용)
    g++ infer_scheduler.cpp -o infer_scheduler -lhailort -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "컴파일 실패!"; return; fi

    rm -f "$TRACES_DIR"/hailort_*.hrtt
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1

    ./infer_scheduler     # 로그에 [적용확인] threshold [OK] 확인

    # HRTT 이름 붙이기 (verify_params로 재확인용)
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
    echo ""
}

run_case ctrl 1  1 1
run_case test 16 1 1

echo "===== 데모 완료! (ctrl vs test) ====="
