#!/bin/bash
# ========== threshold 실험 부분 재실험 (priority=0 포함 조건만) ==========
# 수정: priority=0 모델은 threshold=1(기존값), 15/31은 그대로 (cpp THRESHOLD_EQ_PRIORITY=1)
# priority=0을 하나라도 포함하는 조건만 재실행 (34조건 × 3회 = 102회)
# priority가 전부 15/31인 20조건은 기존 결과 그대로 사용(재실행 안 함)
cd ~/hailo_cpp_test

REPEAT=3

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_threshold_rerun_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1))
    EXP_DIR="experiments/${EXP_DATE}_threshold_rerun_exp${EXP_NUM}"
done
mkdir -p "$EXP_DIR/traces"
CSV_FILE="$HOME/hailo_cpp_test/${EXP_DIR}/results_all.csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"

echo "실험 폴더: $EXP_DIR"
echo "조건: priority=0 -> threshold=1, 그 외 threshold=priority"
echo "대상: priority=0 포함 조건만 (34조건 × ${REPEAT}회)"
echo ""

compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local PRI_D=$4 PRI_S=$5 PRI_P=$6
    local RUN_ID=$7

    echo "======================================"
    echo "실험: Det=$USE_D($PRI_D), Seg=$USE_S($PRI_S), Pose=$USE_P($PRI_P) | Run=$RUN_ID"
    echo "======================================"

    mkdir -p "$TRACES_DIR"

    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      1/" infer_scheduler.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      0/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD_EQ_PRIORITY.*/#define THRESHOLD_EQ_PRIORITY 1/" infer_scheduler.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    $USE_D/" infer_scheduler.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    $USE_S/" infer_scheduler.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   $USE_P/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    $PRI_D/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    $PRI_S/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   $PRI_P/" infer_scheduler.cpp

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

    kill $NPU_PID
    sleep 1
    rm -f /tmp/hmon_files/*

    LATEST_HRTT=""
    for i in $(seq 1 35); do
        LATEST_HRTT=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST_HRTT" ] && break
        sleep 1
    done

    if [ -n "$LATEST_HRTT" ]; then
        TIMESTAMP=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
        NEW_NAME="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_run${RUN_ID}_${TIMESTAMP}.hrtt"
        mv "$LATEST_HRTT" "$NEW_NAME"
        echo "HRTT: $NEW_NAME"
    else
        echo "경고: HRTT 파일이 생성되지 않았습니다."
    fi

    python3 parse_npu_log.py "$CSV_FILE"
    echo "완료! (Run $RUN_ID)"
    echo ""
}

priorities=(0 15 31)

echo "===== 2개 Multi (priority=0 포함 조건만) ====="
# Det+Seg : pd, ps 중 하나라도 0
for pd in "${priorities[@]}"; do
  for ps in "${priorities[@]}"; do
    if [ $pd -ne 0 ] && [ $ps -ne 0 ]; then continue; fi
    for run in $(seq 1 $REPEAT); do compile_and_run 1 1 0 $pd $ps 0 $run; done
  done
done
# Det+Pose : pd, pp 중 하나라도 0
for pd in "${priorities[@]}"; do
  for pp in "${priorities[@]}"; do
    if [ $pd -ne 0 ] && [ $pp -ne 0 ]; then continue; fi
    for run in $(seq 1 $REPEAT); do compile_and_run 1 0 1 $pd 0 $pp $run; done
  done
done
# Seg+Pose : ps, pp 중 하나라도 0
for ps in "${priorities[@]}"; do
  for pp in "${priorities[@]}"; do
    if [ $ps -ne 0 ] && [ $pp -ne 0 ]; then continue; fi
    for run in $(seq 1 $REPEAT); do compile_and_run 0 1 1 0 $ps $pp $run; done
  done
done

echo "===== 3개 Multi (priority=0 포함 조건만) ====="
for pd in "${priorities[@]}"; do
  for ps in "${priorities[@]}"; do
    for pp in "${priorities[@]}"; do
      if [ $pd -ne 0 ] && [ $ps -ne 0 ] && [ $pp -ne 0 ]; then continue; fi
      for run in $(seq 1 $REPEAT); do compile_and_run 1 1 1 $pd $ps $pp $run; done
    done
  done
done

echo "===== 부분 재실험 완료! (priority=0 포함 34조건 × ${REPEAT}회 = 102회) ====="
