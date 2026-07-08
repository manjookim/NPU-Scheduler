#!/bin/bash
# ===== priority × threshold 독립 조합 실험 (2모델 + 3모델, 1회) =====
# 각 모델: priority ∈ {0,15,31}, threshold ∈ {1,15,31} 독립 조합
# 2모델 243 (81×3쌍) + 3모델 729 = 972조건 × 1회
# batch=1, timeout=0, 이미지 673장(NUM_IMAGES=0)
cd ~/hailo_cpp_test

REPEAT=1

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_prithr_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_prithr_exp${EXP_NUM}"
done
mkdir -p "$EXP_DIR/traces"
CSV_FILE="$HOME/hailo_cpp_test/${EXP_DIR}/results_all.csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"
echo "실험 폴더: $EXP_DIR"
echo "조건: priority{0,15,31} × threshold{1,15,31} 독립 조합, 2+3모델, ${REPEAT}회"
echo ""

priorities=(0 15 31)
thresholds=(1 15 31)

compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local PRI_D=$4 PRI_S=$5 PRI_P=$6
    local THR_D=$7 THR_S=$8 THR_P=$9
    local RUN_ID=${10}

    echo "=== D=$USE_D(p$PRI_D,t$THR_D) S=$USE_S(p$PRI_S,t$THR_S) P=$USE_P(p$PRI_P,t$THR_P) run$RUN_ID ==="
    mkdir -p "$TRACES_DIR"

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
        NEW="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_${THR_D}TD-${THR_S}TS-${THR_P}TP_run${RUN_ID}_${TS}.hrtt"
        mv "$LATEST_HRTT" "$NEW"; echo "HRTT: $(basename $NEW)"
    else
        echo "경고: HRTT 미생성"
    fi
    python3 parse_npu_log.py "$CSV_FILE"
}

echo "===== 2모델 (243조건) ====="
# Det+Seg
for pd in "${priorities[@]}"; do for td in "${thresholds[@]}"; do
  for ps in "${priorities[@]}"; do for ts in "${thresholds[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 1 0 $pd $ps 0 $td $ts 1 $run; done
  done; done
done; done
# Det+Pose
for pd in "${priorities[@]}"; do for td in "${thresholds[@]}"; do
  for pp in "${priorities[@]}"; do for tp in "${thresholds[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 0 1 $pd 0 $pp $td 1 $tp $run; done
  done; done
done; done
# Seg+Pose
for ps in "${priorities[@]}"; do for ts in "${thresholds[@]}"; do
  for pp in "${priorities[@]}"; do for tp in "${thresholds[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 0 1 1 0 $ps $pp 1 $ts $tp $run; done
  done; done
done; done

echo "===== 3모델 (729조건) ====="
for pd in "${priorities[@]}"; do for td in "${thresholds[@]}"; do
  for ps in "${priorities[@]}"; do for ts in "${thresholds[@]}"; do
    for pp in "${priorities[@]}"; do for tp in "${thresholds[@]}"; do
      for run in $(seq 1 $REPEAT); do compile_and_run 1 1 1 $pd $ps $pp $td $ts $tp $run; done
    done; done
  done; done
done; done

echo "===== 완료! 972조건 × ${REPEAT}회 ====="
