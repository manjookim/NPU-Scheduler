#!/bin/bash
# =============================================================================
# threshold ≤ batch_size 경계 실험
#   - batch_size = 32 (세 모델 통일)
#   - threshold = 31 / 32 / 33 (세 모델 통일) — 3케이스
#   - priority = 0 (세 모델 동일), timeout = 0, 3모델 동시
#   - 각 케이스 1회 → 총 3회 실험
#
# 목적: threshold ≤ batch 제약 확인
#   31 ≤ 32 → 적용 예상,  32 = 32 → 적용 예상,  33 > 32 → 거부(기본값 1로 남음) 예상
#   실행 로그 [적용확인]의 [OK]/[실패], 그리고 HRTT core_op_set_value로 확정.
#
# 주의: priority가 모두 같아(0) Round-Robin 공평 스케줄 → starvation/hang 없음.
#       threshold의 "효과"까지 보려면 timeout>0 + 입력속도 제한이 필요하지만,
#       이 실험은 "적용 여부(경계)"만 보는 것이므로 timeout=0으로 둔다.
# =============================================================================
set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 ~/hailo_cpp_test 없음"; exit 1; }

BATCH=32
NUM_IMAGES=0            # 0 = sampled_val2017 전체
THRESHOLDS=(31 32 33)

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_thr_batch32_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do EXP_NUM=$((EXP_NUM+1)); EXP_DIR="experiments/${EXP_DATE}_thr_batch32_exp${EXP_NUM}"; done
CSV_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"
CSV_FILE="$CSV_DIR/results_thr_batch32.csv"

echo "실험 폴더: $EXP_DIR"
echo "batch=$BATCH, threshold={31,32,33}, priority=0, 3모델, 각 1회"
echo ""

run_one() {
    local THR=$1 RUN_ID=$2
    echo "===== batch=$BATCH  threshold=$THR  (3모델, priority=0)  run$RUN_ID ====="

    # ---- 파라미터 주입 ----
    sed -i "s/^#define BATCH_DET .*/#define BATCH_DET       $BATCH/"   infer_scheduler.cpp
    sed -i "s/^#define BATCH_SEG .*/#define BATCH_SEG       $BATCH/"   infer_scheduler.cpp
    sed -i "s/^#define BATCH_POSE .*/#define BATCH_POSE      $BATCH/"  infer_scheduler.cpp
    sed -i "s/^#define THRESHOLD_DET .*/#define THRESHOLD_DET   $THR/"    infer_scheduler.cpp
    sed -i "s/^#define THRESHOLD_SEG .*/#define THRESHOLD_SEG   $THR/"    infer_scheduler.cpp
    sed -i "s/^#define THRESHOLD_POSE .*/#define THRESHOLD_POSE  $THR/"   infer_scheduler.cpp
    sed -i "s/^#define TIMEOUT_DET_MS .*/#define TIMEOUT_DET_MS   0/"   infer_scheduler.cpp
    sed -i "s/^#define TIMEOUT_SEG_MS .*/#define TIMEOUT_SEG_MS   0/"   infer_scheduler.cpp
    sed -i "s/^#define TIMEOUT_POSE_MS .*/#define TIMEOUT_POSE_MS  0/"  infer_scheduler.cpp
    sed -i "s/^#define PRIORITY_DET .*/#define PRIORITY_DET    0/"   infer_scheduler.cpp
    sed -i "s/^#define PRIORITY_SEG .*/#define PRIORITY_SEG    0/"   infer_scheduler.cpp
    sed -i "s/^#define PRIORITY_POSE .*/#define PRIORITY_POSE   0/"  infer_scheduler.cpp
    sed -i "s/^#define USE_DET .*/#define USE_DET    1/"   infer_scheduler.cpp
    sed -i "s/^#define USE_SEG .*/#define USE_SEG    1/"   infer_scheduler.cpp
    sed -i "s/^#define USE_POSE .*/#define USE_POSE   1/"  infer_scheduler.cpp
    sed -i "s/^#define NUM_IMAGES .*/#define NUM_IMAGES      $NUM_IMAGES/" infer_scheduler.cpp

    # ---- 컴파일 ----
    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "  [!] 컴파일 실패 — 건너뜀"; return; fi

    # ---- 모니터 + 트레이스 ----
    > npu_log.txt; rm -f /tmp/hmon_files/*; rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate 2>/dev/null
    python3 hailo_utilization.py & local NPU_PID=$!; sleep 2
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1

    ./infer_scheduler "$RUN_ID" "$CSV_FILE"

    kill $NPU_PID 2>/dev/null; sleep 1; rm -f /tmp/hmon_files/*

    # ---- npu_percent 채우기 ----
    python3 - "$CSV_FILE" npu_log.txt <<'PY'
import sys, csv, re
c, l = sys.argv[1], sys.argv[2]; v = []
try:
    for line in open(l):
        m = re.search(r'NPU:\s*([\d.]+)%', line)
        if m and float(m.group(1)) > 0: v.append(float(m.group(1)))
except FileNotFoundError: pass
if v:
    rows = list(csv.reader(open(c)))
    if len(rows) >= 2 and 'npu_percent' in rows[0]:
        rows[-1][rows[0].index('npu_percent')] = f"{sum(v)/len(v):.4f}"
        with open(c, 'w', newline='') as f: csv.writer(f).writerows(rows)
PY

    # ---- HRTT 조건명으로 보존 ----
    local LATEST=""
    for i in $(seq 1 35); do LATEST=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1); [ -n "$LATEST" ] && break; sleep 1; done
    if [ -n "$LATEST" ]; then
        TS=$(basename "$LATEST" .hrtt | sed 's/hailort_//')
        mv "$LATEST" "${TRACES_DIR}/3model_b${BATCH}_thr${THR}_run${RUN_ID}_${TS}.hrtt"
        echo "  HRTT: 3model_b${BATCH}_thr${THR}_run${RUN_ID}_${TS}.hrtt"
    else
        echo "  [!] HRTT 미생성"
    fi
    echo ""
}

for THR in "${THRESHOLDS[@]}"; do run_one "$THR" 1; done

echo "===== 완료: threshold 31/32/33 각 1회 (총 3회) ====="
echo "CSV: $CSV_FILE"
echo ""
echo "[확인 포인트]"
echo " - 실행 로그의 [적용확인]에서 각 모델 threshold [OK]/[실패]:"
echo "     threshold=31 → OK 예상,  32 → OK 예상,  33 → [실패] 예상(batch 32 초과)"
echo " - HRTT로 확정: python3 tools/hrtt/verify_params.py $TRACES_DIR"
echo "     33은 core_op_set_value에 안 남으면 '(미적용!)'로 확인됨"
