#!/bin/bash
# =============================================================================
# results_2model_b63.csv에서 빠진 2개 조건 채우기
#   Det+Pose, batch=63, priority_det=15, priority_pose=31, run_id=2,3
#   (run_id=1은 이미 존재함 — auto_experiment_batch_priority.sh 실행 로그 확인 결과
#    이 조건만 run2/run3가 CSV에 기록되지 않았음)
#
# 대상 CSV/traces 폴더는 원래 exp2 캠페인 폴더를 그대로 사용해서 이어붙인다.
# =============================================================================
set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 없음"; exit 1; }

SRC=infer_scheduler.cpp
BIN=infer_scheduler

# 원래 exp2 캠페인 폴더 (날짜가 다르면 이 줄만 실제 폴더명으로 바꿀 것)
EXP_DIR="experiments/2026-07-26_batch_priority_exp1"
CSV_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"

if [ ! -d "$CSV_DIR" ]; then
    echo "[오류] $CSV_DIR 없음 — EXP_DIR 값을 실제 폴더명으로 수정할 것"
    exit 1
fi

cp "$SRC" "${SRC}.bak"

TARGET_CSV="$CSV_DIR/results_2model_b63.csv"
echo "대상 CSV: $TARGET_CSV"
echo "패치 전 행 수: $(tail -n +2 "$TARGET_CSV" | wc -l)"

for RUN_ID in 2 3; do
    USE_D=1 USE_S=0 USE_P=1
    PRI_D=15 PRI_S=0 PRI_P=31
    BATCH=63

    echo "=== D=1(p15) S=0 P=1(p31) batch=63 run${RUN_ID} 재실행 ==="

    sed -i "s/^#define BATCH_DET .*/#define BATCH_DET       $BATCH/"   "$SRC"
    sed -i "s/^#define BATCH_SEG .*/#define BATCH_SEG       $BATCH/"   "$SRC"
    sed -i "s/^#define BATCH_POSE .*/#define BATCH_POSE      $BATCH/"  "$SRC"
    sed -i "s/^#define THRESHOLD_DET .*/#define THRESHOLD_DET   1/"    "$SRC"
    sed -i "s/^#define THRESHOLD_SEG .*/#define THRESHOLD_SEG   1/"    "$SRC"
    sed -i "s/^#define THRESHOLD_POSE .*/#define THRESHOLD_POSE  1/"   "$SRC"
    sed -i "s/^#define TIMEOUT_DET_MS .*/#define TIMEOUT_DET_MS   0/"  "$SRC"
    sed -i "s/^#define TIMEOUT_SEG_MS .*/#define TIMEOUT_SEG_MS   0/"  "$SRC"
    sed -i "s/^#define TIMEOUT_POSE_MS .*/#define TIMEOUT_POSE_MS  0/" "$SRC"
    sed -i "s/^#define USE_DET .*/#define USE_DET    $USE_D/"    "$SRC"
    sed -i "s/^#define USE_SEG .*/#define USE_SEG    $USE_S/"    "$SRC"
    sed -i "s/^#define USE_POSE .*/#define USE_POSE   $USE_P/"   "$SRC"
    sed -i "s/^#define PRIORITY_DET .*/#define PRIORITY_DET    $PRI_D/"   "$SRC"
    sed -i "s/^#define PRIORITY_SEG .*/#define PRIORITY_SEG    $PRI_S/"   "$SRC"
    sed -i "s/^#define PRIORITY_POSE .*/#define PRIORITY_POSE   $PRI_P/"  "$SRC"
    sed -i "s/^#define NUM_IMAGES .*/#define NUM_IMAGES      0/" "$SRC"

    g++ "$SRC" -o "$BIN" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "  [!] 컴파일 실패"; continue; fi

    > npu_log.txt
    rm -f /tmp/hmon_files/*
    rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate 2>/dev/null
    python3 hailo_utilization.py &
    NPU_PID=$!
    sleep 2

    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1

    ./"$BIN" "$RUN_ID" "$TARGET_CSV"

    kill $NPU_PID 2>/dev/null; sleep 1; rm -f /tmp/hmon_files/*

    python3 - "$TARGET_CSV" npu_log.txt <<'PY'
import sys, csv, re
csv_path, log_path = sys.argv[1], sys.argv[2]
vals = []
try:
    for line in open(log_path):
        m = re.search(r'NPU:\s*([\d.]+)%', line)
        if m:
            v = float(m.group(1))
            if v > 0: vals.append(v)
except FileNotFoundError:
    pass
if not vals:
    print("  [npu] 활성 구간 없음"); sys.exit(0)
avg = sum(vals) / len(vals)
rows = list(csv.reader(open(csv_path)))
if len(rows) >= 2:
    header, last = rows[0], rows[-1]
    if 'npu_percent' in header:
        last[header.index('npu_percent')] = f"{avg:.4f}"
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        print(f"  [npu] npu_percent={avg:.2f}% 기록")
PY

    LATEST_HRTT=""
    for i in $(seq 1 35); do
        LATEST_HRTT=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST_HRTT" ] && break; sleep 1
    done
    if [ -n "$LATEST_HRTT" ]; then
        TS=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
        NEW="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_b${BATCH}_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_run${RUN_ID}_${TS}.hrtt"
        mv "$LATEST_HRTT" "$NEW"; echo "  HRTT: $(basename "$NEW")"
    else
        echo "  [!] HRTT 미생성"
    fi
done

cp "${SRC}.bak" "$SRC"
echo ""
echo "패치 후 행 수: $(tail -n +2 "$TARGET_CSV" | wc -l) (81이어야 정상)"
echo "완료"
