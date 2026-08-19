#!/bin/bash
# =============================================================================
# [2026-08-05] 조교님 요청: 단일모델(Det/Seg/Pose 각각 단독)만 대상으로
# INPUT_FPS를 {5,10,15,20,25} 5개 값으로 바꿔가며 재실험.
# 나머지 파라미터는 직전 batch1_pri0_fps30 실험과 동일하게 고정:
# batch=1, threshold=1, timeout=0ms, priority=0(세 모델 다 동일).
#
# 실험 구성: single(Det/Seg/Pose 단독 3개) x FPS(5,10,15,20,25) 5개 x 3회 반복
#          = 15조건 x 3회 = 45회 실행. (2조합/3조합은 이번엔 대상 아님)
#
# CSV는 batch_priority 실험 때처럼 FPS별로 파일을 분리해서 저장
# (results_singlemodel_fps5.csv, ..._fps25.csv) — csv_writer.hpp에 INPUT_FPS
# 컬럼이 없어서, 같은 파일에 여러 FPS를 같이 넣으면 나중에 구분이 안 되기 때문.
#
# 실행 위치: 보드(rpi1, ~/hailo_cpp_test/), infer_scheduler.cpp / postprocess_8l.hpp와
# 같은 디렉토리. 실행 전 최신 버전으로 scp 해뒀는지 반드시 확인할 것.
# 사용법: chmod +x auto_experiment_singlemodel_fps_sweep.sh
#         nohup bash auto_experiment_singlemodel_fps_sweep.sh > singlemodel_fps_sweep_log.txt 2>&1 &
#         disown
# =============================================================================

set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 ~/hailo_cpp_test 없음"; exit 1; }

SRC=infer_scheduler.cpp
BIN=infer_scheduler

if [ ! -f "$SRC" ]; then
    echo "[오류] $SRC 를 찾을 수 없음."
    exit 1
fi
if [ ! -f postprocess_8l.hpp ]; then
    echo "[오류] postprocess_8l.hpp 를 찾을 수 없음 — $SRC 와 같은 디렉토리에 scp 해둘 것."
    exit 1
fi
cp "$SRC" "${SRC}.bak"
echo "[백업] ${SRC} -> ${SRC}.bak"

REPEAT=3
BATCH=1
THRESHOLD=1
TIMEOUT_MS=0
PRIORITY=0
FPS_LIST="5 10 15 20 25"

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_singlemodel_fps_sweep_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_singlemodel_fps_sweep_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"
TRACES_DIR="$OUTDIR/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"

echo "실험 폴더 : $EXP_DIR"
echo "설정      : batch=$BATCH, threshold=$THRESHOLD, timeout=${TIMEOUT_MS}ms, priority=$PRIORITY, FPS={${FPS_LIST// /,}}"
echo "조건      : single(3) x FPS(5개) = 15조건 x ${REPEAT}회 = $((15 * REPEAT))회"
echo ""

# ---------------- 한 조건 1회 실행 ----------------
# 인자: USE_D USE_S USE_P  FPS  RUN_ID
compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3 FPS=$4 RUN_ID=$5
    local TARGET_CSV="$CSV_DIR/results_singlemodel_fps${FPS}.csv"

    echo "=== D=$USE_D S=$USE_S P=$USE_P fps=$FPS run${RUN_ID} (batch=$BATCH, pri=$PRIORITY) ==="

    sed -i "s/^#define BATCH_DET .*/#define BATCH_DET       $BATCH/"   "$SRC"
    sed -i "s/^#define BATCH_SEG .*/#define BATCH_SEG       $BATCH/"   "$SRC"
    sed -i "s/^#define BATCH_POSE .*/#define BATCH_POSE      $BATCH/"  "$SRC"
    sed -i "s/^#define THRESHOLD_DET .*/#define THRESHOLD_DET   $THRESHOLD/"   "$SRC"
    sed -i "s/^#define THRESHOLD_SEG .*/#define THRESHOLD_SEG   $THRESHOLD/"   "$SRC"
    sed -i "s/^#define THRESHOLD_POSE .*/#define THRESHOLD_POSE  $THRESHOLD/"  "$SRC"
    sed -i "s/^#define TIMEOUT_DET_MS .*/#define TIMEOUT_DET_MS   $TIMEOUT_MS/"  "$SRC"
    sed -i "s/^#define TIMEOUT_SEG_MS .*/#define TIMEOUT_SEG_MS   $TIMEOUT_MS/"  "$SRC"
    sed -i "s/^#define TIMEOUT_POSE_MS .*/#define TIMEOUT_POSE_MS  $TIMEOUT_MS/" "$SRC"
    sed -i "s/^#define PRIORITY_DET .*/#define PRIORITY_DET    $PRIORITY/"   "$SRC"
    sed -i "s/^#define PRIORITY_SEG .*/#define PRIORITY_SEG    $PRIORITY/"   "$SRC"
    sed -i "s/^#define PRIORITY_POSE .*/#define PRIORITY_POSE   $PRIORITY/"  "$SRC"
    sed -i "s/^#define USE_DET .*/#define USE_DET    $USE_D/"    "$SRC"
    sed -i "s/^#define USE_SEG .*/#define USE_SEG    $USE_S/"    "$SRC"
    sed -i "s/^#define USE_POSE .*/#define USE_POSE   $USE_P/"   "$SRC"
    sed -i "s/^#define INPUT_FPS .*/#define INPUT_FPS       $FPS/" "$SRC"

    g++ "$SRC" -o "$BIN" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "  [!] 컴파일 실패 — 이 조건 건너뜀"; return; fi

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
        NEW="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_fps${FPS}_run${RUN_ID}_${TS}.hrtt"
        mv "$LATEST_HRTT" "$NEW"; echo "  HRTT: $(basename "$NEW")"
    else
        echo "  [!] HRTT 미생성"
    fi
}

# ================= 실행 루프 =================
# FPS 바깥, 모델 안쪽 순서 — 같은 FPS끼리 묶어서 실행(재컴파일 최소화 목적은 아니고 가독성)

for FPS in $FPS_LIST; do
    echo "----- INPUT_FPS=$FPS -----"
    for run in $(seq 1 $REPEAT); do compile_and_run 1 0 0 $FPS $run; done   # Det
    for run in $(seq 1 $REPEAT); do compile_and_run 0 1 0 $FPS $run; done   # Seg
    for run in $(seq 1 $REPEAT); do compile_and_run 0 0 1 $FPS $run; done   # Pose
done

# ---------- 원본 설정 복구 ----------
cp "${SRC}.bak" "$SRC"
echo ""
echo "[복구] ${SRC} 원본 설정으로 복구 완료"
echo "===== 완료! 15개 조건(단독3 x fps5) × ${REPEAT}회 = $((15 * REPEAT))회 ====="
echo "CSV: $CSV_DIR/results_singlemodel_fps{5,10,15,20,25}.csv"
echo "HRTT 트레이스: $TRACES_DIR"
