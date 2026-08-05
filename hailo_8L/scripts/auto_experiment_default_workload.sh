#!/bin/bash
# =============================================================================
# [2026-07-28] 조교님 요청: "8L에서 코드 수정 후 priority, batch size, timeout,
# threshold 모두 default 값에서 single, multi workload 실험만 추가로 해주세요.
# 지금 메모리 사용률이 이상해서 확인해야하거든요"
#
# 이번 실행은 방금 반영한 코드 변경(후처리 decode_det/pose/seg 추가, 출력 FLOAT32 통일,
# 전처리 프레임별/모델별 독립 측정)이 적용된 infer_scheduler.cpp로 도는 첫 실기 테스트도
# 겸한다 — 정상 동작 확인 + 조교님이 요청한 메모리 사용률 점검을 한 번에 수행.
#
# "모두 default 값"의 구체적 수치 — HailoRT 공식 소스(GitHub hailo-ai/hailort)로 재검증:
#   - batch     = 0   [2026-08 정정] HAILO_DEFAULT_BATCH_SIZE=0(hailort.h) — "batch is
#                     determined by HailoRT automatically". 최초 실행(exp1/exp2)은 프로젝트
#                     관례값인 1로 돌렸었는데, 공식 문서 기준 진짜 기본값은 0(auto)임.
#                     threshold(1) > batch(0)이라 set_scheduler_threshold 호출은 실패하지만,
#                     그 경우 threshold는 SDK 자체 기본값 1로 남으므로(model_setup.hpp 주석
#                     참고) 결과적으로 동일 — 0으로 둬도 안전함. 이후 재실행(exp3~)부터 적용.
#   - threshold = 1   (network_group.hpp doxygen: "The default threshold is 1." 확인)
#   - timeout   = 0   (network_group.hpp doxygen: "The default timeout is 0ms." 확인)
#   - priority  = 16  (hailort.h: #define HAILO_SCHEDULER_PRIORITY_NORMAL (16) 확인)
#     [주의] 이 프로젝트가 다른 실험(priority_batch_size 등)에서 관례적으로 써온 15가
#     아니라 진짜 SDK 기본값 16을 씀 — "default" 요청이라 의도적으로 다르게 설정.
#
# 실험 구성(사용자 확인 완료): single(Det/Seg/Pose 단독 3개) + 2조합(Det+Seg, Det+Pose,
# Seg+Pose 3개) + 3조합(전체 동시 1개) = 7개 조건 × 3회 반복 = 21회 실행.
# NUM_IMAGES는 건드리지 않음(파일에 이미 있는 값 그대로 사용, 현재 600).
#
# 실행 위치: 보드(rpi1, ~/hailo_cpp_test/), infer_scheduler.cpp / postprocess_8l.hpp와
# 같은 디렉토리. 실행 전 두 파일을 최신 버전으로 scp 해뒀는지 반드시 확인할 것.
# 사용법: chmod +x auto_experiment_default_workload.sh
#         nohup bash auto_experiment_default_workload.sh > default_workload_log.txt 2>&1 &
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
BATCH=0
THRESHOLD=1
TIMEOUT_MS=0
PRIORITY=16

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_default_workload_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_default_workload_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"
TRACES_DIR="$OUTDIR/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"
TARGET_CSV="$CSV_DIR/results_default_workload.csv"

echo "실험 폴더 : $EXP_DIR"
echo "설정      : batch=$BATCH, threshold=$THRESHOLD, timeout=${TIMEOUT_MS}ms, priority=$PRIORITY (전부 default, 스윕 없음)"
echo "조건      : single(3) + 2조합(3) + 3조합(1) = 7개 조건 × ${REPEAT}회 = $((7 * REPEAT))회"
echo "CSV       : $TARGET_CSV"
echo ""

# ---------------- 한 조건 1회 실행 ----------------
# 인자: USE_D USE_S USE_P  RUN_ID
compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3 RUN_ID=$4

    echo "=== D=$USE_D S=$USE_S P=$USE_P run${RUN_ID} (batch=$BATCH, pri=$PRIORITY) ==="

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
        NEW="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_default_run${RUN_ID}_${TS}.hrtt"
        mv "$LATEST_HRTT" "$NEW"; echo "  HRTT: $(basename "$NEW")"
    else
        echo "  [!] HRTT 미생성"
    fi
}

# ================= 실행 루프 =================

echo "----- single (단독 3개) -----"
for run in $(seq 1 $REPEAT); do compile_and_run 1 0 0 $run; done   # Det
for run in $(seq 1 $REPEAT); do compile_and_run 0 1 0 $run; done   # Seg
for run in $(seq 1 $REPEAT); do compile_and_run 0 0 1 $run; done   # Pose

echo "----- 2조합 (3개) -----"
for run in $(seq 1 $REPEAT); do compile_and_run 1 1 0 $run; done   # Det+Seg
for run in $(seq 1 $REPEAT); do compile_and_run 1 0 1 $run; done   # Det+Pose
for run in $(seq 1 $REPEAT); do compile_and_run 0 1 1 $run; done   # Seg+Pose

echo "----- 3조합 (1개) -----"
for run in $(seq 1 $REPEAT); do compile_and_run 1 1 1 $run; done   # Det+Seg+Pose

# ---------- 원본 설정 복구 ----------
cp "${SRC}.bak" "$SRC"
echo ""
echo "[복구] ${SRC} 원본 설정으로 복구 완료"
echo "===== 완료! 7개 조건 × ${REPEAT}회 = $((7 * REPEAT))회 ====="
echo "CSV: $TARGET_CSV"
echo "HRTT 트레이스: $TRACES_DIR"
