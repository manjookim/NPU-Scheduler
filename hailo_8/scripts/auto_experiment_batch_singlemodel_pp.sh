#!/bin/bash
# =============================================================================
# [2026-07-29] 배치사이즈 스윕(싱글모델) x 후처리 ON/OFF 비교 실험 (Hailo-8)
# (Hailo-8L의 auto_experiment_batch_singlemodel_pp.sh와 완전히 동일한 조건 —
#  두 보드를 1:1로 비교하기 위해 batch/threshold/timeout/priority/구조 전부 통일함)
#
# 요청 내용:
#   - batch = {1, 17, 32, 63} 만 스윕 (63이 HailoRT batch_size 상한)
#   - 나머지 파라미터는 전부 default: threshold=1, timeout=0, priority=16
#   - 싱글 모델만 (Det/Seg/Pose 각각 단독, 2/3조합 없음)
#   - 케이스별 3회 반복
#   - 후처리(decode_det/pose/seg)는 ENABLE_POSTPROCESS 매크로로 ON/OFF 두 버전 다 실행
#   - 전처리는 이미 프레임별(1장씩) 구조로 되어 있음(코드 변경 불필요)
#
# 경우의 수: batch(4) x 모델(3) x REPEAT(3) x pp모드(2) = 216회
#
# CSV 라우팅: pp ON/OFF 별로 별도 파일(results_pp_on.csv / results_pp_off.csv)
#
# 실행 위치: 보드(rpi4, ~/hailo_cpp_test/), infer_scheduler_hailo8.cpp / postprocess_hailo8.hpp와 같은 디렉토리.
# 사용법: chmod +x auto_experiment_batch_singlemodel_pp.sh
#         nohup bash auto_experiment_batch_singlemodel_pp.sh > batch_singlemodel_pp_log.txt 2>&1 &
#         disown
# =============================================================================

set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 ~/hailo_cpp_test 없음"; exit 1; }

SRC=infer_scheduler_hailo8.cpp
BIN=infer_scheduler_hailo8

if [ ! -f "$SRC" ]; then echo "[오류] $SRC 없음"; exit 1; fi
if [ ! -f postprocess_hailo8.hpp ]; then echo "[오류] postprocess_hailo8.hpp 없음"; exit 1; fi
cp "$SRC" "${SRC}.bak"
echo "[백업] ${SRC} -> ${SRC}.bak"

REPEAT=3
batches=(1 17 32 63)
THRESHOLD=1
TIMEOUT_MS=0
PRIORITY=16

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_batch_singlemodel_pp_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_batch_singlemodel_pp_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"
TRACES_DIR="$OUTDIR/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"

echo "실험 폴더 : $EXP_DIR"
echo "설정      : batch{1,17,32,63}, threshold=$THRESHOLD, timeout=${TIMEOUT_MS}ms, priority=$PRIORITY, 싱글모델만"
echo "총 실행   : batch(4) x 모델(3) x ${REPEAT}회 x pp(2) = $((4*3*REPEAT*2))회"
echo ""

# ---------------- 한 조건 1회 실행 ----------------
# 인자: USE_D USE_S USE_P  BATCH  PP(1/0)  RUN_ID
compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3 BATCH=$4 PP=$5 RUN_ID=$6
    local PP_TAG; if [ "$PP" -eq 1 ]; then PP_TAG="on"; else PP_TAG="off"; fi
    local TARGET_CSV="$CSV_DIR/results_pp_${PP_TAG}.csv"

    echo "=== D=$USE_D S=$USE_S P=$USE_P batch=$BATCH pp=$PP_TAG run${RUN_ID} ==="

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
    sed -i "s/^#define ENABLE_POSTPROCESS .*/#define ENABLE_POSTPROCESS  $PP/" "$SRC"
    # NUM_IMAGES/IMG_DIR 등 나머지는 건드리지 않음(기존 값 유지)

    g++ "$SRC" -o "$BIN" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "  [!] 컴파일 실패 — 이 조건 건너뜀"; return; fi

    > npu_log.txt
    rm -f /tmp/hmon_files/*
    rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate 2>/dev/null
    python3 hailo_utilization_hailo8.py &
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
        NEW="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_b${BATCH}_pp${PP_TAG}_run${RUN_ID}_${TS}.hrtt"
        mv "$LATEST_HRTT" "$NEW"; echo "  HRTT: $(basename "$NEW")"
    else
        echo "  [!] HRTT 미생성"
    fi
}

# ================= 실행 루프 =================
for PP in 1 0; do
    echo "########## 후처리 ${PP} (1=ON, 0=OFF) ##########"
    for BATCH in "${batches[@]}"; do
        for run in $(seq 1 $REPEAT); do compile_and_run 1 0 0 $BATCH $PP $run; done   # Det
        for run in $(seq 1 $REPEAT); do compile_and_run 0 1 0 $BATCH $PP $run; done   # Seg
        for run in $(seq 1 $REPEAT); do compile_and_run 0 0 1 $BATCH $PP $run; done   # Pose
    done
done

# ---------- 원본 설정 복구 ----------
cp "${SRC}.bak" "$SRC"
echo ""
echo "[복구] ${SRC} 원본 설정으로 복구 완료"
echo "===== 완료! batch(4) x 모델(3) x ${REPEAT}회 x pp(2) = $((4*3*REPEAT*2))회 ====="
echo "CSV(2개): $CSV_DIR"
echo "HRTT 트레이스: $TRACES_DIR"
