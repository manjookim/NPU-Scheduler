#!/bin/bash
# =============================================================================
# Hailo-8 배치사이즈 스윕 실험: {1,16,32,48,63} x 단일/2조합/3조합 모델
# -----------------------------------------------------------------------------
# 원래 조교님 과제 스펙: "장당 전체 추론시간(전처리-추론-후처리)"을
#   배치사이즈 {1,16,32,48,63} x 단일/2-way/3-way 모델 조합(Det/Seg/Pose)
#   에 대해 측정. priority/threshold/timeout은 이번 실험 대상이 아니므로
#   infer_scheduler_hailo8.cpp에 이미 들어있는 기본값을 그대로 둔다
#   (이 스크립트는 BATCH_DET/SEG/POSE 와 USE_DET/SEG/POSE 만 sed로 바꿈).
#
# 경우의 수:
#   - 단일 : 3 모델 × 배치 5개                    =  15
#   - 2조합: 3 쌍   × 배치 5×5(각 모델 독립)       =  75
#   - 3조합: 1      × 배치 5×5×5(각 모델 독립)     = 125
#   합계 215 조건 × REPEAT(3) = 645 회 실행
#
# 비활성 모델의 batch 값은 의미 없으므로 항상 1로 고정(불필요한 재컴파일 방지).
#
# CSV 라우팅: infer_scheduler_hailo8.cpp의 save_csv()가 이미
#   batch_det/batch_seg/batch_pose 를 별도 컬럼으로 기록하므로,
#   8L처럼 batch별로 파일을 쪼갤 필요 없이 "활성 모델 수"로만 라우팅한다
#   (results_1model.csv / results_2model.csv / results_3model.csv).
#
# 실행 위치: 보드(~/hailo_cpp_test/), infer_scheduler_hailo8.cpp와 같은 디렉토리.
# 사용법: chmod +x auto_experiment_batch_sweep.sh
#         nohup bash auto_experiment_batch_sweep.sh > sweep_log.txt 2>&1 &
#         disown
# =============================================================================

set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 ~/hailo_cpp_test 없음"; exit 1; }

SRC=infer_scheduler_hailo8.cpp
BIN=infer_scheduler_hailo8

if [ ! -f "$SRC" ]; then
    echo "[오류] $SRC 를 찾을 수 없음."
    exit 1
fi
cp "$SRC" "${SRC}.bak"
echo "[백업] ${SRC} -> ${SRC}.bak"

# ---------------- 설정 ----------------
REPEAT=3
batches=(1 16 32 48 63)

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_batch_sweep_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_batch_sweep_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"
TRACES_DIR="$OUTDIR/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"

echo "실험 폴더 : $EXP_DIR"
echo "CSV 폴더  : $CSV_DIR (활성모델수 1/2/3 → 시트 3개)"
echo "조건      : batch{1,16,32,48,63} 활성모델별 독립 교차, 단일+2조합+3조합"
echo "총 실행   : 215 조건 × ${REPEAT}회 = $((215 * REPEAT))회"
echo "priority/threshold/timeout: ${SRC}의 기존 기본값 유지(이번 실험에서 안 건드림)"
echo ""

# ---------------- 한 조건 1회 실행 ----------------
# 인자: USE_D USE_S USE_P  BATCH_D BATCH_S BATCH_P  RUN_ID
compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local BATCH_D=$4 BATCH_S=$5 BATCH_P=$6
    local RUN_ID=$7

    local NMODEL=$((USE_D + USE_S + USE_P))
    local TARGET_CSV="$CSV_DIR/results_${NMODEL}model.csv"

    echo "=== D=$USE_D(b$BATCH_D) S=$USE_S(b$BATCH_S) P=$USE_P(b$BATCH_P) run$RUN_ID -> $(basename "$TARGET_CSV") ==="

    sed -i "s/^#define BATCH_DET .*/#define BATCH_DET       $BATCH_D/"   "$SRC"
    sed -i "s/^#define BATCH_SEG .*/#define BATCH_SEG       $BATCH_S/"   "$SRC"
    sed -i "s/^#define BATCH_POSE .*/#define BATCH_POSE      $BATCH_P/"  "$SRC"
    sed -i "s/^#define USE_DET .*/#define USE_DET    $USE_D/"    "$SRC"
    sed -i "s/^#define USE_SEG .*/#define USE_SEG    $USE_S/"    "$SRC"
    sed -i "s/^#define USE_POSE .*/#define USE_POSE   $USE_P/"   "$SRC"
    # priority/threshold/timeout/NUM_IMAGES 는 건드리지 않음(기존 값 유지)

    g++ "$SRC" -o "$BIN" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "  [!] 컴파일 실패 - 이 조건 건너뜀"; return; fi

    # ---- NPU 모니터 시작 ----
    > npu_log.txt
    rm -f /tmp/hmon_files/*
    rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate 2>/dev/null
    python3 hailo_utilization_hailo8.py &
    NPU_PID=$!
    sleep 2

    # ---- HRTT 트레이스 환경변수 ----
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1

    ./"$BIN" "$RUN_ID" "$TARGET_CSV"

    kill $NPU_PID 2>/dev/null; sleep 1; rm -f /tmp/hmon_files/*

    # ---- npu_percent 채우기 ----
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
    print("  [npu] 활성 구간 없음 - npu_percent NaN 유지")
    sys.exit(0)
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

    # ---- HRTT 트레이스 조건명으로 보존 ----
    LATEST_HRTT=""
    for i in $(seq 1 35); do
        LATEST_HRTT=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST_HRTT" ] && break; sleep 1
    done
    if [ -n "$LATEST_HRTT" ]; then
        TS=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
        NEW="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_bD${BATCH_D}-bS${BATCH_S}-bP${BATCH_P}_run${RUN_ID}_${TS}.hrtt"
        mv "$LATEST_HRTT" "$NEW"; echo "  HRTT: $(basename "$NEW")"
    else
        echo "  [!] HRTT 미생성"
    fi
}

# ================= 실행 루프 =================

# ---------- 단일 모델 (3 모델 × 배치 5개) ----------
echo "----- 단일 모델 -----"
for b in "${batches[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 0 0 $b 1 1 $run; done   # Det
done
for b in "${batches[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 0 1 0 1 $b 1 $run; done   # Seg
done
for b in "${batches[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 0 0 1 1 1 $b $run; done   # Pose
done

# ---------- 2조합 (3 쌍 × 배치 5×5) ----------
echo "----- 2조합 -----"
# Det + Seg
for bd in "${batches[@]}"; do for bs in "${batches[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 1 0 $bd $bs 1 $run; done
done; done
# Det + Pose
for bd in "${batches[@]}"; do for bp in "${batches[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 0 1 $bd 1 $bp $run; done
done; done
# Seg + Pose
for bs in "${batches[@]}"; do for bp in "${batches[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 0 1 1 1 $bs $bp $run; done
done; done

# ---------- 3조합 (배치 5×5×5) ----------
echo "----- 3조합 -----"
for bd in "${batches[@]}"; do for bs in "${batches[@]}"; do for bp in "${batches[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 1 1 $bd $bs $bp $run; done
done; done; done

# ---------- 원본 설정 복구 ----------
cp "${SRC}.bak" "$SRC"
echo ""
echo "[복구] ${SRC} 원본 설정으로 복구 완료"
echo "===== 완료! 215 조건 × ${REPEAT}회 = $((215 * REPEAT))회 ====="
echo "CSV(3개): $CSV_DIR"
echo "HRTT 트레이스: $TRACES_DIR"
