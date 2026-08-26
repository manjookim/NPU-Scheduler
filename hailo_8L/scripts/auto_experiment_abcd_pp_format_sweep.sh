#!/bin/bash
# =============================================================================
# [2026-08-26] 조교님 코드(infer_scheduler_new.cpp)와의 latency 차이 원인을 좁히기 위한
# 4개 조건(A/B/C/D) x 7개 모델조합(single 3 + 2조합 3 + 3조합 1) x 1회 = 28회 실험.
#
# 조건 정의:
#   A: 우리 코드, 후처리 O + FLOAT32 강제           (ENABLE_POSTPROCESS=1, FORCE_OUTPUT_FLOAT32=1) — 평소 기본 상태
#   B: 우리 코드, 후처리 X + AUTO 포맷              (ENABLE_POSTPROCESS=0, FORCE_OUTPUT_FLOAT32=0) — 조교님과 동일조건
#   C: 우리 코드, 후처리 X + FLOAT32 강제(변환비용만 남김) (ENABLE_POSTPROCESS=0, FORCE_OUTPUT_FLOAT32=1)
#   D: 조교님 코드(infer_scheduler_new.cpp) 그대로   (스케줄러 파라미터 적용 안 됨, AUTO 포맷, 후처리 없음)
#
# A/B/C는 infer_scheduler.cpp를 sed로 편집+재컴파일, D는 infer_scheduler_new.cpp를 그대로
# 씀(USE_DET/SEG/POSE만 조합별로 sed 편집). batch=1, threshold=1, timeout=0ms, priority=15,
# NUM_IMAGES=600은 네 조건 모두 동일(D도 이미 이 값으로 맞춰져 있음).
#
# 실행 위치: npu-rpi1, ~/hailo_cpp_test/ (infer_scheduler.cpp, infer_scheduler_new.cpp,
# postprocess_8l.hpp, model_*.hpp, image_utils.hpp, output_classify.hpp, sys_monitor.hpp,
# csv_writer.hpp 전부 최신 버전으로 scp 해둔 상태여야 함).
# 사용법: chmod +x auto_experiment_abcd_pp_format_sweep.sh
#         nohup bash auto_experiment_abcd_pp_format_sweep.sh > abcd_sweep_log.txt 2>&1 &
#         disown
# =============================================================================

set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 ~/hailo_cpp_test 없음"; exit 1; }

SRC_OURS=infer_scheduler.cpp
BIN_OURS=infer_scheduler
SRC_TA=infer_scheduler_new.cpp
BIN_TA=infer_scheduler_new

for f in "$SRC_OURS" "$SRC_TA" postprocess_8l.hpp model_types.hpp sys_monitor.hpp \
         image_utils.hpp output_classify.hpp model_setup.hpp model_runner.hpp csv_writer.hpp; do
    if [ ! -f "$f" ]; then
        echo "[오류] $f 를 찾을 수 없음 — 최신 버전으로 scp 해둘 것."
        exit 1
    fi
done

cp "$SRC_OURS" "${SRC_OURS}.bak"
cp "$SRC_TA" "${SRC_TA}.bak"
echo "[백업] ${SRC_OURS}.bak, ${SRC_TA}.bak 생성"

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_abcd_pp_format_sweep_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_abcd_pp_format_sweep_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"
TRACES_DIR="$OUTDIR/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"

echo "실험 폴더 : $EXP_DIR"
echo "조건      : A(후처리O+F32) / B(후처리X+AUTO) / C(후처리X+F32) / D(조교님코드)"
echo "조합      : single(3) + 2조합(3) + 3조합(1) = 7개 x 4조건 x 1회 = 28회"
echo ""

export HAILO_TRACE=scheduler
export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
export HAILO_TRACE_PATH="$TRACES_DIR"
export HAILO_MONITOR=1

# HRTT 파일을 라벨 붙여서 옮기는 공통 함수. 인자: LABEL
collect_hrtt() {
    local LABEL=$1
    local LATEST_HRTT=""
    for i in $(seq 1 35); do
        LATEST_HRTT=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST_HRTT" ] && break; sleep 1
    done
    if [ -n "$LATEST_HRTT" ]; then
        TS=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
        NEW="${TRACES_DIR}/${LABEL}_${TS}.hrtt"
        mv "$LATEST_HRTT" "$NEW"
        echo "  HRTT: $(basename "$NEW")"
    else
        echo "  [!] HRTT 미생성 (${LABEL})"
    fi
}

# ---------------- 우리 코드(A/B/C) 1회 실행 ----------------
# 인자: COND(A/B/C) ENABLE_PP FORCE_F32 USE_D USE_S USE_P LABEL
run_ours() {
    local COND=$1 ENABLE_PP=$2 FORCE_F32=$3 USE_D=$4 USE_S=$5 USE_P=$6 LABEL=$7
    echo "=== [$COND] D=$USE_D S=$USE_S P=$USE_P ($LABEL) ==="

    sed -i "s/^#define ENABLE_POSTPROCESS .*/#define ENABLE_POSTPROCESS  $ENABLE_PP/" "$SRC_OURS"
    sed -i "s/^#define FORCE_OUTPUT_FLOAT32 .*/#define FORCE_OUTPUT_FLOAT32  $FORCE_F32/" "$SRC_OURS"
    sed -i "s/^#define USE_DET .*/#define USE_DET    $USE_D/"  "$SRC_OURS"
    sed -i "s/^#define USE_SEG .*/#define USE_SEG    $USE_S/"  "$SRC_OURS"
    sed -i "s/^#define USE_POSE .*/#define USE_POSE   $USE_P/" "$SRC_OURS"

    g++ "$SRC_OURS" -o "$BIN_OURS" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "  [!] 컴파일 실패 — 이 조건 건너뜀"; return; fi

    rm -f "$TRACES_DIR"/hailort_*.hrtt
    ./"$BIN_OURS" 1 "$CSV_DIR/results_${COND}.csv"
    collect_hrtt "${COND}_${USE_D}D-${USE_S}S-${USE_P}P"
}

# ---------------- 조교님 코드(D) 1회 실행 ----------------
# 인자: USE_D USE_S USE_P LABEL
run_ta() {
    local USE_D=$1 USE_S=$2 USE_P=$3 LABEL=$4
    echo "=== [D] D=$USE_D S=$USE_S P=$USE_P ($LABEL) ==="

    sed -i "s/^#define USE_DET .*/#define USE_DET    $USE_D/"  "$SRC_TA"
    sed -i "s/^#define USE_SEG .*/#define USE_SEG    $USE_S/"  "$SRC_TA"
    sed -i "s/^#define USE_POSE .*/#define USE_POSE   $USE_P/" "$SRC_TA"

    g++ "$SRC_TA" -o "$BIN_TA" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "  [!] 컴파일 실패 — 이 조건 건너뜀"; return; fi

    rm -f "$TRACES_DIR"/hailort_*.hrtt
    ./"$BIN_TA" 1 "$CSV_DIR/results_D.csv"
    collect_hrtt "D_${USE_D}D-${USE_S}S-${USE_P}P"
}

# ================= 실행 루프 =================
# (USE_D USE_S USE_P) 7개 조합: single 3 + 2조합 3 + 3조합 1
COMBOS=(
  "1 0 0"   # Det
  "0 1 0"   # Seg
  "0 0 1"   # Pose
  "1 1 0"   # Det+Seg
  "1 0 1"   # Det+Pose
  "0 1 1"   # Seg+Pose
  "1 1 1"   # Det+Seg+Pose
)

echo "----- [A] 후처리 O + FLOAT32 (평소 기본 상태) -----"
for c in "${COMBOS[@]}"; do run_ours A 1 1 $c "A"; done

echo "----- [B] 후처리 X + AUTO (조교님과 동일조건) -----"
for c in "${COMBOS[@]}"; do run_ours B 0 0 $c "B"; done

echo "----- [C] 후처리 X + FLOAT32 (변환비용만 남김) -----"
for c in "${COMBOS[@]}"; do run_ours C 0 1 $c "C"; done

echo "----- [D] 조교님 코드 (infer_scheduler_new.cpp) -----"
for c in "${COMBOS[@]}"; do run_ta $c "D"; done

# ---------- 원본 설정 복구 ----------
cp "${SRC_OURS}.bak" "$SRC_OURS"
cp "${SRC_TA}.bak" "$SRC_TA"
echo ""
echo "[복구] ${SRC_OURS}, ${SRC_TA} 원본 설정으로 복구 완료"
echo "===== 완료! 4조건 x 7조합 x 1회 = 28회 ====="
echo "CSV    : $CSV_DIR/results_{A,B,C,D}.csv"
echo "HRTT   : $TRACES_DIR"
