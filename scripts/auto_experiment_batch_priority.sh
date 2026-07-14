#!/bin/bash
# =============================================================================
# batch × priority 전체 조합 실험 (threshold / timeout 제외)
# -----------------------------------------------------------------------------
# 요구사항:
#   1) batch_size ∈ {1, 10, 50, 63}  — 한 실행에서 모든 활성 모델에 통일 적용
#      (HailoRT batch_size 상한 63 — 100 이상은 configure 단계에서 거부됨)
#   2) priority   ∈ {0, 15, 31}       — 활성 모델별로 독립 교차
#   3) 단일 / 2모델 / 3모델 (활성 모델의 모든 조합) 실험
#   4) threshold / timeout 은 이번 실험 대상이 아니므로 둘 다 0 고정
#
# 경우의 수 (활성 모델 = 켜진 모델 기준):
#   - 단일 : 3 모델 × 4 batch × 3 priority                =  36
#   - 2모델: 3 쌍   × 4 batch × 3² priority (=9)          = 108
#   - 3모델: 1      × 4 batch × 3³ priority (=27)         = 108
#   합계 252 조건 × REPEAT(3) = 756 회 실행
#
# 주의:
#   - threshold=0 (제외 대상). 만약 HailoRT가 0을 거부하면 set_scheduler_threshold가
#     [실패]로 찍히고 기본값(1)로 남는다 — 실행 로그의 [적용확인]에서 확인 가능.
#   - 입력 데이터셋: 조교 제공 sampled_val2017 (RPi에 이미 전송됨).
#     infer_scheduler.cpp 의 IMG_DIR 이 실제 저장 경로와 다르면 그 #define 만 수정할 것.
#   - infer_scheduler 는 argv[2] 로 CSV 경로를 받아 "추론 중 측정값"만 저장한다.
#     npu_percent 는 이 스크립트가 npu_log.txt 에서 채우고,
#     나머지 HRTT 전용 컬럼(activation/avg_fps/avg_latency/max_latency/idle_time/switches_per_s)은
#     보존된 .hrtt 트레이스에서 이후 단계에 채운다.
# =============================================================================

set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 ~/hailo_cpp_test 없음"; exit 1; }

# ---------------- 설정 ----------------
REPEAT=3
batches=(1 10 50 63)     # HailoRT batch_size 상한 63
priorities=(0 15 31)
NUM_IMAGES=0            # 0 = sampled_val2017 전체 사용

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_batch_priority_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_batch_priority_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"          # 시트별(사용모델수 × batch) CSV 12개가 여기에 생성됨
TRACES_DIR="$OUTDIR/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"

echo "실험 폴더 : $EXP_DIR"
echo "CSV 폴더  : $CSV_DIR  (사용모델수 1/2/3 × batch 4종 = 시트 12개)"
echo "조건      : batch{1,10,50,63} × priority{0,15,31} 활성모델 교차, 단일+2모델+3모델"
echo "총 실행   : 252 조건 × ${REPEAT}회 = $((252 * REPEAT))회"
echo ""

# ---------------- 한 조건 1회 실행 ----------------
# 인자: USE_D USE_S USE_P  PRI_D PRI_S PRI_P  BATCH  RUN_ID
compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local PRI_D=$4 PRI_S=$5 PRI_P=$6
    local BATCH=$7 RUN_ID=$8

    # 시트 라우팅: 사용 모델 수 × batch → 12개 CSV 중 하나로 저장
    local NMODEL=$((USE_D + USE_S + USE_P))
    local TARGET_CSV="$CSV_DIR/results_${NMODEL}model_b${BATCH}.csv"

    echo "=== D=$USE_D(p$PRI_D) S=$USE_S(p$PRI_S) P=$USE_P(p$PRI_P) batch=$BATCH run$RUN_ID → $(basename "$TARGET_CSV") ==="

    # ---- infer_scheduler.cpp 파라미터 주입 ----
    # batch: 세 모델 통일
    sed -i "s/^#define BATCH_DET .*/#define BATCH_DET       $BATCH/"   infer_scheduler.cpp
    sed -i "s/^#define BATCH_SEG .*/#define BATCH_SEG       $BATCH/"   infer_scheduler.cpp
    sed -i "s/^#define BATCH_POSE .*/#define BATCH_POSE      $BATCH/"  infer_scheduler.cpp
    # threshold: 제외 대상 → 0 고정
    sed -i "s/^#define THRESHOLD_DET .*/#define THRESHOLD_DET   0/"    infer_scheduler.cpp
    sed -i "s/^#define THRESHOLD_SEG .*/#define THRESHOLD_SEG   0/"    infer_scheduler.cpp
    sed -i "s/^#define THRESHOLD_POSE .*/#define THRESHOLD_POSE  0/"   infer_scheduler.cpp
    # timeout: 제외 대상 → 0 고정
    sed -i "s/^#define TIMEOUT_DET_MS .*/#define TIMEOUT_DET_MS   0/"  infer_scheduler.cpp
    sed -i "s/^#define TIMEOUT_SEG_MS .*/#define TIMEOUT_SEG_MS   0/"  infer_scheduler.cpp
    sed -i "s/^#define TIMEOUT_POSE_MS .*/#define TIMEOUT_POSE_MS  0/" infer_scheduler.cpp
    # 활성 모델
    sed -i "s/^#define USE_DET .*/#define USE_DET    $USE_D/"    infer_scheduler.cpp
    sed -i "s/^#define USE_SEG .*/#define USE_SEG    $USE_S/"    infer_scheduler.cpp
    sed -i "s/^#define USE_POSE .*/#define USE_POSE   $USE_P/"   infer_scheduler.cpp
    # priority (활성 모델별 교차, 비활성은 0 — CSV 기록용, 스케줄링엔 영향 없음)
    sed -i "s/^#define PRIORITY_DET .*/#define PRIORITY_DET    $PRI_D/"   infer_scheduler.cpp
    sed -i "s/^#define PRIORITY_SEG .*/#define PRIORITY_SEG    $PRI_S/"   infer_scheduler.cpp
    sed -i "s/^#define PRIORITY_POSE .*/#define PRIORITY_POSE   $PRI_P/"  infer_scheduler.cpp
    # 사용 이미지 수
    sed -i "s/^#define NUM_IMAGES .*/#define NUM_IMAGES      $NUM_IMAGES/" infer_scheduler.cpp

    # ---- 컴파일 ----
    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then echo "  [!] 컴파일 실패 — 이 조건 건너뜀"; return; fi

    # ---- NPU/CPU/MEM 모니터 시작 ----
    > npu_log.txt
    rm -f /tmp/hmon_files/*
    rm -f "$TRACES_DIR"/hailort_*.hrtt
    source ~/hailo_platform_venv/bin/activate 2>/dev/null
    python3 hailo_utilization.py &
    NPU_PID=$!
    sleep 2

    # ---- HRTT 트레이스 환경변수 ----
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1

    # ---- 추론 실행 (cpp 가 해당 시트 CSV 에 한 행 append: 측정값 + NaN placeholder) ----
    ./infer_scheduler "$RUN_ID" "$TARGET_CSV"

    # ---- 모니터 종료 ----
    kill $NPU_PID 2>/dev/null; sleep 1; rm -f /tmp/hmon_files/*

    # ---- npu_percent 채우기: 방금 append 된 마지막 행의 NaN → npu_log 평균 ----
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
    print("  [npu] 활성 구간 없음 — npu_percent NaN 유지")
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

    # ---- HRTT 트레이스 조건명으로 보존 (이후 HRTT 전용 컬럼 채우기용) ----
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
}

# ================= 실행 루프 =================
for BATCH in "${batches[@]}"; do

  echo "########## batch=$BATCH ##########"

  # ---------- 단일 모델 (3 모델 × 3 priority) ----------
  echo "----- 단일 모델 -----"
  for p in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 0 0 $p 0 0 $BATCH $run; done   # Det
  done
  for p in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 0 1 0 0 $p 0 $BATCH $run; done   # Seg
  done
  for p in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 0 0 1 0 0 $p $BATCH $run; done   # Pose
  done

  # ---------- 2모델 (3 쌍 × 3×3 priority) ----------
  echo "----- 2모델 -----"
  # Det + Seg
  for pd in "${priorities[@]}"; do for ps in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 1 0 $pd $ps 0 $BATCH $run; done
  done; done
  # Det + Pose
  for pd in "${priorities[@]}"; do for pp in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 0 1 $pd 0 $pp $BATCH $run; done
  done; done
  # Seg + Pose
  for ps in "${priorities[@]}"; do for pp in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 0 1 1 0 $ps $pp $BATCH $run; done
  done; done

  # ---------- 3모델 (3×3×3 priority) ----------
  echo "----- 3모델 -----"
  for pd in "${priorities[@]}"; do for ps in "${priorities[@]}"; do for pp in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do compile_and_run 1 1 1 $pd $ps $pp $BATCH $run; done
  done; done; done

done

echo ""
echo "===== 완료! 252 조건 × ${REPEAT}회 = $((252 * REPEAT))회 ====="
echo "시트 CSV(12개): $CSV_DIR"
echo "HRTT 트레이스   : $TRACES_DIR (activation/fps/avg·max_latency/idle_time/switches_per_s 채우기용)"

# ---- 12개 시트 CSV → 시트 12개짜리 xlsx 로 합치기 (openpyxl 있으면) ----
XLSX="$OUTDIR/results_batch_priority.xlsx"
if python3 - "$CSV_DIR" "$XLSX" <<'PY'
import sys, os, csv, glob
try:
    from openpyxl import Workbook
except ImportError:
    print("  [xlsx] openpyxl 없음 — CSV 12개만 생성됨(PC에서 combine_to_xlsx.py 실행 가능)"); sys.exit(1)
csv_dir, xlsx_path = sys.argv[1], sys.argv[2]
wb = Workbook(); wb.remove(wb.active)
# 사용모델수(1,2,3) × batch(1,10,50,100) 순서로 시트 생성
for n in (1, 2, 3):
    for b in (1, 10, 50, 63):
        path = os.path.join(csv_dir, f"results_{n}model_b{b}.csv")
        ws = wb.create_sheet(title=f"{n}model_b{b}")
        if os.path.exists(path):
            for row in csv.reader(open(path)):
                ws.append(row)
wb.save(xlsx_path)
print(f"  [xlsx] 시트 12개 저장: {xlsx_path}")
PY
then :; fi
