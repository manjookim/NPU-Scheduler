#!/bin/bash
# ------------------------------------------------------------------------
# batch=1 조건에서 "single model" vs "multi model 동시실행" latency 비교용
# 진단 스크립트. 조교님 지적사항(batch=1, single model latency가 multi model보다
# 10배 이상 크게 나오는 이유 확인) 대응을 위해, 아래 4개 조합을 각각 3회씩
# 자동으로 빌드+실행해서 CSV 한 파일에 누적한다.
#
#   1) Detection 단독   (USE_DET=1, USE_SEG=0, USE_POSE=0)
#   2) Segmentation 단독 (USE_DET=0, USE_SEG=1, USE_POSE=0)
#   3) Pose 단독        (USE_DET=0, USE_SEG=0, USE_POSE=1)
#   4) 세 모델 동시 실행 (USE_DET=1, USE_SEG=1, USE_POSE=1)
#
# 4개 조합 모두 BATCH_DET=BATCH_SEG=BATCH_POSE=1 로 고정.
# threshold/timeout/priority는 기존 기본값(THRESHOLD_*=1, TIMEOUT_*=0, PRIORITY_*=15) 그대로 둠.
#
# 실행 위치: 보드(~/hailo_cpp_test/), infer_scheduler_hailo8.cpp와 같은 디렉토리.
# 사용법: chmod +x run_batch1_investigation.sh && ./run_batch1_investigation.sh
#
# [주의] 이 스크립트는 infer_scheduler_hailo8.cpp의 #define 값을 sed로 직접 고쳐서
# 재컴파일한다. 실행 전 원본을 .bak로 백업해두고, 스크립트 종료 시 원래 상태로
# 복구한다(중간에 Ctrl+C로 중단하면 .bak에서 수동 복구할 것:
#   cp infer_scheduler_hailo8.cpp.bak infer_scheduler_hailo8.cpp).
# ------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

SRC=infer_scheduler_hailo8.cpp
BIN=infer_scheduler_hailo8_b1
CSV=results_batch1_investigation.csv
REPEATS=3
START_RUN_ID=100

if [ ! -f "$SRC" ]; then
    echo "[오류] $SRC 를 찾을 수 없음. hailo_cpp_test 디렉토리에서 실행할 것."
    exit 1
fi

cp "$SRC" "${SRC}.bak"
echo "[백업] ${SRC} -> ${SRC}.bak"

set_defines() {
    local use_det=$1 use_seg=$2 use_pose=$3
    sed -i "s/^#define BATCH_DET.*/#define BATCH_DET       1/"  "$SRC"
    sed -i "s/^#define BATCH_SEG.*/#define BATCH_SEG       1/"  "$SRC"
    sed -i "s/^#define BATCH_POSE.*/#define BATCH_POSE      1/" "$SRC"
    sed -i "s/^#define USE_DET.*/#define USE_DET    ${use_det}/"   "$SRC"
    sed -i "s/^#define USE_SEG.*/#define USE_SEG    ${use_seg}/"   "$SRC"
    sed -i "s/^#define USE_POSE.*/#define USE_POSE   ${use_pose}/" "$SRC"
}

build() {
    echo "  빌드 중..."
    g++ "$SRC" -o "$BIN" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
}

run_id=$START_RUN_ID

# "1 0 0:det_only" 형식: USE_DET USE_SEG USE_POSE : 라벨
CONFIGS=(
    "1 0 0:det_only"
    "0 1 0:seg_only"
    "0 0 1:pose_only"
    "1 1 1:all_three"
)

for cfg in "${CONFIGS[@]}"; do
    params="${cfg%%:*}"
    label="${cfg##*:}"
    read -r use_det use_seg use_pose <<< "$params"

    echo ""
    echo "========================================"
    echo " 조합: $label (USE_DET=$use_det USE_SEG=$use_seg USE_POSE=$use_pose, batch=1)"
    echo "========================================"
    set_defines "$use_det" "$use_seg" "$use_pose"
    build

    for rep in $(seq 1 $REPEATS); do
        echo ""
        echo "  --- $label 반복 $rep/$REPEATS (run_id=$run_id) ---"
        ./"$BIN" "$run_id" "$CSV"
        run_id=$((run_id + 1))
    done
done

# 원본 설정 복구
cp "${SRC}.bak" "$SRC"
echo ""
echo "[복구] ${SRC} 원본 설정으로 복구 완료"
echo "[완료] 결과: $CSV (run_id ${START_RUN_ID}~$((run_id-1)))"
