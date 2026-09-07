#!/bin/bash
# =============================================================================
# run_inflight_sweep.sh — in-flight 깊이 스윕 (H10 전용, 8L에는 대응물이 없음)
#
# 왜 필요한가: 8L VStreams 파이프라인의 실측 in-flight는 약 11프레임이었고, 기존 H10
# 코드(run_async 직후 wait)는 0.77프레임이었다. latency 10배 차이의 정체가 이 깊이 차이다.
# D를 1 -> 16으로 올리며 (FPS, latency, npu_percent)를 찍으면
#   - FPS가 포화되는 지점 = NNC가 실제로 꽉 차는 깊이
#   - latency가 D에 비례해 늘어나는 구간 = 큐 체류시간이 지배하는 구간
# 이 두 가지가 한 표에서 보인다. 8L 값과 나란히 놓을 D도 여기서 정한다.
#
# 확장 컬럼(backend/inflight_depth/service_latency 등)이 필요하면 소스에서
# CSV_EXTENSION_COLUMNS를 1로 바꾸고 다시 빌드할 것(기본값 0 = 8L과 100% 동일한 49컬럼).
# =============================================================================
set -u
cd "$(dirname "$0")" || exit 1

BIN=./infer_scheduler_h10
MODELS=${MODELS:-det}
IMG_DIR=${IMG_DIR:-$HOME/datasets/sampled_val2017/}   # 끝의 / 필수 (8L get_image_files 규약)
NUM_IMAGES=${NUM_IMAGES:-600}
REPEAT=${REPEAT:-3}
DEPTHS=${DEPTHS:-"1 2 4 8 12 16"}

label=$(echo "$MODELS" | tr ',' '_')
# 깊이별로 파일을 나눈다 — 코어 49컬럼에 D를 적을 칸이 없어서, 한 파일에 모으면 구분 불가.
CSV_PREFIX=${CSV_PREFIX:-csv/results_h10_inflight}

mkdir -p csv logs

echo "모델=$MODELS / 깊이=$DEPTHS / 이미지=$NUM_IMAGES장 / 반복=$REPEAT회"
echo "[주의] CSV 코어 49컬럼에는 D를 적을 칸이 없어 깊이별로 파일을 나눠 저장한다:"
echo "       ${CSV_PREFIX}<D>_${label}.csv"
echo

for d in $DEPTHS; do
    for run in $(seq 1 $REPEAT); do
        echo "===== [inflight=$d] run $run/$REPEAT ====="
        "$BIN" "$run" "${CSV_PREFIX}${d}_${label}.csv" \
               --models "$MODELS" \
               --img_dir "$IMG_DIR" \
               --num_images "$NUM_IMAGES" \
               --api infer --inflight "$d" \
            2>&1 | tee "logs/inflight${d}_${label}_run${run}.log"
        echo ""
        sleep 1
    done
done

echo "===== 전체 완료 ====="
echo "각 로그의 '[in-flight 추정]' 줄이 실제로 D만큼 찼는지 확인할 것."
ls -1 ${CSV_PREFIX}*_${label}.csv 2>/dev/null
