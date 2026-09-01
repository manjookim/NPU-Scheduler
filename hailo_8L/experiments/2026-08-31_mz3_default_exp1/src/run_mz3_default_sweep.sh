#!/bin/bash
# =============================================================================
# Hailo-8L: Model Zoo 3모델 워크로드 조합 실험 — "파라미터 미설정(기본값)" 조건
#   ssd_mobilenet_v1 (Detection, 300x300, single context)
#   deeplab_v3_mobilenet_v2_wo_dilation (Segmentation, 513x513, multi context x3)
#   mobilenet_v2_1.0 (Classification, 224x224, single context)
#
# 조건: 단독 3 + 쌍 3 + 트리플 1 = 7조건, 각 3회 반복 = 21회 실행
#       (직전 Hailo-10H 실험 run_workload_sweep.sh 와 동일한 설계 — 결과 비교 가능하게 맞춤)
# 스케줄러 파라미터(priority/threshold/timeout/batch)는 전혀 건드리지 않음 — HailoRT 기본값.
# 데이터셋: ~/datasets/sampled_val2017 (673장)
#
# 실행: chmod +x run_mz3_default_sweep.sh && ./run_mz3_default_sweep.sh
# =============================================================================
set -u
cd "$HOME/mz3_exp" || exit 1

BIN=./mz3_sched_bench
CSV="$HOME/mz3_exp/csv/results_mz3_default.csv"
IMG_DIR="$HOME/datasets/sampled_val2017"
TRACES_DIR="$HOME/mz3_exp/traces"
FRAMES=673       # sampled_val2017 전량
REPEAT=3

mkdir -p csv logs "$TRACES_DIR"

# HRTT 트레이싱 (기본 ON 방침). 30초 bounded dump.
export HAILO_TRACE=scheduler
export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
export HAILO_TRACE_PATH="$TRACES_DIR"

# "라벨:ssd,deeplab,mnv2" 플래그
CONDITIONS=(
  "ssd:1,0,0"
  "deeplab:0,1,0"
  "mnv2:0,0,1"
  "ssd_deeplab:1,1,0"
  "ssd_mnv2:1,0,1"
  "deeplab_mnv2:0,1,1"
  "ssd_deeplab_mnv2:1,1,1"
)

echo "===== MZ 3모델 기본값 스윕 시작: $(date) ====="
for cond in "${CONDITIONS[@]}"; do
    label="${cond%%:*}"
    flags="${cond##*:}"
    IFS=',' read -r F_SSD F_DEEPLAB F_MNV2 <<< "$flags"

    for run in $(seq 1 $REPEAT); do
        echo "===== [$label] run $run/$REPEAT ====="
        "$BIN" --ssd "$F_SSD" --deeplab "$F_DEEPLAB" --mnv2 "$F_MNV2" \
               --fps 0 --frames "$FRAMES" --run-id "$run" \
               --tag "$label" --csv "$CSV" --images "$IMG_DIR" \
               2>&1 | tee "logs/${label}_run${run}.log"
        echo ""
        sleep 2
    done
done

echo "===== 전체 완료: $(date) ====="
echo "CSV: $CSV"
wc -l "$CSV"
