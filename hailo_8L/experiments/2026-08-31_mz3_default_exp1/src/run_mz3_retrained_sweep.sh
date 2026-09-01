#!/bin/bash
# =============================================================================
# Hailo-8L: MZ 3모델 Default 스윕 — "재학습판" 조건
#
#   ssd_mobilenet_v1                     : Model Zoo 사전학습 (COCO) — 재학습 불필요
#   deeplab_v3_mobilenet_v2_wo_dilation  : Model Zoo 사전학습 (PASCAL VOC) — 재학습 대기중
#   mobilenet_v2_1.0                     : ★ GTSRB 43클래스 재학습판 ★
#
# 2026-08-31_mz3_default_exp1(사전학습 3종)과 **완전히 동일한 설계**로 돌려서
# "같은 칩 · 같은 구조 · 다른 가중치"일 때 스케줄링 특성이 변하는지 본다.
#   조건 7 x 3회 x 입력속도 2종(fps=0 / fps=30) = 42런
#
# [코드 수정 불필요] mz3_sched_bench.cpp 는 출력을 디코딩하지 않고 버퍼만 읽으며
# vstream 포맷도 AUTO 라, 출력 클래스가 1001 -> 43 으로 바뀌어도 그대로 동작한다.
# HEF 파일명만 원본과 같게 맞춰 별도 폴더에 두면 --res 로 전환된다.
#
# 사전 준비 (RPi, npu-rpi1):
#   mkdir -p ~/mz3_exp/resources_retrained
#   cp ~/mz3_exp/resources/ssd_mobilenet_v1.hef                    ~/mz3_exp/resources_retrained/
#   cp ~/mz3_exp/resources/deeplab_v3_mobilenet_v2_wo_dilation.hef ~/mz3_exp/resources_retrained/
#   cp <재학습>/mobilenet_v2_1.0_gtsrb_sc.hef ~/mz3_exp/resources_retrained/mobilenet_v2_1.0.hef
#
# 실행: chmod +x run_mz3_retrained_sweep.sh && ./run_mz3_retrained_sweep.sh
# =============================================================================
set -u
cd "$HOME/mz3_exp" || exit 1

BIN=./mz3_sched_bench
RES="$HOME/mz3_exp/resources_retrained"
IMG_DIR="$HOME/datasets/sampled_val2017"
TRACES_DIR="$HOME/mz3_exp/traces_retrained"
FRAMES=673
REPEAT=3

mkdir -p csv logs "$TRACES_DIR"

# 재학습 HEF 가 실제로 43클래스인지 먼저 확인 (원본을 잘못 복사하는 실수 방지)
echo "===== HEF 확인 ====="
for h in ssd_mobilenet_v1 deeplab_v3_mobilenet_v2_wo_dilation mobilenet_v2_1.0; do
    if [ ! -f "$RES/$h.hef" ]; then
        echo "[에러] 없음: $RES/$h.hef"; exit 1
    fi
    echo "--- $h"
    hailortcli parse-hef "$RES/$h.hef" 2>/dev/null | grep -iE "context|output|shape" | head -8
done
echo ""
echo "위에서 mobilenet_v2_1.0 의 출력이 43 이어야 재학습판입니다. 1001 이면 원본입니다."
read -p "계속하려면 Enter, 중단하려면 Ctrl+C: " _

# HRTT 트레이싱. 가장 짧은 조건(수초)보다 작아야 덤프가 트리거되므로 3으로 둔다
# (2026-08-08 기록: 30 이면 짧은 조건에서 .hrtt 가 아예 안 생김)
export HAILO_TRACE=scheduler
export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=3
export HAILO_TRACE_PATH="$TRACES_DIR"

CONDITIONS=(
  "ssd:1,0,0"
  "deeplab:0,1,0"
  "mnv2:0,0,1"
  "ssd_deeplab:1,1,0"
  "ssd_mnv2:1,0,1"
  "deeplab_mnv2:0,1,1"
  "ssd_deeplab_mnv2:1,1,1"
)

run_set () {
    local FPS="$1" CSV="$2" PREFIX="$3"
    echo ""
    echo "############ 입력속도 fps=$FPS  ->  $CSV ############"
    for cond in "${CONDITIONS[@]}"; do
        local label="${cond%%:*}" flags="${cond##*:}"
        IFS=',' read -r F_SSD F_DEEPLAB F_MNV2 <<< "$flags"
        for run in $(seq 1 $REPEAT); do
            echo "===== [$label] fps=$FPS run $run/$REPEAT ====="
            "$BIN" --ssd "$F_SSD" --deeplab "$F_DEEPLAB" --mnv2 "$F_MNV2" \
                   --fps "$FPS" --frames "$FRAMES" --run-id "$run" \
                   --tag "$label" --csv "$CSV" --images "$IMG_DIR" --res "$RES" \
                   2>&1 | tee "logs/${PREFIX}_${label}_run${run}.log"
            echo ""
            sleep 2
        done
    done
    echo "완료: $CSV  ($(wc -l < "$CSV") 행, 헤더 포함)"
}

echo "===== 재학습판 스윕 시작: $(date) ====="
run_set 0  "$HOME/mz3_exp/csv/results_mz3_retrained.csv"       "rt"
run_set 30 "$HOME/mz3_exp/csv/results_mz3_retrained_fps30.csv" "rt_fps30"
echo "===== 전체 완료: $(date) ====="
echo ""
echo "PC 로 가져오기:"
echo "  scp -P 40021 'rpi1@155.230.16.157:~/mz3_exp/csv/results_mz3_retrained*.csv' ."
