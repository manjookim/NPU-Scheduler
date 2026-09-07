#!/bin/bash
# =============================================================================
# run_workload_sweep_h10.sh — Hailo-8L "default_workload" 실험과 같은 7조건 스윕
#   단독 3개(det/seg/pose) + 쌍 3개 + 트리플 1개 = 7조건, 각 3회 = 21회 실행
#
# 8L 대응 스크립트: hailo_8L/scripts/auto_experiment_default_workload.sh
#   - 8L은 sed로 #define USE_DET/USE_SEG/USE_POSE를 패치하고 매번 재컴파일했다.
#   - 이 이식본은 --models 플래그로 같은 일을 한다. batch와 NUM_IMAGES는 소스의 #define
#     값을 그대로 쓴다.
#
# [H10 백엔드] --api infer 가 기본이자 유일한 선택지다. VStreams/ConfiguredNetworkGroup
#   경로는 H10에서 create_configure_params가 HAILO_NOT_IMPLEMENTED(7)로 실패한다
#   (HailoRT 안내: "use InferModel instead", 실측 2026-08-31).
#
# [실행 조건] 후처리 없음(출력 포맷 AUTO) + 스케줄러 파라미터 setter 주석 처리
#   => HailoRT 기본값(threshold=1, timeout=0ms, priority=16)으로 동작한다.
#      8L 쪽 비교 대상은 B조건 또는 D조건(ENABLE_POSTPROCESS=0, FORCE_OUTPUT_FLOAT32=0) CSV.
#
# CSV: 8L csv_writer.hpp와 동일한 49컬럼. 8L 결과 CSV와 그대로 concat 가능.
# npu_percent: 벤치가 hailortcli monitor를 자식 프로세스로 띄워 채운다(별도 터미널 불필요).
# =============================================================================
set -u
cd "$(dirname "$0")" || exit 1

BIN=./infer_scheduler_h10
API=${API:-infer}             # infer(H10 기본/유일) | vstreams(H10 미지원, 8L 대조용)
# in-flight 깊이.
#   auto (기본) = 조합별로 8L 실측 L에 맞춘 모델별 D를 벤치가 스스로 고른다.
#                 (infer_scheduler_h10.cpp의 L8L 표 — results_B.csv에서 역산한 값)
#   12          = 예전처럼 전 모델 공통. det에만 맞고 seg/3모델은 15~17% 어긋난다.
#   11,9,9      = 모델별 직접 지정.
INFLIGHT=${INFLIGHT:-auto}
TAG=$INFLIGHT; [ "$TAG" = "auto" ] && TAG=matched8L
# [주의] CSV 코어 49컬럼에는 in-flight 깊이를 적는 칸이 없다(8L 스키마 유지). 그래서
#        깊이를 파일명에 넣어 구분한다. 안 그러면 D=12와 D=8 결과가 한 파일에 섞여
#        나중에 구분할 방법이 없다. (CSV_EXTENSION_COLUMNS=1로 빌드하면 컬럼으로도 남는다)
CSV=${CSV:-csv/results_h10_workload_${API}_if${TAG}.csv}
IMG_DIR=${IMG_DIR:-$HOME/datasets/sampled_val2017/}   # 끝의 / 필수 (8L get_image_files 규약)
NUM_IMAGES=${NUM_IMAGES:-600} # 8L 기존 실험과 동일. 반드시 맞출 것.
REPEAT=${REPEAT:-3}

CONDITIONS=("det" "seg" "pose" "det,seg" "det,pose" "seg,pose" "det,seg,pose")

mkdir -p csv logs

EXTRA=""
if [ "$API" = "infer" ]; then EXTRA="--inflight $INFLIGHT"; fi

echo "백엔드=$API $EXTRA / 이미지=$NUM_IMAGES장 / 반복=$REPEAT회 / CSV=$CSV"
echo

for cond in "${CONDITIONS[@]}"; do
    label=$(echo "$cond" | tr ',' '_')
    for run in $(seq 1 $REPEAT); do
        echo "===== [$label] run $run/$REPEAT ====="
        "$BIN" "$run" "$CSV" \
               --models "$cond" \
               --img_dir "$IMG_DIR" \
               --num_images "$NUM_IMAGES" \
               --api "$API" $EXTRA \
            2>&1 | tee "logs/${API}_${TAG}_${label}_run${run}.log"
        echo ""
        sleep 1
    done
done

echo "===== 전체 완료 ====="
echo "CSV: $CSV"
wc -l "$CSV"
echo
echo "[확인] 각 로그 끝의 'in-flight 검증' 표가 전부 OK인지 볼 것:"
echo "  grep -A6 'in-flight 검증' logs/${API}_${TAG}_*_run*.log | grep -E 'det|seg|pose|=>'"
