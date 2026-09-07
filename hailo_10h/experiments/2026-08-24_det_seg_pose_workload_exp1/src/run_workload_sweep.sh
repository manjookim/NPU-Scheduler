#!/bin/bash
# =============================================================================
# Hailo10H: det(yolov8s)/seg(yolov8s_seg)/pose(yolov8s_pose) 워크로드 조합 실험
# 조건: 단독 3개(det/seg/pose) + 쌍 3개(det+seg/det+pose/seg+pose) + 트리플 1개 = 7조건
#       각 조건 3회 반복(나중에 평균) = 21회 실행
# 스케줄러 파라미터(priority/threshold/timeout/batch)는 전혀 건드리지 않음 — HailoRT 기본값 그대로.
# CSV 컬럼은 hailo_8L/csv_writer.hpp와 동일 구성(HRTT 전용 컬럼 제외). npu_percent는 벤치가
# hailortcli monitor를 자식 프로세스로 띄워 채운다 — 별도 모니터 터미널을 띄울 필요 없음.
# 데이터셋: ~/datasets/sampled_val2017 (RPi1/Hailo-8L 프로젝트와 동일한 673장, 새로 안 만듦)
#
# 실행 위치: ~/hailo10h_sched_exp1/ (resources/에 3개 hef 있어야 함)
# 사용법: chmod +x run_workload_sweep.sh && ./run_workload_sweep.sh
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1   # src/에서 실행해도 ~/hailo10h_sched_exp1으로 이동
BIN=./hailo10h_sched_bench
CSV=csv/results_det_seg_pose_workload_full.csv   # 컬럼 구성이 8L과 맞춰지며 바뀌어서 새 파일로 분리
                                                # (예전 스키마 파일에 append하면 벤치가 헤더 불일치로 멈춘다)
IMG_DIR="$HOME/datasets/sampled_val2017"
REPEAT=3

CONDITIONS=(
  "det"
  "seg"
  "pose"
  "det,seg"
  "det,pose"
  "seg,pose"
  "det,seg,pose"
)

mkdir -p csv logs traces

for cond in "${CONDITIONS[@]}"; do
    label=$(echo "$cond" | tr ',' '_')
    for run in $(seq 1 $REPEAT); do
        echo "===== [$label] run $run/$REPEAT ====="
        "$BIN" --models "$cond" --label "$label" --run_id "$run" \
               --csv "$CSV" --img_dir "$IMG_DIR" --trace_dir traces 2>&1 | tee "logs/${label}_run${run}.log"
        echo ""
        sleep 1
    done
done

echo "===== 전체 완료 ====="
echo "CSV: $CSV"
wc -l "$CSV"
