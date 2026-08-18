#!/bin/bash
# ------------------------------------------------------------------------
# "Detection만 YOLOv5로 바꿔서 NPU 후처리 vs CPU 후처리를, det / det+seg / det+pose /
# det+seg+pose 4가지 조합 각각에서 비교" 실험의 실행기. (범위 확정 경위는
# infer_scheduler_hailo8_v5det_cpu.cpp 상단 주석 / PROJECT_SUMMARY.md §6 참고 —
# Detection만 v5로 바꾸고 Seg/Pose는 기존 YOLOv8(CPU) 그대로 두기로 함. 근거: Seg는
# v5 앵커기반 디코딩 새로 짜야 해서 실기 검증 없이 위험하고, Pose는 애초에 공식
# YOLOv5-Pose 모델 자체가 없음.)
#
# run_v5det_seg_pose.sh(Detection NPU 후처리)와 run_v5det_cpu_seg_pose.sh(Detection
# CPU 후처리)를 같은 조건(batch=1/threshold=1/timeout=0/priority=0)·같은 4조합으로
# 순서대로 실행한다(각 스크립트 내부에서 4조합 × 3회 = 12회씩, 총 24회). 두 스크립트를
# 각각 따로 실행해도 되고, 이 파일로 한 번에 돌려도 된다. 크래시로 중간에 멈춰도 각
# 스크립트가 이미 모은 결과 기준으로 재개하므로 이 파일을 그냥 다시 실행하면 된다.
#
# 실행 위치: 보드(rpi4, ~/hailo_cpp_test/), hailo_8/scripts/와 같은 위치 관계 유지.
# 사용법: chmod +x run_v5det_npu_vs_cpu_compare.sh && ./run_v5det_npu_vs_cpu_compare.sh
# ------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

echo "############################################################"
echo "# [1/2] Detection = YOLOv5 NPU 후처리 (yolov5xs_wo_spp_nms_core.hef) — 4조합 x 3회"
echo "############################################################"
./run_v5det_seg_pose.sh

echo ""
echo "############################################################"
echo "# [2/2] Detection = YOLOv5 CPU 후처리 (yolov5xs_wo_spp.hef) — 4조합 x 3회"
echo "############################################################"
./run_v5det_cpu_seg_pose.sh

echo ""
echo "[완료] 두 결과 CSV/HRTT 위치 (2026-08-07 #2: HRTT 재수집판, _hrtt 폴더):"
echo "  NPU: hailo_8/experiments/2026-08-07_v5det_seg_pose_hrtt_exp1/csv/results_v5det_seg_pose.csv"
echo "       HRTT: hailo_8/experiments/2026-08-07_v5det_seg_pose_hrtt_exp1/traces/"
echo "  CPU: hailo_8/experiments/2026-08-07_v5det_cpu_seg_pose_hrtt_exp1/csv/results_v5det_cpu_seg_pose.csv"
echo "       HRTT: hailo_8/experiments/2026-08-07_v5det_cpu_seg_pose_hrtt_exp1/traces/"
echo "각 CSV 안에서 use_seg/use_pose 컬럼으로 4조합(det/det+seg/det+pose/det+seg+pose)이 구분됨."
echo "같은 조합끼리 두 CSV를 det_latency_ms / postprocess_ms_det / total_time_ms_det / cpu_percent 컬럼 기준으로 대조할 것."
