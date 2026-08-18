#!/bin/bash
# WSL2(또는 Dataflow Compiler/hailo CLI가 설치된 x86 호스트)에서 실행:
# HRTT 파일을 실험별 HTML로 변환. hailo_8L/scripts/convert_hrtt_to_html.sh와 완전히
# 동일한 로직(보드/칩 종류와 무관하게 동작 — 변환 도구는 .hrtt 파일 자체를
# 파싱하는 도구라 8/8L 구분이 없음). hailo_8 실험 경로에 맞춰 재사용.
#
# [주의] 이 변환 자체는 rpi4(HailoRT 런타임만 있음)에서는 못 돌림 — `hailo` CLI는
# Hailo Dataflow Compiler 패키지에 들어있는 도구라 별도 설치된 WSL2/Linux 호스트
# (가상환경 ~/hailo_venv)에서 실행해야 함. .hrtt 파일 자체는 rpi4에서
# HAILO_TRACE=scheduler 등 환경변수를 설정하고 추론 프로그램을 실행해야 생성됨.
#
# [수정 이력 2026-08-08] 이전 버전은 존재하지 않는 `hailort_report` 명령을 호출하고
# 출력 위치도 $HRTT_DIR로 잘못 가정했음. 실제 도구는 `hailo runtime-profiler <file>.hrtt`
# 이며, 실행한 cwd에 항상 runtime_report.html을 생성함(입력 파일 위치와 무관).
# (근거: hailo_8L/PROJECT_HANDOFF.md §5, hailo_8L/docs/RUN_threshold_exp.md §5)
#
# 사용법: bash convert_hrtt_to_html.sh [hrtt_dir] [html_dir]
# 실행 전: source ~/hailo_venv/bin/activate
# 예시(v5det 실험 트레이스 변환 시):
#   bash convert_hrtt_to_html.sh hailo_8/experiments/2026-08-07_v5det_seg_pose_hrtt_exp1/traces \
#                                 hailo_8/experiments/2026-08-07_v5det_seg_pose_hrtt_exp1/html

HRTT_DIR="${1:-./hrtt_reports/hrtt}"
HTML_DIR="${2:-./hrtt_reports/html}"

mkdir -p "$HTML_DIR"

found=0
for hrtt in "$HRTT_DIR"/*.hrtt; do
    [ -f "$hrtt" ] || continue
    name=$(basename "$hrtt" .hrtt)

    echo "변환 중: $name"
    hailo runtime-profiler "$hrtt"

    if [ -f "./runtime_report.html" ]; then
        mv "./runtime_report.html" "$HTML_DIR/${name}.html"
        echo "  → $HTML_DIR/${name}.html"
        found=$((found + 1))
    else
        echo "  경고: runtime_report.html 생성 안됨, 건너뜀"
    fi
done

echo ""
echo "변환 완료: ${found}개"
