#!/bin/bash
# WSL2에서 실행: HRTT 파일을 실험별 HTML로 변환
# 사용법: bash convert_hrtt_to_html.sh [hrtt_dir] [html_dir]
# 예시: bash convert_hrtt_to_html.sh ./hrtt_reports/hrtt ./hrtt_reports/html

HRTT_DIR="${1:-./hrtt_reports/hrtt}"
HTML_DIR="${2:-./hrtt_reports/html}"

mkdir -p "$HTML_DIR"

found=0
for hrtt in "$HRTT_DIR"/*.hrtt; do
    [ -f "$hrtt" ] || continue
    name=$(basename "$hrtt" .hrtt)

    echo "변환 중: $name"
    hailort_report "$hrtt"

    if [ -f "$HRTT_DIR/runtime_report.html" ]; then
        cp "$HRTT_DIR/runtime_report.html" "$HTML_DIR/${name}.html"
        echo "  → $HTML_DIR/${name}.html"
        found=$((found + 1))
    else
        echo "  경고: runtime_report.html 생성 안됨, 건너뜀"
    fi
done

echo ""
echo "변환 완료: ${found}개"
