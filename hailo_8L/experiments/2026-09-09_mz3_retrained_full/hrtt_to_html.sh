#!/bin/bash
# hrtt_to_html.sh [2026-09-09]
# hrtt/ 의 .hrtt 21개 -> html/ 에 같은 이름의 .html 리포트로 변환.
# WSL 에서 hailo_venv 활성화 후 실행할 것 (hailo CLI 가 필요).
set -u
E="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$E/html"
ok=0; fail=0
for f in "$E"/hrtt/*.hrtt; do
  b="$(basename "$f" .hrtt)"
  if hailo runtime-profiler --output-path "$E/html/$b.html" "$f" >/dev/null 2>&1; then
    ok=$((ok+1)); echo "  OK   $b.html"
  else
    fail=$((fail+1)); echo "  FAIL $b"
  fi
done
echo "변환 완료: $ok 성공 / $fail 실패 (총 $((ok+fail)))"
ls -1 "$E/html" | wc -l
