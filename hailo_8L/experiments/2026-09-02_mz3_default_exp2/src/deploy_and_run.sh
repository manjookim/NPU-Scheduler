#!/bin/bash
# =============================================================================
# deploy_and_run.sh — exp2 벤치를 npu-rpi1 에 배포하고 스윕까지 돌린다.
# 노트북(WSL) 또는 SSH 가 되는 곳에서 실행. 이 저장소 루트 기준 상대경로 사용.
#
#   ./deploy_and_run.sh deploy      # 소스 전송 + 빌드 + smoke test (약 2분)
#   ./deploy_and_run.sh pretrained  # 사전학습 세트 42런 (약 15분)
#   ./deploy_and_run.sh retrained   # 재학습 세트 (같은 계측 조건)
#   ./deploy_and_run.sh retrained_sc# 컨텍스트 수만 다른 대조군
#   ./deploy_and_run.sh fetch       # 결과 CSV/manifest/traces 회수
#
# ★ 순서 주의: 사전학습 세트를 **반드시 다시 돌린다.** 기존 42런은 HAILO_MONITOR 없이
#   측정됐고 재학습 18런은 있는 상태로 측정됐다. 계측 조건이 다르면 두 세트의
#   fps=0 비교(+113%/+125%)가 재학습 효과인지 모니터 오버헤드인지 분리되지 않는다.
# =============================================================================
set -euo pipefail
HOST="${HOST:-npu-rpi1@155.230.16.157}"
PORT="${PORT:-40021}"
SSH="ssh -p $PORT $HOST"
# 로컬에서 준 값을 원격 스윕으로 전달 (deeplab 재학습 HEF 가 없을 때 SKIP_DEEPLAB=1)
PASS="SKIP_DEEPLAB=${SKIP_DEEPLAB:-0} REPEAT=${REPEAT:-3} MONITOR=${MONITOR:-1} BOUNDED_DUMP=${BOUNDED_DUMP:-60} WARMUP=${WARMUP:-30}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
EXP="$REPO/hailo_8L/experiments/2026-09-02_mz3_default_exp2/src"
CMD="${1:-deploy}"

case "$CMD" in
deploy)
    echo "== 전송 =="
    scp -P "$PORT" "$EXP/mz3_sched_bench.cpp" "$EXP/run_mz3_sweep.sh" \
        "$REPO/hailo_8L/sys_monitor.hpp" \
        "$REPO/tools/monitoring/hailo_utilization2.py" \
        "$HOST:~/mz3_exp"
    echo "== 빌드 =="
    $SSH 'cd ~/mz3_exp && chmod +x run_mz3_sweep.sh && \
        for f in scheduler_mon_pb2.py; do [ -f $f ] || cp ~/hailo_cpp_test/$f . 2>/dev/null || true; done && \
        g++ -O2 -std=c++17 mz3_sched_bench.cpp -o mz3_sched_bench \
            $(pkg-config --cflags --libs opencv4) -lhailort -lpthread && \
        echo "빌드 성공" && ls -la mz3_sched_bench'
    echo "== smoke test (ssd 단독 60프레임, CSV 는 /tmp 로) =="
    $SSH 'cd ~/mz3_exp && ./mz3_sched_bench --ssd 1 --deeplab 0 --mnv2 0 --fps 30 \
            --frames 60 --warmup 10 --tag smoke --csv /tmp/smoke.csv \
            --manifest /tmp/smoke.json --instr smoke'
    echo
    echo "위 [params] 줄의 batch_size 값이 인계문서 Next Action #3 의 1차 답이다."
    echo "late/err 경고가 없으면 스윕으로 진행."
    ;;
pretrained)
    $SSH 'cd ~/mz3_exp && mkdir -p logs csv && '"$PASS"' SET_NAME=pretrained RES_DIR=resources ASSUME_YES=1 \
          nohup ./run_mz3_sweep.sh > logs/sweep_pretrained.out 2>&1 &
          echo "시작됨. tail -f ~/mz3_exp/logs/sweep_pretrained.out"'
    ;;
retrained)
    $SSH 'cd ~/mz3_exp && mkdir -p logs csv && '"$PASS"' SET_NAME=retrained RES_DIR=resources_retrained \
          MNV2_HEF=mobilenet_v2_1.0_gtsrb.hef ASSUME_YES=1 \
          nohup ./run_mz3_sweep.sh > logs/sweep_retrained.out 2>&1 &
          echo "시작됨."'
    ;;
retrained_sc)
    $SSH 'cd ~/mz3_exp && mkdir -p logs csv && '"$PASS"' SET_NAME=retrained_sc RES_DIR=resources_retrained \
          MNV2_HEF=mobilenet_v2_1.0_gtsrb_sc.hef ASSUME_YES=1 \
          nohup ./run_mz3_sweep.sh > logs/sweep_retrained_sc.out 2>&1 &
          echo "시작됨."'
    ;;
status)
    # 스윕이 살아 있는지 / 어디까지 갔는지 한 번에 본다.
    # 인터넷이 끊겨도 nohup 으로 띄웠으므로 RPi 쪽 작업은 계속 돈다.
    $SSH 'set +e
      echo "=== 프로세스 ==="
      pgrep -af "run_mz3_sweep|mz3_sched_bench|hailo_utilization" || echo "  (실행 중인 프로세스 없음 = 끝났거나 죽음)"
      echo
      echo "=== 진행 로그 (마지막 15줄) ==="
      tail -n 15 ~/mz3_exp/logs/sweep_*.out 2>/dev/null
      echo
      echo "=== CSV 행 수 (헤더 포함. 42런 완료 시 각각 22행) ==="
      wc -l ~/mz3_exp/csv/results_mz3_*.csv 2>/dev/null
      echo
      echo "=== npu_percent ==="
      wc -l ~/mz3_exp/csv/npu_percent_*.csv 2>/dev/null
      echo
      echo "=== HRTT 트레이스 (런별 디렉터리) ==="
      for d in ~/mz3_exp/traces/*/; do
        echo "  $d : $(find "$d" -name "*.hrtt" 2>/dev/null | wc -l) 파일 / $(find "$d" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | wc -l) 런디렉터리"
      done
      echo
      echo "=== manifest ==="
      ls ~/mz3_exp/manifests/ 2>/dev/null | wc -l | xargs echo "  개수:"
      echo
      echo "=== 완료 여부 ==="
      grep -h "전체 완료\|실패 런" ~/mz3_exp/logs/sweep_*.out 2>/dev/null || echo "  아직 완료 문구 없음 (진행 중)"'
    ;;
fetch)
    D="$REPO/hailo_8L/experiments/2026-09-02_mz3_default_exp2"
    mkdir -p "$D/csv" "$D/manifests" "$D/traces" "$D/logs"
    scp -P "$PORT" "$HOST:~/mz3_exp/csv/*"          "$D/csv/"       || true
    scp -P "$PORT" -r "$HOST:~/mz3_exp/manifests/*" "$D/manifests/" || true
    scp -P "$PORT" -r "$HOST:~/mz3_exp/traces/*"    "$D/traces/"    || true
    scp -P "$PORT" "$HOST:~/mz3_exp/logs/hef_report_*.txt" "$D/logs/" || true
    echo "회수 완료 -> $D"
    echo
    echo "다음: 병합·요약"
    echo "  python tools/mz3/merge_mz3_results.py \\"
    echo "    --new $D/csv/results_mz3_pretrained.csv $D/csv/results_mz3_pretrained_fps30.csv \\"
    echo "    --npu $D/csv/npu_percent_pretrained.csv \\"
    echo "    --traces $D/traces/pretrained \\"
    echo "    --out $D/csv/merged_pretrained.csv --summary $D/csv/summary_pretrained.csv"
    ;;
*)
    echo "사용: $0 {deploy|pretrained|retrained|retrained_sc|status|fetch}"; exit 1 ;;
esac
