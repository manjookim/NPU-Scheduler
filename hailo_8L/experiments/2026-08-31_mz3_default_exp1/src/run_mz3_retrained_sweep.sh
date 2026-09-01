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
HEFS="ssd_mobilenet_v1 deeplab_v3_mobilenet_v2_wo_dilation mobilenet_v2_1.0"
[ "${SKIP_DEEPLAB:-0}" = "1" ] && HEFS="ssd_mobilenet_v1 mobilenet_v2_1.0"
for h in $HEFS; do
    if [ ! -f "$RES/$h.hef" ]; then
        echo "[에러] 없음: $RES/$h.hef"; exit 1
    fi
    echo "--- $h"
    hailortcli parse-hef "$RES/$h.hef" 2>/dev/null | grep -iE "context|output|shape" | head -8
done
echo ""
echo "위에서 mobilenet_v2_1.0 의 출력이 43 이어야 재학습판입니다. 1001 이면 원본입니다."
read -p "계속하려면 Enter, 중단하려면 Ctrl+C: " _

# ── NPU 사용률 측정 (기존 8L 방식 그대로) ──────────────────────────────
#   HAILO_MONITOR=1 을 주면 libhailort 가 /tmp/hmon_files 에 ProtoMon 을 떨구고,
#   hailo_utilization.py 가 1초마다 그걸 읽어 npu_log.txt 에 기록한다.
#   parse_npu_log.py 와 동일한 정의(NPU>0 인 샘플만 평균)로 조건별 평균을 낸다.
export HAILO_MONITOR=1

MON_PY=""
for c in "$HOME/mz3_exp/hailo_utilization.py" "$HOME/hailo_cpp_test/hailo_utilization.py"; do
    [ -f "$c" ] && MON_PY="$c" && break
done
MON_DIR=$(dirname "${MON_PY:-/nonexistent}")

# [2026-09-01] 시스템 python3 의 protobuf 가 scheduler_mon_pb2.py 를 만든 protoc 보다
# 낡으면 "cannot import name 'runtime_version'" 로 죽는다(2026-08-08 rpi4 에서 겪은 건과 동일).
# hailo_platform_venv 가 있으면 그쪽 python 을 우선 쓴다.
MON_PYBIN="python3"
[ -x "$HOME/hailo_platform_venv/bin/python3" ] && MON_PYBIN="$HOME/hailo_platform_venv/bin/python3"

# 모니터가 실제로 도는지 미리 확인 (import 에러를 조용히 삼키지 않는다)
if [ -n "$MON_PY" ]; then
    MON_ERR=$(cd "$MON_DIR" && "$MON_PYBIN" -c "import scheduler_mon_pb2" 2>&1)
    if [ -n "$MON_ERR" ]; then
        echo "[경고] NPU 모니터 import 실패 — npu_percent 는 NaN 이 됩니다."
        echo "$MON_ERR" | tail -3
        echo "  해결: $MON_PYBIN -m pip install --upgrade protobuf   (또는 --break-system-packages)"
        MON_PY=""
    fi
fi
NPU_LOG="$HOME/hailo_cpp_test/npu_log.txt"     # hailo_utilization.py 가 쓰는 고정 경로
NPU_CSV="$HOME/mz3_exp/csv/npu_percent_retrained.csv"

if [ -z "$MON_PY" ] || [ ! -f "$MON_DIR/scheduler_mon_pb2.py" ]; then
    echo "[경고] hailo_utilization.py 또는 scheduler_mon_pb2.py 를 못 찾았습니다."
    echo "       npu_percent 는 NaN 으로 기록됩니다."
    echo "       필요 파일: $MON_DIR/hailo_utilization.py, $MON_DIR/scheduler_mon_pb2.py"
    MON_PY=""
else
    echo "NPU 모니터: $MON_PY  (python=$MON_PYBIN)"
    mkdir -p "$(dirname "$NPU_LOG")"
    [ -f "$NPU_CSV" ] || echo "tag,run_id,input_fps,npu_percent,n_samples" > "$NPU_CSV"
fi

# HRTT 트레이싱. 가장 짧은 조건(수초)보다 작아야 덤프가 트리거되므로 3으로 둔다
# (2026-08-08 기록: 30 이면 짧은 조건에서 .hrtt 가 아예 안 생김)
export HAILO_TRACE=scheduler
export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=3
export HAILO_TRACE_PATH="$TRACES_DIR"

# SKIP_DEEPLAB=1 로 실행하면 deeplab 이 들어간 조건 4개를 건너뛴다.
#   → 남는 건 ssd / mnv2 / ssd_mnv2 3조건 (= 2모델의 2^2-1)
#   deeplab 은 아직 Cityscapes 재학습 전이라, 재학습 효과만 보고 싶을 때 쓴다.
CONDITIONS=(
  "ssd:1,0,0"
  "deeplab:0,1,0"
  "mnv2:0,0,1"
  "ssd_deeplab:1,1,0"
  "ssd_mnv2:1,0,1"
  "deeplab_mnv2:0,1,1"
  "ssd_deeplab_mnv2:1,1,1"
)

if [ "${SKIP_DEEPLAB:-0}" = "1" ]; then
    FILTERED=()
    for c in "${CONDITIONS[@]}"; do
        case "${c%%:*}" in *deeplab*) ;; *) FILTERED+=("$c") ;; esac
    done
    CONDITIONS=("${FILTERED[@]}")
    echo ">>> SKIP_DEEPLAB=1 : deeplab 제외, ${#CONDITIONS[@]}조건만 실행"
    echo ">>>   ${CONDITIONS[*]%%:*}"
fi

run_set () {
    local FPS="$1" CSV="$2" PREFIX="$3"
    echo ""
    echo "############ 입력속도 fps=$FPS  ->  $CSV ############"
    for cond in "${CONDITIONS[@]}"; do
        local label="${cond%%:*}" flags="${cond##*:}"
        IFS=',' read -r F_SSD F_DEEPLAB F_MNV2 <<< "$flags"
        for run in $(seq 1 $REPEAT); do
            echo "===== [$label] fps=$FPS run $run/$REPEAT ====="

            MON_PID=""
            if [ -n "$MON_PY" ]; then
                : > "$NPU_LOG"                      # 이 런 구간만 남기려고 비운다
                ( cd "$MON_DIR" && "$MON_PYBIN" "$MON_PY" ) >/dev/null 2>>"$HOME/mz3_exp/logs/npu_monitor.log" &
                MON_PID=$!
                sleep 1
            fi

            "$BIN" --ssd "$F_SSD" --deeplab "$F_DEEPLAB" --mnv2 "$F_MNV2" \
                   --fps "$FPS" --frames "$FRAMES" --run-id "$run" \
                   --tag "$label" --csv "$CSV" --images "$IMG_DIR" --res "$RES" \
                   2>&1 | tee "logs/${PREFIX}_${label}_run${run}.log"

            if [ -n "$MON_PID" ]; then
                kill "$MON_PID" 2>/dev/null
                wait "$MON_PID" 2>/dev/null
                # parse_npu_log.py 와 같은 정의: NPU>0 인 샘플만 평균
                read -r NPU NS <<< "$(awk -F'NPU: ' '
                    NF>1 { split($2, a, "%"); if (a[1]+0 > 0) { s += a[1]; n++ } }
                    END  { if (n>0) printf "%.4f %d", s/n, n; else printf "NaN 0" }' "$NPU_LOG")"
                echo "$label,$run,$FPS,$NPU,$NS" >> "$NPU_CSV"
                echo "  NPU 사용률 = ${NPU}%  (활성 샘플 ${NS}개)"
                if [ "$NS" = "0" ]; then
                    if [ ! -d /tmp/hmon_files ]; then
                        echo "    [원인] /tmp/hmon_files 가 없음 — HAILO_MONITOR=1 이 안 먹었거나 런이 너무 짧음"
                    elif [ ! -s "$NPU_LOG" ]; then
                        echo "    [원인] npu_log.txt 가 비어 있음 — 모니터가 죽었을 수 있음"
                        tail -3 "$HOME/mz3_exp/logs/npu_monitor.log" 2>/dev/null | sed "s/^/      /"
                    else
                        echo "    [원인] 샘플은 있으나 전부 NPU=0.00% — 런이 짧아 1Hz 폴링이 활성 구간을 못 잡음"
                    fi
                fi
            fi

            echo ""
            sleep 2
        done
    done
    echo "완료: $CSV  ($(wc -l < "$CSV") 행, 헤더 포함)"
}

echo "===== 재학습판 스윕 시작: $(date) ====="
SUF=""; [ "${SKIP_DEEPLAB:-0}" = "1" ] && SUF="_nodeeplab"
run_set 0  "$HOME/mz3_exp/csv/results_mz3_retrained${SUF}.csv"       "rt${SUF}"
run_set 30 "$HOME/mz3_exp/csv/results_mz3_retrained${SUF}_fps30.csv" "rt${SUF}_fps30"
echo "===== 전체 완료: $(date) ====="
echo ""
echo "npu_percent: $NPU_CSV"
echo ""
echo "PC 로 가져오기:"
echo "  scp -P 40021 'npu-rpi1@155.230.16.157:~/mz3_exp/csv/results_mz3_retrained*.csv' ."
