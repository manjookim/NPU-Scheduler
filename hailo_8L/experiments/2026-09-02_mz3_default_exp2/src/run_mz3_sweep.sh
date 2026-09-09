#!/bin/bash
# =============================================================================
# run_mz3_sweep.sh  (v2, 2026-09-02)  — Hailo-8L MZ3 다중조합(7조건) Default 스윕
#
# v1 의 세 스크립트(run_mz3_default_sweep.sh / _fps30_ / _retrained_)를 하나로 합쳤다.
# **가장 중요한 변경**: 사전학습 세트와 재학습 세트를 "완전히 같은 계측 조건"으로 돌린다.
#   v1 은 사전학습 42런에는 HAILO_MONITOR 가 없었고 재학습 18런에는 있었다.
#   즉 RPi5 4코어에서 python 모니터 프로세스 1개 + libhailort monitor 계측이
#   재학습 세트에만 얹혀 있었다. "재학습하니 fps=0 에서 +113%/+125% 느려졌다"는
#   결론은 이 차이와 교락(confound)되어 있다. 이제 MONITOR 는 두 세트에 동일 적용되고,
#   실제 적용값이 CSV 의 instr 컬럼과 manifest 에 기록된다.
#
# 그 밖의 수정
#   · HRTT 트레이스를 **런별 하위 디렉터리**에 쓴다 -> .hrtt 파일에 조건 라벨이 없어서
#     매핑을 못 하던 문제(4-D)가 원천 해소. traces/<set>/<label>/run<N>/ 안의 파일 = 그 런.
#   · set -o pipefail + 종료코드 검사. v1 은 `$BIN | tee` 라 벤치가 죽어도 성공으로 보였다.
#   · NPU 모니터를 exec 로 띄우고 프로세스 그룹째 죽인다. v1 은 `( cd .. && python ) &` 의
#     서브셸 PID 만 kill 해서 python 손자 프로세스가 살아남아 다음 런의 npu_log 를
#     오염시키고 CPU 를 계속 먹었다.
#   · npu_log 를 런별 파일로 분리(고정 경로 공유 제거).
#   · read -p 를 대화형일 때만. nohup/원격 배치에서 멈추지 않는다.
#   · 기존 CSV 가 있으면 덮어쓰지 않고 타임스탬프로 보관(데이터 오염 방지).
#
# 사용:
#   ./run_mz3_sweep.sh                       # 사전학습 세트 (resources/)
#   RES_DIR=resources_retrained MNV2_HEF=mobilenet_v2_1.0_gtsrb.hef ./run_mz3_sweep.sh
#   SKIP_DEEPLAB=1 ./run_mz3_sweep.sh        # deeplab 제외(3조건)
#   MONITOR=0 ./run_mz3_sweep.sh             # NPU 모니터 끄기(두 세트 모두 같게 할 것!)
#   ASSUME_YES=1 nohup ./run_mz3_sweep.sh &  # 무인 실행
# =============================================================================
set -u -o pipefail
cd "$HOME/mz3_exp" || exit 1

SET_NAME="${SET_NAME:-pretrained}"          # 결과 파일/트레이스 폴더 접두사
RES_NAME="${RES_DIR:-resources}"            # resources | resources_retrained
MNV2_HEF="${MNV2_HEF:-}"                    # 지정 시 RES 로 복사(재학습판 교체)
FRAMES="${FRAMES:-673}"
WARMUP="${WARMUP:-30}"                      # 통계에서 제외할 앞쪽 프레임(activate/첫 페이지폴트)
REPEAT="${REPEAT:-3}"
MONITOR="${MONITOR:-1}"
TRACE="${TRACE:-1}"
BOUNDED_DUMP="${BOUNDED_DUMP:-60}"          # 링버퍼 길이(초). 최장 런(22.4s)보다 넉넉히.
FPS_LIST="${FPS_LIST:-0 30}"

BIN=./mz3_sched_bench
RES="$HOME/mz3_exp/$RES_NAME"
IMG_DIR="${IMG_DIR:-$HOME/datasets/sampled_val2017}"
OUT="$HOME/mz3_exp"
mkdir -p "$OUT/csv" "$OUT/logs" "$OUT/manifests" "$RES"

[ -x "$BIN" ] || { echo "[에러] $BIN 없음. 먼저 빌드하세요:"; \
  echo "  g++ -O2 -std=c++17 mz3_sched_bench.cpp -o mz3_sched_bench \$(pkg-config --cflags --libs opencv4) -lhailort -lpthread"; exit 1; }

# ── mnv2 HEF 교체 (재학습판) ──
SUF=""
if [ -n "$MNV2_HEF" ]; then
    SRC="$HOME/mz3_exp/resources/$MNV2_HEF"
    [ -f "$SRC" ] || { echo "[에러] 없음: $SRC"; exit 1; }
    for h in ssd_mobilenet_v1.hef deeplab_v3_mobilenet_v2_wo_dilation.hef; do
        [ -f "$RES/$h" ] || cp "$HOME/mz3_exp/resources/$h" "$RES/" 2>/dev/null
    done
    cp "$SRC" "$RES/mobilenet_v2_1.0.hef"
    case "$MNV2_HEF" in
        *_sc.hef)             SUF="_sc"   ;;
        mobilenet_v2_1.0.hef) SUF="_orig" ;;
        *)                    SUF="_gtsrb";;
    esac
    echo "mnv2 HEF: $MNV2_HEF  (접미사='$SUF')"
fi
[ "${SKIP_DEEPLAB:-0}" = "1" ] && SUF="${SUF}_nodeeplab"

# ── HEF 확인 (컨텍스트 수는 실험의 핵심 변수이므로 매번 기록) ──
echo "===== HEF 확인 ====="
HEFS="ssd_mobilenet_v1 deeplab_v3_mobilenet_v2_wo_dilation mobilenet_v2_1.0"
[ "${SKIP_DEEPLAB:-0}" = "1" ] && HEFS="ssd_mobilenet_v1 mobilenet_v2_1.0"
HEF_REPORT="$OUT/logs/hef_report_${SET_NAME}${SUF}.txt"
: > "$HEF_REPORT"
for h in $HEFS; do
    [ -f "$RES/$h.hef" ] || { echo "[에러] 없음: $RES/$h.hef"; exit 1; }
    { echo "--- $h  ($(stat -c '%s bytes, mtime %y' "$RES/$h.hef"))"
      hailortcli parse-hef "$RES/$h.hef" 2>/dev/null | grep -iE "context|Operation|output|shape" | head -12
    } | tee -a "$HEF_REPORT"
done
echo "(HEF 리포트: $HEF_REPORT)"
echo "위 mobilenet_v2_1.0 출력이 43 이면 재학습판, 1001 이면 원본입니다."
if [ "${ASSUME_YES:-0}" != "1" ] && [ -t 0 ]; then
    read -r -p "계속하려면 Enter, 중단하려면 Ctrl+C: " _
fi

# ── 계측 환경 (두 세트에 **동일하게** 적용될 것) ──
INSTR="mon${MONITOR}-trace${TRACE}-dump${BOUNDED_DUMP}"
if [ "$MONITOR" = "1" ]; then export HAILO_MONITOR=1; else unset HAILO_MONITOR; fi

# hailo_utilization2.py 를 우선 쓴다(로그경로 인자 + 5Hz 폴링 + 낡은 hmon 파일 방지).
# 없으면 구버전으로 폴백하되, 구버전은 로그 경로가 /home/rpi1/ 로 하드코딩돼 있어
# npu_percent 가 통째로 안 남을 수 있음을 경고한다.
MON_PY=""; MON_PYBIN="python3"; MON_V2=0
MON_INTERVAL="${MON_INTERVAL:-0.2}"
if [ "$MONITOR" = "1" ]; then
    for c in "$HOME/mz3_exp/hailo_utilization2.py" "$HOME/hailo_cpp_test/hailo_utilization2.py"; do
        [ -f "$c" ] && MON_PY="$c" && MON_V2=1 && break
    done
    [ -z "$MON_PY" ] && for c in "$HOME/mz3_exp/hailo_utilization.py" "$HOME/hailo_cpp_test/hailo_utilization.py"; do
        [ -f "$c" ] && MON_PY="$c" && break
    done
    [ "$MON_V2" = "0" ] && [ -n "$MON_PY" ] && \
        echo "[경고] 구버전 모니터입니다. 로그 경로가 하드코딩(/home/rpi1)돼 있고 1Hz 고정입니다."
    [ "$MON_V2" = "0" ] && [ -n "$MON_PY" ] && \
        echo "        tools/monitoring/hailo_utilization2.py 를 ~/mz3_exp/ 로 복사해 쓰세요."
    [ -x "$HOME/hailo_platform_venv/bin/python3" ] && MON_PYBIN="$HOME/hailo_platform_venv/bin/python3"
    if [ -n "$MON_PY" ]; then
        MON_DIR=$(dirname "$MON_PY")
        if ! (cd "$MON_DIR" && "$MON_PYBIN" -c "import scheduler_mon_pb2" >/dev/null 2>&1); then
            echo "[경고] scheduler_mon_pb2 import 실패 -> npu_percent 는 NaN."
            echo "       해결: $MON_PYBIN -m pip install --upgrade protobuf"
            MON_PY=""
        fi
    fi
    [ -z "$MON_PY" ] && echo "[경고] NPU 모니터 비활성 (npu_percent = NaN)"
fi
NPU_CSV="$OUT/csv/npu_percent_${SET_NAME}${SUF}.csv"
[ -f "$NPU_CSV" ] || echo "tag,run_id,input_fps,npu_percent,n_samples,run_s,reliable" > "$NPU_CSV"

CONDITIONS=( "ssd:1,0,0" "deeplab:0,1,0" "mnv2:0,0,1"
             "ssd_deeplab:1,1,0" "ssd_mnv2:1,0,1" "deeplab_mnv2:0,1,1"
             "ssd_deeplab_mnv2:1,1,1" )
if [ "${SKIP_DEEPLAB:-0}" = "1" ]; then
    F=(); for c in "${CONDITIONS[@]}"; do case "${c%%:*}" in *deeplab*) ;; *) F+=("$c");; esac; done
    CONDITIONS=("${F[@]}")
    echo ">>> SKIP_DEEPLAB=1 : ${#CONDITIONS[@]}조건"
fi

archive_if_exists () {  # 기존 결과에 덧붙여 두 실험이 한 파일에 섞이는 사고 방지
    [ -f "$1" ] && mv "$1" "$1.$(date +%Y%m%d_%H%M%S).bak" && echo "  (기존 CSV 보관: $1.*.bak)"
    return 0
}

FAILED=0
run_set () {
    local FPS="$1" CSV="$2" PREFIX="$3"
    archive_if_exists "$CSV"
    echo ""; echo "########## fps=$FPS -> $CSV ##########"
    for cond in "${CONDITIONS[@]}"; do
        local label="${cond%%:*}" flags="${cond##*:}"
        IFS=',' read -r F_SSD F_DEEPLAB F_MNV2 <<< "$flags"
        for run in $(seq 1 "$REPEAT"); do
            echo "===== [$label] fps=$FPS run $run/$REPEAT ====="

            # ★ HRTT: 런마다 별도 디렉터리 -> 파일-조건 매핑이 자명해진다 (4-D 해결)
            local TDIR="$OUT/traces/${SET_NAME}${SUF}/fps${FPS}/${label}/run${run}"
            if [ "$TRACE" = "1" ]; then
                rm -rf "$TDIR"; mkdir -p "$TDIR"
                export HAILO_TRACE=scheduler
                export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP="$BOUNDED_DUMP"
                export HAILO_TRACE_PATH="$TDIR"
            else
                unset HAILO_TRACE HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP HAILO_TRACE_PATH
            fi

            local NPU_LOG="$OUT/logs/npu_${PREFIX}_${label}_run${run}.txt"
            local MON_PID="" T0 T1
            if [ -n "$MON_PY" ]; then
                : > "$NPU_LOG"
                # setsid + exec: 서브셸이 아니라 python 자신이 리더가 되어 그룹째 확실히 죽는다
                if [ "$MON_V2" = "1" ]; then
                    setsid bash -c 'cd "$1" && exec "$2" "$3" "$4" --interval "$5"' _ \
                        "$(dirname "$MON_PY")" "$MON_PYBIN" "$MON_PY" "$NPU_LOG" "$MON_INTERVAL" \
                        >/dev/null 2>>"$OUT/logs/npu_monitor.log" &
                else
                    NPU_LOG="$HOME/hailo_cpp_test/npu_log.txt"; : > "$NPU_LOG" 2>/dev/null
                    HAILO_UTIL_LOG="$NPU_LOG" setsid bash -c 'cd "$1" && exec "$2" "$3"' _ \
                        "$(dirname "$MON_PY")" "$MON_PYBIN" "$MON_PY" \
                        >/dev/null 2>>"$OUT/logs/npu_monitor.log" &
                fi
                MON_PID=$!
                sleep 1
            fi

            T0=$(date +%s.%N)
            "$BIN" --ssd "$F_SSD" --deeplab "$F_DEEPLAB" --mnv2 "$F_MNV2" \
                   --fps "$FPS" --frames "$FRAMES" --warmup "$WARMUP" --run-id "$run" \
                   --tag "$label" --csv "$CSV" --images "$IMG_DIR" --res "$RES" \
                   --instr "$INSTR" \
                   --manifest "$OUT/manifests/${PREFIX}_${label}_run${run}.json" \
                   > "$OUT/logs/${PREFIX}_${label}_run${run}.log" 2>&1
            local RC=$?
            T1=$(date +%s.%N)
            tail -n 12 "$OUT/logs/${PREFIX}_${label}_run${run}.log"
            if [ "$RC" -ne 0 ]; then
                echo "  [실패] 종료코드=$RC  로그: $OUT/logs/${PREFIX}_${label}_run${run}.log"
                FAILED=$((FAILED+1))
            fi

            if [ -n "$MON_PID" ]; then
                kill -TERM -"$MON_PID" 2>/dev/null || kill "$MON_PID" 2>/dev/null
                wait "$MON_PID" 2>/dev/null
                local DUR; DUR=$(awk "BEGIN{printf \"%.2f\", $T1-$T0}")
                # parse_npu_log.py 와 동일 정의(NPU>0 샘플만 평균) + 신뢰도 플래그
                read -r NPU NS <<< "$(awk -F'NPU: ' '
                    NF>1 { split($2,a,"%"); if (a[1]+0>0) { s+=a[1]; n++ } }
                    END  { if (n>0) printf "%.4f %d", s/n, n; else printf "NaN 0" }' "$NPU_LOG")"
                local REL="yes"; [ "${NS:-0}" -lt 10 ] && REL="no"
                echo "$label,$run,$FPS,$NPU,$NS,$DUR,$REL" >> "$NPU_CSV"
                echo "  NPU=${NPU}%  샘플=${NS}개 / ${DUR}s  신뢰=$REL"
                [ "$REL" = "no" ] && echo "    [주의] 1Hz 폴링 샘플이 10개 미만입니다. 이 값으로 조건 간 비교를 하지 마세요."
            fi
            echo ""; sleep 2
        done
    done
    echo "완료: $CSV  ($(wc -l < "$CSV") 행)"
}

echo "===== 스윕 시작: $(date)  set=${SET_NAME}${SUF}  instr=$INSTR ====="
for FPS in $FPS_LIST; do
    if [ "$FPS" = "0" ]; then
        run_set 0  "$OUT/csv/results_mz3_${SET_NAME}${SUF}.csv"       "${SET_NAME}${SUF}"
    else
        run_set "$FPS" "$OUT/csv/results_mz3_${SET_NAME}${SUF}_fps${FPS}.csv" "${SET_NAME}${SUF}_fps${FPS}"
    fi
done
echo "===== 전체 완료: $(date)  실패 런=$FAILED ====="
echo "npu_percent : $NPU_CSV"
echo "manifests   : $OUT/manifests/${SET_NAME}${SUF}*"
echo "traces      : $OUT/traces/${SET_NAME}${SUF}/  (fps/조건/run 별 디렉터리 = 매핑 자명)"
echo ""
echo "PC 로 가져오기:"
echo "  scp -P 40021 -r 'npu-rpi1@155.230.16.157:~/mz3_exp/csv/*' ."
echo "  scp -P 40021 -r 'npu-rpi1@155.230.16.157:~/mz3_exp/manifests' ."
echo "  scp -P 40021 -r 'npu-rpi1@155.230.16.157:~/mz3_exp/traces' ."
exit $(( FAILED > 0 ? 1 : 0 ))
