#!/bin/bash
# =============================================================================
# [2026-08-19] Det 단일모델 NPU vs CPU 후처리 비교 — FPS 스윕판 (Hailo-8L, rpi1)
#
# run_det_v8s_vs_v5npu_fps60.sh(INPUT_FPS=60 고정)의 후속. 사용자 요청으로
# INPUT_FPS를 60 대신 {0(제한없음/최대속도), 30}으로 바꿔서 돈다. 또한 이 버전은
# HRTT(NPU 스케줄러 트레이스)도 같이 캡처한다 — 기존 3모델 default_workload류
# 실험(auto_experiment_default_workload.sh)의 HAILO_MONITOR=1 + hailo_utilization.py
# 캡처 패턴을 그대로 가져왔다.
#
# HEF 2종(및 각 후처리 담당 프로세서)은 fps60판과 동일:
#   - yolov8s_h8l.hef               : engine=cpu (640x640)             -> CPU 후처리
#   - yolov5xs_wo_spp_nms_core.hef  : engine=nn_core/auto (512x512)    -> NPU 후처리
#
# 조건: Det 단독 x {v5-NPU, v8s-CPU} x INPUT_FPS{0,30} x 3회 반복 = 12회 실행
# batch=1, threshold=1, timeout=0ms, priority=0 (infer_yolov5_hailo8l.cpp 기본값, 안 건드림)
#
# 측정값: infer_yolov5_hailo8l.cpp가 자체 chrono 타이머로 직접 측정하는
#   avg_preprocess_ms/avg_latency_ms/avg_postprocess_ms/avg_total_time_ms/total_time_s
#   에 더해, 이번 버전은 HRTT 기반 지표(npu_percent/switches_per_s/idle_time_pct/
#   avg_fps_hrtt/avg_latency_hrtt/max_latency_hrtt/activation_hrtt) 컬럼도 CSV에
#   추가돼 있다(csv 헤더 변경분, infer_yolov5_hailo8l.cpp 참고). npu_percent는 이
#   스크립트가 실행 직후 바로 채우고, 나머지는 다운로드 후
#   fill_hrtt_columns_det_v8s_vs_v5npu.py로 채운다.
#
# 실행 위치: 보드(rpi1, ~/hailo_cpp_test/), infer_yolov5_hailo8l.cpp와 같은 디렉토리.
# 사전 준비물(같은 디렉토리에 scp 되어 있어야 함): infer_yolov5_hailo8l.cpp,
#   postprocess_8l.hpp, model_types.hpp, sys_monitor.hpp, image_utils.hpp,
#   output_classify.hpp, model_setup.hpp, model_runner.hpp, hailo_utilization.py
# 사용법: chmod +x run_det_v8s_vs_v5npu_fpssweep.sh
#         nohup bash run_det_v8s_vs_v5npu_fpssweep.sh > det_v8s_vs_v5npu_fpssweep_log.txt 2>&1 &
#         disown
# =============================================================================
set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 ~/hailo_cpp_test 없음"; exit 1; }

SRC=infer_yolov5_hailo8l.cpp
BIN=infer_yolov5_hailo8l

for req in "$SRC" postprocess_8l.hpp model_types.hpp sys_monitor.hpp image_utils.hpp \
           output_classify.hpp model_setup.hpp model_runner.hpp hailo_utilization.py; do
    if [ ! -f "$req" ]; then
        echo "[오류] $req 를 찾을 수 없음 — hailo_8L/에서 ~/hailo_cpp_test/로 scp 해둘 것."
        exit 1
    fi
done
cp "$SRC" "${SRC}.bak"
echo "[백업] ${SRC} -> ${SRC}.bak"

REPEAT=3
FPS_LIST=(0 30)
MAX_ATTEMPTS=3

HEF_DIR="/home/rpi1/hailo-rpi5-examples/resources"
V5_HEF_NAME=yolov5xs_wo_spp_nms_core.hef
V5_HEF_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/${V5_HEF_NAME}"
V8S_HEF="${HEF_DIR}/yolov8s_h8l.hef"

mkdir -p "$HEF_DIR"

if [ ! -f "$V8S_HEF" ]; then
    echo "[오류] $V8S_HEF 없음 — 기존 Det 기준 모델이 필요함(다른 실험에서 이미 받아둔 파일)."
    cp "${SRC}.bak" "$SRC"
    exit 1
fi

if [ ! -f "${HEF_DIR}/${V5_HEF_NAME}" ]; then
    echo "[HEF 없음] ${HEF_DIR}/${V5_HEF_NAME} 를 공식 hailo_model_zoo S3(hailo8l 타겟)에서 받는다..."
    wget -q --show-progress -O "${HEF_DIR}/${V5_HEF_NAME}" "$V5_HEF_URL"
fi

echo "[HEF 확인] yolov5xs_wo_spp_nms_core.hef (hailo8l) parse-hef:"
hailortcli parse-hef "${HEF_DIR}/${V5_HEF_NAME}" | head -20
echo ""
echo "[중요] 위 출력의 Architecture가 HAILO8L인지 반드시 확인할 것."
# nohup/백그라운드(&)로 실행되면 stdin이 터미널에 연결돼 있어도 이 프로세스가
# foreground job이 아니라서 read가 SIGTTIN으로 멈춰버린다(백그라운드에서 tty 읽기 시도 시
# 셸이 정지시킴). tty가 실제로 연결된 대화형 세션(-t 0)일 때만 확인 프롬프트를 띄우고,
# 아니면(nohup/백그라운드/파이프) 자동으로 넘어간다 — Architecture 확인은 로그에 이미
# 찍혀 있으니 사후에 grep해서 확인할 것.
if [ -t 0 ]; then
    read -p "계속 진행하려면 Enter, 중단하려면 Ctrl+C: " _
else
    echo "[비대화형 실행 감지 — 확인 프롬프트 자동 통과] 위 Architecture 출력을 로그에서 반드시 확인할 것."
fi

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_det_v8s_vs_v5npu_fpssweep_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_det_v8s_vs_v5npu_fpssweep_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"
TRACES_DIR="$OUTDIR/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"
CSV="${CSV_DIR}/results_det_v8s_vs_v5npu_fpssweep.csv"

echo "실험 폴더: $EXP_DIR"
echo "설정     : Det 단독, FPS={${FPS_LIST[*]}}, batch=1/threshold=1/timeout=0/priority=0, ${REPEAT}회 반복 x 2조건 x 2FPS = $((REPEAT * 2 * ${#FPS_LIST[@]}))회"
echo ""

count_rows() {
    if [ ! -f "$CSV" ]; then echo 0; return; fi
    local n
    n=$(($(wc -l < "$CSV") - 1))
    [ "$n" -lt 0 ] && n=0
    echo "$n"
}

RUN_ID=0

# 인자: BASELINE(0=v5 NPU, 1=v8s CPU) LABEL LABEL_TAG(hrtt 파일명용) FPS
run_condition() {
    local BASELINE=$1 LABEL=$2 LABEL_TAG=$3 FPS=$4

    sed -i "s/^#define USE_CPU_BASELINE_INSTEAD .*/#define USE_CPU_BASELINE_INSTEAD $BASELINE/" "$SRC"
    sed -i "s/^#define INPUT_FPS .*/#define INPUT_FPS           $FPS/" "$SRC"
    echo "### [$LABEL] FPS=$FPS 빌드 중 (USE_CPU_BASELINE_INSTEAD=$BASELINE) ###"
    g++ "$SRC" -o "$BIN" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then
        echo "[!] 컴파일 실패 ($LABEL, FPS=$FPS) — 이 조건 전체 건너뜀"
        return
    fi

    for run in $(seq 1 $REPEAT); do
        RUN_ID=$((RUN_ID + 1))
        local attempt=1
        local before
        before=$(count_rows)
        while [ $attempt -le $MAX_ATTEMPTS ]; do
            echo "--- [$LABEL] FPS=$FPS 반복 $run/$REPEAT (run_id=$RUN_ID, 시도 $attempt/$MAX_ATTEMPTS) ---"

            > npu_log.txt
            rm -f /tmp/hmon_files/*
            rm -f "$TRACES_DIR"/hailort_*.hrtt
            source ~/hailo_platform_venv/bin/activate 2>/dev/null
            python3 hailo_utilization.py &
            NPU_PID=$!
            sleep 2

            export HAILO_TRACE=scheduler
            export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
            export HAILO_TRACE_PATH="$TRACES_DIR"
            export HAILO_MONITOR=1

            ./"$BIN" "$RUN_ID" "$CSV"

            kill $NPU_PID 2>/dev/null; sleep 1; rm -f /tmp/hmon_files/*

            # npu_percent를 방금 기록된 마지막 CSV 행에 바로 채움 (auto_experiment_default_workload.sh와 동일 패턴)
            python3 - "$CSV" npu_log.txt <<'PY'
import sys, csv, re
csv_path, log_path = sys.argv[1], sys.argv[2]
vals = []
try:
    for line in open(log_path):
        m = re.search(r'NPU:\s*([\d.]+)%', line)
        if m:
            v = float(m.group(1))
            if v > 0: vals.append(v)
except FileNotFoundError:
    pass
if not vals:
    print("  [npu] 활성 구간 없음"); sys.exit(0)
avg = sum(vals) / len(vals)
rows = list(csv.reader(open(csv_path)))
if len(rows) >= 2:
    header, last = rows[0], rows[-1]
    if 'npu_percent' in header:
        last[header.index('npu_percent')] = f"{avg:.4f}"
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        print(f"  [npu] npu_percent={avg:.2f}% 기록")
PY

            local after
            after=$(count_rows)
            if [ "$after" -gt "$before" ]; then
                echo "  [OK] CSV 행 $before -> $after"
                break
            fi
            echo "  [!] CSV 행이 늘지 않음 — 재시도"
            attempt=$((attempt + 1))
            sleep 2
        done

        # HRTT 파일을 조건/FPS/반복회차 구분되게 rename
        LATEST_HRTT=""
        for i in $(seq 1 35); do
            LATEST_HRTT=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
            [ -n "$LATEST_HRTT" ] && break; sleep 1
        done
        if [ -n "$LATEST_HRTT" ]; then
            TS=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
            NEW="${TRACES_DIR}/det_${LABEL_TAG}_fps${FPS}_run${run}_${TS}.hrtt"
            mv "$LATEST_HRTT" "$NEW"; echo "  HRTT: $(basename "$NEW")"
        else
            echo "  [!] HRTT 미생성 (run_id=$RUN_ID)"
        fi

        sleep 1
    done
}

for FPS in "${FPS_LIST[@]}"; do
    echo "===== INPUT_FPS=$FPS ====="
    run_condition 0 "YOLOv5-NPU후처리(nms_core)"        "v5npu"  "$FPS"
    run_condition 1 "YOLOv8s_h8l-CPU후처리(engine=cpu)" "v8scpu" "$FPS"
done

cp "${SRC}.bak" "$SRC"
echo ""
echo "[복구] ${SRC} 원본 설정(INPUT_FPS=60, USE_CPU_BASELINE_INSTEAD=0)으로 복구 완료"

echo ""
echo "===== 완료: $(count_rows)/$((REPEAT * 2 * ${#FPS_LIST[@]}))행 수집 ====="
echo "CSV: $CSV"
echo "HRTT 트레이스: $TRACES_DIR"

echo ""
echo "[다음 단계] PC로 결과 다운로드 (PowerShell에서):"
echo "  scp -P 40021 rpi1@<host>:~/hailo_cpp_test/${EXP_DIR}/csv/*.csv ."
echo "  scp -P 40021 \"rpi1@<host>:~/hailo_cpp_test/${EXP_DIR}/traces/*.hrtt\" ."
