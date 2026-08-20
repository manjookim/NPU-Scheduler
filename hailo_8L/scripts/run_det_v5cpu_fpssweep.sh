#!/bin/bash
# =============================================================================
# [2026-08-20] Det 단일모델 v5-CPU후처리 벤치마크 — FPS 스윕 (Hailo-8L, rpi1)
#
# run_det_v8s_vs_v5npu_fpssweep.sh(v5-NPU vs v8s-CPU 비교판)의 후속. 사용자 요청으로
# "v5 모델을 CPU로 후처리하는" 조건을 추가로 측정한다 — v5-NPU(yolov5xs_wo_spp_nms_core.hef,
# engine=nn_core/auto)와 같은 v5xs 아키텍처지만 "_nms_core" 접미사가 없는
# yolov5xs_wo_spp.hef(engine=cpu, hailo_model_zoo에서 host CPU가 NMS까지 전부 수행하도록
# 컴파일된 버전)를 사용한다. infer_yolov5_hailo8l.cpp의 MODEL_MODE=2가 이 조건.
#
# 조건: Det 단독, v5-CPU후처리 x INPUT_FPS{0(제한없음/최대속도), 30} x 3회 반복 = 6회 실행
# batch=1, threshold=1, timeout=0ms, priority=0 (infer_yolov5_hailo8l.cpp 기본값, 안 건드림 —
# 사용자 요청: "4개 파라미터는 default 값으로").
#
# 측정값: run_det_v8s_vs_v5npu_fpssweep.sh와 동일 — chrono 기반 avg_preprocess_ms/
#   avg_latency_ms/avg_postprocess_ms/avg_total_time_ms/total_time_s + npu_percent(스크립트가
#   즉시 채움) + HRTT 기반 switches_per_s/idle_time_pct/avg_fps_hrtt/avg_latency_hrtt/
#   max_latency_hrtt/activation_hrtt(다운로드 후 fill_hrtt_columns_det_v5cpu.py로 채움).
#
# 실행 위치: 보드(rpi1, ~/hailo_cpp_test/), infer_yolov5_hailo8l.cpp와 같은 디렉토리.
# 사전 준비물(같은 디렉토리에 scp 되어 있어야 함): infer_yolov5_hailo8l.cpp,
#   postprocess_8l.hpp, model_types.hpp, sys_monitor.hpp, image_utils.hpp,
#   output_classify.hpp, model_setup.hpp, model_runner.hpp, hailo_utilization.py
# 사용법: chmod +x run_det_v5cpu_fpssweep.sh
#         nohup bash run_det_v5cpu_fpssweep.sh > det_v5cpu_fpssweep_log.txt 2>&1 &
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

HEF_DIR="$HOME/hailo-rpi5-examples/resources"
V5_CPU_HEF_NAME=yolov5xs_wo_spp.hef
V5_CPU_HEF_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/${V5_CPU_HEF_NAME}"

mkdir -p "$HEF_DIR"

if [ ! -f "${HEF_DIR}/${V5_CPU_HEF_NAME}" ]; then
    echo "[HEF 없음] ${HEF_DIR}/${V5_CPU_HEF_NAME} 를 공식 hailo_model_zoo S3(hailo8l 타겟)에서 받는다..."
    wget -q --show-progress -O "${HEF_DIR}/${V5_CPU_HEF_NAME}" "$V5_CPU_HEF_URL"
fi

if [ ! -s "${HEF_DIR}/${V5_CPU_HEF_NAME}" ]; then
    echo "[오류] ${V5_CPU_HEF_NAME} 다운로드 실패(파일이 없거나 0바이트). URL 확인 필요: $V5_CPU_HEF_URL"
    cp "${SRC}.bak" "$SRC"
    exit 1
fi

echo "[HEF 확인] yolov5xs_wo_spp.hef (hailo8l) parse-hef:"
hailortcli parse-hef "${HEF_DIR}/${V5_CPU_HEF_NAME}" | head -20
echo ""
echo "[중요] 위 출력의 Architecture가 HAILO8L인지, output 포맷이 host CPU NMS(engine=cpu)에"
echo "       맞게 'HAILO NMS BY CLASS'류로 나오는지 반드시 확인할 것."
# nohup/백그라운드(&)로 실행되면 stdin이 터미널에 연결돼 있어도 이 프로세스가
# foreground job이 아니라서 read가 SIGTTIN으로 멈춰버린다. tty가 실제로 연결된 대화형
# 세션(-t 0)일 때만 확인 프롬프트를 띄우고, 아니면 자동으로 넘어간다.
if [ -t 0 ]; then
    read -p "계속 진행하려면 Enter, 중단하려면 Ctrl+C: " _
else
    echo "[비대화형 실행 감지 — 확인 프롬프트 자동 통과] 위 Architecture/포맷 출력을 로그에서 반드시 확인할 것."
fi

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_det_v5cpu_fpssweep_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_det_v5cpu_fpssweep_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"
TRACES_DIR="$OUTDIR/traces"
mkdir -p "$CSV_DIR" "$TRACES_DIR"
CSV="${CSV_DIR}/results_det_v5cpu_fpssweep.csv"

echo "실험 폴더: $EXP_DIR"
echo "설정     : Det 단독(v5-CPU후처리), FPS={${FPS_LIST[*]}}, batch=1/threshold=1/timeout=0/priority=0, ${REPEAT}회 반복 x ${#FPS_LIST[@]}FPS = $((REPEAT * ${#FPS_LIST[@]}))회"
echo ""

count_rows() {
    if [ ! -f "$CSV" ]; then echo 0; return; fi
    local n
    n=$(($(wc -l < "$CSV") - 1))
    [ "$n" -lt 0 ] && n=0
    echo "$n"
}

RUN_ID=0

# 인자: FPS
run_condition() {
    local FPS=$1
    local LABEL="YOLOv5-CPU후처리(wo_spp)"
    local LABEL_TAG="v5cpu"

    sed -i "s/^#define MODEL_MODE .*/#define MODEL_MODE 2/" "$SRC"
    sed -i "s/^#define INPUT_FPS .*/#define INPUT_FPS           $FPS/" "$SRC"
    echo "### [$LABEL] FPS=$FPS 빌드 중 (MODEL_MODE=2) ###"
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
            "$HOME/hailo_platform_venv/bin/python3" hailo_utilization.py &
            NPU_PID=$!
            sleep 2

            export HAILO_TRACE=scheduler
            export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
            export HAILO_TRACE_PATH="$TRACES_DIR"
            export HAILO_MONITOR=1

            ./"$BIN" "$RUN_ID" "$CSV"

            kill $NPU_PID 2>/dev/null; sleep 1; rm -f /tmp/hmon_files/*

            # npu_percent를 방금 기록된 마지막 CSV 행에 바로 채움
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
    run_condition "$FPS"
done

cp "${SRC}.bak" "$SRC"
echo ""
echo "[복구] ${SRC} 원본 설정(INPUT_FPS=60, MODEL_MODE=0)으로 복구 완료"

echo ""
echo "===== 완료: $(count_rows)/$((REPEAT * ${#FPS_LIST[@]}))행 수집 ====="
echo "CSV: $CSV"
echo "HRTT 트레이스: $TRACES_DIR"

echo ""
echo "[다음 단계] PC로 결과 다운로드 (PowerShell/bash에서):"
echo "  scp -P 40021 rpi1@<host>:~/hailo_cpp_test/${EXP_DIR}/csv/*.csv ."
echo "  scp -P 40021 \"rpi1@<host>:~/hailo_cpp_test/${EXP_DIR}/traces/*.hrtt\" ."
