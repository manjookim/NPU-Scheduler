#!/bin/bash
# =============================================================================
# [2026-08-19] Det 단일모델 NPU vs CPU 후처리 비교 (Hailo-8L, rpi1)
#
# 목적: Detection 후처리를 NPU(neural core)에서 하는 것과 CPU(host)에서 하는 것 중
# 어느 쪽이 더 빠른지 비교. HEF 2종:
#   - yolov8s_h8l.hef               : engine=cpu (기존 Det 기준 모델, 640x640)              -> CPU 후처리
#   - yolov5xs_wo_spp_nms_core.hef  : engine=nn_core/auto (공식 hailo_model_zoo hailo8l 타겟,
#                                      512x512) -> NPU 후처리
# (YOLOv8은 애초에 NPU측 후처리를 지원하지 않아 v5로 바꾼 것 — PROJECT_SUMMARY.md §6 참고.
#  즉 "후처리 담당 프로세서"와 "모델 버전(v8s/v5, img_size 640/512)"이 함께 바뀌는 것은
#  이 두 HEF를 쓰는 이상 피할 수 없는 제약 — 사용자도 이를 인지하고 그 외 나머지
#  (batch/threshold/timeout/priority/INPUT_FPS/데이터셋)는 전부 통일하기로 함.)
#
# [원래 Hailo-8(rpi4)용으로 준비했던 실험을 사용자 요청으로 Hailo-8L(rpi1)로 이관함 —
# 앞으로는 8L에서만 실험, Hailo-8은 사용 안 함.]
#
# 실행 프로그램: infer_yolov5_hailo8l.cpp (신규 — hailo_8/infer_yolov5_hailo8.cpp를 8L로
# 포팅. USE_CPU_BASELINE_INSTEAD 매크로로 두 HEF를 스위칭하도록 만들어져 있음). 이 스크립트는
# 그 매크로를 sed로 토글해가며 양쪽을 순서대로 빌드/실행한다(INPUT_FPS=60은 파일 기본값에
# 이미 반영되어 있어 따로 sed할 필요 없음, 필요시 스크립트 상단 FPS 변수만 바꿔서 재적용 가능).
#
# 조건: Det 단독(Seg/Pose 없음) x {v5-NPU, v8s-CPU} 2가지 x FPS=60 x 3회 반복 = 6회 실행
# batch=1, threshold=1, timeout=0ms, priority=0 (infer_yolov5_hailo8l.cpp 기본값, 안 건드림)
#
# 측정값 (infer_yolov5_hailo8l.cpp가 자체 chrono 타이머로 직접 측정 — 별도 HRTT/NPU
# 모니터링 없이도 CSV에 바로 기록됨):
#   avg_preprocess_ms (전처리) / avg_latency_ms (추론) / avg_postprocess_ms (후처리) /
#   avg_total_time_ms (장당 전체=위 세 값의 합) / total_time_s (데이터셋 전체 처리 총 시간)
#
# [참고] Hailo-8L(rpi1, Ubuntu 24.04, 커널 6.8.x)은 Hailo-8(rpi4)에서 겪었던 hailo_pci
# find_vma 커널 크래시 이슈가 보고된 적 없는 보드다(QUESTION_FOR_TA.md — 그 버그는 rpi4의
# 커널 6.12.x 계열에서만 재현됨). 그래도 안전하게, 각 실행 후 CSV 행이 실제로 늘었는지
# 확인해서 안 늘면 자동 재시도하는 최소한의 안전장치는 넣어둠(최대 MAX_ATTEMPTS회).
#
# 실행 위치: 보드(rpi1, ~/hailo_cpp_test/), infer_yolov5_hailo8l.cpp와 같은 디렉토리
# (hailo_8L/scripts/*.sh의 기존 관례를 따라 cwd와 무관하게 스크립트가 직접 cd한다).
# 사용법: chmod +x run_det_v8s_vs_v5npu_fps60.sh
#         nohup bash run_det_v8s_vs_v5npu_fps60.sh > det_v8s_vs_v5npu_fps60_log.txt 2>&1 &
#         disown
# =============================================================================
set -u
cd ~/hailo_cpp_test || { echo "작업 폴더 ~/hailo_cpp_test 없음"; exit 1; }

SRC=infer_yolov5_hailo8l.cpp
BIN=infer_yolov5_hailo8l

if [ ! -f "$SRC" ]; then
    echo "[오류] $SRC 를 찾을 수 없음 — hailo_8L/infer_yolov5_hailo8l.cpp를 ~/hailo_cpp_test/에 scp 해둘 것."
    exit 1
fi
if [ ! -f postprocess_8l.hpp ]; then
    echo "[오류] postprocess_8l.hpp 를 찾을 수 없음 — $SRC 와 같은 디렉토리에 scp 해둘 것."
    exit 1
fi
cp "$SRC" "${SRC}.bak"
echo "[백업] ${SRC} -> ${SRC}.bak"

REPEAT=3
FPS=60
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
echo "[중요] 위 출력의 Architecture가 HAILO8L인지 반드시 확인할 것 (HAILO8이면 잘못된 파일 —"
echo "       8L 보드에서 8용 HEF는 로드 자체가 실패하거나 오동작할 수 있음)."
read -p "계속 진행하려면 Enter, 중단하려면 Ctrl+C: " _

EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_det_v8s_vs_v5npu_fps60_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1)); EXP_DIR="experiments/${EXP_DATE}_det_v8s_vs_v5npu_fps60_exp${EXP_NUM}"
done
OUTDIR="$HOME/hailo_cpp_test/${EXP_DIR}"
CSV_DIR="$OUTDIR/csv"
mkdir -p "$CSV_DIR"
CSV="${CSV_DIR}/results_det_v8s_vs_v5npu_fps60.csv"

echo "실험 폴더: $EXP_DIR"
echo "설정     : Det 단독, FPS=$FPS, batch=1/threshold=1/timeout=0/priority=0, ${REPEAT}회 반복 x 2조건 = $((REPEAT * 2))회"
echo ""

# INPUT_FPS는 두 조건 공통이라 미리 한 번만 sed (파일 기본값이 이미 60이지만, 혹시 값이
# 다르게 바뀌어 있을 경우를 대비해 명시적으로 맞춰둔다)
sed -i "s/^#define INPUT_FPS .*/#define INPUT_FPS           $FPS/" "$SRC"

count_rows() {
    if [ ! -f "$CSV" ]; then echo 0; return; fi
    local n
    n=$(($(wc -l < "$CSV") - 1))
    [ "$n" -lt 0 ] && n=0
    echo "$n"
}

# 인자: BASELINE(0=v5 NPU, 1=v8s CPU) LABEL RUN_ID_START
run_condition() {
    local BASELINE=$1 LABEL=$2 RUN_ID_START=$3

    sed -i "s/^#define USE_CPU_BASELINE_INSTEAD .*/#define USE_CPU_BASELINE_INSTEAD $BASELINE/" "$SRC"
    echo "### [$LABEL] 빌드 중 (USE_CPU_BASELINE_INSTEAD=$BASELINE, INPUT_FPS=$FPS) ###"
    g++ "$SRC" -o "$BIN" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
    if [ $? -ne 0 ]; then
        echo "[!] 컴파일 실패 ($LABEL) — 이 조건 전체 건너뜀"
        return
    fi

    for run in $(seq 1 $REPEAT); do
        local run_id=$((RUN_ID_START + run - 1))
        local attempt=1
        local before
        before=$(count_rows)
        while [ $attempt -le $MAX_ATTEMPTS ]; do
            echo "--- [$LABEL] 반복 $run/$REPEAT (run_id=$run_id, 시도 $attempt/$MAX_ATTEMPTS) ---"
            ./"$BIN" "$run_id" "$CSV"
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
        sleep 1
    done
}

run_condition 0 "YOLOv5-NPU후처리(nms_core)"    1
run_condition 1 "YOLOv8s_h8l-CPU후처리(engine=cpu)" $((REPEAT + 1))

cp "${SRC}.bak" "$SRC"
echo ""
echo "[복구] ${SRC} 원본 설정(INPUT_FPS=60 그대로, USE_CPU_BASELINE_INSTEAD=0)으로 복구 완료"

echo ""
echo "===== 완료: $(count_rows)/$((REPEAT * 2))행 수집 ====="
echo "CSV: $CSV"

if [ -f scripts/make_avg_csv_det_v8s_vs_v5npu.py ]; then
    echo "평균 계산 중..."
    python3 scripts/make_avg_csv_det_v8s_vs_v5npu.py "$CSV"
else
    echo "[참고] make_avg_csv_det_v8s_vs_v5npu.py를 hailo_8L/scripts/에 같이 두면 평균 CSV까지 자동 생성됨"
fi

echo ""
echo "[다음 단계] PC로 결과 다운로드 (README.md의 rpi1 접속 정보 사용):"
echo "  scp -P 40021 rpi1@155.230.16.157:~/hailo_cpp_test/${EXP_DIR}/csv/*.csv ."
echo "  이후 PC에서: python3 hailo_8L/scripts/build_xlsx_det_v8s_vs_v5npu.py results_det_v8s_vs_v5npu_fps60.csv results_det_v8s_vs_v5npu_fps60_avg.csv"
