#!/bin/bash
# ========== 전체 자동 실험 스크립트 (63개 조건 × 3회 반복 = 189회) ==========
cd ~/hailo_cpp_test

REPEAT=3  # 반복 횟수

# ── 실험 폴더 생성 (날짜 + 회차 자동 증가) ──
EXP_DATE=$(date +%Y-%m-%d)
EXP_NUM=1
EXP_DIR="experiments/${EXP_DATE}_priority_exp${EXP_NUM}"
while [ -d "$EXP_DIR" ]; do
    EXP_NUM=$((EXP_NUM + 1))
    EXP_DIR="experiments/${EXP_DATE}_priority_exp${EXP_NUM}"
done
mkdir -p "$EXP_DIR/traces"
CSV_FILE="$HOME/hailo_cpp_test/${EXP_DIR}/results_all.csv"
TRACES_DIR="$HOME/hailo_cpp_test/${EXP_DIR}/traces"

echo "실험 폴더: $EXP_DIR"
echo "CSV 저장: $CSV_FILE"
echo "HRTT 저장: $TRACES_DIR"
echo ""

compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local PRI_D=$4 PRI_S=$5 PRI_P=$6
    local RUN_ID=$7

    echo "======================================"
    echo "실험: Det=$USE_D($PRI_D), Seg=$USE_S($PRI_S), Pose=$USE_P($PRI_P) | Run=$RUN_ID"
    echo "======================================"

    mkdir -p "$TRACES_DIR"

    # 파라미터 수정
    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      1/" infer_scheduler.cpp
    sed -i "s/#define THRESHOLD .*/#define THRESHOLD       1/" infer_scheduler.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      0/" infer_scheduler.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    $USE_D/" infer_scheduler.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    $USE_S/" infer_scheduler.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   $USE_P/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    $PRI_D/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    $PRI_S/" infer_scheduler.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   $PRI_P/" infer_scheduler.cpp

    echo "파라미터 확인:"
    grep "#define BATCH_SIZE\|#define THRESHOLD\|#define TIMEOUT\|#define USE\|#define PRIORITY" infer_scheduler.cpp

    # 컴파일 (조건당 1회만 컴파일, run 반복 시 재컴파일 불필요하나 안전하게 유지)
    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread
    if [ $? -ne 0 ]; then echo "컴파일 실패!"; return; fi

    # npu_log 초기화
    > npu_log.txt
    rm -f /tmp/hmon_files/*

    # 이전 unnamed HRTT 파일 제거
    rm -f "$TRACES_DIR"/hailort_*.hrtt

    # NPU 모니터링 백그라운드
    source ~/hailo_platform_venv/bin/activate
    python3 hailo_utilization.py &
    NPU_PID=$!
    sleep 2

    # HRTT 설정
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH="$TRACES_DIR"
    export HAILO_MONITOR=1

    # 추론 실행 (run_id, csv_path 인자 전달)
    ./infer_scheduler $RUN_ID "$CSV_FILE"

    # NPU 종료
    kill $NPU_PID
    sleep 1
    rm -f /tmp/hmon_files/*

    # HRTT 파일 대기 (최대 35초)
    LATEST_HRTT=""
    for i in $(seq 1 35); do
        LATEST_HRTT=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST_HRTT" ] && break
        sleep 1
    done

    if [ -n "$LATEST_HRTT" ]; then
        TIMESTAMP=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
        NEW_NAME="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_run${RUN_ID}_${TIMESTAMP}.hrtt"
        mv "$LATEST_HRTT" "$NEW_NAME"
        echo "HRTT: $NEW_NAME"

        if [ -f "${TRACES_DIR}/runtime_report.html" ]; then
            HTML_NAME="${TRACES_DIR}/${USE_D}D-${USE_S}S-${USE_P}P_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_run${RUN_ID}_${TIMESTAMP}.html"
            cp "${TRACES_DIR}/runtime_report.html" "$HTML_NAME"
            echo "HTML: $HTML_NAME"
        fi
    else
        echo "경고: HRTT 파일이 생성되지 않았습니다."
    fi

    # NPU 파싱 (csv 경로 전달)
    python3 parse_npu_log.py "$CSV_FILE"

    echo "완료! (Run $RUN_ID)"
    echo ""
}

priorities=(0 15 31)

echo "===== Single 실험 (9개 조건 × ${REPEAT}회) ====="

for p in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do
        compile_and_run 1 0 0 $p 0 0 $run
    done
done
for p in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do
        compile_and_run 0 1 0 0 $p 0 $run
    done
done
for p in "${priorities[@]}"; do
    for run in $(seq 1 $REPEAT); do
        compile_and_run 0 0 1 0 0 $p $run
    done
done

echo "===== 2개 Multi 실험 (27개 조건 × ${REPEAT}회) ====="

for pd in "${priorities[@]}"; do
    for ps in "${priorities[@]}"; do
        for run in $(seq 1 $REPEAT); do
            compile_and_run 1 1 0 $pd $ps 0 $run
        done
    done
done
for pd in "${priorities[@]}"; do
    for pp in "${priorities[@]}"; do
        for run in $(seq 1 $REPEAT); do
            compile_and_run 1 0 1 $pd 0 $pp $run
        done
    done
done
for ps in "${priorities[@]}"; do
    for pp in "${priorities[@]}"; do
        for run in $(seq 1 $REPEAT); do
            compile_and_run 0 1 1 0 $ps $pp $run
        done
    done
done

echo "===== 3개 Multi 실험 (27개 조건 × ${REPEAT}회) ====="

for pd in "${priorities[@]}"; do
    for ps in "${priorities[@]}"; do
        for pp in "${priorities[@]}"; do
            for run in $(seq 1 $REPEAT); do
                compile_and_run 1 1 1 $pd $ps $pp $run
            done
        done
    done
done

echo "===== 모든 실험 완료! (63개 조건 × ${REPEAT}회 = $((63 * REPEAT))회) ====="
