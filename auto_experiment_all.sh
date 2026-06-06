#!/bin/bash
# ========== 전체 자동 실험 스크립트 (63개) ==========
cd ~/hailo_cpp_test

compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local PRI_D=$4 PRI_S=$5 PRI_P=$6

    echo "======================================"
    echo "실험: Det=$USE_D($PRI_D), Seg=$USE_S($PRI_S), Pose=$USE_P($PRI_P)"
    echo "======================================"

    # traces 디렉토리 생성
    mkdir -p traces

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

    # 파라미터 확인
    echo "파라미터 확인:"
    grep "#define BATCH_SIZE\|#define THRESHOLD\|#define TIMEOUT\|#define USE\|#define PRIORITY" infer_scheduler.cpp

    # 컴파일
    g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread
    if [ $? -ne 0 ]; then echo "컴파일 실패!"; return; fi

    # npu_log 초기화
    > npu_log.txt
    rm -f /tmp/hmon_files/*

    # 이전 실험의 unnamed HRTT 파일 제거 (잘못된 파일 선택 방지)
    rm -f traces/hailort_*.hrtt

    # NPU 모니터링 백그라운드
    source ~/hailo_platform_venv/bin/activate
    python3 hailo_utilization.py &
    NPU_PID=$!
    sleep 2

    # HRTT 설정
    export HAILO_TRACE=scheduler
    export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
    export HAILO_TRACE_PATH=/home/rpi1/hailo_cpp_test/traces
    export HAILO_MONITOR=1

    # 추론 실행
    ./infer_scheduler

    # NPU 종료
    kill $NPU_PID
    sleep 1
    rm -f /tmp/hmon_files/*

    # HRTT 파일이 생성될 때까지 대기 (최대 35초)
    LATEST_HRTT=""
    for i in $(seq 1 35); do
        LATEST_HRTT=$(ls -t traces/hailort_*.hrtt 2>/dev/null | head -1)
        [ -n "$LATEST_HRTT" ] && break
        sleep 1
    done

    if [ -n "$LATEST_HRTT" ]; then
        TIMESTAMP=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
        NEW_NAME="traces/${USE_D}D-${USE_S}S-${USE_P}P_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_${TIMESTAMP}.hrtt"
        mv "$LATEST_HRTT" "$NEW_NAME"
        echo "HRTT: $NEW_NAME"

        # runtime_report.html을 실험별로 즉시 복사 저장 (덮어쓰이기 전에)
        if [ -f "traces/runtime_report.html" ]; then
            HTML_NAME="traces/${USE_D}D-${USE_S}S-${USE_P}P_${PRI_D}PD-${PRI_S}PS-${PRI_P}PP_${TIMESTAMP}.html"
            cp "traces/runtime_report.html" "$HTML_NAME"
            echo "HTML: $HTML_NAME"
        fi
    else
        echo "경고: HRTT 파일이 생성되지 않았습니다."
    fi

    # NPU 파싱
    python3 parse_npu_log.py

    echo "완료!"
    echo ""
}

priorities=(0 15 31)

echo "===== Single 실험 (9개) ====="

for p in "${priorities[@]}"; do
    compile_and_run 1 0 0 $p 0 0
done
for p in "${priorities[@]}"; do
    compile_and_run 0 1 0 0 $p 0
done
for p in "${priorities[@]}"; do
    compile_and_run 0 0 1 0 0 $p
done

echo "===== 2개 Multi 실험 (27개) ====="

for pd in "${priorities[@]}"; do
    for ps in "${priorities[@]}"; do
        compile_and_run 1 1 0 $pd $ps 0
    done
done
for pd in "${priorities[@]}"; do
    for pp in "${priorities[@]}"; do
        compile_and_run 1 0 1 $pd 0 $pp
    done
done
for ps in "${priorities[@]}"; do
    for pp in "${priorities[@]}"; do
        compile_and_run 0 1 1 0 $ps $pp
    done
done

echo "===== 3개 Multi 실험 (27개) ====="

for pd in "${priorities[@]}"; do
    for ps in "${priorities[@]}"; do
        for pp in "${priorities[@]}"; do
            compile_and_run 1 1 1 $pd $ps $pp
        done
    done
done

echo "===== 모든 실험 완료! (63개) ====="