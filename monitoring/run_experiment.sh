#!/bin/bash
# ========== 실험 실행 스크립트 ==========

cd ~/hailo_cpp_test

# npu_log 초기화
> npu_log.txt

# 파라미터 값 읽기 (infer_scheduler.cpp에서)
BATCH=$(grep "#define BATCH_SIZE" infer_scheduler.cpp | awk '{print $3}')
THRESHOLD=$(grep "#define THRESHOLD" infer_scheduler.cpp | awk '{print $3}')
TIMEOUT=$(grep "#define TIMEOUT_MS" infer_scheduler.cpp | awk '{print $3}')
PRIORITY=$(grep "#define PRIORITY" infer_scheduler.cpp | awk '{print $3}')

echo "실험 시작: batch=$BATCH, threshold=$THRESHOLD, timeout=$TIMEOUT, priority=$PRIORITY"

# HRTT 환경변수 설정
export HAILO_TRACE=scheduler
export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
export HAILO_TRACE_PATH=/home/rpi1/hailo_cpp_test/traces
export HAILO_MONITOR=1

# 추론 실행
./infer_scheduler

# 추론 완료 후 HRTT 파일명 변경
LATEST_HRTT=$(ls -t traces/*.hrtt 2>/dev/null | head -1)
if [ ! -z "$LATEST_HRTT" ]; then
    TIMESTAMP=$(basename $LATEST_HRTT .hrtt | sed 's/hailort_//')
    NEW_NAME="traces/(${BATCH}B-${THRESHOLD}T-${TIMEOUT}TO-${PRIORITY}P)_${TIMESTAMP}.hrtt"
    mv "$LATEST_HRTT" "$NEW_NAME"
    echo "HRTT 파일명 변경: $NEW_NAME"
fi

# NPU 로그 파싱
source ~/hailo_platform_venv/bin/activate
python3 parse_npu_log.py

echo "실험 완료!"
