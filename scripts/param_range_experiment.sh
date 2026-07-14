#!/bin/bash
# ============================================================
# param_range_experiment.sh
# 1. threshold 유효 상한 이진 탐색  (Det+Seg, 두 모델 모두 완료 여부 판정)
# 2. timeout_ms 유효 하한 이진 탐색 (Det+Seg, 프로그램 정상 종료 여부 판정)
# 3. threshold × timeout × batch × priority 결합 실험
# 4. 발견된 범위 내에서 랜덤 파라미터 7회 실험
# ============================================================

WORK_DIR=~/hailo_cpp_test
SRC=$WORK_DIR/infer_scheduler.cpp
RANGE_CSV=$WORK_DIR/param_range_results.csv
COMBINED_CSV=$WORK_DIR/param_combined_results.csv

# ── 이진 탐색 시 이미지 수 (빠른 판정용) ──
SEARCH_IMAGES=300
# ── 결합 실험 시 이미지 수 ──
COMBINED_IMAGES=1000
# ── 프로그램 hang 판정 타임아웃 (초) ──
RUN_TIMEOUT=360

# ── 결합 실험 파라미터 (필요에 따라 수정) ──
COMBINED_BATCHES=(1 2 4 8)
COMBINED_THRESHOLDS=(1 10 50 200)
COMBINED_TIMEOUTS=(0 100 1000 10000)
COMBINED_PRIORITIES=(0 15 31)

# ── 이진 탐색 결과 저장 (함수 간 공유) ──
THRESH_MAX=1
TIMEOUT_MIN=0

# ============================================================
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 소스를 패치 후 컴파일, 실행
# 인자: USE_D USE_S USE_P  BATCH THRESH TIMEOUT  PRI_D PRI_S PRI_P  MAX_IMGS  OUT_CSV
# 반환: 0=성공(결과 CSV에 행 추가됨), 1=실패/hang/컴파일 오류
compile_and_run() {
    local USE_D=$1 USE_S=$2 USE_P=$3
    local BATCH=$4 THRESH=$5 TIMEOUT=$6
    local PRI_D=$7 PRI_S=$8 PRI_P=$9
    local MAX_IMG=${10}
    local OUT_CSV=${11}

    cd $WORK_DIR
    cp $SRC /tmp/infer_test.cpp

    # 파라미터 패치
    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE      $BATCH/"    /tmp/infer_test.cpp
    sed -i "s/#define THRESHOLD .*/#define THRESHOLD       $THRESH/"   /tmp/infer_test.cpp
    sed -i "s/#define TIMEOUT_MS.*/#define TIMEOUT_MS      $TIMEOUT/"  /tmp/infer_test.cpp
    sed -i "s/#define USE_DET.*/#define USE_DET    $USE_D/"            /tmp/infer_test.cpp
    sed -i "s/#define USE_SEG.*/#define USE_SEG    $USE_S/"            /tmp/infer_test.cpp
    sed -i "s/#define USE_POSE.*/#define USE_POSE   $USE_P/"           /tmp/infer_test.cpp
    sed -i "s/#define PRIORITY_DET.*/#define PRIORITY_DET    $PRI_D/"  /tmp/infer_test.cpp
    sed -i "s/#define PRIORITY_SEG.*/#define PRIORITY_SEG    $PRI_S/"  /tmp/infer_test.cpp
    sed -i "s/#define PRIORITY_POSE.*/#define PRIORITY_POSE   $PRI_P/" /tmp/infer_test.cpp

    # CSV 경로 패치
    sed -i "s|#define CSV_PATH.*|#define CSV_PATH \"$OUT_CSV\"|" /tmp/infer_test.cpp

    # 이미지 수 제한 패치: get_image_files 호출 다음 줄에 resize 삽입
    sed -i '/std::vector<std::string> images = get_image_files/a\    if (images.size() > (size_t)'"$MAX_IMG"') images.resize((size_t)'"$MAX_IMG"');' \
        /tmp/infer_test.cpp

    # 컴파일
    g++ /tmp/infer_test.cpp -o /tmp/infer_test \
        -lhailort $(pkg-config --cflags --libs opencv4) -lpthread 2>/tmp/compile_err.txt
    if [ $? -ne 0 ]; then
        log "  컴파일 실패 → /tmp/compile_err.txt 확인"
        return 1
    fi

    # 실행 (hang 방지: RUN_TIMEOUT 초 제한)
    local before_lines
    before_lines=$(wc -l < "$OUT_CSV" 2>/dev/null || echo 0)

    # NPU 모니터링
    source ~/hailo_platform_venv/bin/activate
    export HAILO_MONITOR=1
    > npu_log.txt
    rm -f /tmp/hmon_files/*
    python3 hailo_utilization.py &
    local NPU_PID=$!
    sleep 2

    local TMP_OUT=/tmp/infer_out_$$.txt
    timeout $RUN_TIMEOUT /tmp/infer_test 2>&1 | tee "$TMP_OUT"
    local exit_code=${PIPESTATUS[0]}

    kill $NPU_PID 2>/dev/null
    wait $NPU_PID 2>/dev/null

    # NPU 퍼센트 파싱
    python3 parse_npu_log.py 2>/dev/null

    local after_lines
    after_lines=$(wc -l < "$OUT_CSV" 2>/dev/null || echo 0)

    # 오버플로우 감지: threshold/timeout이 음수로 출력되면 FAIL
    if grep -qE "threshold=-[0-9]|timeout=-[0-9]" "$TMP_OUT" 2>/dev/null; then
        log "  → OVERFLOW 감지 (파라미터 음수 표시) → FAIL 처리"
        rm -f "$TMP_OUT"
        return 1
    fi
    rm -f "$TMP_OUT"

    if [ $exit_code -eq 124 ]; then
        log "  → TIMEOUT (${RUN_TIMEOUT}s 초과, 스케줄러 starvation 의심)"
        return 1
    elif [ $exit_code -ne 0 ]; then
        log "  → 비정상 종료 (exit=$exit_code)"
        return 1
    elif [ "$after_lines" -le "$before_lines" ]; then
        log "  → CSV 행 미추가 (실행은 됐으나 결과 없음)"
        return 1
    fi

    return 0
}

# ============================================================
# 1. THRESHOLD 상한 이진 탐색
#    판정: threshold가 너무 크면 Det+Seg 중 하나가 starvation → hang
# ============================================================
find_threshold_range() {
    log "====== THRESHOLD 범위 탐색 시작 ======"

    log "--- 하한 확인 (0, 1 시도) ---"
    for v in 0 1; do
        log "  threshold=$v 시도..."
        if compile_and_run 1 1 0  1 $v 0  0 0 0  $SEARCH_IMAGES $RANGE_CSV; then
            log "  threshold=$v → OK"
        else
            log "  threshold=$v → FAIL"
        fi
    done

    log "--- 상한 이진 탐색 [2, 4294967295 (uint32 max)] ---"
    local lo=2 hi=4294967295 last_ok=1

    while [ $((hi - lo)) -gt 1 ]; do
        local mid=$(( (lo + hi) / 2 ))
        log "  threshold=$mid 시도... (lo=$lo hi=$hi)"
        if compile_and_run 1 1 0  1 $mid 0  0 0 0  $SEARCH_IMAGES $RANGE_CSV; then
            log "  threshold=$mid → OK"
            last_ok=$mid
            lo=$mid
        else
            log "  threshold=$mid → FAIL"
            hi=$mid
        fi
    done

    THRESH_MAX=$last_ok
    log ">>> Threshold 유효 최대값 추정: $THRESH_MAX"
    log "====== THRESHOLD 탐색 완료 ======"
}

# ============================================================
# 2. TIMEOUT_MS 하한 이진 탐색
#    판정: timeout이 너무 작으면 타임아웃 오류/비정상 종료 발생 가능
#    추가로 상한도 큰 값들로 확인
# ============================================================
find_timeout_range() {
    log "====== TIMEOUT_MS 범위 탐색 시작 ======"

    log "--- 기준값 확인 (0=비활성화) ---"
    log "  timeout=0 시도..."
    compile_and_run 1 1 0  1 1 0  0 0 0  $SEARCH_IMAGES $RANGE_CSV && \
        log "  timeout=0 → OK" || log "  timeout=0 → FAIL"

    log "--- 하한 이진 탐색 [1, 1000ms] ---"
    local lo=1 hi=1000 last_ok=0 first_fail=-1

    while [ $((hi - lo)) -gt 1 ]; do
        local mid=$(( (lo + hi) / 2 ))
        log "  timeout=${mid}ms 시도... (lo=$lo hi=$hi)"
        if compile_and_run 1 1 0  1 1 $mid  0 0 0  $SEARCH_IMAGES $RANGE_CSV; then
            log "  timeout=${mid}ms → OK"
            last_ok=$mid
            hi=$mid
        else
            log "  timeout=${mid}ms → FAIL"
            first_fail=$mid
            lo=$mid
        fi
    done

    if [ $first_fail -eq -1 ]; then
        TIMEOUT_MIN=1
        log ">>> Timeout 최솟값: 1ms (1~1000 전 범위 OK)"
    else
        TIMEOUT_MIN=$last_ok
        log ">>> Timeout 유효 최솟값 추정: ${TIMEOUT_MIN}ms"
    fi

    log "--- 상한 확인 (큰 값들, uint32 max까지) ---"
    for v in 5000 100000 10000000 1000000000 4294967295; do
        log "  timeout=${v}ms 시도..."
        if compile_and_run 1 1 0  1 1 $v  0 0 0  $SEARCH_IMAGES $RANGE_CSV; then
            log "  timeout=${v}ms → OK"
        else
            log "  timeout=${v}ms → FAIL (상한 초과로 추정)"
            break
        fi
    done

    log "====== TIMEOUT_MS 탐색 완료 ======"
}

# ============================================================
# 3. 결합 실험: threshold × timeout × batch × priority
#    모델: Det+Seg (1 1 0) 고정
# ============================================================
run_combined_experiment() {
    log "====== 결합 파라미터 실험 시작 ======"
    log "Batches   : ${COMBINED_BATCHES[*]}"
    log "Thresholds: ${COMBINED_THRESHOLDS[*]}"
    log "Timeouts  : ${COMBINED_TIMEOUTS[*]}"
    log "Priorities: ${COMBINED_PRIORITIES[*]}"

    local total=$(( ${#COMBINED_BATCHES[@]} * ${#COMBINED_THRESHOLDS[@]} * \
                    ${#COMBINED_TIMEOUTS[@]} * ${#COMBINED_PRIORITIES[@]} ))
    local count=0

    for batch in "${COMBINED_BATCHES[@]}"; do
        for thresh in "${COMBINED_THRESHOLDS[@]}"; do
            for timeout in "${COMBINED_TIMEOUTS[@]}"; do
                for pri in "${COMBINED_PRIORITIES[@]}"; do
                    count=$((count + 1))
                    log "[$count/$total] batch=$batch thresh=$thresh timeout=${timeout}ms priority=$pri"
                    if compile_and_run \
                        1 1 0 \
                        $batch $thresh $timeout \
                        $pri $pri 0 \
                        $COMBINED_IMAGES $COMBINED_CSV; then
                        log "  → 성공"
                    else
                        log "  → 실패 (결과 없음)"
                    fi
                done
            done
        done
    done

    log "====== 결합 실험 완료 ($count개) ======"
}

# ============================================================
# 4. 랜덤 파라미터 실험 (이진 탐색으로 찾은 범위 내에서 7회)
# ============================================================
run_random_experiments() {
    local RANDOM_CSV=$WORK_DIR/param_random_results.csv
    > $RANDOM_CSV

    log "====== 랜덤 파라미터 실험 7회 시작 ======"
    log "사용 범위 — threshold: [1, $THRESH_MAX], timeout_min: ${TIMEOUT_MIN}ms"

    # threshold 랜덤 상한: 너무 크면 실험 시간이 길어지므로 1000으로 제한
    local thresh_hi=$THRESH_MAX
    [ $thresh_hi -gt 1000 ] && thresh_hi=1000

    # timeout 랜덤 상한: 10000ms로 제한 (실험 시간 고려)
    local timeout_hi=10000
    [ $TIMEOUT_MIN -gt $timeout_hi ] && timeout_hi=$TIMEOUT_MIN

    local BATCH_OPTS=(1 2 4 8)

    for i in $(seq 1 7); do
        # 모델 조합: 최소 1개 이상 활성
        local use_d use_s use_p
        while true; do
            use_d=$((RANDOM % 2))
            use_s=$((RANDOM % 2))
            use_p=$((RANDOM % 2))
            [ $((use_d + use_s + use_p)) -gt 0 ] && break
        done

        # batch: {1, 2, 4, 8} 중 랜덤
        local batch=${BATCH_OPTS[$((RANDOM % 4))]}

        # threshold: [1, thresh_hi] 랜덤
        local thresh_range=$((thresh_hi - 1))
        local thresh=1
        [ $thresh_range -gt 0 ] && thresh=$((RANDOM % thresh_range + 1))

        # timeout: 50% 확률로 0(비활성), 나머지는 [TIMEOUT_MIN, timeout_hi] 랜덤
        local timeout=0
        if [ $((RANDOM % 2)) -eq 1 ]; then
            local t_range=$((timeout_hi - TIMEOUT_MIN + 1))
            timeout=$((RANDOM % t_range + TIMEOUT_MIN))
        fi

        # priority: 각 모델 [0, 31] 랜덤
        local pri_d=$((RANDOM % 32))
        local pri_s=$((RANDOM % 32))
        local pri_p=$((RANDOM % 32))

        log "[$i/7] use=(${use_d}D,${use_s}S,${use_p}P) batch=$batch thresh=$thresh timeout=${timeout}ms pri=(${pri_d},${pri_s},${pri_p})"
        if compile_and_run \
            $use_d $use_s $use_p \
            $batch $thresh $timeout \
            $pri_d $pri_s $pri_p \
            $COMBINED_IMAGES $RANDOM_CSV; then
            log "  → 성공"
        else
            log "  → 실패"
        fi
    done

    log "====== 랜덤 실험 완료 ======"
    log "결과: cat $RANDOM_CSV | column -t -s,"
}

# ============================================================
# 메인
# ============================================================
cd $WORK_DIR
mkdir -p traces

# CSV 초기화 (헤더는 save_csv가 자동 작성)
> $RANGE_CSV
> $COMBINED_CSV

log "=========================================="
log "파라미터 범위 탐색 + 결합 실험 시작"
log "Range CSV   : $RANGE_CSV"
log "Combined CSV: $COMBINED_CSV"
log "Range 이미지 수: $SEARCH_IMAGES"
log "Combined 이미지 수: $COMBINED_IMAGES"
log "=========================================="

find_threshold_range
find_timeout_range
run_combined_experiment
run_random_experiments

log "=========================================="
log "모든 실험 완료!"
log "결과 확인:"
log "  cat $RANGE_CSV | column -t -s,"
log "  cat $COMBINED_CSV | column -t -s,"
log "  cat $WORK_DIR/param_random_results.csv | column -t -s,"
log "=========================================="
