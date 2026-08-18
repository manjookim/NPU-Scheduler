#!/bin/bash
# ------------------------------------------------------------------------
# Detection=YOLOv5(CPU측 후처리, engine=cpu) 고정, Segmentation/Pose(둘 다 YOLOv8,
# CPU 후처리)는 조합별로 켜고 끄면서 4가지 워크로드를 비교:
#   det / det+seg / det+pose / det+seg+pose   (파라미터는 전부 사용자 지정 기본값:
#   batch=1, threshold=1, timeout=0ms, priority=0, 세 모델 동일)
#
# run_v5det_seg_pose.sh(Detection NPU 후처리 버전)와 완전히 같은 조건·같은 4조합으로
# 돌아가는 짝 스크립트 - 같은 조합끼리 두 CSV를 나란히 비교하면 "Detection 후처리를
# NPU에서 vs CPU에서" 차이만 보는 통제 실험이 된다.
#
# infer_scheduler_hailo8_v5det_cpu.cpp의 #define USE_SEG / USE_POSE 두 줄을 조합마다
# sed로 토글 → 재빌드 → 실행. 결과는 전부 같은 CSV 하나에 누적되고(csv_writer.hpp가
# use_det/use_seg/use_pose 컬럼으로 조합을 구분해서 씀), 조합별로 몇 개나 모였는지는
# 그 컬럼 기준으로 집계한다.
#
# 실행 위치: 보드(rpi4, ~/hailo_cpp_test/), infer_scheduler_hailo8_v5det_cpu.cpp와 같은 디렉토리.
# 사용법: chmod +x run_v5det_cpu_seg_pose.sh && ./run_v5det_cpu_seg_pose.sh
#
# [2026-08-07 추가 #2] HRTT 트레이스(.hrtt) + NPU 사용률 로그를 이번엔 같이 모은다.
# run_v5det_seg_pose.sh(NPU판)와 완전히 동일한 모니터링 로직 — 상세 설명은 그쪽 헤더 참고.
# 결과 폴더는 기존 exp1(3/3 다 채워진 상태)과 겹치지 않게 "_hrtt" 접미사로 새로 분리.
#
# [2026-08-08 버그 수정 — run_v5det_seg_pose.sh와 동일, 상세 설명은 그쪽 헤더 참고]
#   1) HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP 90 -> 30 (다른 모든 스크립트와 통일,
#      90은 90초보다 짧게 끝나는 조합에서 .hrtt가 아예 안 만들어지고 fps도 저하시킴).
#   2) count_existing()에 count_traces() 추가 — CSV 행 수뿐 아니라 .hrtt 개수도
#      REPEATS에 도달해야 그 조합을 "완료"로 침.
#   3) 실행 끝에 fill_hrtt_columns_v5det.py 자동 호출.
# ------------------------------------------------------------------------
set -e
cd "$(dirname "$0")/.."   # hailo_8/ 기준

SRC=infer_scheduler_hailo8_v5det_cpu.cpp
BIN=infer_scheduler_hailo8_v5det_cpu
# [2026-08-07 추가] rpi4 보드에서 hailort.hpp가 표준 경로(/usr/include)가 아니라 이 경로에
# 있는 게 실기에서 확인됨 -I 없이 빌드하면 첫 include부터 못 찾아서 대량 연쇄 에러가 남.
# .so 자체는 /usr/lib/libhailort.so(ldconfig 캐시에 등록됨)라 -L은 필요 없음.
HAILORT_INCLUDE="/home/rpi4/hm/monitor/hailort/hailort/libhailort/include"
HEF_DIR="${HOME}/hailo_cpp_test/resources"
HEF_NAME=yolov5xs_wo_spp.hef
HEF_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/${HEF_NAME}"
# [2026-08-07 #2] HRTT 재수집을 위해 기존 exp1과 분리된 새 폴더(처음부터 3회씩 다시 채움).
OUTDIR="experiments/2026-08-07_v5det_cpu_seg_pose_hrtt_exp1"
CSV="${OUTDIR}/csv/results_v5det_cpu_seg_pose.csv"
TRACES_DIR="${OUTDIR}/traces"
REPEATS=3   # 조합마다 3회 반복

ENGINE_TAG="cpu"

if [ ! -f "$SRC" ]; then
    echo "[오류] $SRC 를 찾을 수 없음. hailo_8/ 디렉토리에서 실행할 것."
    exit 1
fi
for req in "${HEF_DIR}/yolov8s_seg.hef" "${HEF_DIR}/yolov8s_pose.hef"; do
    if [ ! -f "$req" ]; then
        echo "[오류] $req 없음 — Segmentation/Pose HEF는 기존 실험에서 이미 받아뒀어야 함."
        exit 1
    fi
done

mkdir -p "$HEF_DIR" "${OUTDIR}/csv" "$TRACES_DIR"

if [ ! -f hailo_utilization_hailo8.py ]; then
    echo "[경고] hailo_utilization_hailo8.py 를 찾을 수 없음 — NPU 사용률 로그(npu_percent)는 안 채워짐."
    echo "       (HRTT .hrtt 파일 수집 자체는 이 파일과 무관하게 계속 진행됨.)"
fi

if [ ! -f "${HEF_DIR}/${HEF_NAME}" ]; then
    echo "[HEF 없음] ${HEF_DIR}/${HEF_NAME} 를 공식 hailo_model_zoo S3에서 받는다..."
    wget -q --show-progress -O "${HEF_DIR}/${HEF_NAME}" "$HEF_URL"
fi

echo "[HEF 확인] hailortcli parse-hef 결과:"
hailortcli parse-hef "${HEF_DIR}/${HEF_NAME}" | head -20
echo ""
read -p "Architecture=HAILO8 확인되면 Enter, 아니면 Ctrl+C: " _

# CSV에서 use_seg=$1, use_pose=$2 인 행이 몇 개 있는지 센다(헤더 제외).
count_existing() {
    local seg=$1 pose=$2
    if [ ! -f "$CSV" ]; then echo 0; return; fi
    awk -F, -v seg="$seg" -v pose="$pose" 'NR>1 && $3==seg && $4==pose {c++} END{print c+0}' "$CSV"
}

# [2026-08-08 추가] 해당 조합의 .hrtt 트레이스가 몇 개 있는지 센다(run_v5det_seg_pose.sh 참고).
count_traces() {
    local combo_name=$1
    ls "$TRACES_DIR"/*"_${ENGINE_TAG}_${combo_name}_run"*.hrtt 2>/dev/null | wc -l | tr -d ' '
}

# CSV에 있는 run_id 중 최댓값+1 (없으면 1).
next_run_id() {
    if [ ! -f "$CSV" ]; then echo 1; return; fi
    local maxid
    maxid=$(awk -F, 'NR>1 {print $1}' "$CSV" | sort -n | tail -1)
    [ -z "$maxid" ] && maxid=0
    echo $((maxid + 1))
}

# [2026-08-07 추가] rpi4 보드가 간헐적으로 크래시함(hailo_pci 드라이버 레이스 컨디션 또는
# 프로세스 종료 시점 세그폴트 — QUESTION_FOR_TA.md 참고). 두 경우 다 CSV에 결과가 먼저
# 저장된 "뒤"에 죽는 패턴이라, "CSV에 새 줄이 실제로 늘었는지"로 성공 여부를 판단하고
# set -e로 스크립트 전체가 멈추지 않도록 재시도 가능하게 만든다.
run_combo() {
    local combo_name=$1 use_seg=$2 use_pose=$3
    local csv_ok trace_ok success attempt max_attempts run_id
    csv_ok=$(count_existing "$use_seg" "$use_pose")
    trace_ok=$(count_traces "$combo_name")
    success=$csv_ok
    [ "$trace_ok" -lt "$success" ] && success=$trace_ok
    echo ""
    echo "=== 조합: $combo_name (USE_SEG=$use_seg, USE_POSE=$use_pose) — CSV ${csv_ok}개 / 트레이스 ${trace_ok}개 → 유효 ${success}/${REPEATS} ==="
    if [ "$success" -ge "$REPEATS" ]; then
        echo "[스킵] 이미 목표(${REPEATS}, CSV+HRTT 둘 다)를 채움."
        return
    fi

    sed -i "s/^#define USE_SEG.*/#define USE_SEG    ${use_seg}/" "$SRC"
    sed -i "s/^#define USE_POSE.*/#define USE_POSE   ${use_pose}/" "$SRC"
    echo "빌드 중... ($combo_name)"
    g++ "$SRC" -o "$BIN" -I"$HAILORT_INCLUDE" -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17

    attempt=0
    max_attempts=$((REPEATS * 4))
    while [ "$success" -lt "$REPEATS" ] && [ "$attempt" -lt "$max_attempts" ]; do
        attempt=$((attempt + 1))
        run_id=$(next_run_id)
        echo ""
        echo "--- [$combo_name] 시도 $attempt (run_id=$run_id, 목표 $((success + 1))/$REPEATS) ---"
        local before after

        # --- HRTT/NPU 모니터링 준비 (auto_experiment_batch1_pri0_fps30.sh와 동일 패턴) ---
        > npu_log.txt
        rm -f /tmp/hmon_files/* 2>/dev/null
        rm -f "$TRACES_DIR"/hailort_*.hrtt
        NPU_PID=""
        if [ -f hailo_utilization_hailo8.py ]; then
            ( source ~/hailo_platform_venv/bin/activate 2>/dev/null; python3 hailo_utilization_hailo8.py ) > /tmp/hailo_util_mon.log 2>&1 &
            NPU_PID=$!
            sleep 2
        fi
        export HAILO_TRACE=scheduler
        # [2026-08-08 재조정] run_v5det_seg_pose.sh와 동일한 이유로 30 -> 3 (상세 설명은
        # 그쪽 헤더 참고). 실기에서 CPU 조건 det 단독이 15번 시도해도 트레이스 0/3이었음.
        export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=3
        export HAILO_TRACE_PATH="$TRACES_DIR"
        export HAILO_MONITOR=1

        before=0
        [ -f "$CSV" ] && before=$(wc -l < "$CSV")
        ./"$BIN" "$run_id" "$CSV" || echo "[경고] run_id=$run_id 비정상 종료(세그폴트/드라이버 크래시 가능) — 결과 저장 여부 아래서 확인."
        after=0
        [ -f "$CSV" ] && after=$(wc -l < "$CSV")

        if [ -n "$NPU_PID" ]; then
            kill "$NPU_PID" 2>/dev/null || true
        fi
        sleep 1
        rm -f /tmp/hmon_files/* 2>/dev/null || true

        if [ "$after" -gt "$before" ]; then
            echo "[확인] run_id=$run_id 결과 CSV에 저장됨."
            csv_ok=$((csv_ok + 1))

            # npu_log.txt의 "NPU: X%" 평균을 CSV 마지막 행의 npu_percent 컬럼에 채움.
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
    print("  [npu] 활성 구간 없음")
else:
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

            # 새로 생긴 .hrtt를 조합/엔진/run_id가 드러나는 이름으로 rename.
            LATEST_HRTT=""
            for i in $(seq 1 35); do
                LATEST_HRTT=$(ls -t "$TRACES_DIR"/hailort_*.hrtt 2>/dev/null | head -1)
                [ -n "$LATEST_HRTT" ] && break
                sleep 1
            done
            if [ -n "$LATEST_HRTT" ]; then
                TS=$(basename "$LATEST_HRTT" .hrtt | sed 's/hailort_//')
                NEW="${TRACES_DIR}/1D-${use_seg}S-${use_pose}P_0PD-0PS-0PP_${ENGINE_TAG}_${combo_name}_run${run_id}_${TS}.hrtt"
                mv "$LATEST_HRTT" "$NEW"
                echo "  HRTT: $(basename "$NEW")"
                trace_ok=$((trace_ok + 1))
            else
                echo "  [!] HRTT 미생성(run_id=$run_id) — CSV엔 저장됐지만 트레이스가 없어서 이 조합은 재시도 계속됨."
            fi

            success=$csv_ok
            [ "$trace_ok" -lt "$success" ] && success=$trace_ok
        else
            echo "[재시도] run_id=$run_id 결과가 저장 안 됨(초반에 죽었을 가능성) — 다시 시도."
            rm -f "$TRACES_DIR"/hailort_*.hrtt   # 실패한 시도의 트레이스는 정리
            sleep 2
        fi
        sleep 1
    done

    if [ "$success" -lt "$REPEATS" ]; then
        echo "[경고][$combo_name] ${max_attempts}회 시도했지만 목표(${REPEATS})를 못 채움(CSV ${csv_ok}개 / 트레이스 ${trace_ok}개)."
        echo "       dmesg / hailortcli fw-control identify로 디바이스 상태 확인 필요."
    else
        echo "[완료][$combo_name] CSV ${csv_ok}개 / 트레이스 ${trace_ok}개 — 목표(${REPEATS}) 달성."
    fi
}

run_combo "det"          0 0
run_combo "det_seg"      1 0
run_combo "det_pose"     0 1
run_combo "det_seg_pose" 1 1

# 원본 파일을 다음 실행에서도 그대로 쓸 수 있도록 마지막엔 3조합(전체 활성) 상태로 복원.
sed -i "s/^#define USE_SEG.*/#define USE_SEG    1/" "$SRC"
sed -i "s/^#define USE_POSE.*/#define USE_POSE   1/" "$SRC"

echo ""
echo "========== 전체 완료 (CPU 후처리, Detection=YOLOv5 engine=cpu) =========="
for combo in "det:0:0" "det_seg:1:0" "det_pose:0:1" "det_seg_pose:1:1"; do
    name="${combo%%:*}"; rest="${combo#*:}"; seg="${rest%%:*}"; pose="${rest#*:}"
    n=$(count_existing "$seg" "$pose")
    t=$(count_traces "$name")
    echo "  $name: CSV ${n}/${REPEATS}, 트레이스 ${t}/${REPEATS}"
done
echo "결과 CSV: $CSV"
echo "HRTT 트레이스: $TRACES_DIR"
echo "[참고] run_v5det_seg_pose.sh(Detection NPU 후처리) 결과와 같은 조합끼리"
echo "       det_latency_ms / postprocess_ms_det / total_time_ms_det / cpu_percent 컬럼 대조."

# [2026-08-08 추가] 방금 모인 .hrtt를 바로 CSV에 반영. venv 우선 시도(protobuf 버전
# 이슈 대응, run_v5det_seg_pose.sh와 동일 — 상세 설명은 그쪽 참고).
echo ""
echo "[HRTT 컬럼 채우기] fill_hrtt_columns_v5det.py 실행 중..."
FILL_OK=0
if [ -f ~/hailo_platform_venv/bin/activate ]; then
    if ( source ~/hailo_platform_venv/bin/activate 2>/dev/null; python3 scripts/fill_hrtt_columns_v5det.py "$OUTDIR" "$(basename "$CSV")" ); then
        FILL_OK=1
    fi
fi
if [ "$FILL_OK" -eq 0 ]; then
    python3 scripts/fill_hrtt_columns_v5det.py "$OUTDIR" "$(basename "$CSV")" && FILL_OK=1
fi
if [ "$FILL_OK" -eq 1 ]; then
    echo "  완료."
else
    echo "  [경고] 실패(venv/시스템 python3 둘 다) — protobuf 버전 문제일 가능성 높음:"
    echo "    pip3 install --upgrade protobuf --break-system-packages"
    echo "    수동 재실행: python3 scripts/fill_hrtt_columns_v5det.py $OUTDIR $(basename "$CSV")"
fi
