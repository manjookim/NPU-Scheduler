#!/bin/bash
# =============================================================================
# run_deeplab_pathA.sh  [2026-09-08]
# DeepLabV3+MNv2(Cityscapes 재학습) -> Hailo-8L HEF 컴파일 · 경로 A(hailomz)
#
# ─────────────────────────────────────────────────────────────────────────────
# 왜 이 스크립트인가 (2026-09-08 원인 재진단 결과)
# ─────────────────────────────────────────────────────────────────────────────
# 실패 원인은 할당기(allocator) 옵션이 아니라 **컴파일 경로**였다.
#
#   · 공식 HEF 는 513x513 bilinear + argmax 를 칩의 device pre/post layer(전용 HW)로
#     처리한다. 그 지시는 Model Zoo **yaml** 의 postprocessing.device_pre_post_layers
#     에 있고, .alls 에는 없다.
#   · 그런데 compile_deeplab_ladder.py 는 ClientRunner + .alls 만 쓴다.
#     ClientRunner 는 yaml 을 읽지 않는다  ->  지시가 전달된 적이 없다.
#   · 그래서 DFC 는 ONNX 의 Resize 를 평범한 NN core 레이어로 배치했고,
#     17x17x19 -> 513x513x19 (약 500만 원소) 를 클러스터 microcode 로 처리하려다
#     `microcode exceeded size for resize1_sd0` 로 죽었다.
#
#   근거 3가지:
#     (1) 공식 generic alls 는 3줄뿐인데도 컴파일된다 (allocator_param/defuse 전무).
#     (2) 공식 HEF = 3 contexts, 출력 argmax1 UINT8 NHW(513x513).
#         NN core 로 resize 를 돌렸다면 3줄 alls 로 절대 불가능한 결과다.
#     (3) 우리 HN 은 resize1 이 engine 배정을 못 받은 일반 레이어로 남아 있었다.
#         (tools/dfc/deeplab_tail_report.txt)
#     보조 근거: 우리 모델은 19클래스로 공식(21클래스)보다 **작다**.
#               "커서 안 들어간다"는 설명 자체가 성립하지 않는다.
#
#   부수 원인(악화 요인): export_deeplab.py 의
#     F.interpolate(..., align_corners=False)  ->  ONNX coordinate_transformation_mode
#     = half_pixel  ->  실효 배율 513/17 = 30.176..., gcd(17,513)=1 이라 보간 가중치
#     패턴이 513칸 내내 한 번도 반복되지 않는다. 공식(TF, align_corners=True)은
#     (513-1)/(17-1) = 정확히 32배라 32칸 주기로 반복된다.
#     -> automatic_resize_reshapes 류 최적화가 무력했던 이유이기도 하다.
#     경로 A 는 logits ONNX(Resize 노드 0개)를 쓰므로 이 문제도 자동으로 사라진다.
#
# ─────────────────────────────────────────────────────────────────────────────
# 이 스크립트가 하는 일
# ─────────────────────────────────────────────────────────────────────────────
#   0단계  환경 점검 (venv / DFC / hailomz / MZ 버전 / 캘리브 이미지)
#   1단계  Model Zoo cfg 설치 (yaml + alls). base yaml 존재 확인 후 자동 폴백
#   2단계  hailomz parse                                          (약 3분)
#   3단계  ★게이트★ 파싱된 HAR 의 꼬리가 device layer 로 접혔는지 판정
#          -> 실패면 여기서 즉시 중단한다. optimize/compile 로 시간 낭비 금지.
#   4단계  hailomz optimize (양자화)                              (약 20~40분)
#   5단계  hailomz compile                                        (약 10~40분)
#   6단계  HEF 검증 — context 수 / 출력 형식이 공식과 같은지
#
# ★ 3단계가 이 스크립트 전체의 핵심이다. 예전에는 이 판정 없이 바로 컴파일에
#   들어가서 5시간 44분짜리 실패를 반복했다. 이제는 3분 만에 성패를 안다.
#
# ─────────────────────────────────────────────────────────────────────────────
# 실행 위치: WSL2 (x86). DFC 는 x86 전용이라 RPi 에서는 못 돌린다.
# 사용법:
#   source ~/hailo_venv/bin/activate
#   cd /mnt/c/Users/sset0/jungmin-claude/StudentExperiment/NPUscheduler/hailo_8L/experiments/2026-08-31_mz3_retrain/src
#   chmod +x run_deeplab_pathA.sh
#   ./run_deeplab_pathA.sh                 # 전체 실행
#   ./run_deeplab_pathA.sh --stop-at-gate  # 0~3단계만 (약 4분, 성패 판정만)
#   ./run_deeplab_pathA.sh --from 4        # 이미 parse 가 끝났으면 4단계부터
# =============================================================================
set -u

# ───────────────────────── 설정 (필요하면 여기만 수정) ─────────────────────────
REPO="${REPO:-/mnt/c/Users/sset0/jungmin-claude/StudentExperiment/NPUscheduler}"
EXP="$REPO/hailo_8L/experiments/2026-08-31_mz3_retrain"
CALIB="${CALIB:-$REPO/calib513/calib513}"
MZ="${MZ:-$HOME/hailo_model_zoo}"

NET="deeplab_v3_mnv2_wo_dilation_cityscapes"
ARCH="hailo8l"
# 경로 A 는 반드시 **logits ONNX** 를 쓴다. Resize/ArgMax 는 그래프가 아니라
# device_pre_post_layers 가 붙여준다 — 공식이 하는 방식이 정확히 이것이다.
ONNX="${ONNX:-$EXP/artifacts/deeplab_cs_logits.onnx}"
START_NODE="/backbone/backbone.0/backbone.0.0/Conv"
END_NODE="/classifier/Conv"

# 공식 조건과 일치시킨다. 512 는 우리가 임의로 올렸던 값이고, 공식 alls 는 64 다.
# (optimize 시간도 8배 줄어든다.)
CALIBSET_SIZE="${CALIBSET_SIZE:-64}"

WORK="$EXP/artifacts/pathA"
LOG="$WORK/pathA_$(date +%Y%m%d_%H%M%S).log"

STOP_AT_GATE=0
FROM_STEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --stop-at-gate) STOP_AT_GATE=1; shift ;;
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "알 수 없는 인자: $1"; exit 1 ;;
    esac
done

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

hr() { printf '=%.0s' $(seq 1 78); echo; }
step() { hr; echo "== $*"; hr; }
die() { echo; echo "[중단] $*"; echo "[로그] $LOG"; exit 1; }

echo "로그: $LOG"
echo "작업 폴더: $WORK"
echo

# ───────────────────────────── 0단계 · 환경 점검 ─────────────────────────────
if [ "$FROM_STEP" -le 0 ]; then
step "0단계 · 환경 점검"

python -c "import hailo_sdk_client" 2>/dev/null \
    || die "hailo_venv 가 활성화되지 않았습니다.  source ~/hailo_venv/bin/activate  후 다시 실행하세요."
echo "[OK] DFC(hailo_sdk_client) 임포트 성공"
hailo --version 2>/dev/null | head -3 || echo "  (hailo --version 사용 불가 — 무시 가능)"

command -v hailomz >/dev/null 2>&1 \
    || die "hailomz 를 찾을 수 없습니다. Model Zoo(v2.19.0 이하)를 설치하세요:
       cd $MZ && pip install -e ."
echo "[OK] hailomz 발견: $(command -v hailomz)"
hailomz --version 2>/dev/null | head -3 || true

[ -d "$MZ/hailo_model_zoo/cfg/networks" ] \
    || die "Model Zoo cfg 폴더가 없습니다: $MZ/hailo_model_zoo/cfg/networks
       MZ 환경변수로 경로를 지정하세요:  MZ=/path/to/hailo_model_zoo ./run_deeplab_pathA.sh"
echo "[OK] Model Zoo: $MZ"

[ -f "$ONNX" ] || die "ONNX 가 없습니다: $ONNX"
echo "[OK] ONNX: $ONNX ($(du -h "$ONNX" | cut -f1))"

# 경로 A 는 Resize 가 그래프에 있으면 안 된다. 실수 방지용 확인.
if grep -qa "half_pixel\|align_corners" "$ONNX" 2>/dev/null; then
    echo
    echo "[경고] 이 ONNX 안에 Resize 좌표변환 모드 문자열이 보입니다."
    echo "       경로 A 는 Resize 가 **없는** logits ONNX 를 써야 합니다."
    echo "       deeplab_cs_resize_argmax.onnx 를 지정하신 것은 아닌지 확인하세요."
    echo "       (그대로 진행하면 예전과 같은 microcode 초과가 재현됩니다.)"
    read -r -p "       그래도 계속하려면 Enter, 중단하려면 Ctrl+C: " _
fi

N_CALIB=$(find "$CALIB" -maxdepth 2 -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)
[ "$N_CALIB" -gt 0 ] || die "캘리브 이미지가 없습니다: $CALIB"
echo "[OK] 캘리브 이미지 ${N_CALIB}장 (사용할 개수: $CALIBSET_SIZE)"
echo
fi

# ─────────────────────── 1단계 · Model Zoo cfg 설치 ───────────────────────
if [ "$FROM_STEP" -le 1 ]; then
step "1단계 · Model Zoo cfg 설치 (yaml + alls)"

SRC_YAML="$EXP/mz_cfg/${NET}.yaml"
[ -f "$SRC_YAML" ] || die "yaml 이 없습니다: $SRC_YAML"

DST_YAML="$MZ/hailo_model_zoo/cfg/networks/${NET}.yaml"
DST_ALLS="$MZ/hailo_model_zoo/cfg/alls/generic/${NET}.alls"
mkdir -p "$(dirname "$DST_ALLS")"

# yaml 의 base 가 실제로 존재하는지 확인하고, 없으면 폴백한다.
BASE_REF=$(grep -A2 '^base:' "$SRC_YAML" | grep -o 'base/[a-z_]*\.yaml' | head -1)
echo "yaml 이 참조하는 base: ${BASE_REF:-(없음)}"
if [ -n "$BASE_REF" ] && [ ! -f "$MZ/hailo_model_zoo/cfg/$BASE_REF" ]; then
    echo "[!] $BASE_REF 가 Model Zoo 에 없습니다. 사용 가능한 base 목록:"
    ls "$MZ/hailo_model_zoo/cfg/base/" 2>/dev/null | head -20
    for CAND in cityscapes.yaml pascal.yaml segmentation.yaml; do
        if [ -f "$MZ/hailo_model_zoo/cfg/base/$CAND" ]; then
            echo "    -> base/$CAND 로 자동 대체합니다 (classes: 19 는 그대로 유지)."
            sed "s|base/[a-z_]*\.yaml|base/$CAND|" "$SRC_YAML" > "$DST_YAML"
            BASE_REF="base/$CAND"
            break
        fi
    done
    [ -f "$DST_YAML" ] || die "쓸 수 있는 base yaml 을 찾지 못했습니다. $SRC_YAML 의 base: 항목을 직접 고치세요."
else
    cp "$SRC_YAML" "$DST_YAML"
fi
echo "[OK] yaml 설치: $DST_YAML"

# alls 는 공식과 동일하게 3줄로 되돌린다.
# 할당기 옵션(allocator_param/defuse/performance_param)은 **넣지 않는다** —
# 공식이 3줄로 되는 것을 우리가 3줄로 못 할 이유가 없고, 옵션을 넣으면
# "공식과 동일 조건" 이라는 실험 전제가 깨진다.
cat > "$DST_ALLS" <<ALLS
normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
model_optimization_config(calibration, batch_size=1, calibset_size=${CALIBSET_SIZE})
pre_quantization_optimization(equalization, policy=disabled)
ALLS
echo "[OK] alls 설치 (공식과 동일한 3줄): $DST_ALLS"
cat "$DST_ALLS" | sed 's/^/    /'

echo
echo "device_pre_post_layers 설정 확인:"
grep -n "device_pre_post_layers" "$DST_YAML" | sed 's/^/    /' \
    || die "yaml 에 device_pre_post_layers 가 없습니다. 이게 없으면 이번에도 실패합니다."
echo
fi

# ───────────────────────────── 2단계 · parse ─────────────────────────────
PARSED_HAR="$WORK/${NET}.har"
if [ "$FROM_STEP" -le 2 ]; then
step "2단계 · hailomz parse (약 3분)"
cd "$WORK" || die "작업 폴더 진입 실패: $WORK"

echo "실행: hailomz parse $NET --hw-arch $ARCH --ckpt $ONNX"
hailomz parse "$NET" --hw-arch "$ARCH" --ckpt "$ONNX"
RC=$?
if [ $RC -ne 0 ]; then
    echo
    echo "[!] 기본 parse 실패 (rc=$RC). start/end 노드를 명시해서 재시도합니다."
    hailomz parse "$NET" --hw-arch "$ARCH" --ckpt "$ONNX" \
        --start-node-names "$START_NODE" --end-node-names "$END_NODE" \
        || die "parse 실패. 위 에러 메시지를 확인하세요.
       흔한 원인: (a) yaml 의 parser.nodes 이름이 ONNX 실제 노드명과 불일치
                  (b) Model Zoo 버전이 v2.19.0 초과 (master 에는 hailo8l 지원이 없음)"
fi

# hailomz 가 어디에 HAR 을 떨궜는지 확인
FOUND=$(find "$WORK" -maxdepth 1 -name "*.har" -newermt "-10 minutes" 2>/dev/null | head -1)
[ -n "$FOUND" ] || FOUND=$(find "$WORK" -maxdepth 1 -name "*.har" 2>/dev/null | head -1)
[ -n "$FOUND" ] || die "parse 는 끝났는데 .har 를 찾지 못했습니다. $WORK 를 확인하세요."
PARSED_HAR="$FOUND"
echo "[OK] parsed HAR: $PARSED_HAR"
echo
fi

# ──────────────────────── 3단계 · ★게이트★ 꼬리 판정 ────────────────────────
if [ "$FROM_STEP" -le 3 ]; then
step "3단계 · ★게이트★ — 꼬리(bilinear+argmax)가 device layer 로 접혔는가"

[ -f "$PARSED_HAR" ] || PARSED_HAR=$(find "$WORK" -maxdepth 1 -name "*.har" | head -1)
[ -f "$PARSED_HAR" ] || die "판정할 HAR 이 없습니다."

python - "$PARSED_HAR" <<'PYGATE'
import sys
from hailo_sdk_client import ClientRunner

har = sys.argv[1]
r = ClientRunner(har=har)
layers = r.get_hn()["layers"]

outs, resizes, argmaxes = [], [], []
for name, v in layers.items():
    t = v.get("type")
    if t == "output_layer":
        outs.append((name, v.get("output_shapes"), v.get("engine")))
    elif t == "resize":
        resizes.append((name, v.get("output_shapes"), v.get("engine")))
    elif t == "argmax":
        argmaxes.append((name, v.get("output_shapes"), v.get("engine")))

print(f"  총 레이어 수 : {len(layers)}")
print(f"  resize 레이어: {[n for n, _, _ in resizes] or '없음'}")
print(f"  argmax 레이어: {[n for n, _, _ in argmaxes] or '없음'}")
for n, s, e in resizes + argmaxes:
    print(f"     {n:36s} out={s}  engine={e}")
print("  출력 레이어  :")
for n, s, e in outs:
    print(f"     {n:36s} out={s}  engine={e}")
print()

def tail_dims(shapes):
    """[[-1,H,W,C]] -> (H, W, C)"""
    try:
        s = shapes[0]
        return tuple(s[-3:])
    except Exception:
        return None

dims = tail_dims(outs[0][1]) if outs else None
print(f"  최종 출력 형태: {dims}")
print()

if dims == (513, 513, 1):
    print("  ★ 판정: PASS — 꼬리가 칩(device layer)으로 접혔습니다.")
    print("     공식 HEF 와 동일한 구조입니다. 4단계(optimize)로 진행하세요.")
    sys.exit(0)
elif dims == (17, 17, 19):
    print("  ★ 판정: FAIL — 꼬리가 안 붙었습니다 (bilinear/argmax 없음).")
    print("     device_pre_post_layers 가 적용되지 않은 것입니다.")
    print("     확인할 것:")
    print("       1) yaml 이 실제로 Model Zoo cfg/networks/ 에 복사됐는가")
    print("       2) yaml 의 postprocessing.device_pre_post_layers 에")
    print("          bilinear: true, argmax: true 가 있는가")
    print("       3) Model Zoo 버전이 v2.19.0 이하인가 (master 는 hailo8l 미지원)")
    print("     이 상태로 컴파일하면 17x17x19 출력이 나와 호스트 후처리가 생기고,")
    print("     공식 HEF 와 비교 대상이 되지 못합니다.")
    sys.exit(2)
elif dims == (513, 513, 19):
    print("  ★ 판정: FAIL — resize 가 NN core 일반 레이어로 남았습니다.")
    print("     이대로 컴파일하면 'microcode exceeded size for resize1_sd0' 가")
    print("     그대로 재현됩니다. (예전 5번의 실패와 동일한 상태)")
    print("     원인: logits ONNX 가 아니라 resize_argmax ONNX 를 쓴 것일 가능성이 큽니다.")
    sys.exit(2)
else:
    print("  ★ 판정: 알 수 없음 — 예상 밖의 출력 형태입니다.")
    print("     위 레이어 목록을 그대로 공유해 주세요.")
    sys.exit(3)
PYGATE

GATE_RC=$?
echo
if [ $GATE_RC -ne 0 ]; then
    echo "──────────────────────────────────────────────────────────────"
    echo "게이트 실패. **여기서 멈춥니다.**"
    echo "이 상태로 optimize/compile 을 돌리면 예전처럼 몇 시간을 버리게 됩니다."
    echo "위 안내를 따라 원인을 고친 뒤  ./run_deeplab_pathA.sh --from 1  로 재시도하세요."
    echo "[로그] $LOG"
    echo "──────────────────────────────────────────────────────────────"
    exit $GATE_RC
fi
echo "게이트 통과. 계속 진행합니다."
echo

if [ "$STOP_AT_GATE" -eq 1 ]; then
    echo "--stop-at-gate 지정 — 여기까지만 실행합니다."
    echo "이어서 돌리려면:  ./run_deeplab_pathA.sh --from 4"
    exit 0
fi
fi

# ────────────────────────── 4단계 · optimize(양자화) ──────────────────────────
QUANT_HAR="$WORK/${NET}_optimized.har"
if [ "$FROM_STEP" -le 4 ]; then
step "4단계 · hailomz optimize / 양자화 (약 20~40분)"
cd "$WORK" || die "작업 폴더 진입 실패"

[ -f "$PARSED_HAR" ] || PARSED_HAR=$(find "$WORK" -maxdepth 1 -name "${NET}.har" | head -1)
echo "실행: hailomz optimize $NET --har $PARSED_HAR --calib-path $CALIB"
hailomz optimize "$NET" --har "$PARSED_HAR" --calib-path "$CALIB" \
    || die "optimize 실패. 캘리브 경로/이미지 크기(513x513)를 확인하세요."

FOUND=$(find "$WORK" -maxdepth 1 -name "*optimized*.har" -o -maxdepth 1 -name "*quantized*.har" 2>/dev/null | head -1)
[ -n "$FOUND" ] && QUANT_HAR="$FOUND"
echo "[OK] quantized HAR: $QUANT_HAR"
echo
fi

# ───────────────────────────── 5단계 · compile ─────────────────────────────
HEF="$WORK/${NET}.hef"
if [ "$FROM_STEP" -le 5 ]; then
step "5단계 · hailomz compile (약 10~40분)"
cd "$WORK" || die "작업 폴더 진입 실패"

[ -f "$QUANT_HAR" ] || QUANT_HAR=$(find "$WORK" -maxdepth 1 -name "*optimized*.har" | head -1)
echo "실행: hailomz compile $NET --har $QUANT_HAR --hw-arch $ARCH"
hailomz compile "$NET" --har "$QUANT_HAR" --hw-arch "$ARCH" \
    || die "compile 실패.
       에러가 여전히 'microcode exceeded ... resize1' 계열이면 3단계 게이트를 다시 확인하세요
       (꼬리가 실제로는 안 접혔을 수 있습니다)."

FOUND=$(find "$WORK" -maxdepth 1 -name "*.hef" 2>/dev/null | head -1)
[ -n "$FOUND" ] && HEF="$FOUND"
[ -f "$HEF" ] || die "compile 은 끝났는데 .hef 를 찾지 못했습니다."
cp "$HEF" "$EXP/artifacts/${NET}.hef" 2>/dev/null && echo "[OK] artifacts 에도 복사: $EXP/artifacts/${NET}.hef"
echo "[OK] HEF: $HEF ($(du -h "$HEF" | cut -f1))"
echo
fi

# ──────────────────────────── 6단계 · HEF 검증 ────────────────────────────
step "6단계 · HEF 검증 — 공식과 같은 구조인가"

[ -f "$HEF" ] || HEF=$(find "$WORK" -maxdepth 1 -name "*.hef" | head -1)

echo "공식 HEF(사전학습판) 기준값:"
echo "    Number of contexts : 3"
echo "    Output             : argmax1  UINT8, NHW(513x513)"
echo
if command -v hailortcli >/dev/null 2>&1; then
    echo "우리 HEF:"
    hailortcli parse-hef "$HEF" | grep -iE "context|output|architecture|network group" | sed 's/^/    /'
    echo
    echo "위 두 값이 일치하면 사전학습 베이스라인(42런)과 나란히 비교할 수 있습니다."
    echo "context 수가 다르면 스케줄링 거동이 달라지므로 보고서에 반드시 명시하세요."
else
    echo "[참고] 이 환경에 hailortcli 가 없습니다. HEF 를 보드로 옮겨서 확인하세요:"
    echo "    scp -P 40021 $HEF npu-rpi1@155.230.16.157:~/hailo-rpi5-examples/resources"
    echo "    ssh npu-rpi1@155.230.16.157 -p 40021 \\"
    echo "        'hailortcli parse-hef ~/hailo-rpi5-examples/resources/${NET}.hef'"
fi

echo
hr
echo "완료. [로그] $LOG"
echo
echo "다음 할 일:"
echo "  1) HEF 를 보드로 전송 (위 scp 명령)"
echo "  2) mz3 3모델 벤치마크에 재학습판 deeplab 을 넣고 측정"
echo "  3) mIoU 재측정 — 칩은 align_corners=True 계열로 업샘플하므로,"
echo "     export_deeplab.py::predict_labels 의 align_corners 를 True 로 바꿔"
echo "     --eval 을 한 번 더 돌려 배포 기준 mIoU 를 확정할 것"
hr
