#!/bin/bash
# =============================================================================
# run_deeplab_fix.sh  [2026-09-09]
# deeplab Cityscapes 재학습판 -> Hailo-8L HEF.  **확정된 원인에 대한 수정판.**
#
# ─────────────────────────────────────────────────────────────────────────────
# 확정된 원인 (공식 HEF 와 우리 HN 을 직접 뜯어 나란히 비교한 결과)
# ─────────────────────────────────────────────────────────────────────────────
#   공식  .../resize2 : input 17x17x21 -> output 513x513x21
#         params = {method:'bilinear', resize_bilinear_pixels_mode:'align_corners',
#                   resize_h_ratio_list:[15.0, 2.011764705882353]}   <- **2단 분해됨**
#
#   우리  .../resize1 : input 17x17x19 -> output 513x513x19
#         params = {method:'bilinear', resize_bilinear_pixels_mode:'half_pixels',
#                   resize_h_ratio_list:[30.176470588235293]}        <- **단일 단계**
#
#   align_corners 는 src = dst*(17-1)/(513-1) = dst/32 라, 17->255->513 으로 쪼개도
#   합성이 정확히 원래 매핑과 같다 (i*127/256 * 16/254 = i/32). 그래서 DFC 가 큰
#   업샘플을 두 단계로 분해할 수 있고, 각 단계 microcode 가 작아 3줄 alls 로도 컴파일된다.
#
#   half_pixel 은 src = (dst+0.5)*17/513 - 0.5 로 ±0.5 오프셋 때문에 단계 합성이
#   성립하지 않는다. DFC 가 분해를 못 하고 x30.176 을 한 방에 처리해야 하는데,
#   gcd(17,513)=1 이라 보간 가중치가 513칸 내내 한 번도 반복되지 않는다.
#   -> 주소생성 microcode 폭발 = `microcode exceeded size for resize1_sd0`.
#
#   원인은 PyTorch export 의 F.interpolate(..., align_corners=False) 한 줄이었다.
#   export_deeplab.py 를 True 로 고쳤다(2026-09-09).
#
# ─────────────────────────────────────────────────────────────────────────────
# 폐기된 이전 가설들 (기록용 — 다시 시도하지 말 것)
# ─────────────────────────────────────────────────────────────────────────────
#   x  할당기 옵션 튜닝 (L1~L6 사다리)  -> 약 14시간 소모, 전부 실패. 층위가 틀렸다.
#   x  "device_pre_post_layers 미적용이 원인"  -> 오진.
#      MZ 소스 확인 결과 이 키는 get_postprocessing_callback()/postprocessing_callback()
#      에서만 쓰인다 = **평가 시 호스트 후처리를 건너뛸지** 정하는 용도이고,
#      그래프에 레이어를 추가하지 않는다.
#   x  "logits ONNX 를 써야 한다"  -> 오진. 공식 yaml 의 parser.nodes 는
#      [MobilenetV2/Conv/Conv2D, ArgMax] 로 **ArgMax 까지 파싱**한다. 즉 공식도
#      우리처럼 꼬리를 ONNX 그래프에서 가져온다. resize_argmax ONNX 가 맞다.
#
# ─────────────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────────────
#   # torch 가 있는 환경에서 (재export)
#   ./run_deeplab_fix.sh --export-only --ckpt-dir <best.pt 가 있는 폴더>
#
#   # hailo_venv 에서 (컴파일)
#   source ~/hailo_venv/bin/activate
#   ./run_deeplab_fix.sh --skip-export
#
#   # 한 환경에 둘 다 있으면 그냥
#   ./run_deeplab_fix.sh --ckpt-dir <폴더>
#
#   # 게이트까지만 (약 2분, 성패 판정만)
#   ./run_deeplab_fix.sh --skip-export --stop-at-gate
# =============================================================================
set -u

EXP="${EXP:-$(cd "$(dirname "$0")/.." && pwd)}"
SRC="$EXP/src"
ART="$EXP/artifacts"
REPO="${REPO:-$(cd "$EXP/../../.." && pwd)}"
CALIB="${CALIB:-$REPO/calib513/calib513}"

NET="deeplab_v3_mnv2_wo_dilation_cityscapes"
ARCH="hailo8l"
ONNX="$ART/deeplab_cs_resize_argmax_ac.onnx"   # ac = align_corners 판 (신규)
START_NODE="/backbone/backbone.0/backbone.0.0/Conv"
END_NODE="/ArgMax"
CALIBSET_SIZE="${CALIBSET_SIZE:-64}"           # 공식 alls 와 동일

CKPT_DIR="${CKPT_DIR:-./runs/dlv3_mnv2}"
DO_EXPORT=1; DO_COMPILE=1; STOP_AT_GATE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-export)  DO_EXPORT=0; shift ;;
        --export-only)  DO_COMPILE=0; shift ;;
        --stop-at-gate) STOP_AT_GATE=1; shift ;;
        --ckpt-dir)     CKPT_DIR="$2"; shift 2 ;;
        *) echo "알 수 없는 인자: $1"; exit 1 ;;
    esac
done

WORK="$ART/fix"
mkdir -p "$WORK"
LOG="$WORK/fix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

hr()  { printf -- '-%.0s' $(seq 1 78); echo; }
sec() { echo; printf '=%.0s' $(seq 1 78); echo; echo "== $*"; printf '=%.0s' $(seq 1 78); echo; }
die() { echo; echo "[중단] $*"; echo "[로그] $LOG"; exit 1; }

echo "로그: $LOG"

# ───────────────────────── 1단계 · ONNX 재export ─────────────────────────
if [ "$DO_EXPORT" -eq 1 ]; then
sec "1단계 · align_corners=True 로 ONNX 재export"
cd "$SRC" || die "src 진입 실패"

python -c "import torch" 2>/dev/null \
    || die "이 환경에 torch 가 없습니다.
       torch 가 있는 환경에서 먼저 다음을 실행하세요:
         ./run_deeplab_fix.sh --export-only --ckpt-dir $CKPT_DIR
       그 다음 hailo_venv 에서:
         ./run_deeplab_fix.sh --skip-export"

grep -q "align_corners=True" export_deeplab.py \
    || die "export_deeplab.py 가 아직 수정 전입니다 (align_corners=True 가 없음).
       2026-09-09 수정본으로 교체한 뒤 다시 실행하세요."
echo "[OK] export_deeplab.py 수정본 확인"

# ── 체크포인트 자동 탐색 ──
# --ckpt-dir 를 안 줬거나 거기에 best.pt/last.pt 가 없으면 흔한 위치를 훑는다.
if [ ! -f "$CKPT_DIR/best.pt" ] && [ ! -f "$CKPT_DIR/last.pt" ]; then
    echo "[탐색] $CKPT_DIR 에 best.pt/last.pt 가 없어 자동으로 찾습니다..."
    FOUND_CKPT=""
    for ROOT in "$SRC" "$EXP" "$HOME" "$REPO"; do
        [ -d "$ROOT" ] || continue
        FOUND_CKPT=$(find "$ROOT" -maxdepth 5 -name "best.pt" -o -maxdepth 5 -name "last.pt" 2>/dev/null | head -1)
        [ -n "$FOUND_CKPT" ] && break
    done
    if [ -n "$FOUND_CKPT" ]; then
        CKPT_DIR=$(dirname "$FOUND_CKPT")
        echo "[OK] 체크포인트 발견: $FOUND_CKPT"
        echo "     --ckpt-dir = $CKPT_DIR 로 진행합니다."
    else
        die "체크포인트(best.pt / last.pt)를 찾지 못했습니다.
       학습 결과 폴더를 직접 지정하세요:
         ./run_deeplab_fix.sh --export-only --ckpt-dir /경로/runs/dlv3_mnv2
       어디 있는지 모르겠으면 이 명령으로 찾으세요:
         find ~ -name 'best.pt' 2>/dev/null
         find /mnt/c/Users/sset0 -maxdepth 6 -name 'best.pt' 2>/dev/null
       [주의] 위 사용법의 <...> 는 자리표시자입니다. 꺾쇠까지 그대로 붙여넣으면
              bash 가 리다이렉션으로 해석해 syntax error 가 납니다."
    fi
fi

echo "실행: python export_deeplab.py --out $CKPT_DIR --mode resize_argmax --onnx $ONNX"
python export_deeplab.py --out "$CKPT_DIR" --mode resize_argmax --onnx "$ONNX" \
    || die "export 실패. --ckpt-dir 로 best.pt 가 있는 폴더를 지정했는지 확인하세요."

[ -f "$ONNX" ] || die "ONNX 가 생성되지 않았습니다: $ONNX"
echo "[OK] $ONNX ($(du -h "$ONNX" | cut -f1))"

# 바이트 수준 확인 — align_corners 문자열이 실제로 들어갔는가
hr
if grep -qa "align_corners" "$ONNX"; then
    echo "[OK] ONNX 안에 align_corners 확인"
else
    echo "[!!] ONNX 에 align_corners 가 없습니다."
    grep -qa "half_pixel" "$ONNX" && echo "     대신 half_pixel 이 있습니다 — 수정이 반영되지 않았습니다."
    die "export 가 수정 전 코드로 돌았습니다."
fi
fi

[ "$DO_COMPILE" -eq 1 ] || { echo; echo "--export-only 지정 — 여기까지."; echo "다음: hailo_venv 에서 ./run_deeplab_fix.sh --skip-export"; exit 0; }

# ───────────────────────── 2단계 · 파싱 ─────────────────────────
sec "2단계 · 파싱 (약 1분)"
cd "$WORK" || die "작업 폴더 진입 실패"
python -c "import hailo_sdk_client" 2>/dev/null \
    || die "hailo_venv 를 활성화하세요:  source ~/hailo_venv/bin/activate"
[ -f "$ONNX" ] || die "ONNX 가 없습니다: $ONNX  (먼저 --export-only 로 재export 하세요)"

PARSED="$WORK/${NET}_ac_parsed.har"
python - "$ONNX" "$PARSED" "$ARCH" "$START_NODE" "$END_NODE" "$NET" <<'PYPARSE'
import sys
from hailo_sdk_client import ClientRunner
onnx, out, arch, start, end, name = sys.argv[1:7]
r = ClientRunner(hw_arch=arch)
r.translate_onnx_model(onnx, name, start_node_names=[start], end_node_names=[end])
r.save_har(out)
print(f"  파싱 완료 -> {out}")
PYPARSE
[ -f "$PARSED" ] || die "파싱 실패."

# ────────────────── 3단계 · ★게이트★ resize 파라미터 판정 ──────────────────
sec "3단계 · ★게이트★ — resize 가 공식과 같은 형태로 파싱됐는가"
echo "판정 기준 (공식 HN 실측값):"
echo "    resize_bilinear_pixels_mode = align_corners"
echo "    resize_h_ratio_list         = 원소 2개 이상 (= DFC 가 다단 분해함)"
hr

python - "$PARSED" <<'PYGATE'
import sys
from hailo_sdk_client import ClientRunner
layers = ClientRunner(har=sys.argv[1]).get_hn()["layers"]

resizes = [(n, v) for n, v in layers.items() if v.get("type") == "resize"]
outs    = [(n, v) for n, v in layers.items() if v.get("type") == "output_layer"]

for n, v in resizes:
    p = v.get("params", {})
    print(f"  [resize] {n}")
    print(f"      in={v.get('input_shapes')}  out={v.get('output_shapes')}")
    print(f"      mode  = {p.get('resize_bilinear_pixels_mode')}")
    print(f"      ratio = {p.get('resize_h_ratio_list')}")
for n, v in outs:
    print(f"  [output] {n}  out={v.get('output_shapes')}")
print()

if not resizes:
    print("  ★ 판정: FAIL — resize 레이어가 없습니다.")
    print("     logits ONNX 를 지정했을 가능성이 큽니다. resize_argmax 판을 쓰세요.")
    sys.exit(2)

big = max(resizes, key=lambda kv: (kv[1].get("output_shapes") or [[0,0,0,0]])[0][1])
p = big[1].get("params", {})
mode  = p.get("resize_bilinear_pixels_mode")
ratio = p.get("resize_h_ratio_list") or []

ok_mode  = (mode == "align_corners")
ok_split = (len(ratio) >= 2)

print(f"  큰 resize: {big[0]}   mode={mode}   ratio 단계수={len(ratio)}")
print(f"    mode  align_corners : {'OK' if ok_mode else 'NG  (현재: %s)' % mode}")
print(f"    다단 분해            : {'OK' if ok_split else 'NG  (단일 단계 -> microcode 폭발)'}")
print()

if ok_mode and ok_split:
    print("  ★ 판정: PASS — 공식과 동일한 형태입니다. 컴파일이 통과할 조건을 갖췄습니다.")
    sys.exit(0)
if not ok_mode:
    print("  ★ 판정: FAIL — 아직 half_pixel 입니다.")
    print("     export_deeplab.py 의 align_corners=True 수정이 ONNX 에 반영되지 않았습니다.")
    print("     1단계(재export)를 다시 하세요.")
    sys.exit(2)
print("  ★ 판정: 주의 — align_corners 인데도 분해가 1단입니다.")
print("     컴파일이 실패할 수 있습니다. 그래도 시도할 가치는 있으니 진행은 가능하지만,")
print("     30분 넘게 걸리면 중단하고 이 출력을 공유하세요.")
sys.exit(1)
PYGATE
GATE=$?
echo
if [ $GATE -eq 2 ]; then
    echo "게이트 실패 — 여기서 멈춥니다. 위 안내대로 고친 뒤 재실행하세요."
    echo "[로그] $LOG"; exit 2
fi
[ $GATE -eq 1 ] && echo "경고 상태지만 계속 진행합니다."
[ "$STOP_AT_GATE" -eq 1 ] && { echo; echo "--stop-at-gate 지정 — 여기까지."; echo "이어서: ./run_deeplab_fix.sh --skip-export"; exit 0; }

# ────────────────── 4단계 · 양자화 + 컴파일 (공식과 동일한 3줄 alls) ──────────────────
sec "4단계 · 양자화 + 컴파일 (약 20~50분)"
echo "alls 는 공식 generic 과 동일한 3줄만 사용합니다."
echo "할당기 옵션(allocator_param/defuse/performance_param)은 넣지 않습니다 —"
echo "원인이 해결됐다면 공식처럼 3줄로 통과해야 정상입니다."
hr

N_CALIB=$(find "$CALIB" -maxdepth 2 -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)
[ "$N_CALIB" -gt 0 ] || die "캘리브 이미지가 없습니다: $CALIB"
echo "캘리브 이미지 ${N_CALIB}장 중 ${CALIBSET_SIZE}장 사용"

HEF="$ART/${NET}.hef"
python - "$PARSED" "$CALIB" "$CALIBSET_SIZE" "$HEF" "$WORK" "$NET" <<'PYCOMPILE'
import os, sys, glob, time
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from hailo_sdk_client import ClientRunner

parsed, calib_dir, n_calib, hef_out, work, name = sys.argv[1:7]
n_calib = int(n_calib)
SIZE = 513

ALLS = (
    "normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])\n"
    f"model_optimization_config(calibration, batch_size=1, calibset_size={n_calib})\n"
    "pre_quantization_optimization(equalization, policy=disabled)\n"
)

def load_calib():
    import cv2
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
    files.sort()
    step = max(1, len(files) // n_calib)
    arr = []
    for f in files[::step][:n_calib]:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.shape[0] != SIZE or img.shape[1] != SIZE:
            img = cv2.resize(img, (SIZE, SIZE))
        arr.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(f"  캘리브 {len(arr)}장 로드 (풀 {len(files)}장)", flush=True)
    return np.asarray(arr, dtype=np.float32)

q_har = os.path.join(work, f"{name}_ac_quantized.har")
if os.path.exists(q_har):
    print(f"  기존 quantized HAR 재사용: {q_har}", flush=True)
else:
    print("  [1/2] 양자화 시작...", flush=True)
    t0 = time.time()
    r = ClientRunner(har=parsed)
    r.load_model_script(ALLS)
    r.optimize(load_calib())
    r.save_har(q_har)
    print(f"  [1/2] 양자화 완료 {time.time()-t0:.0f}s -> {q_har}", flush=True)

print("  [2/2] 컴파일 시작... (공식과 동일한 3줄 alls, 할당기 옵션 없음)", flush=True)
t0 = time.time()
r = ClientRunner(har=q_har)
try:
    hef = r.compile()
except Exception as e:
    print(f"\n  [실패] 컴파일 에러 ({time.time()-t0:.0f}s):", flush=True)
    for line in str(e).splitlines()[:15]:
        print("     | " + line[:190], flush=True)
    print("\n  에러에 'resize' 가 또 나오면 3단계 게이트 출력을 함께 공유해 주세요.", flush=True)
    sys.exit(1)

with open(hef_out, "wb") as f:
    f.write(hef)
print(f"\n  ★★★ 컴파일 성공 ({time.time()-t0:.0f}s)", flush=True)
print(f"      HEF: {hef_out} ({os.path.getsize(hef_out)/1e6:.2f} MB)", flush=True)
PYCOMPILE
RC=$?
[ $RC -eq 0 ] || die "컴파일 실패. 위 에러를 확인하세요."

# ───────────────────────── 5단계 · HEF 검증 ─────────────────────────
sec "5단계 · HEF 검증"
echo "공식(사전학습판) 기준값:  contexts = 3,  출력 = argmax1 UINT8 NHW(513x513)"
hr
if command -v hailortcli >/dev/null 2>&1; then
    hailortcli parse-hef "$HEF" | grep -iE "context|output|architecture" | sed 's/^/  /'
else
    echo "  이 환경에 hailortcli 없음. 보드에서 확인:"
    echo "    scp -P 40021 $HEF npu-rpi1@155.230.16.157:~/hailo-rpi5-examples/resources"
    echo "    ssh npu-rpi1@155.230.16.157 -p 40021 'hailortcli parse-hef ~/hailo-rpi5-examples/resources/${NET}.hef'"
fi

echo
echo "다음 할 일:"
echo "  1) context 수가 3인지 확인 (다르면 보고서에 명시)"
echo "  2) HEF 를 보드로 전송하고 mz3 3모델 벤치마크에 투입"
echo "  3) mIoU 재측정 — align_corners 가 바뀌었으므로 배포 기준 값이 달라진다:"
echo "     python export_deeplab.py --out $CKPT_DIR --mode resize_argmax --eval --data <Cityscapes>"
echo "[로그] $LOG"
