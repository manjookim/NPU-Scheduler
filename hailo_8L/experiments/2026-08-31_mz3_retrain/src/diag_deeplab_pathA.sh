#!/bin/bash
# =============================================================================
# diag_deeplab_pathA.sh  [2026-09-09]
# 경로 A 게이트 FAIL 원인 규명 — 읽기 전용 진단. 아무것도 바꾸지 않는다.
#
# 2026-09-09 02:03 실행 결과:
#   Model Zoo v2.17.0 (v2.19.0 이하 = 버전 조건 충족)
#   base/cityscapes.yaml 없음 -> base/pascal.yaml 로 자동 대체됨
#   parse 후 HN: 65 레이어, resize 없음, argmax 없음, 출력 17x17x19
#   => device_pre_post_layers 가 parse 단계에서 적용되지 않았다.
#
# 가설 3개를 순서대로 검증한다:
#   H1. device_pre_post_layers 는 애초에 parse 가 아니라 optimize/compile 단계에서
#       적용된다  ->  그렇다면 게이트가 너무 이른 시점을 본 것이고, 실패가 아니다.
#   H2. base/pascal.yaml 로 대체된 것이 postprocessing 병합을 깨뜨렸다.
#   H3. yaml 의 다른 키(network_type / meta_arch)가 공식과 달라서 MZ 가
#       segmentation 후처리 경로를 타지 않았다.
#
# ★ H1 이 참이면 지금 상태 그대로 4단계로 넘어가면 된다. 그래서 H1 부터 본다.
#
# 사용:
#   source ~/hailo_venv/bin/activate
#   cd .../2026-08-31_mz3_retrain/src
#   sed -i 's/\r$//' diag_deeplab_pathA.sh && chmod +x diag_deeplab_pathA.sh
#   ./diag_deeplab_pathA.sh 2>&1 | tee ../artifacts/pathA/diag.txt
# =============================================================================
set -u
MZ="${MZ:-$HOME/hailo_model_zoo}"
CFG="$MZ/hailo_model_zoo/cfg"
CORE="$MZ/hailo_model_zoo/core"
EXP="${EXP:-$(cd "$(dirname "$0")/.." && pwd)}"

hr() { printf -- '-%.0s' $(seq 1 78); echo; }
sec() { echo; printf '=%.0s' $(seq 1 78); echo; echo "== $*"; printf '=%.0s' $(seq 1 78); echo; }

sec "A · device_pre_post_layers 가 MZ 소스 어디에서 소비되는가  [H1 검증]"
echo "이 키를 읽는 함수가 parse 쪽인지 optimize/compile 쪽인지가 핵심이다."
hr
grep -rn "device_pre_post_layers" "$CORE"/main_utils.py 2>/dev/null | sed 's/^/  /'
hr
echo "위 라인 주변 문맥 (어느 함수 안인지 확인):"
python3 - "$CORE/main_utils.py" <<'PY'
import re, sys
path = sys.argv[1]
try:
    lines = open(path, encoding="utf-8").read().splitlines()
except Exception as e:
    print(f"  [!] 읽기 실패: {e}"); sys.exit(0)
hits = [i for i, l in enumerate(lines) if "device_pre_post_layers" in l]
for i in hits:
    # 이 라인을 감싸는 가장 가까운 def 를 위로 거슬러 찾는다
    fn = "(모듈 최상위)"
    for j in range(i, -1, -1):
        m = re.match(r"\s*def\s+(\w+)", lines[j])
        if m:
            fn = m.group(1); break
    print(f"  L{i+1:<5} def {fn}()   |  {lines[i].strip()[:100]}")
print()
print("  판정 힌트: 함수 이름에 parse 가 들어가면 parse 단계,")
print("             optimize/quantize/compile 이 들어가면 그 단계에서 적용된다.")
PY

sec "B · 공식 wo_dilation yaml 전문  [H2/H3 검증용 정답지]"
OFF="$CFG/networks/deeplab_v3_mobilenet_v2_wo_dilation.yaml"
if [ -f "$OFF" ]; then
    cat "$OFF" | sed 's/^/  /'
else
    echo "  [!] 없음: $OFF"
fi

sec "C · 우리 yaml vs 공식 yaml 차이"
OURS="$CFG/networks/deeplab_v3_mnv2_wo_dilation_cityscapes.yaml"
if [ -f "$OFF" ] && [ -f "$OURS" ]; then
    diff -u "$OFF" "$OURS" | sed 's/^/  /' || true
else
    echo "  [!] 비교 불가 (파일 없음)"
fi

sec "D · 공식이 참조하는 base yaml 들의 postprocessing 설정"
for B in $(grep -A5 '^base:' "$OFF" 2>/dev/null | grep -o 'base/[a-z0-9_]*\.yaml'); do
    echo "  --- $B ---"
    if [ -f "$CFG/$B" ]; then
        grep -n -A8 "postprocessing\|network_type\|meta_arch" "$CFG/$B" | head -30 | sed 's/^/    /'
    else
        echo "    (없음)"
    fi
done

sec "E · Cityscapes 계열 공식 모델은 어떤 base 를 쓰는가"
echo "우리는 base/cityscapes.yaml 이 없어서 base/pascal.yaml 로 대체했다."
echo "MZ v2.17.0 에 실제로 있는 Cityscapes 세그멘테이션 모델들의 base 를 본다:"
hr
for N in stdc1 fcn8_resnet_v1_18 segformer_b0_bn deeplab_v3_mobilenet_v2; do
    Y="$CFG/networks/$N.yaml"
    if [ -f "$Y" ]; then
        echo "  [$N]"
        grep -A4 '^base:' "$Y" | grep -o 'base/[a-z0-9_]*\.yaml' | sed 's/^/      base: /'
        grep -n "classes:\|network_type:\|meta_arch:" "$Y" | sed 's/^/      /'
        grep -n -A2 "device_pre_post_layers" "$Y" | sed 's/^/      /'
        echo
    fi
done
echo "  cfg/base/ 에 실제로 있는 세그멘테이션 관련 base:"
ls "$CFG/base/" | grep -iE "cityscape|pascal|segment|voc" | sed 's/^/      /' || echo "      (해당 없음)"

sec "F · ★대조군★ 공식 모델을 같은 방식으로 파싱하면 꼬리가 붙는가"
echo "이게 H1 을 최종 확정한다."
echo "  · 공식도 parse 후 17x17x21 이면  -> 꼬리는 나중 단계에서 붙는다(우리 게이트가 이른 것)"
echo "  · 공식은 513x513x1 로 나오면    -> parse 단계에서 붙는 게 맞고, 우리 yaml 이 문제다"
hr
WORK="$EXP/artifacts/pathA/control"
mkdir -p "$WORK" && cd "$WORK" || exit 1
echo "실행: hailomz parse deeplab_v3_mobilenet_v2_wo_dilation --hw-arch hailo8l"
echo "(공식 ckpt 를 자동 다운로드합니다. 네트워크 필요, 1~3분)"
hr
hailomz parse deeplab_v3_mobilenet_v2_wo_dilation --hw-arch hailo8l 2>&1 | tail -20
echo
CTRL_HAR=$(find "$WORK" -maxdepth 1 -name "*.har" | head -1)
if [ -n "$CTRL_HAR" ]; then
    echo "대조군 HAR: $CTRL_HAR"
    hr
    python3 - "$CTRL_HAR" <<'PY'
import sys
from hailo_sdk_client import ClientRunner
layers = ClientRunner(har=sys.argv[1]).get_hn()["layers"]
outs    = [(n, v.get("output_shapes"), v.get("engine")) for n, v in layers.items() if v.get("type") == "output_layer"]
resizes = [(n, v.get("output_shapes"), v.get("engine")) for n, v in layers.items() if v.get("type") == "resize"]
amax    = [(n, v.get("output_shapes"), v.get("engine")) for n, v in layers.items() if v.get("type") == "argmax"]
print(f"  [공식] 총 레이어 : {len(layers)}")
print(f"  [공식] resize    : {[n for n,_,_ in resizes] or '없음'}")
print(f"  [공식] argmax    : {[n for n,_,_ in amax] or '없음'}")
for n, s, e in outs:
    print(f"  [공식] 출력      : {n}  out={s}  engine={e}")
print()
dims = tuple(outs[0][1][0][-3:]) if outs else None
print(f"  [공식] 최종 출력 형태: {dims}")
print()
if dims and dims[0] == 17:
    print("  ★★ 결론: 공식도 parse 단계에서는 꼬리가 없다.")
    print("     => 우리 게이트가 너무 이른 시점을 본 것이다. FAIL 이 아니다.")
    print("     => 그대로  ./run_deeplab_pathA.sh --from 4  로 진행하면 된다.")
elif dims and dims[0] == 513:
    print("  ★★ 결론: 공식은 parse 단계에서 이미 513 꼬리가 붙는다.")
    print("     => 우리 yaml/base 설정이 문제다. 위 C·D·E 의 diff 를 보고 맞춰야 한다.")
else:
    print("  ★★ 예상 밖의 형태. 위 출력을 그대로 공유할 것.")
PY
else
    echo "  [!] 대조군 HAR 생성 실패 — 위 hailomz 출력을 확인하세요."
    echo "      (네트워크 차단이면 이 단계만 건너뛰고 A~E 결과만 공유해 주세요.)"
fi

sec "진단 끝"
echo "A 섹션의 함수 이름과 F 섹션의 ★★ 결론, 이 두 개가 핵심입니다."
echo "출력 전체를 그대로 공유해 주세요."
