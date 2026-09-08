#!/bin/bash
# =============================================================================
# run_deeplab_compile.sh — deeplab HEF 컴파일 런북 (WSL2 Ubuntu 22.04 + DFC 3.33)
#
# 이 세션(클라우드 컨테이너)에서 이미 끝난 것:
#   · deeplab_cs_513_sim.onnx 구조 확인 — 꼬리가 Conv(/classifier/Conv) -> Resize(/Resize)
#     -> ArgMax(/ArgMax) -> Cast(/Cast) 이고 그래프 출력 텐서 이름이 'ArgMax'(INT32).
#     즉 **'ArgMax' 라는 이름의 노드는 없다.** 인계문서/가이드의
#     `--end-node-names ArgMax`, `--start-node-names MobilenetV2/Conv/Conv2D` 는 둘 다 없는 이름이다.
#   · Resize 가 1개뿐 -> 컴파일 에러의 `resize1_sd0` 가 바로 이것. 진단 확정.
#   · 파라미터 실측 = 2,231,161 (2.231M), 공식 2.10M 의 1.06배. (인계문서 "2.25M 추정" 대체)
#   · artifacts/deeplab_cs_logits.onnx        : /classifier/Conv 까지 (출력 1x19x17x17)
#   · artifacts/deeplab_cs_resize_argmax.onnx : /ArgMax 까지, Cast 제거 (출력 1x1x513x513)
#   · mz_cfg/*.yaml, *.alls                   : device_pre_post_layers 포함 (경로 A용)
#   · 캘리브 세트 calib513/calib513/ 512장(513x513 PNG) 확인 — 별도 준비 불필요
#
# 여기서 할 일: DFC 로 컴파일. **반드시 ①->②->③ 순서.** ③만 10분이다.
# =============================================================================
set -e
REPO="${REPO:-/mnt/c/Users/sset0/jungmin-claude/StudentExperiment/NPUscheduler}"
EXP="$REPO/hailo_8L/experiments/2026-08-31_mz3_retrain"
CALIB="${CALIB:-$REPO/calib513/calib513}"
cd "$EXP/src"

echo "== 환경 =="
python -c "import hailo_sdk_client, sys; print('DFC OK', sys.version.split()[0])" \
  || { echo "hailo_venv 를 먼저 activate 하세요: source ~/hailo_venv/bin/activate"; exit 1; }
ls "$CALIB" | wc -l | xargs echo "캘리브 이미지:"

echo
echo "===== ① DFC 가 받아들이는 device post-layer 명령 철자 탐색 (약 1분) ====="
python compile_deeplab.py --onnx ../artifacts/deeplab_cs_logits.onnx --mode logits --probe

echo
echo "===== ② 파싱만 (약 1분) — 여기서 막히면 ③으로 10분 낭비하지 않는다 ====="
python compile_deeplab.py --onnx ../artifacts/deeplab_cs_logits.onnx --mode logits \
       --har-dir ../artifacts --stop-after parse

echo
echo "===== ③ 양자화 + 컴파일 (약 10분) ====="
python compile_deeplab.py --onnx ../artifacts/deeplab_cs_logits.onnx --mode logits \
       --calib-dir "$CALIB" --calib-n 512 --har-dir ../artifacts \
       --out ../artifacts/deeplab_v3_mnv2_wo_dilation_cityscapes.hef

echo
echo "===== ④ 컨텍스트 수 확인 (스케줄링 실험의 핵심 변수) ====="
hailortcli parse-hef ../artifacts/deeplab_v3_mnv2_wo_dilation_cityscapes.hef | grep -iE "context|output|shape" || true
echo "공식 HEF 와 컨텍스트 수가 다르면 보고서에 반드시 명시할 것."

cat <<'NOTE'

──────────────────────────────────────────────────────────────────────
①에서 받아들여진 명령이 하나도 없거나 ②/③이 여전히 실패하면 → 경로 A(hailomz)
──────────────────────────────────────────────────────────────────────
  MZ=~/hailo_model_zoo            # v2.19.0 이하 체크아웃
  cp ../mz_cfg/deeplab_v3_mnv2_wo_dilation_cityscapes.yaml  $MZ/hailo_model_zoo/cfg/networks/
  cp ../mz_cfg/deeplab_v3_mnv2_wo_dilation_cityscapes.alls  $MZ/hailo_model_zoo/cfg/alls/generic/

  hailomz parse    deeplab_v3_mnv2_wo_dilation_cityscapes --hw-arch hailo8l \
      --ckpt ../artifacts/deeplab_cs_logits.onnx \
      --start-node-names /backbone/backbone.0/backbone.0.0/Conv \
      --end-node-names   /classifier/Conv
  hailomz optimize deeplab_v3_mnv2_wo_dilation_cityscapes --har <parsed.har> --calib-path <CALIB>
  hailomz compile  deeplab_v3_mnv2_wo_dilation_cityscapes --har <quantized.har> --hw-arch hailo8l

  ※ base/cityscapes.yaml 이 없으면 yaml 의 base 를 base/pascal.yaml 로 바꾸고 classes: 19 유지.
  ※ --performance 금지. alls 에 resources_param/performance_param/context_switch_param 추가 금지.

  그래도 안 되면 resize_argmax 판으로 한 번 더:
      --ckpt ../artifacts/deeplab_cs_resize_argmax.onnx
      --end-node-names /ArgMax   (yaml 은 *_resize_argmax.yaml 사용)
  파싱 후 HN 출력이 513x513x1 이면 꼬리가 device layer 로 접힌 것,
  513x513x19 로 나오면 접히지 않은 것이라 그대로 컴파일하면 또 microcode 초과로 실패한다.
NOTE
