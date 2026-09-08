#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_deeplab.py (2026-09-02) — deeplab_v3_mnv2_wo_dilation(Cityscapes) HEF 컴파일.

인계문서 4-A 블로커 전용. compile_hailo.py 를 대체하지 않고, deeplab 만 따로 다룬다.

─────────────────────────────────────────────────────────────────────────────
실패 원인 (재진단)
─────────────────────────────────────────────────────────────────────────────
  [error] microcode exceeded size for resize1_sd0 (cluster_index=0, layer_index=0)

`resize1_sd0` 라는 레이어 이름 자체가 답이다. **resize 가 NN core 레이어로 컴파일됐다.**
공식 모델에서 이 연산은 NN core 가 아니라 device pre/post layer 에서 처리된다:

    cfg/networks/deeplab_v3_mobilenet_v2_wo_dilation.yaml
      postprocessing:
        device_pre_post_layers: {bilinear: true, argmax: true}

그런데 compile_hailo.py 는 `ClientRunner.translate_onnx_model()` + `.alls` 만 쓴다.
`device_pre_post_layers` 는 **yaml 쪽 설정**이라 이 경로에는 전달되지 않는다.
지시가 없으니 DFC 는 ONNX 의 Resize 를 평범한 레이어로 배치했고, 17x17x19 -> 513x513x19
(약 500만 원소) 업샘플의 microcode 가 클러스터 한계를 넘었다.

즉 "resize 를 그래프 어디에 두느냐"의 문제가 아니라 **어느 경로로 컴파일하느냐**의 문제다.
argmax 를 앞으로 옮기는 안(인계문서 제안)은 컴파일은 통과할 수 있어도
정확도와 연산 배치가 공식과 달라져서 비교 실험 자체가 무의미해진다(export_deeplab.py 주석 참조).

─────────────────────────────────────────────────────────────────────────────
경로 2개 — 순서대로 시도할 것
─────────────────────────────────────────────────────────────────────────────
 [경로 A · 권장] hailomz (Model Zoo yaml) — 공식 HEF 를 만든 것과 **같은 경로**.
    --emit-yaml 로 yaml/alls 를 생성한 뒤:
      cp <생성된 yaml> $MZ/hailo_model_zoo/cfg/networks/
      cp <생성된 alls> $MZ/hailo_model_zoo/cfg/alls/generic/
      hailomz parse    deeplab_v3_mnv2_wo_dilation_cityscapes --hw-arch hailo8l --ckpt <sim.onnx> \\
                       --start-node-names <입력노드> --end-node-names <꼬리노드>
      hailomz optimize deeplab_v3_mnv2_wo_dilation_cityscapes --har <har> --calib-path <이미지폴더>
      hailomz compile  deeplab_v3_mnv2_wo_dilation_cityscapes --har <quantized.har> --hw-arch hailo8l
    device_pre_post_layers 가 yaml 에 있으므로 bilinear/argmax 가 확실히 칩으로 간다.

 [경로 B · DFC 직접] 이 스크립트. .alls 로 device post layer 를 요청한다.
    DFC 버전마다 model script 명령 철자가 다를 수 있어, **설치된 DFC 에 실제로 먹는 철자를
    자동 탐색**한다(--probe). 지어내지 않고 load_model_script() 가 받아들이는 것만 쓴다.

─────────────────────────────────────────────────────────────────────────────
사용 (컴파일 1회 = 약 10분. 반드시 parse 먼저)
─────────────────────────────────────────────────────────────────────────────
  python compile_deeplab.py --onnx deeplab_cs_logits_sim.onnx --mode logits --probe
  python compile_deeplab.py --onnx deeplab_cs_logits_sim.onnx --mode logits --stop-after parse
  python compile_deeplab.py --onnx deeplab_cs_logits_sim.onnx --mode logits \\
         --calib-dir /mnt/c/datasets/Cityscapes/leftImg8bit/train --calib-n 512 \\
         --out deeplab_v3_mnv2_wo_dilation_cityscapes.hef
"""
import argparse
import glob
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SIZE = 513
NAME = "deeplab_v3_mnv2_wo_dilation_cityscapes"

# 공식 alls (v2.19.0, cfg/alls/generic/deeplab_v3_mobilenet_v2_wo_dilation.alls) 3줄.
# calibset_size 만 우리 캘리브 장수에 맞춰 조정한다.
BASE_ALLS = """\
normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
model_optimization_config(calibration, batch_size=1, calibset_size={calibset})
pre_quantization_optimization(equalization, policy=disabled)
"""

# bilinear+argmax 를 device post layer 로 넘기라고 DFC 에 알리는 model-script 후보들.
# DFC 3.3x 의 정확한 철자는 버전마다 다르다. **추측을 코드에 박지 않고** 설치된 DFC 가
# load_model_script() 에서 받아들이는 것을 실제로 시험해서 고른다(--probe).
DEVICE_PP_CANDIDATES = [
    ("logits_layer(argmax)",
     "logits_layer(argmax)\n"),
    ("resize+argmax (logits_layer)",
     "resize1 = resize(resize_shapes=[{s},{s}], resize_method=bilinear)\n"
     "logits_layer(argmax)\n"),
    ("bilinear+argmax (device_pre_post_layers)",
     "device_pre_post_layers(bilinear=true, argmax=true)\n"),
    ("post_process(argmax)",
     "post_process(argmax)\n"),
]

FORBIDDEN = ["performance_param", "resources_param", "allocator_param", "context_switch_param"]


def load_calib(calib_dir, n, size):
    """캘리브: HEF 입력이 uint8 raw 이고 정규화는 칩(normalization1)이 한다.
    따라서 여기서도 0~255 원본 스케일 그대로 넣는다(학습 전처리와 규약 일치)."""
    import cv2
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
    files.sort()
    if not files:
        raise SystemExit(f"[에러] 캘리브레이션 이미지가 없습니다: {calib_dir}")
    step = max(1, len(files) // n)
    picked = files[::step][:n]
    arr, bad = [], 0
    for f in picked:
        img = cv2.imread(f)
        if img is None:                       # [수정] 구버전은 실패해도 0 배열을 그대로 넣었다
            bad += 1
            continue
        img = cv2.resize(img, (size, size))
        arr.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if bad:
        print(f"  [경고] 읽기 실패 {bad}장 제외")
    if not arr:
        raise SystemExit("[에러] 유효한 캘리브 이미지가 0장입니다.")
    print(f"  캘리브레이션: {len(arr)}장 (풀 {len(files)}장에서 균등 stride 샘플)")
    return np.asarray(arr, dtype=np.float32)


def onnx_edge_nodes(path):
    """--start-node-names / --end-node-names 에 넣을 **실제** 노드 이름을 ONNX 에서 읽는다.
    (v1 은 output_names='ArgMax' 라는 텐서 이름만 보고 노드 이름이라 가정했다.)"""
    try:
        import onnx
    except ImportError:
        return None, None
    m = onnx.load(path)
    start = m.graph.node[0].name or m.graph.node[0].output[0]
    end = m.graph.node[-1].name or m.graph.node[-1].output[0]
    print(f"  ONNX 시작 노드: {m.graph.node[0].op_type} '{start}'")
    print(f"  ONNX 끝   노드: {m.graph.node[-1].op_type} '{end}'")
    for o in m.graph.output:
        print(f"  ONNX 출력 '{o.name}' shape="
              f"{[d.dim_value for d in o.type.tensor_type.shape.dim]}")
    return start, end


def probe_alls(runner, calibset):
    """설치된 DFC 가 실제로 받아들이는 device post-layer 명령을 찾는다."""
    print("\n[probe] device pre/post layer 명령 철자 탐색")
    ok = []
    for label, snippet in DEVICE_PP_CANDIDATES:
        script = BASE_ALLS.format(calibset=calibset) + snippet.format(s=SIZE)
        try:
            runner.load_model_script(script)
            print(f"  [OK  ] {label}")
            ok.append((label, snippet))
        except Exception as e:
            msg = str(e).splitlines()[0][:120]
            print(f"  [실패] {label}  -> {msg}")
    if not ok:
        print("\n  받아들여진 명령이 없습니다. 경로 A(hailomz + yaml)를 쓰십시오.")
        print("  또는 DFC 문서에서 이 버전의 post-processing 명령을 확인해 "
              "DEVICE_PP_CANDIDATES 에 추가하세요:")
        print("    python -c \"import hailo_sdk_client, os; "
              "print(os.path.dirname(hailo_sdk_client.__file__))\"")
    return ok


def emit_yaml(onnx_path, start, end, out_dir):
    """경로 A 용 Model Zoo yaml/alls 생성 (device_pre_post_layers 포함)."""
    os.makedirs(out_dir, exist_ok=True)
    yml = f"""base:
- base/cityscapes.yaml
network:
  network_name: {NAME}
paths:
  network_path:
  - {onnx_path}
  alls_script: {NAME}.alls
parser:
  nodes:
  - {start}
  - {end}
  normalization_params:
    normalize_in_net: true
    mean_list: [127.5, 127.5, 127.5]
    std_list: [127.5, 127.5, 127.5]
postprocessing:
  device_pre_post_layers: {{max_finder: false, bilinear: true, argmax: true, softmax: false}}
preprocessing:
  network_type: segmentation
  meta_arch: fcn_resnet
evaluation:
  labels_offset: 0
  classes: 19
info:
  task: segmentation
  input_shape: {SIZE}x{SIZE}x3
  training_data: cityscapes train
"""
    yp = os.path.join(out_dir, f"{NAME}.yaml")
    ap_ = os.path.join(out_dir, f"{NAME}.alls")
    open(yp, "w", encoding="utf-8").write(yml)
    open(ap_, "w", encoding="utf-8").write(BASE_ALLS.format(calibset=512))
    print(f"\n[경로 A] 생성: {yp}\n           생성: {ap_}")
    print("  cp 로 Model Zoo cfg/networks, cfg/alls/generic 에 넣고 hailomz parse 부터 진행하세요.")
    print("  (base/cityscapes.yaml 이 없으면 base/pascal.yaml 로 바꾸고 classes 만 19로 두면 됩니다.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mode", choices=["logits", "resize_argmax"], default="logits")
    ap.add_argument("--arch", default="hailo8l")
    ap.add_argument("--out", default=f"{NAME}.hef")
    ap.add_argument("--calib-dir")
    ap.add_argument("--calib-n", type=int, default=512)
    ap.add_argument("--start-node")
    ap.add_argument("--end-node")
    ap.add_argument("--alls", help="직접 만든 .alls 파일 경로 (지정 시 자동 탐색 안 함)")
    ap.add_argument("--stop-after", choices=["parse", "optimize", "compile"], default="compile")
    ap.add_argument("--probe", action="store_true", help="alls 명령 철자만 시험하고 종료")
    ap.add_argument("--emit-yaml", metavar="DIR", help="경로 A(hailomz)용 yaml/alls 생성")
    ap.add_argument("--har-dir", default=".", help="중간 HAR 저장 위치(재시도 시 파싱/양자화 재활용)")
    ap.add_argument("--allow-nondefault", action="store_true",
                    help="performance_param 등 '기본값 조건'을 깨는 명령을 허용")
    a = ap.parse_args()

    if not os.path.exists(a.onnx):
        raise SystemExit(f"[에러] ONNX 없음: {a.onnx}")

    print(f"[0/4] ONNX 점검: {a.onnx}")
    s_auto, e_auto = onnx_edge_nodes(a.onnx)
    start = a.start_node or s_auto
    end = a.end_node or e_auto

    if a.emit_yaml:
        emit_yaml(os.path.abspath(a.onnx), start, end, a.emit_yaml)
        if a.probe is False and a.stop_after == "parse":
            return

    from hailo_sdk_client import ClientRunner
    runner = ClientRunner(hw_arch=a.arch)

    kw = {}
    if start:
        kw["start_node_names"] = [start]
    if end:
        kw["end_node_names"] = [end]
    print(f"\n[1/4] 파싱 (arch={a.arch}, nodes={start} .. {end})")
    runner.translate_onnx_model(a.onnx, NAME, **kw)
    print("      파싱 완료")
    try:                                   # 파싱 결과 출력 shape 확인 — 여기서 513x513x1 이 목표
        hn = runner.get_hn()
        outs = [(k, v.get("output_shapes")) for k, v in hn["layers"].items()
                if v.get("type") == "output_layer"]
        print(f"      HN 출력 레이어: {outs}")
    except Exception as e:
        print(f"      (HN 출력 확인 생략: {e})")

    har_parsed = os.path.join(a.har_dir, f"{NAME}_parsed.har")
    runner.save_har(har_parsed)
    print(f"      HAR 저장: {har_parsed}  (재시도 시 ClientRunner(har=...) 로 재활용)")

    calibset = a.calib_n
    if a.alls:
        script = open(a.alls, encoding="utf-8").read()
    elif a.mode == "logits":
        cands = probe_alls(runner, calibset)
        if a.probe:
            return
        if not cands:
            raise SystemExit("[중단] device post layer 명령을 못 찾았습니다. 경로 A 를 쓰십시오.")
        label, snippet = cands[0]
        print(f"\n  선택된 명령: {label}")
        script = BASE_ALLS.format(calibset=calibset) + snippet.format(s=SIZE)
    else:
        # resize_argmax: 꼬리가 이미 그래프에 있다. 파서가 device layer 로 접었는지
        # 위의 'HN 출력 레이어' 로그로 확인할 것. 513x513x19 로 나오면 접히지 않은 것이고,
        # 그대로 컴파일하면 v1 과 똑같이 microcode 초과로 실패한다.
        script = BASE_ALLS.format(calibset=calibset)
        if a.probe:
            probe_alls(runner, calibset)
            return

    bad = [c for c in FORBIDDEN if c in script]
    if bad and not a.allow_nondefault:
        raise SystemExit(f"[중단] '기본값 조건' 실험인데 alls 에 {bad} 가 있습니다. "
                         f"(의도한 것이면 --allow-nondefault)")

    print(f"\n[2/4] model script:\n{script}")
    runner.load_model_script(script)

    if a.stop_after == "parse":
        print("[중단] --stop-after parse — 여기까지 통과하면 그래프 구조는 문제없습니다.")
        return

    print("[3/4] 최적화(양자화)")
    if not a.calib_dir:
        raise SystemExit("--calib-dir 이 필요합니다 (예: <Cityscapes>/leftImg8bit/train)")
    runner.optimize(load_calib(a.calib_dir, a.calib_n, SIZE))
    har_opt = os.path.join(a.har_dir, f"{NAME}_quantized.har")
    runner.save_har(har_opt)
    print(f"      최적화 완료. HAR: {har_opt}")
    if a.stop_after == "optimize":
        return

    print("[4/4] 컴파일 -> HEF  (10분 내외)")
    hef = runner.compile()
    with open(a.out, "wb") as f:
        f.write(hef)
    print(f"  -> 저장: {a.out} ({os.path.getsize(a.out)/1e6:.2f} MB)")
    print("\n반드시 확인 (컨텍스트 수는 이 실험의 핵심 변수):")
    print(f"  hailortcli parse-hef {a.out} | grep -i context")
    print("  공식 HEF 와 컨텍스트 수가 다르면 그 사실을 보고서에 명시할 것.")


if __name__ == "__main__":
    main()
