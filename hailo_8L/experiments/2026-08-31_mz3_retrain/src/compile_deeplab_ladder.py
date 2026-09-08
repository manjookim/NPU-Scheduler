#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_deeplab_ladder.py (2026-09-02) — deeplab HEF 컴파일 사다리.

[여기까지 확정]
  · 우리 HN = 공식과 동일: conv36(17x17x19) -> resize1(513x513x19) -> argmax1(513x513x1) -> output.
    학습/ONNX/파싱 전부 정상. Cast 무관.
  · 실패는 오직 할당: `microcode exceeded size for resize1_sd0`.
  · MZ 에 hailo8l/base/..._wo_dilation.alls 는 **없다** -> 공식 8L HEF 는 3줄 generic alls 로 컴파일됐다.
    같은 폴더의 dilated 판만 `performance_param(compiler_optimization_level=max)` 를 갖고 있다.
  · `resize` model-script 명령은 engine / hw_layer_type 인자를 받는다
    (hailo_sdk_client/.../model_modifications_commands.py).

[전략]
  양자화(optimize)는 **한 번만** 하고 quantized HAR 로 저장한 뒤,
  컴파일만 옵션을 바꿔가며 재시도한다. 재시도마다 양자화를 다시 하지 않으므로 훨씬 빠르다.

사용:
  # 문법 탐색만 (초 단위) — resize 의 engine 값 후보를 실제로 시험한다
  python compile_deeplab_ladder.py --probe-engine

  # 본 실행 (양자화 1회 + 컴파일 사다리). 백그라운드 권장
  nohup python compile_deeplab_ladder.py \
      --onnx ../artifacts/deeplab_cs_resize_argmax.onnx \
      --calib-dir ../../../../calib513/calib513 \
      --out ../artifacts/deeplab_v3_mnv2_wo_dilation_cityscapes.hef \
      > ../artifacts/ladder.log 2>&1 &
  tail -f ../artifacts/ladder.log
"""
import argparse
import glob
import os
import time
import traceback

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SIZE = 513
NAME = "deeplab_v3_mnv2_wo_dilation_cityscapes"
START = "/backbone/backbone.0/backbone.0.0/Conv"
END = "/ArgMax"

# 공식 wo_dilation generic alls (3줄) — 양자화 조건은 이것으로 고정한다.
BASE_ALLS = """\
normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
model_optimization_config(calibration, batch_size=1, calibset_size={calibset})
pre_quantization_optimization(equalization, policy=disabled)
"""

# 컴파일 단계에서만 얹는 할당 옵션 사다리. 위에서부터 시도.
# 주: 이 옵션들은 **컴파일러 자원배치**에 관한 것이고, 실험 조건인 **스케줄러 파라미터**
#     (threshold/timeout/priority/batch)와는 다른 층위다. 공식 HEF 도
#     hailo8l/base 에서 performance_param 을 쓰므로 이걸 금지하면 오히려 조건이 벌어진다.
LADDER = [
    ("L1 optimization_level=max",
     "performance_param(compiler_optimization_level=max)\n"),

    ("L2 max + 넉넉한 timeout",
     "performance_param(compiler_optimization_level=max)\n"
     "allocator_param(timeout=3600, cluster_timeout=1800, splitter_timeout=1800)\n"),

    ("L3 max + resize/spatial reshape 자동화 허용",
     "performance_param(compiler_optimization_level=max)\n"
     "allocator_param(timeout=3600, cluster_timeout=1800, splitter_timeout=1800,\n"
     "                automatic_resize_reshapes=enabled, automatic_reshapes=enabled,\n"
     "                enable_auto_spatial_reshapes=True, enable_auto_spatial_collapse=True)\n"),

    ("L4 L3 + defuse 여유 + width splitter",
     "performance_param(compiler_optimization_level=max)\n"
     "allocator_param(timeout=3600, cluster_timeout=1800, splitter_timeout=1800,\n"
     "                automatic_resize_reshapes=enabled, automatic_reshapes=enabled,\n"
     "                enable_auto_spatial_reshapes=True, enable_auto_spatial_collapse=True,\n"
     "                max_auto_defuse=4, width_splitter_defuse=enabled, enable_defuse_with_slack=True)\n"),

    ("L5 L4 + resize1 명시 defuse(4분할)",
     "performance_param(compiler_optimization_level=max)\n"
     "allocator_param(timeout=3600, cluster_timeout=1800, splitter_timeout=1800,\n"
     "                automatic_resize_reshapes=enabled, enable_auto_spatial_reshapes=True,\n"
     "                max_auto_defuse=4, width_splitter_defuse=enabled)\n"
     f"{NAME}/resize1_fs, {NAME}/resize1_d0, {NAME}/resize1_d1, {NAME}/resize1_d2,"
     f" {NAME}/resize1_d3, {NAME}/resize1_dc = defuse({NAME}/resize1, 4, defuse_type=INPUT_FEATURES)\n"),

    # [2026-09-07 추가] L5는 컴파일이 즉시(2초) "Defuse of type INPUT_FEATURES is not
    # supported in resize1"로 거부됐다 — defuse_type 자체가 이 레이어에 안 맞았던 것.
    # tools/dfc/alloc_grammar_report.txt의 D2 probe(파싱만 검증, 컴파일은 미검증)에서
    # COMPUTE_LANES + axis=FEATURES는 같은 resize1에 대해 파싱이 통과했으므로 다음으로
    # 시도할 값. LHS 변수 개수(3개: fs/중간/c)는 probe에 실제로 통과한 문구를 그대로
    # 재사용한 것 — N을 4로 올리면 몇 개가 맞는지 검증된 바 없어 우선 N=2로 시도.
    ("L6 L4 + resize1 COMPUTE_LANES defuse(2분할, 미검증-파싱만 확인됨)",
     "performance_param(compiler_optimization_level=max)\n"
     "allocator_param(timeout=3600, cluster_timeout=1800, splitter_timeout=1800,\n"
     "                automatic_resize_reshapes=enabled, enable_auto_spatial_reshapes=True,\n"
     "                max_auto_defuse=4, width_splitter_defuse=enabled)\n"
     f"{NAME}/resize1_fs, {NAME}/resize1_2, {NAME}/resize1_c = "
     f"defuse({NAME}/resize1, 2, defuse_type=COMPUTE_LANES, axis=FEATURES)\n"),
]

# --probe-engine 에서 시험할 resize engine/hw_layer_type 후보 (파싱만)
ENGINE_CANDS = ["nn_core", "ppu", "sdma", "auto", "device", "post_infra", "dsp"]


def load_calib(calib_dir, n, size):
    import cv2
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
    files.sort()
    if not files:
        raise SystemExit(f"[에러] 캘리브 이미지 없음: {calib_dir}")
    step = max(1, len(files) // n)
    picked = files[::step][:n]
    arr = []
    for f in picked:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size))
        arr.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(f"  캘리브 {len(arr)}장 (풀 {len(files)}장)", flush=True)
    return np.asarray(arr, dtype=np.float32)


def probe_engine(har):
    from hailo_sdk_client import ClientRunner
    print("[probe] resize 의 engine / hw_layer_type 값 탐색 (파싱만)")
    base = BASE_ALLS.format(calibset=16)
    for kind in ("engine", "hw_layer_type"):
        for v in ENGINE_CANDS:
            snip = (f"{NAME}/resize1 = resize({NAME}/resize1, resize_shapes=[{SIZE},{SIZE}], "
                    f"resize_method=bilinear, {kind}={v})\n")
            try:
                r = ClientRunner(har=har) if har and os.path.exists(har) else ClientRunner(hw_arch="hailo8l")
                r.load_model_script(base + snip)
                print(f"  [OK  ] {kind}={v}")
            except Exception as e:
                print(f"  [실패] {kind}={v:12s} {str(e).splitlines()[0][:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="../artifacts/deeplab_cs_resize_argmax.onnx")
    ap.add_argument("--calib-dir", default="../../../../calib513/calib513")
    ap.add_argument("--calib-n", type=int, default=512)
    ap.add_argument("--out", default="../artifacts/deeplab_v3_mnv2_wo_dilation_cityscapes.hef")
    ap.add_argument("--har-dir", default="../artifacts")
    ap.add_argument("--arch", default="hailo8l")
    ap.add_argument("--probe-engine", action="store_true")
    ap.add_argument("--only", help="특정 사다리만 (예: L1)")
    a = ap.parse_args()

    from hailo_sdk_client import ClientRunner
    os.makedirs(a.har_dir, exist_ok=True)
    q_har = os.path.join(a.har_dir, f"{NAME}_quantized.har")

    if a.probe_engine:
        p = os.path.join(a.har_dir, f"{NAME}_parsed.har")
        probe_engine(p if os.path.exists(p) else None)
        return

    # ── 1) 양자화는 한 번만 ──
    if os.path.exists(q_har):
        print(f"[1/2] 기존 quantized HAR 재사용: {q_har}", flush=True)
    else:
        print(f"[1/2] 파싱 + 양자화 (1회만)  onnx={a.onnx}", flush=True)
        r = ClientRunner(hw_arch=a.arch)
        r.translate_onnx_model(a.onnx, NAME, start_node_names=[START], end_node_names=[END])
        outs = [(n, v.get("output_shapes")) for n, v in r.get_hn()["layers"].items()
                if v.get("type") == "output_layer"]
        print(f"      HN 출력: {outs}", flush=True)
        r.load_model_script(BASE_ALLS.format(calibset=a.calib_n))
        t0 = time.time()
        r.optimize(load_calib(a.calib_dir, a.calib_n, SIZE))
        r.save_har(q_har)
        print(f"      양자화 완료 {time.time()-t0:.0f}s -> {q_har}", flush=True)

    # ── 2) 컴파일만 옵션 바꿔가며 재시도 ──
    print("\n[2/2] 컴파일 사다리 시작. 각 단계 약 10분.\n", flush=True)
    for label, snip in LADDER:
        if a.only and not label.startswith(a.only):
            continue
        print("=" * 72, flush=True)
        print(f"== {label}", flush=True)
        print(snip.rstrip(), flush=True)
        print("=" * 72, flush=True)
        t0 = time.time()
        try:
            r = ClientRunner(har=q_har)
            r.load_model_script(snip)
            hef = r.compile()
            with open(a.out, "wb") as f:
                f.write(hef)
            print(f"\n★★★ 성공: {label}  ({time.time()-t0:.0f}s)", flush=True)
            print(f"    HEF: {a.out} ({os.path.getsize(a.out)/1e6:.2f} MB)", flush=True)
            # 성공한 할당 스크립트를 남긴다 (재현성 + 보고서 부록)
            try:
                agp = os.path.join(a.har_dir, f"{NAME}_autogen_alloc.alls")
                r.save_autogen_allocation_script(agp)
                print(f"    자동생성 할당 스크립트: {agp}", flush=True)
            except Exception as e:
                print(f"    (할당 스크립트 저장 실패: {e})", flush=True)
            open(os.path.join(a.har_dir, f"{NAME}_winning.alls"), "w", encoding="utf-8").write(snip)
            print("\n다음: hailortcli parse-hef 로 컨텍스트 수 확인", flush=True)
            return
        except Exception as e:
            msg = str(e).splitlines()
            print(f"\n[실패] {label}  ({time.time()-t0:.0f}s)", flush=True)
            for line in msg[:12]:
                print("   | " + line[:190], flush=True)
            print(flush=True)

    print("사다리 전부 실패. ladder.log 를 공유해 주세요.", flush=True)


if __name__ == "__main__":
    main()
