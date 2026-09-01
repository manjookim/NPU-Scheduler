#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tflite/onnx -> Hailo-8L HEF 컴파일 (DFC ClientRunner 직접 사용).
hailomz 는 model zoo 데이터셋/평가 체계에 묶여 있어서, GTSRB 같은 외부 데이터셋에는
DFC API를 직접 쓰는 편이 예측 가능하다.

사용:
  python compile_hailo.py --model x.tflite --name mobilenet_v2_1_0_gtsrb \
      --calib-dir /path/to/imgs --out x.hef [--calib-n 64] [--random-calib]
"""
import argparse, glob, os, sys
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# mobilenet_v2_1.0 원본 .alls 와 동일 (raw uint8 입력 -> in-net 정규화 [-1,1])
DEFAULT_ALLS = """\
normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
post_quantization_optimization(finetune, policy=disabled)
post_quantization_optimization(bias_correction, policy=enabled)
"""


def load_calib(calib_dir, n, size):
    import cv2
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.ppm"):
        files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
    files.sort()
    if not files:
        raise SystemExit(f"[에러] 캘리브레이션 이미지가 없습니다: {calib_dir}")
    step = max(1, len(files) // n)
    picked = files[::step][:n]
    arr = np.zeros((len(picked), size, size, 3), dtype=np.float32)
    for i, f in enumerate(picked):
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        arr[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  캘리브레이션: {len(picked)}장 (풀 {len(files)}장 중 균등 샘플)")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-dir")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--random-calib", action="store_true",
                    help="스모크 검증용 — 컴파일 경로만 확인, 양자화 품질은 무의미")
    ap.add_argument("--arch", default="hailo8l")
    ap.add_argument("--start-node")
    ap.add_argument("--end-node")
    ap.add_argument("--alls")
    a = ap.parse_args()

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(hw_arch=a.arch)

    kw = {}
    if a.start_node:
        kw["start_node_names"] = [a.start_node]
    if a.end_node:
        kw["end_node_names"] = [a.end_node]

    print(f"[1/4] 파싱: {a.model} (arch={a.arch})")
    if a.model.endswith(".onnx"):
        runner.translate_onnx_model(a.model, a.name, **kw)
    else:
        runner.translate_tf_model(a.model, a.name, **kw)
    print("      파싱 완료")

    alls = a.alls if a.alls else DEFAULT_ALLS
    if os.path.exists(alls):
        alls = open(alls).read()
    print(f"[2/4] model script 적용:\n{alls}")
    runner.load_model_script(alls)

    if a.random_calib:
        print("[3/4] 최적화(양자화) — **랜덤 캘리브레이션**")
        calib = np.random.randint(0, 256, (16, a.size, a.size, 3)).astype(np.float32)
    else:
        print("[3/4] 최적화(양자화)")
        calib = load_calib(a.calib_dir, a.calib_n, a.size)
    runner.optimize(calib)
    print("      최적화 완료")

    print("[4/4] 컴파일 -> HEF")
    hef = runner.compile()
    with open(a.out, "wb") as f:
        f.write(hef)
    print(f"  -> 저장: {a.out} ({os.path.getsize(a.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
