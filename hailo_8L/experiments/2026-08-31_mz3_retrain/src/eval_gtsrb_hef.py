#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
재학습한 mobilenet_v2_1.0 GTSRB HEF 를 **실기(Hailo-8L)에서** 돌려
양자화(int8) 후 정확도를 측정한다.

[왜 필요한가]
eval_gtsrb.py 가 낸 94.14% 는 양자화 **전** float `.keras` 모델의 값이다.
보고서/논문에 "재학습 정확도"로 쓰려면 실제 배포되는 HEF(int8)의 정확도여야 한다.
Hailo 공식 Model Zoo 표기법으로는 float 71.78 / hardware 70.9 처럼 둘을 나눠 적는다.

[전처리 규약 — eval_gtsrb.py 와 반드시 동일하게]
  cv2.imread(BGR) -> resize(224) -> BGR2RGB
  HEF 입력이 uint8 이면 그대로 넣는다. HEF 안의 normalization 층이
  (x-127.5)/127.5 = [-1,1] 로 바꿔주기 때문에 호스트에서 정규화하면 안 된다(이중 정규화).
  → 이 스크립트는 입력 vstream 포맷을 확인해서 uint8 이면 raw, float32 면 [-1,1] 로 맞춘다.

실행 (RPi, npu-rpi1):
  source ~/hailo_platform_venv/bin/activate
  python3 eval_gtsrb_hef.py --hef mobilenet_v2_1.0_gtsrb_sc.hef --data ~/gtsrb

준비물:
  · HEF 1개
  · GTSRB 공식 테스트셋 (PC 에서 scp)
      GT-final_test.csv
      GTSRB/Final_Test/Images/*.ppm   (12,630장, 약 88MB)
    scp -P 40021 -r ~/gtsrb rpi1@155.230.16.157:~/
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("[에러] opencv 가 없습니다: pip install opencv-python-headless")

from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams,
                            FormatType)

IMG = 224


def load_testset(data_dir):
    gt = os.path.join(data_dir, "GT-final_test.csv")
    img_dir = os.path.join(data_dir, "GTSRB", "Final_Test", "Images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(data_dir, "Final_Test", "Images")
    if not os.path.exists(gt):
        sys.exit(f"[에러] 정답 파일 없음: {gt}")
    rows = list(csv.DictReader(open(gt), delimiter=";"))
    return img_dir, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/gtsrb"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="빠른 확인용으로 N장만 (0=전체)")
    ap.add_argument("--out", default="gtsrb_hef_accuracy.txt")
    a = ap.parse_args()

    img_dir, rows = load_testset(a.data)
    if a.limit:
        rows = rows[: a.limit]
    print(f"테스트셋: {len(rows)}장  ({img_dir})")

    hef = HEF(a.hef)
    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        ng_params = ng.create_params()

        in_info = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        print(f"  입력 : {in_info.name}  shape={in_info.shape}")
        print(f"  출력 : {out_info.name} shape={out_info.shape}")
        n_cls = int(np.prod(out_info.shape))
        print(f"  출력 클래스 수 = {n_cls}  (GTSRB 재학습판이면 43)")
        if n_cls == 1001:
            print("  [경고] 1001 입니다 — 재학습 HEF 가 아니라 ImageNet 원본일 수 있습니다.")

        # HEF 가 uint8 입력을 받으면 호스트에서 정규화하지 않는다 (칩의 normalization 층이 처리)
        in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

        correct = total = 0
        t0 = time.time()
        with ng.activate(ng_params):
            with InferVStreams(ng, in_params, out_params) as pipe:
                bx, by = [], []

                def flush():
                    nonlocal correct, total, bx, by
                    if not bx:
                        return
                    arr = np.stack(bx).astype(np.uint8)
                    res = pipe.infer({in_info.name: arr})
                    logits = np.asarray(res[out_info.name]).reshape(len(bx), -1)
                    correct += int((logits.argmax(1) == np.array(by)).sum())
                    total += len(bx)
                    bx, by = [], []

                for i, r in enumerate(rows):
                    im = cv2.imread(os.path.join(img_dir, r["Filename"]))
                    if im is None:
                        continue
                    im = cv2.cvtColor(cv2.resize(im, (IMG, IMG)), cv2.COLOR_BGR2RGB)
                    bx.append(im)                       # uint8 그대로
                    by.append(int(r["ClassId"]))
                    if len(bx) == a.batch:
                        flush()
                    if i and i % 2000 == 0:
                        print(f"    {i}/{len(rows)} ... (현재 {correct/max(total,1)*100:.2f}%)",
                              flush=True)
                flush()

    acc = correct / max(total, 1) * 100
    el = time.time() - t0
    print(f"\n  HEF(int8) GTSRB 테스트 정확도 = {acc:.2f}%  ({correct}/{total})")
    print(f"  소요 {el:.1f}초  ({total/el:.1f} img/s — 정확도 측정용이라 성능 지표 아님)")
    open(a.out, "w").write(f"{acc:.4f}\n")
    print(f"  저장: {a.out}")
    print("\n  보고서 표기 예: float 94.14% / hardware(int8) %.2f%%" % acc)


if __name__ == "__main__":
    main()
