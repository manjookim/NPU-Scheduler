#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""학습된 mnv2_gtsrb.keras 를 GTSRB 공식 테스트셋(12,630장)으로 평가한다."""
import argparse, csv, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf, cv2

IMG = 224

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mnv2_gtsrb.keras")
    ap.add_argument("--data", default=os.path.expanduser("~/gtsrb"))
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    gt = os.path.join(a.data, "GT-final_test.csv")
    img_dir = os.path.join(a.data, "GTSRB", "Final_Test", "Images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(a.data, "Final_Test", "Images")
    rows = list(csv.DictReader(open(gt), delimiter=";"))
    print(f"  테스트 {len(rows)}장")

    model = tf.keras.models.load_model(a.model)
    correct = 0
    batch_x, batch_y = [], []

    def flush():
        nonlocal correct, batch_x, batch_y
        if not batch_x:
            return
        p = model.predict(np.stack(batch_x), verbose=0)
        correct += int((p.argmax(1) == np.array(batch_y)).sum())
        batch_x, batch_y = [], []

    for i, r in enumerate(rows):
        im = cv2.imread(os.path.join(img_dir, r["Filename"]))
        if im is None:
            continue
        im = cv2.cvtColor(cv2.resize(im, (IMG, IMG)), cv2.COLOR_BGR2RGB)
        batch_x.append((im.astype(np.float32) - 127.5) / 127.5)
        batch_y.append(int(r["ClassId"]))
        if len(batch_x) == a.batch:
            flush()
        if i and i % 2000 == 0:
            print(f"    {i}/{len(rows)} ...")
    flush()

    acc = correct / len(rows) * 100
    print(f"\n  GTSRB 공식 테스트 정확도 = {acc:.2f}%  ({correct}/{len(rows)})")
    open("gtsrb_test_accuracy.txt", "w").write(f"{acc:.4f}\n")

if __name__ == "__main__":
    main()
