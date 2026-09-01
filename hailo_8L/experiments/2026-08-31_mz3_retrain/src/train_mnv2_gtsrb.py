#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mobilenet_v2_1.0 을 GTSRB(독일 교통표지판 43클래스)로 재학습(fine-tune)한다.
workload2.xlsx SMART TRAFFIC 시나리오 2 의 "교통표지판 분류" 워크로드용.

[왜 fine-tune인가] 목적은 ImageNet 재현이 아니라 **교통 도메인 적응**이다
(교수님 지적: "ImageNet은 교통 도메인 X"). 따라서 ImageNet 사전학습 가중치에서 출발해
GTSRB 43클래스 헤드만 새로 학습하고, 이어서 상위 블록을 낮은 lr로 함께 미세조정한다.

[입력 규약] Hailo .alls 가 normalization([127.5]*3,[127.5]*3) 을 넣어주므로 HEF는 uint8[0,255]를 받고
그래프 본체는 [-1,1] 을 본다. 그래서 여기서도 학습 입력을 [-1,1] 로 맞춘다 (자체 Rescaling 층 없음).

[검증 분할] GTSRB 학습셋은 같은 실물 표지판을 연속 촬영한 30장이 하나의 track 이다.
파일명이 `{track:05d}_{frame:05d}.ppm` 이므로 **track 단위로 분할**해야 누수가 없다.
프레임 단위로 섞으면 검증 정확도가 비현실적으로 높게 나온다.

사용:
  python train_mnv2_gtsrb.py --data ~/gtsrb --out mnv2_gtsrb.tflite \
      --epochs-head 3 --epochs-ft 5
"""
import argparse
import csv
import os
import random
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

IMG = 224
NUM_CLASSES = 43


# ────────────────────────── 데이터 로딩 ──────────────────────────
def load_train_index(root):
    """GTSRB/Final_Training/Images/000XX/*.ppm 목록과 라벨, track id를 모은다."""
    base = os.path.join(root, "GTSRB", "Final_Training", "Images")
    if not os.path.isdir(base):
        # zip 구조가 다를 수 있어 한 번 더 탐색
        for cand in (os.path.join(root, "Final_Training", "Images"), root):
            if os.path.isdir(cand) and any(d.isdigit() for d in os.listdir(cand)):
                base = cand
                break
    items = []
    for cls_dir in sorted(os.listdir(base)):
        cls_path = os.path.join(base, cls_dir)
        if not (os.path.isdir(cls_path) and cls_dir.isdigit()):
            continue
        label = int(cls_dir)
        for fn in sorted(os.listdir(cls_path)):
            if not fn.lower().endswith(".ppm"):
                continue
            track = fn.split("_")[0]  # 00000_00029.ppm -> 00000
            items.append((os.path.join(cls_path, fn), label, f"{label}_{track}"))
    if not items:
        raise SystemExit(f"[에러] 학습 이미지를 못 찾았습니다: {base}")
    return items


def load_test_index(root):
    """공식 테스트셋 + GT csv."""
    img_dir = os.path.join(root, "GTSRB", "Final_Test", "Images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(root, "Final_Test", "Images")
    gt = None
    for cand in ("GT-final_test.csv", os.path.join("GTSRB", "GT-final_test.csv")):
        p = os.path.join(root, cand)
        if os.path.exists(p):
            gt = p
            break
    if gt is None or not os.path.isdir(img_dir):
        print("  [경고] 공식 테스트셋을 못 찾아 테스트 평가를 건너뜁니다.")
        return []
    items = []
    with open(gt, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            items.append((os.path.join(img_dir, row["Filename"]), int(row["ClassId"]), "test"))
    return items


def split_by_track(items, val_ratio=0.2, seed=42):
    """track 단위 분할 — 같은 표지판의 연속 프레임이 train/val 양쪽에 걸치지 않게."""
    tracks = sorted({t for _, _, t in items})
    rng = random.Random(seed)
    rng.shuffle(tracks)
    n_val = int(len(tracks) * val_ratio)
    val_tracks = set(tracks[:n_val])
    train = [it for it in items if it[2] not in val_tracks]
    val = [it for it in items if it[2] in val_tracks]
    return train, val


def make_ds(items, batch, training, cache_path=None):
    """[-1,1] 정규화된 (image, label) 데이터셋."""
    paths = [p for p, _, _ in items]
    labels = [l for _, l, _ in items]

    def _load(path, label):
        raw = tf.io.read_file(path)
        # GTSRB는 .ppm — tf.io.decode_image가 ppm을 못 읽어서 numpy_function으로 처리
        img = tf.numpy_function(_decode_ppm, [path], tf.uint8)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (IMG, IMG))
        img = (img - 127.5) / 127.5          # [-1, 1]
        return img, label

    def _decode_ppm(path):
        import cv2
        p = path.decode() if isinstance(path, bytes) else path
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            im = np.zeros((IMG, IMG, 3), np.uint8)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_path is not None:
        ds = ds.cache(cache_path)
    if training:
        ds = ds.shuffle(4096, seed=42)
        # 표지판은 좌우 반전하면 의미가 달라지는 클래스가 있으므로 flip은 쓰지 않는다.
        ds = ds.map(lambda x, y: (tf.image.random_brightness(x, 0.2), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ────────────────────────── 모델 ──────────────────────────
def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), alpha=1.0, include_top=False,
        weights="imagenet", pooling="avg")
    inp = tf.keras.Input(shape=(IMG, IMG, 3), name="input_image")
    x = base(inp, training=False)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inp, out, name="mobilenet_v2_1_0_gtsrb"), base


def to_tflite(model, path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = []   # 양자화는 Hailo DFC가 담당
    open(path, "wb").write(conv.convert())
    print(f"  -> tflite 저장: {path} ({os.path.getsize(path)/1e6:.2f} MB)")


# ────────────────────────── main ──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/gtsrb"))
    ap.add_argument("--out", default="mnv2_gtsrb.tflite")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help=">0이면 학습 샘플 수 제한(빠른 검증용)")
    a = ap.parse_args()

    print("=== GTSRB 인덱싱 ===")
    items = load_train_index(a.data)
    train_it, val_it = split_by_track(items)
    if a.limit:
        train_it = train_it[:a.limit]
        val_it = val_it[:max(1, a.limit // 5)]
    test_it = load_test_index(a.data)
    print(f"  train {len(train_it)} / val {len(val_it)} / test {len(test_it)}  (클래스 {NUM_CLASSES})")

    train_ds = make_ds(train_it, a.batch, training=True)
    val_ds = make_ds(val_it, a.batch, training=False)

    model, base = build_model()

    # ── 1단계: 백본 동결, 헤드만 학습 ──
    print("\n=== 1단계: 헤드 학습 (백본 동결) ===")
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_head, verbose=2)

    # ── 2단계: 상위 블록 해제 후 낮은 lr로 미세조정 ──
    print("\n=== 2단계: 상위 블록 미세조정 ===")
    base.trainable = True
    for layer in base.layers[:100]:      # 하위 100개 층은 계속 동결
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_ft, verbose=2)

    # ── 공식 테스트셋 평가 ──
    if test_it:
        print("\n=== 공식 테스트셋 평가 ===")
        test_ds = make_ds(test_it, a.batch, training=False)
        loss, acc = model.evaluate(test_ds, verbose=2)
        print(f"  GTSRB test accuracy = {acc*100:.2f}%  (loss {loss:.4f})")
        with open("gtsrb_test_accuracy.txt", "w") as f:
            f.write(f"{acc*100:.4f}\n")

    model.save("mnv2_gtsrb.keras")
    to_tflite(model, a.out)


if __name__ == "__main__":
    main()
