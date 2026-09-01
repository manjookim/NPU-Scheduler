#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV2(alpha=1.0, 224x224) + GTSRB 43클래스 헤드 모델을 만들고 tflite로 내보낸다.
--smoke 를 주면 학습 없이 랜덤/ImageNet 초기값 그대로 내보낸다 (컴파일 경로 검증용).

[입력 규약] Hailo .alls 가 normalization([127.5]*3, [127.5]*3) 을 넣어주므로
HEF는 raw uint8[0,255]를 받고, 그 다음 단이 (x-127.5)/127.5 = [-1,1] 을 본다.
따라서 이 Keras 모델은 **[-1,1] 입력을 기대**하도록 만들고, 자체 Rescaling 층을 두지 않는다.
"""
import argparse, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

NUM_CLASSES = 43   # GTSRB
IMG = 224


def build(num_classes=NUM_CLASSES, weights="imagenet"):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), alpha=1.0, include_top=False,
        weights=weights, pooling="avg")
    inp = tf.keras.Input(shape=(IMG, IMG, 3), name="input_image")  # [-1,1] 기대
    x = base(inp, training=False)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inp, out, name="mobilenet_v2_1.0_gtsrb"), base


def to_tflite(model, path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    # 양자화는 Hailo DFC가 담당한다 — 여기서는 float32 그대로 내보낸다.
    conv.optimizations = []
    tfl = conv.convert()
    with open(path, "wb") as f:
        f.write(tfl)
    print(f"  -> tflite 저장: {path} ({os.path.getsize(path)/1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mobilenet_v2_1.0_gtsrb.tflite")
    ap.add_argument("--smoke", action="store_true", help="학습 없이 초기값으로 내보내기")
    a = ap.parse_args()

    model, _ = build()
    model.summary(line_length=100)
    to_tflite(model, a.out)

    # tflite 입출력 텐서 이름 확인 (yaml parser nodes 작성용)
    it = tf.lite.Interpreter(model_path=a.out)
    it.allocate_tensors()
    print("\n[tflite 입출력]")
    for d in it.get_input_details():
        print("  IN ", d["name"], d["shape"], d["dtype"].__name__)
    for d in it.get_output_details():
        print("  OUT", d["name"], d["shape"], d["dtype"].__name__)
