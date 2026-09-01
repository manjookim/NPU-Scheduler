---
name: mz3-retrain-baseline-guide
description: "SMART TRAFFIC 시나리오2 3모델(ssd_mobilenet_v1 / deeplab_v3_mobilenet_v2_wo_dilation / mobilenet_v2_1.0) 재학습·HEF 컴파일 가이드라인 + Default 베이스라인 실험 설계(8L/10H) + 트러블슈팅"
metadata:
  node_type: memory
  type: reference
  created: 2026-08-31
---

# Model Zoo 3모델 재학습·컴파일 & Default 베이스라인 가이드

대상: `workload2.xlsx` → Hailo Model Zoo → **SMART TRAFFIC 시나리오 2**

| TASK | 모델 | xlsx 기준 데이터셋 | 입력 | 출력 | Params |
|---|---|---|---|---|---|
| 차량 검출 | `ssd_mobilenet_v1` | COCO | 300x300x3 | 90x8x1 | 6.79M |
| 차선/도로 영역 분할 | `deeplab_v3_mobilenet_v2_wo_dilation` | Cityscapes | 513x513x3 | 513x513x1 | 2.10M |
| 교통표지판 분류 | `mobilenet_v2_1.0` | GTSRB | 224x224x3 | 1001 | 3.49M |

타겟 보드: **Hailo-8L (npu-rpi1)** + **Hailo-10H (npu-rpi5)**

---

## 0. 시작 전 반드시 알아야 할 사실 3가지

공식 저장소를 확인한 결과, 계획 단계의 전제 몇 개가 실제와 다릅니다.

### (1) 이 3모델에는 Hailo 공식 재학습 Docker가 **없다**

`docs/RETRAIN_ON_CUSTOM_DATASET.rst`(v2.19.0 / master 모두 확인)에 등재된 재학습 아키텍처는
다음이 전부입니다:

> YOLOv3/v4/v5/v8/X, DAMO-YOLO, NanoDet, CenterPose, MSPN, **FCN**, YOLACT, YOLOv8_seg, ArcFace

`training/` 디렉터리에도 `arcface / centerpose / damoyolo / fcn / mspn / nanodet / yolact / yolov3 / yolov4 / yolov5 /
yolov8 / yolox` 12개만 있고 **ssd, deeplab, mobilenet 폴더는 존재하지 않습니다**
(v2.19.0 태그에서 경로 프로브로 실측 확인).

→ 따라서 "Hailo Model Zoo 재학습 가이드라인 기반"의 실제 의미는:

```
[원 프레임워크에서 직접 학습]  ← Hailo 관여 없음, 사용자 책임
        ↓ ONNX / TFLite export
[Model Zoo의 공식 yaml + alls 를 그대로 재사용]  ← 여기서부터가 "Model Zoo 가이드라인"
        ↓ hailomz parse → optimize → compile
[HEF]
```

즉 **학습은 각 모델의 원본 저장소 방식**, **컴파일만 Model Zoo 규약**입니다.
이게 mobilenet_v2를 DFC 직접 호출로 처리했던 이유이기도 합니다(그 판단은 결과적으로 옳았음).

> 세그멘테이션에 재학습 Docker가 필요하다면 유일한 공식 선택지는 **FCN**(`fcn8_resnet_v1_18`,
> Cityscapes)입니다. DeepLab 재학습 리스크를 피하고 싶으면 Seg 모델을 FCN으로 바꾸는 것이
> "공식 가이드라인 준수" 측면에선 가장 안전합니다. 다만 workload2.xlsx의 모델 선정이 바뀝니다.

### (2) `deeplab_v3_mobilenet_v2_wo_dilation`의 Model Zoo 학습셋은 **Cityscapes가 아니라 PASCAL VOC**

`cfg/networks/deeplab_v3_mobilenet_v2_wo_dilation.yaml`:

```yaml
base: [base/pascal.yaml]
info:
  training_data: pascal voc train2012
  validation_data: pascal voc val2012
  eval_metric: mIoU
  full_precision_result: 71.46
```

Model Zoo의 Cityscapes 세그멘테이션 모델은 `fcn8_resnet_v1_18`, `stdc1`, `segformer_b0_bn` 3종뿐입니다.
workload2.xlsx의 "Cityscapes"는 **목표 상태**이지 현재 상태가 아니며, 같은 시트의
"고려할 부분 1. Cityscapes, GTSRB 재학습 및 컴파일 필요"와 정확히 일치합니다.

정리하면 재학습이 실제로 필요한 모델은 **2개**입니다.

| 모델 | MZ 현재 학습셋 | 목표 학습셋 | 재학습 필요? |
|---|---|---|---|
| ssd_mobilenet_v1 | COCO | COCO | **불필요** (사전학습 그대로) |
| deeplab_v3_..._wo_dilation | PASCAL VOC (21 cls) | Cityscapes (19 cls) | **필요** |
| mobilenet_v2_1.0 | ImageNet (1001 cls) | GTSRB (43 cls) | **완료** (94.14%) |

### (3) Hailo-8L용과 Hailo-10H용은 **Model Zoo / DFC 버전 라인이 다르다**

master(= v5.4.0)의 세 모델 yaml `supported_hw_arch`에서 hailo8 / hailo8l이 **빠졌습니다**:

| yaml | master(v5.4.0) supported_hw_arch | v2.19.0 supported_hw_arch |
|---|---|---|
| ssd_mobilenet_v1 | hailo15h, hailo15l, hailo10h | **hailo8, hailo8l** |
| deeplab_v3_..._wo_dilation | hailo15h, hailo15l, hailo10h | **hailo8, hailo8l** |
| mobilenet_v2_1.0 | hailo15h, hailo15l, hailo10h | **hailo8, hailo8l** |

→ **두 벌의 컴파일 환경이 필요합니다.**

| 보드 | Model Zoo | DFC | HailoRT | 비고 |
|---|---|---|---|---|
| Hailo-8L (npu-rpi1) | **v2.19.0 이하** | 3.3x (설치본 3.33.0) | 4.23.0 | `docs/setup.md` 환경 그대로 |
| Hailo-10H (npu-rpi5) | **v5.x (master)** | 5.x | 5.3.0 | 별도 venv로 격리할 것 |

같은 머신에 DFC 3.33.0과 5.x를 동시에 넣으면 충돌하므로 **venv를 반드시 분리**하세요
(`~/hailo_venv_dfc33`, `~/hailo_venv_dfc54` 식).

### (0-4) 사전 컴파일 HEF는 두 보드 모두 존재 (재학습 전 베이스라인용)

| 모델 | Hailo-8L (v2.19.0) FPS b1 | Hailo-10H (v5.4.0) FPS b1 |
|---|---|---|
| ssd_mobilenet_v1 | 356 (mAP 22.4 HW) | 463 (mAP 22.2 HW) |
| deeplab_v3_..._wo_dilation | 60.6 (mIoU 71.1 HW) | 372 (mIoU 71.2 HW) |
| mobilenet_v2_1.0 | 1738 (top1 70.9 HW) | 2798 (top1 71.0 HW) |

> workload2.xlsx의 FPS 컬럼(356 / 60.6 / 1738)은 **Hailo-8L 공식 수치와 정확히 일치**합니다.
> 즉 xlsx는 8L 기준으로 작성된 표이며, 10H 비교 시 그대로 쓰면 안 됩니다.

S3 경로 규칙:
`https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<ver>/<arch>/<name>.hef`
(8L은 `v2.18.0`/`v2.19.0` + `hailo8l`, 10H는 `v5.4.0` + `hailo10h`)

> **8L↔10H 격차 주의**: deeplab이 60.6 → 372 FPS (6.1배)로 다른 두 모델보다 격차가 큽니다.
> 8L에서 multi-context(3)로 쪼개지던 모델이 10H에선 단일/저컨텍스트로 들어갔을 가능성이 높으므로,
> 두 보드 결과를 비교할 때 **반드시 `hailortcli parse-hef`로 컨텍스트 수를 먼저 확인**하고
> 그 값을 결과 표에 컬럼으로 남기세요. 이게 이번 워크로드에서 가장 중요한 교란변수입니다.

---

## 1. 모델별 재학습 및 컴파일(HEF 생성) 가이드라인

### 1-0. 공통 파이프라인

```
① 학습 (원 프레임워크)
② export → ONNX(권장) 또는 TFLite
③ hailomz parse    --ckpt <파일> --yaml <cfg> --hw-arch <arch>
                   [--start-node-names / --end-node-names]   → .har (HAR: Hailo ARchive)
④ hailomz optimize --har <har> --calib-path <실제 도메인 이미지 폴더>
                   [--classes N]                              → quantized .har
⑤ hailomz compile  --har <quantized.har> --hw-arch <arch>     → .hef
⑥ hailomz eval     --har <har> --target emulator | hardware   → 정확도 검증
```

**전 단계 공통 원칙 4가지**

1. **`--calib-path`는 반드시 실제 배포 도메인 이미지**를 쓸 것. ImageNet/COCO 캘리브레이션으로
   교통 도메인 모델을 양자화하면 mIoU/accuracy가 크게 떨어집니다. (mobilenet_v2 때 GTSRB
   실제 이미지 1024장으로 캘리브레이션한 것이 정답이었음)
2. **캘리브셋 크기는 최소 64장, 권장 1024장.** deeplab의 공식 alls는 `calibset_size=64`로
   잡혀 있지만 이건 PASCAL VOC 기준입니다. Cityscapes로 바꾸면 늘리세요.
3. **정규화는 반드시 alls의 `normalization()`에 맡길 것.** 학습 코드에 Rescaling 층을 넣으면
   이중 정규화가 됩니다. 세 모델 모두 `normalization([127.5]*3, [127.5]*3)` = 입력 uint8[0,255],
   네트 내부 [-1,1]. (mobilenet_v2 재학습 스크립트가 이미 이 규약을 지키고 있음)
4. **`hailomz parse`가 잘리는 지점(start/end node)이 정확도·구조를 결정**합니다. 재학습 모델은
   원본과 노드 이름이 달라지므로 `--start-node-names` / `--end-node-names`를 직접 줘야 합니다.

---

### 1-1. `ssd_mobilenet_v1` (Detection)

#### Model Zoo 공식 정보

```yaml
# cfg/networks/ssd_mobilenet_v1.yaml (v2.19.0)
base: [base/coco.yaml, base/ssd.yaml]
postprocessing:
  device_pre_post_layers: {argmax: false, softmax: false, bilinear: false, nms: true}
  meta_arch: ssd
  anchors: {predefined: true}
paths:
  network_path: [.../ssd/ssd_mobilenet_v1/pretrained/2023-07-18/mobilenet_ssd.tflite]
parser:
  nodes:
    - FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6;...   # start
    - [BoxPredictor_0/BoxEncodingPredictor/BiasAdd,                  # end (12개)
       BoxPredictor_0/ClassPredictor/BiasAdd, ... BoxPredictor_5/...]
info:
  framework: tensorflow
  full_precision_result: 23.19   # mAP
  source: tensorflow/models research/object_detection TF1 detection zoo
```

```
# cfg/alls/generic/ssd_mobilenet_v1.alls (v2.19.0)
new_normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
post_quantization_optimization(finetune, policy=enabled, dataset_size=4000, epochs=8,
    learning_rate=0.0001, loss_layer_names=[bbox_decoder13, conv14, ... conv16],
    loss_factors=[0.1, ... 1.0, ...], loss_types=[l2 x18])
model_optimization_flavor(compression_level=0)
nms_postprocess(meta_arch=ssd)
```

#### 권장: **재학습하지 않는다**

- xlsx 기준 데이터셋이 COCO이고, Model Zoo 사전학습 모델이 이미 COCO train2017입니다.
- 원본이 **TF1 Object Detection API**(`tf1_detection_zoo`) 산출물입니다. TF1.15 + 구버전
  object_detection 패키지 환경을 재현하는 비용이 매우 큽니다(파이썬 3.7 이하, protobuf 구버전 등).
- 시나리오 2의 서사에서 SSD는 "차량 검출"이고 COCO에 car/bus/truck/traffic light가 이미 있습니다.

#### 그래도 교통 도메인으로 재학습해야 한다면

| 항목 | 내용 |
|---|---|
| 데이터셋 | BDD100K(차량+신호등+표지판, 10만장), KITTI, Mapillary Traffic Sign |
| 학습 경로 A | TF2 Object Detection API의 `ssd_mobilenet_v1_fpn`... **주의: FPN 있는 변형은 다른 모델** |
| 학습 경로 B | TF1 OD API 컨테이너(`tensorflow/tensorflow:1.15.5-gpu-py3`)에서 원본 pipeline.config 수정 |
| 학습 경로 C | torchvision `ssdlite320_mobilenet_v3` 등으로 대체 — **모델이 바뀌므로 xlsx 수정 필요** |

#### 컴파일 시 핵심 주의점

1. **`nms_postprocess(meta_arch=ssd)` + `anchors: predefined: true`**
   → 앵커와 NMS 설정이 **COCO 90클래스 기준으로 고정**되어 있습니다. 클래스 수를 바꾸면
   출력 shape `90x8x1`의 90이 달라지므로 NMS config를 새로 만들어야 합니다.
   `hailomz optimize --classes <N>` 으로 클래스 수를 넘기거나,
   `nms_postprocess("<custom>_nms_config.json", meta_arch=ssd, engine=...)` 형태로
   config JSON을 직접 지정합니다(`cfg/postprocess_config/` 의 기존 json을 템플릿으로 복사).
   → **DFC 버전마다 인자 형식이 다르니 `hailo tutorial`의 model script 레퍼런스로 반드시 확인.**
2. **SSD는 on-chip NMS(`engine=nn_core`)를 지원하는 몇 안 되는 아키텍처**입니다
   (PROJECT_SUMMARY.md §6에 이미 조사 완료: SSD/CenterNet은 nn_core 지원).
   → 이번 워크로드에서 SSD만 후처리가 칩에서 돌고 deeplab/mobilenet은 다릅니다.
   **베이스라인 표에 "후처리 위치(NPU/CPU/없음)" 컬럼을 반드시 넣으세요.** 안 넣으면
   세 모델 latency를 같은 선상에서 비교했다는 지적을 받습니다.
3. `post_quantization_optimization(finetune, policy=enabled, dataset_size=4000)` — 공식 alls가
   **QFT(quantization fine-tuning)를 켜 놓았습니다.** 캘리브셋이 4000장 필요하고 GPU에서
   수십 분~수 시간 걸립니다. 시간이 없으면 `policy=disabled`로 끄되, **끈 사실을 실험 조건에
   기록**하세요(mAP가 1~2점 떨어질 수 있음).
4. 전처리는 letterbox가 아니라 **단순 resize(300x300) + BGR→RGB**. (`mz3_sched_bench.cpp`에
   이미 반영되어 있음)

---

### 1-2. `deeplab_v3_mobilenet_v2_wo_dilation` (Segmentation) — **가장 손이 많이 감**

#### Model Zoo 공식 정보

```yaml
# cfg/networks/deeplab_v3_mobilenet_v2_wo_dilation.yaml (v2.19.0)
base: [base/pascal.yaml]
paths:
  network_path: [.../Segmentation/Pascal/deeplab_v3_mobilenet_v2/pretrained/
                 2025-01-20/deeplab_v3_mobilenet_v2_wo_dilation_sim.onnx]
parser:
  nodes: [MobilenetV2/Conv/Conv2D, ArgMax]      # start, end
  normalization_params:
    normalize_in_net: true
    mean_list: [127.5, 127.5, 127.5]
    std_list:  [127.5, 127.5, 127.5]
postprocessing:
  device_pre_post_layers: {max_finder: false, bilinear: true, argmax: true, softmax: false}
preprocessing: {network_type: segmentation, meta_arch: fcn_resnet}
info:
  training_data: pascal voc train2012
  full_precision_result: 71.46   # mIoU
  source: tensorflow/models research/deeplab
```

```
# cfg/alls/generic/deeplab_v3_mobilenet_v2_wo_dilation.alls (v2.19.0)
normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
model_optimization_config(calibration, batch_size=1, calibset_size=64)
pre_quantization_optimization(equalization, policy=disabled)
```

#### `wo_dilation`이 무엇인지 — 이 실험의 핵심 변수

- DeepLabV3의 표준 구성은 backbone 후반부와 ASPP에서 **atrous(dilated) convolution**을 씁니다.
- Hailo 하드웨어에서 dilated conv는 비효율적이라, Model Zoo는 **dilation을 제거한 변형**을
  따로 만들어 배포합니다.

| 변형 | mIoU (float / HW) | Hailo-8L FPS b1 |
|---|---|---|
| `deeplab_v3_mobilenet_v2` (표준) | 76.0 / 74.6 | 51.8 |
| `deeplab_v3_mobilenet_v2_wo_dilation` | **71.5 / 71.1** | **60.6** |

→ **정확도 4.5점을 내주고 처리량 17%를 얻은 모델**입니다.

**재학습 시 가장 중요한 것: dilation을 그대로 두고 학습하면 "wo_dilation 모델이 아닙니다."**
TF `research/deeplab` 학습 시 `--atrous_rates` 를 주지 않고 `--output_stride=32`(dilation 없이
다운샘플링으로 처리)로 맞춰야 원본과 같은 계열이 됩니다. 이 설정을 실험 노트에 반드시 남기세요.
(학습 후 ONNX를 `hailomz parse` 했을 때 컨텍스트 수가 원본과 크게 다르면 dilation이 남아있다는 신호)

#### Cityscapes 재학습 절차

```bash
# 1) 데이터셋 (회원가입 필요, gtFine + leftImg8bit)
Cityscapes/
├── gtFine/{train,val,test}/
└── leftImg8bit/{train,val,test}/

# 2) 학습 — TF research/deeplab
git clone https://github.com/tensorflow/models.git
cd models/research/deeplab

# 2-a) trainId 라벨 생성이 선행되어야 함 (gtFine의 labelIds → labelTrainIds)
#      cityscapesScripts 의 createTrainIdLabelImgs 를 먼저 돌린다.
#      보통 datasets/convert_cityscapes.sh 가 이 과정 + build_cityscapes_data.py 를 묶어 실행함.
pip install cityscapesscripts
bash datasets/convert_cityscapes.sh      # 내부에서 build_cityscapes_data.py 호출
python train.py \
  --logtostderr --training_number_of_steps=90000 \
  --train_split="train_fine" \
  --model_variant="mobilenet_v2" \
  --output_stride=32 \            # ← dilation 없이. atrous_rates 지정하지 않음
  --train_crop_size="513,513" \
  --train_batch_size=8 \
  --dataset="cityscapes" \
  --tf_initial_checkpoint=<mobilenet_v2 imagenet ckpt> \
  --train_logdir=<dir> --dataset_dir=<tfrecord dir>

# 3) export (ArgMax까지 포함된 frozen graph)
python export_model.py \
  --checkpoint_path=<ckpt> --export_path=<frozen.pb> \
  --model_variant="mobilenet_v2" --output_stride=32 \   # export_model.py 기본값은 8이므로 반드시 명시
  --num_classes=19 \              # ← Cityscapes trainId 19클래스
  --crop_size=513 --crop_size=513 --inference_scales=1.0

# 4) frozen.pb → ONNX (tf2onnx) → simplify (onnxsim)
python -m tf2onnx.convert --graphdef frozen.pb --output deeplab_cs.onnx \
  --inputs ImageTensor:0 --outputs ArgMax:0
python -m onnxsim deeplab_cs.onnx deeplab_cs_sim.onnx
```

**PyTorch 대안**: `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large` — 백본이
MobileNetV3라 **다른 모델**입니다. workload2.xlsx 표를 고쳐야 하므로 권장하지 않습니다.

#### 컴파일

```bash
# 클래스 수가 21 → 19로 바뀌므로 전용 yaml을 새로 만든다
cp hailo_model_zoo/cfg/networks/deeplab_v3_mobilenet_v2_wo_dilation.yaml \
   hailo_model_zoo/cfg/networks/deeplab_v3_mnv2_wo_dilation_cityscapes.yaml
# 안에서 base 를 base/pascal.yaml → base/cityscapes.yaml 로 바꾸고
# network_name / alls_script / info.training_data 수정

python hailo_model_zoo/datasets/create_cityscapes_tfrecord.py calib --data /path/to/Cityscapes/
python hailo_model_zoo/datasets/create_cityscapes_tfrecord.py val   --data /path/to/Cityscapes/

hailomz parse    deeplab_v3_mnv2_wo_dilation_cityscapes --hw-arch hailo8l \
                 --ckpt deeplab_cs_sim.onnx \
                 --start-node-names MobilenetV2/Conv/Conv2D --end-node-names ArgMax
hailomz optimize deeplab_v3_mnv2_wo_dilation_cityscapes --har <har> \
                 --calib-path /path/to/Cityscapes/leftImg8bit/train
hailomz compile  deeplab_v3_mnv2_wo_dilation_cityscapes --har <quantized.har> --hw-arch hailo8l
hailomz eval     deeplab_v3_mnv2_wo_dilation_cityscapes --har <har> --target emulator
```

#### 주의점

1. **`device_pre_post_layers`에 `bilinear: true, argmax: true`** — 업샘플과 argmax를 칩이
   처리합니다. 그래서 출력이 `513x513x1`(클래스 확률맵이 아니라 **라벨 인덱스 맵**)입니다.
   export할 때 반드시 **ArgMax 노드를 end node로 포함**해야 원본과 같은 구조가 됩니다.
   ArgMax를 빼고 자르면 출력이 `513x513x19` float가 되어 **호스트 후처리가 생기고
   latency 비교가 무의미해집니다.**
2. **8L에서 multi-context 모델**(`mz3_sched_bench.cpp` 주석 기준 3 contexts). 재학습 후
   컨텍스트 수가 바뀔 수 있으므로 `hailortcli parse-hef`로 **컴파일 직후 반드시 확인**하고
   원본(3)과 다르면 그 사실을 기록하세요. 컨텍스트 수가 스케줄링 실험의 핵심 변수입니다.
3. `pre_quantization_optimization(equalization, policy=disabled)` — 공식 alls가 equalization을
   **끕니다**. 재학습 모델에서도 그대로 두세요(켜면 이 구조에서 정확도가 나빠집니다).
4. 전처리는 **단순 resize(513x513)**, letterbox 아님.

---

### 1-3. `mobilenet_v2_1.0` (Classification) — **완료 상태 검증**

`hailo_8L/experiments/2026-08-31_mz3_retrain/` 에 이미 산출물이 있습니다.

| 항목 | 값 |
|---|---|
| 데이터셋 | GTSRB 43클래스, train 31,379 / val 7,830 (track 단위 분할) |
| 학습 | head 2 epoch + fine-tune 3 epoch |
| **GTSRB 공식 테스트셋 정확도** | **94.14%** (12,630장) |
| 캘리브레이션 | GTSRB 실제 이미지 1024장 |
| 산출물 | `mobilenet_v2_1.0_gtsrb.tflite` (9.09MB), `mobilenet_v2_1.0_gtsrb_sc.hef` (5.8MB) |

#### 공식 alls 대조 — 일치 확인됨

| 공식 `mobilenet_v2_1.0.alls` (v2.19.0) | 우리 `mobilenet_v2_1.0_gtsrb.alls` |
|---|---|
| `normalization1 = normalization([127.5]*3, [127.5]*3)` | 동일 |
| `post_quantization_optimization(finetune, policy=disabled)` | 동일 |
| `post_quantization_optimization(bias_correction, policy=enabled)` | 동일 |
| — | `resources_param(max_utilization=1.0, ...)` **추가** |
| — | `context_switch_param(mode=disabled)` **추가** |

→ 양자화 설정은 공식과 100% 일치합니다. 추가한 2줄은 **single-context 강제**용입니다.

#### 남은 확인 사항 (실험 전 필수)

- [ ] `hailortcli parse-hef mobilenet_v2_1.0_gtsrb_sc.hef` — 컨텍스트 1개 확인
- [ ] **공식 `mobilenet_v2_1.0.hef`(ImageNet)도 원래 single-context인지 확인.**
      만약 원본이 이미 single-context라면 우리가 추가한 `context_switch_param(mode=disabled)`은
      무해하지만, 원본이 multi-context였다면 **우리 HEF만 조건이 다른 것**이 되어
      사전학습 vs 재학습 비교가 오염됩니다.
- [ ] 출력이 1001 → **43**으로 바뀌었으므로 `mz3_sched_bench.cpp`의 출력 vstream 처리 확인
- [x] `--calib-n 1024` 클래스 편중 여부 — **문제 없음.** `compile_hailo.py::load_calib()`이
      `files.sort()` 후 `files[::step][:n]`(step = 전체/n)로 **균등 stride 샘플링**을 합니다.
      GTSRB `Final_Training/Images/00000~00042/` 구조에서 39,209장을 1024장으로 뽑으면
      step=38이라 43클래스 전체가 고르게 들어갑니다.
- [ ] **양자화 후 정확도를 아직 안 쟀음.** `gtsrb_test_accuracy.txt`의 94.14%는 **float
      `.keras` 모델** 기준(`eval_gtsrb.py`)입니다. HEF(int8) 정확도는 미측정 상태.
      → `hailomz eval --target emulator` 또는 실기 HEF 추론으로 **재측정 필요**.
      논문 표에 "재학습 정확도"를 쓰려면 반드시 HW 정확도여야 합니다.
- [ ] `finish_pipeline.sh`는 `--alls`를 주지 않아 `DEFAULT_ALLS`(공식 3줄)로 컴파일합니다.
      그런데 산출물은 `_sc.hef` + `resources_param`/`context_switch_param`이 든 alls 파일 —
      **즉 별도의 2차 컴파일이 있었고 그 명령이 스크립트에 안 남아 있습니다.**
      재현성을 위해 실제 사용한 명령을 `src/`에 스크립트로 남기세요.

---

### 1-4. `hailomz` 명령어 요약

| 명령 | 용도 | 자주 쓰는 플래그 |
|---|---|---|
| `hailomz info <model>` | 모델 메타정보 확인 | |
| `hailomz parse <model>` | 원본 → HAR | `--hw-arch hailo8l` `--ckpt` `--start-node-names` `--end-node-names` |
| `hailomz optimize <model>` | 양자화 | `--har` `--calib-path` `--classes` `--performance` `--resize` |
| `hailomz compile <model>` | HAR → HEF | `--har` `--hw-arch` `--performance` |
| `hailomz eval <model>` | 정확도 평가 | `--har` `--target emulator\|hardware\|<device_id>` `--data-count` `--visualize` |

TFRecord 생성:

```bash
python hailo_model_zoo/datasets/create_coco_tfrecord.py val2017
python hailo_model_zoo/datasets/create_coco_tfrecord.py calib2017
python hailo_model_zoo/datasets/create_cityscapes_tfrecord.py val   --data /path/to/Cityscapes/
python hailo_model_zoo/datasets/create_cityscapes_tfrecord.py calib --data /path/to/Cityscapes/
python hailo_model_zoo/datasets/create_imagenet_tfrecord.py val --img /path/to/imagenet/val/
```

> **`--performance` 플래그는 이번 실험에서 쓰지 마세요.** 컴파일러가 자원 배분을 자동 탐색해
> "기본값 조건"이 깨집니다. `model_optimization_flavor(compression_level=0)` 기본값 유지.

### 1-5. hailomz ↔ DFC 직접 호출 대응표 (기존 `compile_hailo.py`와의 매핑)

| hailomz | DFC ClientRunner 직접 | 비고 |
|---|---|---|
| `hailomz parse --ckpt x.onnx` | `runner.translate_onnx_model(...)` / `translate_tf_model` | |
| yaml의 `parser.nodes` | `start_node_names=` / `end_node_names=` 인자 | |
| alls 파일 | `runner.load_model_script(alls_string)` | |
| `hailomz optimize --calib-path` | `runner.optimize(calib_numpy_array)` | 직접 방식은 numpy 배열을 만들어야 함 |
| `hailomz compile` | `runner.compile()` → bytes 저장 | |
| `hailomz eval` | **없음** — 직접 평가 코드를 짜야 함 (`eval_gtsrb.py`가 그 역할) |

→ **직접 방식의 유일한 실질적 단점은 `eval`이 없다는 것**입니다. 공식 데이터셋(COCO/Cityscapes/
ImageNet)으로 평가할 거라면 hailomz가, GTSRB 같은 외부 데이터셋이면 직접 방식이 편합니다.

---

## 2. Default(파라미터 미조정) 베이스라인 실험 설계

### 2-1. "Default"의 조작적 정의 — 체크리스트

베이스라인이 흔들리는 지점은 **런타임 파라미터가 아니라 컴파일 옵션과 측정 코드 구조**입니다.
아래 4개 층위를 전부 고정해야 "기본값 조건"이라고 말할 수 있습니다.

#### (A) 런타임 스케줄러 — setter를 **호출하지 않는다**

| 파라미터 | HailoRT 기본값 | 확인 방법 |
|---|---|---|
| priority | 16 (NORMAL) | `set_scheduler_priority()` 미호출 |
| threshold | 1 | `set_scheduler_threshold()` 미호출 |
| timeout | 0 ms | `set_scheduler_timeout()` 미호출 |
| batch_size | 0 (auto) | `configure_params` 미변경 |
| scheduling scheme | ROUND_ROBIN (유일) | — |

- [ ] 세 setter 중 **하나도** 호출하지 않는다 (0을 넣는 것과 호출하지 않는 것은 다름)
- [ ] `power_mode`도 건드리지 않는다
- [ ] 예외 1개: **output/input vstream의 host 측 timeout만 크게** 잡는다.
      이건 스케줄링 정책이 아니라 "굶은 모델이 10초 만에 실패해 프레임이 유실되는 것"을 막는
      호스트 안전장치이며, `infer_scheduler.cpp` 이래 모든 실험에서 동일하게 유지해 온 관례입니다.
      → **이 예외를 논문/보고서에 명시**할 것.
- [ ] 기존 발견 반영: `timeout=0`이면 threshold가 사실상 무력화되고, priority 차이 15 이상이면
      starvation이 발생합니다. **기본값(전 모델 priority 16 동일)에서는 starvation이 구조적으로
      발생할 수 없으므로**, seg latency가 ~10,000ms로 찍히면 그건 starvation이 아니라 다른 문제입니다.

#### (B) 컴파일 측 기본값 — 세 모델의 조건을 맞춘다

- [ ] `hailomz` 사용 시 **`--performance` 플래그 금지** (자원 배분 자동 탐색 → 조건 붕괴)
- [ ] alls에 `resources_param` / `allocator_param` / `performance_param` 추가 금지
      → **단, 현재 `mobilenet_v2_1.0_gtsrb.alls`에는 `resources_param`과
      `context_switch_param(mode=disabled)`이 들어 있습니다.** 이대로면 이 모델만 조건이 다릅니다.
      세 모델 전부 넣든지 전부 빼든지 **통일**하고 그 사실을 기록하세요.
- [ ] `model_optimization_flavor(compression_level=0)` 유지
- [ ] **컨텍스트 수를 반드시 기록**: `hailortcli parse-hef <hef> | grep -i context`
- [ ] 세 HEF의 DFC 버전 통일 (`parse-hef`의 `HEF Compiler Version`)

#### (C) 측정 코드 구조 — 세 모델을 같은 선에 놓는다

이 워크로드는 **후처리 위치가 모델마다 다릅니다.** 이게 이번 실험 최대의 함정입니다.

| 모델 | NMS/후처리 위치 | 근거 |
|---|---|---|
| ssd_mobilenet_v1 | **NPU 추정(on-chip NMS)** ⚠️ | alls에 `nms_postprocess(meta_arch=ssd)` — **`engine=` 인자가 없어 DFC 기본값에 달려 있음.** `hailortcli parse-hef`로 NMS 파라미터가 HEF에 baked-in 됐는지 실측 확인 필요 |
| deeplab_v3_..._wo_dilation | **NPU(bilinear + argmax)** | yaml `device_pre_post_layers: {bilinear: true, argmax: true}` |
| mobilenet_v2_1.0 | **후처리 없음**(softmax까지 그래프 내부) | parser end node = `MobilenetV2/Predictions/Softmax` |

→ 셋 다 호스트 후처리가 사실상 없습니다. **YOLOv8 워크로드와 정반대 성격**이며,
이것 자체가 논문에서 쓸 만한 대비점입니다("후처리가 CPU에 있는 워크로드 vs 없는 워크로드").

- [ ] `mz3_sched_bench.cpp`의 latency 정의를 명시: `enq_ts[i] = now → input.write()`,
      `output.read() → deq_ts[i]`, `latency = mean(deq - enq)`.
      **이 값은 `write()`가 큐 포화로 블로킹된 시간을 포함**합니다(2026-08-08에 확인한 대로
      HRTT의 순수 device latency와 최대 50배까지 차이날 수 있음).
- [ ] 전처리는 세 모델 모두 **letterbox 아닌 단순 resize + BGR→RGB** (YOLO 실험과 다름)
- [ ] reader 스레드가 `read() → 후처리 → 다음 read()` 순차 구조인지 확인.
      후처리가 없으므로 이번엔 문제가 없지만, 나중에 후처리를 넣으면 latency가 오염됩니다.

#### (D) 실행 환경

- [ ] `hailortcli fw-control identify` — Architecture / FW 버전
- [ ] `lspci -vv | grep LnkSta` — **PCIe Gen/lane 기록**.
      기존 조사(`memory/2026-08-24_v5_npu_vs_cpu_pcie_investigation.md`)에서 RPi5가 Gen3 **x1**이고
      Hailo 공식 Model Zoo FPS는 Gen3 **x2** 기준이라 대역폭이 절반임을 확인했습니다.
      → **공식 FPS(356/60.6/…)와 우리 실측이 안 맞는 건 정상**이며, 이 사실을 결과 해설에 넣어야 합니다.
- [ ] 온도/스로틀링: `vcgencmd get_throttled`, `hailortcli monitor`의 On Die Temperature
- [ ] 데이터셋 `~/datasets/sampled_val2017` 673장 — **두 보드 동일 파일**인지 확인 (md5)
- [ ] 백그라운드 프로세스 정리(이전 nohup 실험이 살아있는지 `ps aux | grep sched`)

---

### 2-2. 측정 메트릭 — 정의와 보드별 측정 방법

> **핵심**: Hailo-10H는 추론 스택이 디바이스 내부 Linux의 `hailort_server`에서 돌기 때문에
> `.hrtt` 트레이스가 **원리적으로 0바이트**입니다(`HRTT_ON_HAILO10H.md` 참조).
> 따라서 8L에서 되던 측정 중 일부는 10H에서 대체 수단이 필요합니다.

| 메트릭 | 정의 | Hailo-8L 측정법 | Hailo-10H 측정법 |
|---|---|---|---|
| **End-to-end latency** | 전처리+추론+(후처리) 장당 시간 | `total_time_ms_*` (앱 실측) | 동일 (앱 실측) |
| **앱 관측 추론 latency** | `write()` → `read()` 왕복 (큐 대기 포함) | `*_latency_ms` | 동일 |
| **NPU 순수 latency** | H2D→D2H 디바이스 내부 시간 | **HRTT** `frame_dequeue` 페어링 → `avg_latency_*` | **불가**. 대체: `hailortcli run2 --measure-latency` (**단일 모델만**) |
| **FPS / TPS** | 프레임수 ÷ 워커 전체시간 (전처리 포함) | HRTT `avg_fps_*` 또는 호스트 실측 | 호스트 실측 (`avg_fps_*`) |
| **CPU 사용률** | `/proc/stat` 300ms 샘플 평균 | `sys_monitor.hpp` | 동일 |
| **Memory 사용률** | `/proc/meminfo` | `sys_monitor.hpp` | 동일 |
| **NPU utilization** | 디바이스 가동률 | `HAILO_MONITOR=1` → `/tmp/hmon_files` → `hailo_utilization.py`, 또는 HRTT `device_usage_pct` 합 | `hailortcli monitor`를 자식 프로세스로 띄워 파싱 (`npu_monitor.hpp`). **/tmp/hmon_files 없음** |
| **OS context switch** | 워커 스레드 vol/nonvol 증가분 | `/proc/thread-self/status` | 동일 |
| **스케줄러 core_op 전환 횟수·사유** | `switch_core_op_decision` (over_threshold / over_timeout / switch_because_idle) | **HRTT만 가능** | **불가능** |
| **모델별 디바이스 점유율(동시 실행)** | core_op 활성 시간 비율 | HRTT `device_usage_pct` | **불가능** (monitor는 전체 합계만) |
| **activation 소요 시간** | core_op 활성화 오버헤드 | HRTT `activation_ms` | **불가능** |
| **HEF 컨텍스트 수** | 정적 정보 | `hailortcli parse-hef` | 동일 |
| **전력** | — | 8L 보드 INA 센서 유무 확인 필요 | **불가** (센서 없음, `HAILO_OPEN_FILE_FAILURE`) |

#### "Context Switch Overhead"는 3가지 다른 것을 가리킨다 — 반드시 구분할 것

논문·보고서에서 이 용어가 섞여 쓰이면 리뷰에서 지적받습니다. 이번 워크로드에서는 특히 위험합니다.

| # | 무엇 | 어디서 발생 | 측정 | 이번 워크로드에서의 의미 |
|---|---|---|---|---|
| **(a)** | **OS 스레드 컨텍스트 스위치** | 호스트 CPU | `/proc/thread-self/status` vol/nonvol | 두 보드 다 측정 가능. 호스트 부하 지표 |
| **(b)** | **HailoRT 스케줄러의 core_op 전환** | NPU 스케줄러 | HRTT `switch_core_op_decision` | **8L 전용.** 멀티모델 스케줄링 연구의 핵심 지표 |
| **(c)** | **HEF 내부 multi-context 전환** | 칩 내부 (weight 재로드) | `parse-hef`로 정적 확인, 동적 측정 불가 | **deeplab만 3-context.** 다른 두 모델은 single |

**(c)가 이번 실험의 숨은 주인공입니다.** deeplab 혼자 multi-context라
"모델을 하나 더 켰을 때 느려지는 정도"가 다른 모델과 질적으로 다를 수 있습니다.
→ **조건별 표에 "활성 모델 중 multi-context 모델 포함 여부" 컬럼을 넣으세요.**
→ 가능하면 대조군으로 `deeplab_v3_mobilenet_v2`(dilation 있는 표준판, 8L FPS 51.8)도 한 번
   돌려 컨텍스트/성능 차이를 보면 "wo_dilation 선택"의 정당화가 됩니다.

---

### 2-3. 실험 조건 매트릭스

기존 `2026-08-31_mz3_default_exp1`의 설계(7조건 × 3회 × fps 2종)를 그대로 유지하면
Hailo-10H `2026-08-24_det_seg_pose_workload_exp1`과도 직접 비교 가능합니다.

```
조건 7개: ssd / deeplab / mnv2 / ssd+deeplab / ssd+mnv2 / deeplab+mnv2 / ssd+deeplab+mnv2
반복    3회
입력속도 fps=0 (무제한) / fps=30 (제한)   → 2세트
프레임  673장 (sampled_val2017 전량)
```

| 단계 | 보드 | HEF | 런 수 | 상태 |
|---|---|---|---|---|
| S1 | 8L | Model Zoo 사전학습 3종 | 42 | **완료** (`2026-08-31_mz3_default_exp1`) |
| S2 | 10H | Model Zoo 사전학습 3종 (v5.4.0/hailo10h) | 42 | 미실행 |
| S3 | 8L | 재학습판 (ssd=COCO 그대로, deeplab=Cityscapes, mnv2=GTSRB) | 42 | deeplab 재학습 대기 |
| S4 | 10H | 재학습판 | 42 | 대기 |

- **S1↔S2** = 하드웨어 비교 (같은 모델, 다른 칩)
- **S1↔S3** = 도메인 재학습이 스케줄링 특성에 주는 영향 (같은 칩, 다른 가중치)
  → 구조가 같으면 latency는 거의 안 변해야 정상. **변한다면 컨텍스트 수가 바뀐 것**이므로
  `parse-hef` 결과를 먼저 확인하세요.

> **fps=0 세트의 해석 주의**: 기존 발견대로 입력을 최대 속도로 밀어넣으면 큐가 항상 포화라
> 단일 모델에서도 NPU 사용률이 99~100%로 찍힙니다. 이건 경합 때문이 아닙니다.
> **스케줄링 경합 자체를 보려면 fps=30 세트가 주(主)**, fps=0은 처리량 상한 측정용 보조입니다.

### 2-4. CSV 스키마

`hailo_8L/csv_writer.hpp` 관례를 유지하되 이번 워크로드용으로 아래 컬럼을 **추가** 권장:

| 추가 컬럼 | 이유 |
|---|---|
| `hef_variant` | `pretrained` / `retrained` 구분 |
| `board` | `hailo8l` / `hailo10h` |
| `ctx_count_ssd/deeplab/mnv2` | `parse-hef` 값. (c) 컨텍스트 전환 분석의 기준 |
| `pp_location_*` | `npu` / `cpu` / `none` — 후처리 위치 명시 |
| `pcie_link` | `gen3x1` 등. 보드 간 비교 시 필수 |
| `mz_version`, `dfc_version` | 8L(v2.19/DFC3.33)과 10H(v5.4/DFC5.4)가 다르므로 |

---

## 3. 노션 정리 페이지 구조 (제안)

> **규칙 확인**: `memory/notion_workflow.md`에 따라 Notion MCP로 페이지를 직접 만들지 않습니다.
> 아래는 사용자가 본인 Notion 페이지에 **붙여넣을 구조 제안**입니다.

### 페이지 트리

```
📄 NPU Scheduler — SMART TRAFFIC 워크로드 (MZ 3모델)
├─ 0. 한눈에 보기 (Status Board)
├─ 1. 실험 목적 & 워크로드 선정 근거
├─ 2. 환경 세팅
│   ├─ 2-1. 보드/런타임 스펙 표
│   └─ 2-2. 컴파일 환경 (MZ/DFC 버전 2벌)
├─ 3. 모델별 재학습 로그
│   ├─ 3-1. ssd_mobilenet_v1
│   ├─ 3-2. deeplab_v3_mobilenet_v2_wo_dilation
│   └─ 3-3. mobilenet_v2_1.0
├─ 4. HEF 변환 결과
├─ 5. 베이스라인 측정 결과
├─ 6. 트러블슈팅 로그 (DB)
└─ 7. 다음 할 일 / 열린 질문
```

### 각 섹션 템플릿

#### 0. 한눈에 보기

| 항목 | 상태 | 담당 | 최종 갱신 |
|---|---|---|---|
| mobilenet_v2 GTSRB 재학습 | ✅ 완료 (float 94.14%) | 전정민 | 2026-08-31 |
| mobilenet_v2 HEF 정확도 검증 | ⬜ 미측정 | | |
| deeplab Cityscapes 재학습 | ⬜ 미착수 | | |
| ssd_mobilenet_v1 | ✅ 재학습 불필요 (COCO 유지) | | |
| 8L 사전학습 베이스라인 (42런) | ✅ 완료 | 전정민 | 2026-08-31 |
| 10H 사전학습 베이스라인 | ⬜ 미실행 | | |
| 재학습판 베이스라인 | ⬜ 대기 | | |

#### 1. 실험 목적 & 워크로드 선정 근거

- 시나리오: SMART TRAFFIC (workload2.xlsx 시나리오 2)
- 선정 이유 (xlsx 원문 인용): 재현성 문제 없음 / 기존 논문 실험과 형식 일치
- 고려할 부분 (xlsx 원문 인용): 환경 구축 시간, 통찰의 한계, 입력 크기 상이
- **이 페이지에서 추가로 답해야 할 질문**: 세 모델이 후처리를 전부 NPU에서 하거나 안 하는
  워크로드라는 점이 기존 YOLOv8 워크로드(후처리 CPU)와 어떤 대비를 만드는가

#### 2. 환경 세팅

| 항목 | Hailo-8L (npu-rpi1) | Hailo-10H (npu-rpi5) |
|---|---|---|
| 보드 | Raspberry Pi 5 | Raspberry Pi 5 |
| OS | Ubuntu 24.04 | |
| HailoRT | 4.23.0 | 5.3.0 |
| PCIe | Gen3 x1 | Gen3 x1 |
| Model Zoo | v2.19.0 | v5.4.0 |
| DFC | 3.33.0 | 5.4.0 |
| 트레이싱 | HRTT 가능 | **HRTT 불가** (host_trace 대체) |

#### 3. 모델별 재학습 로그 (모델당 1개 토글)

```
▸ <모델명>
   · 원 프레임워크 / 저장소 링크
   · 데이터셋: 출처, 클래스 수, train/val/test 장수, 분할 방식
   · 하이퍼파라미터: epochs, batch, lr, optimizer, 입력 정규화 규약
   · 학습 로그 (스크린샷 또는 코드블록)
   · Float 정확도: __
   · export: 노드 이름(start/end), 파일 크기
   · 재현 명령어 (복붙 가능한 코드블록)
   · 겪은 문제 → 6번 트러블슈팅 DB로 링크
```

#### 4. HEF 변환 결과

| 모델 | arch | alls 요약 | 캘리브셋 | 컨텍스트 수 | HEF 크기 | HW 정확도 | 비고 |
|---|---|---|---|---|---|---|---|
| ssd_mobilenet_v1 | hailo8l | 공식 + QFT | COCO 4000 | | | mAP __ | on-chip NMS |
| deeplab_..._wo_dilation | hailo8l | 공식(equalization off) | Cityscapes __ | | | mIoU __ | bilinear+argmax on-chip |
| mobilenet_v2_1.0_gtsrb | hailo8l | 공식3줄 + resources/context_switch | GTSRB 1024 | 1 | 5.8MB | **미측정** | single-context 강제 |

#### 5. 베이스라인 측정 결과

- 조건 정의(§2-1 체크리스트)를 **토글로 접어서** 페이지 상단에 고정
- 표 1: 조건별 3회 평균 (14행 = 7조건 × fps 2종) — 보드별로 각각
- 표 2: 8L vs 10H 비교 (같은 조건 나란히)
- 표 3: 사전학습 vs 재학습 비교
- 그래프: 모델 수(1/2/3)에 따른 latency 스케일링, fps=30 기준
- **캡션에 반드시 명시**: latency 정의(큐 대기 포함), 후처리 위치, 컨텍스트 수, PCIe Gen3 x1

#### 6. 트러블슈팅 로그 — Notion **데이터베이스**로

| 속성 | 타입 | 값 예시 |
|---|---|---|
| 제목 | Title | "hailomz compile: supported_hw_arch에 hailo8l 없음" |
| 발생 단계 | Select | 환경설치 / 학습 / export / parse / optimize / compile / 실기실행 / 분석 |
| 보드 | Select | 8L / 10H / PC(WSL) |
| 증상 | Text | 에러 메시지 원문 |
| 원인 | Text | |
| 해결 | Text | |
| 상태 | Select | 해결 / 우회 / 미해결 |
| 날짜 | Date | |
| 관련 파일 | Text | 커밋 해시나 경로 |

→ 이렇게 해두면 나중에 논문 Limitation 절과 부록을 이 DB에서 바로 뽑아낼 수 있습니다.

#### 7. 다음 할 일 / 열린 질문

- 체크박스 리스트 + 각 항목에 담당/기한
- "열린 질문"은 조교님·교수님께 물어볼 것을 따로 모아두면 미팅 때 그대로 사용 가능

---

## 4. 트러블슈팅 FAQ

### 4-A. 환경 / 버전

| 증상 | 원인 | 해결 |
|---|---|---|
| `hailomz compile --hw-arch hailo8l` 이 "not supported" | master(v5.x) yaml의 `supported_hw_arch`에 hailo8/8l이 없음 | **Model Zoo v2.19.0 이하 + DFC 3.3x** 사용. 두 라인을 별도 venv로 분리 |
| DFC와 HailoRT 버전 불일치 경고 | HEF 컴파일러 버전 > 런타임 버전 | HEF는 상위 호환이 보장되지 않음. `hailortcli parse-hef`의 `HEF Compiler Version`을 보드 HailoRT와 맞출 것 |
| DFC를 RPi에 설치 불가 | DFC는 Linux **x86** wheel만 배포. Hailo 측이 "ARM 이식 계획 없음"이라고 공식 답변(2025-04). 문서 사양표는 "Ubuntu 22.04/24.04 64bit(WSL2 가능)"로만 적혀 있음 | RPi에서 컴파일 불가. PC/WSL에서 컴파일하고 **HEF만 scp**. HailoRT(런타임)만 ARM 지원 |
| `ImportError: cannot import name 'runtime_version'` (protobuf) | 시스템 protobuf가 protoc 생성 코드보다 구버전 | `pip3 install --upgrade protobuf --break-system-packages` 또는 `~/hailo_platform_venv` 사용 (2026-08-08 기록) |

### 4-B. 파싱 (`hailomz parse` / `translate_*_model`)

| 증상 | 원인 | 해결 |
|---|---|---|
| `Unable to find node <name>` | 재학습 모델의 노드 이름이 원본과 다름 | `netron`으로 실제 이름 확인 후 `--start-node-names` / `--end-node-names` 명시 |
| 파싱은 되는데 출력 shape이 다름 | end node를 잘못 잡음 (예: deeplab에서 ArgMax 누락) | **deeplab은 `ArgMax`, ssd는 12개 BoxPredictor BiasAdd, mnv2는 `Softmax`** — yaml의 `parser.nodes`를 기준으로 맞출 것 |
| `Unsupported layer/op` | ONNX opset이나 커스텀 op | opset 11~13으로 낮춰 재export, `onnxsim`으로 단순화. TF는 frozen graph 경유 |
| ONNX에 넣은 broadcast/expand가 HAR에서 사라짐 | DFC가 elementwise 자동 broadcast를 전부 지원하지 않음 | export 전에 `tile`/`expand`로 **명시적 shape 정렬** 후 재export |

### 4-C. 양자화 (`optimize`)

| 증상 | 원인 | 해결 |
|---|---|---|
| 양자화 후 정확도 급락 | 캘리브셋이 배포 도메인과 다름 | `--calib-path`를 **실제 도메인 이미지**로. GTSRB 모델은 GTSRB, Cityscapes 모델은 Cityscapes |
| 정확도가 여전히 낮음 | 정규화 이중 적용 | alls의 `normalization()`이 있으면 **학습 모델에는 Rescaling 층을 넣지 말 것**. 입력 uint8[0,255] → 네트 내부 [-1,1] |
| 클래스별로 정확도가 들쭉날쭉 | 캘리브셋 클래스 편중 | 균등 stride 또는 클래스별 균등 샘플링 (`compile_hailo.py`는 stride 방식으로 이미 대응됨) |
| optimize가 수 시간 걸림 | 공식 alls의 QFT (`post_quantization_optimization(finetune, policy=enabled, dataset_size=4000, epochs=8)`) — ssd가 해당 | GPU 사용, 또는 `policy=disabled`로 끄고 **끈 사실을 기록** (mAP 1~2점 손실 가능) |
| `equalization` 켜니 더 나빠짐 | deeplab 계열 특성 | 공식 alls대로 `pre_quantization_optimization(equalization, policy=disabled)` 유지 |
| `NMSConfigPostprocessException` | 클래스 수/앵커가 NMS config와 불일치 | `--classes N` 지정, 또는 `cfg/postprocess_config/`의 json을 복사·수정해 `nms_postprocess("<json>", meta_arch=ssd)` 로 지정 |

### 4-D. 컴파일 (`compile`)

| 증상 | 원인 | 해결 |
|---|---|---|
| `Failed to produce compiled graph` / `BackendAllocatorException: No successful assignment for: <layer>` | 자원 배치 실패 | ① 입력 해상도/배치 축소 ② `resources_param(max_..._utilization=...)` 완화 ③ 문제 레이어를 다른 컨텍스트로 분리 ④ **자동 컨텍스트 분할 허용** |
| single-context로 강제했더니 컴파일 실패 | 모델이 칩 메모리에 안 들어감 | `context_switch_param(mode=disabled)` 제거. **deeplab(513x513)에 이걸 쓰면 거의 확실히 실패** |
| 컴파일은 됐는데 컨텍스트 수가 예상과 다름 | 모델 구조 변경(예: dilation 남음) | `hailortcli parse-hef`로 확인. 원본과 다르면 export 설정을 되짚을 것 |
| 컴파일 시간이 수십 분 | 정상 (mapping 단계) | `--performance`를 주면 더 오래 걸림. 베이스라인엔 쓰지 말 것 |

### 4-E. 실기 실행 / 멀티모델 병목

| 증상 | 원인 | 해결 |
|---|---|---|
| `HAILO_TIMEOUT(4)` (timeout=10000ms) 반복 | 해당 모델이 스케줄링 기회를 못 받음(starvation) | priority 차이 15 이상일 때 발생. **기본값 조건(전부 16)에서는 발생하면 안 됨** → 발생하면 다른 원인(드라이버/큐) |
| latency가 HRTT 값과 수십 배 차이 | 앱 측 latency는 `write()` 블로킹 대기를 포함 | 두 지표를 **다른 컬럼으로 둘 다 기록**. 어느 쪽도 틀린 게 아님 |
| 단일 모델인데 NPU 사용률 99~100% | `INPUT_FPS=0`이라 큐가 항상 포화 | 경합 지표로 쓰지 말 것. fps 제한 세트에서 해석 |
| `.hrtt`가 안 생김 / 0바이트 (8L) | `HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP` 값이 실행시간보다 큼 | 가장 짧은 조건보다 **작게** 설정. 2026-08-08 기록상 30 → **3**으로 낮춰야 짧은 조건도 덤프됨 |
| `.hrtt`가 0바이트 (10H) | **구조적 한계** — 추론 스택이 디바이스 내부에 있음 | 대안 없음. `host_trace` + `hailortcli monitor` 사용 (`HRTT_ON_HAILO10H.md`) |
| `/tmp/hmon_files`가 안 생김 (10H) | 호스트 libhailort에 스케줄러 상태가 없음 | `hailortcli monitor`를 자식 프로세스로 띄워 파싱 |
| 프로세스 종료 시 세그폴트 (결과는 저장됨) | VDevice/네트워크그룹 정리 순서 이슈로 추정 | CSV 행 증가로 성공 판정 후 자동 재시도하는 스크립트 패턴 사용 |
| `hailo_pci` 드라이버 크래시 (`find_vma`) | rpi4 커널 6.12.x 한정 이슈 | 8L(rpi1)에선 미보고. 발생 시 `hailortcli fw-control identify`로 복구 여부 확인 |
| deeplab을 켰을 때만 전체가 크게 느려짐 | **multi-context 모델의 weight 재로드** — 스케줄러가 전환할 때마다 컨텍스트를 다시 올림 | 정상 동작. 단, 이걸 "스케줄링 오버헤드"라고 쓰면 안 됨. `parse-hef` 컨텍스트 수를 근거로 **(c) 유형**으로 분류해 서술 |
| 세 모델 입력 크기가 달라 전처리 시간이 제각각 | 300 / 513 / 224로 다름 | `avg_preprocess_ms_*`를 모델별로 독립 기록(이미 그렇게 되어 있음). FPS 비교 시 전처리 포함 여부 명시 |
| 메모리 사용률이 조합마다 크게 다름 | HEF 가중치가 호스트/디바이스에 상주 | `mem_percent`는 호스트 기준. 디바이스 RAM은 `hailortcli monitor`로 별도 확인 |

---

## 5. 다음 단계 (우선순위)

1. **`hailortcli parse-hef`로 3개 HEF의 컨텍스트 수 확보** — 이게 없으면 §2 분석이 전부 추정입니다.
   사전학습 HEF 3개 + 재학습 mnv2 HEF, 8L/10H 양쪽.
2. **mnv2 재학습 HEF의 HW(int8) 정확도 측정** — 지금 94.14%는 float 값입니다.
3. **`mobilenet_v2_1.0_gtsrb.alls`의 `context_switch_param` 처리 결정** — 세 모델 조건 통일.
4. **10H 사전학습 베이스라인(S2) 실행** — 코드는 `hailo10h_sched_bench.cpp`가 이미 있으므로
   HEF 3개(v5.4.0/hailo10h)만 받으면 됨.
5. deeplab Cityscapes 재학습 착수 (가장 오래 걸림). 리스크가 크면 **FCN으로 대체**하는 안을
   교수님/조교님과 먼저 합의.
6. `2026-08-31_mz3_default_exp1` / `_mz3_retrain` 두 폴더 git 커밋 (현재 untracked).

---

## 참고 (확인 출처)

- hailo_model_zoo `docs/RETRAIN_ON_CUSTOM_DATASET.rst` (v2.19.0 / master) — 재학습 지원 아키텍처 목록
- hailo_model_zoo `cfg/networks/{ssd_mobilenet_v1, deeplab_v3_mobilenet_v2_wo_dilation, mobilenet_v2_1.0}.yaml`
- hailo_model_zoo `cfg/alls/generic/*.alls` (v2.19.0)
- hailo_model_zoo `docs/public_models/HAILO8L/*.rst`, `docs/public_models/HAILO10H/*.rst`
- hailo_model_zoo `docs/GETTING_STARTED.rst`, `docs/DATA.rst`
- 저장소 내부: `PROJECT_SUMMARY.md` §5·§6, `memory/findings.md`,
  `memory/2026-08-24_v5_npu_vs_cpu_pcie_investigation.md`,
  `hailo_10h/experiments/2026-08-24_det_seg_pose_workload_exp1/HRTT_ON_HAILO10H.md`,
  `workload2.xlsx` (Scenario 시트)
