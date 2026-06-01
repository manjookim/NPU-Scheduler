# NPU Scheduler - Hailo-8L on Raspberry Pi 5

## 환경

| 항목 | 내용 |
|---|---|
| Device | Raspberry Pi 5 |
| OS | Ubuntu 24.04.x |
| Connection | SSH (Galaxy Book5 Pro / Desktop) |
| NPU | Hailo-8L (M.2 AI HAT+) |
| Architecture | HAILO8L |
| Firmware | 4.23.0 |

## 설치한 SW

| 패키지 | 설명 |
|---|---|
| `hailort-pcie-driver_4.23.0_all.deb` | Hailo PCIe 드라이버 |
| `hailort_4.23.0_arm64.deb` | HailoRT 런타임 |
| `hailo-tappas-core_5.2.0_arm64.deb` | Hailo TAPPAS Core |
| `hailort-4.23.0-cp312-cp312-linux_aarch64.whl` | HailoRT Python 바인딩 |
| `hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl` | Hailo Dataflow Compiler (PC용) |

## 사전 준비

### 1. 필수 패키지 설치 (Raspberry Pi)
```bash
sudo apt install git -y
sudo apt install pkgconf -y
sudo apt install libopencv-dev -y
pip install Pillow --break-system-packages
```

### 2. Hailo 장치 인식 확인
```bash
hailortcli fw-control identify
```
정상 출력:

Device Architecture: HAILO8L
Firmware Version: 4.23.0

### 3. HEF 파일 다운로드
```bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
cd hailo-rpi5-examples
export DEVICE_ARCHITECTURE=HAILO8L
bash download_resources.sh
```
다운로드되는 HEF 파일:
- `yolov8s_h8l.hef` → Object Detection
- `yolov8s_pose_h8l.hef` → Pose Estimation
- `yolov5n_seg_h8l.hef` → Segmentation (yolov8s_seg H8L 미제공으로 대체)

### 4. COCO val2017 검증 데이터셋 다운로드
```bash
mkdir -p ~/datasets/coco
cd ~/datasets/coco
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```
- 총 5,000장 이미지

## 컴파일 및 실행

### 컴파일
```bash
cd ~/hailo_cpp_test
g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4)
```

### 실행 (NPU 모니터링 포함)
터미널 1 (NPU 모니터링):
```bash
hailortcli monitor
```

터미널 2 (추론 실행):
```bash
HAILO_MONITOR=1 ./infer_scheduler
```

## 파라미터 변경 방법

`infer_scheduler.cpp` 상단의 아래 값만 수정 후 재컴파일:
```cpp
#define BATCH_SIZE     1   // 변경 가능: 1, 2, 4, 8, 16, 32, 63
#define THRESHOLD      0   // 변경 가능: 0, 2, 4, 8, 16, 32, 64
```

## 실험 결과

결과는 `~/hailo_cpp_test/results.csv`에 자동 저장됩니다.

| batch | threshold | Det Latency(ms) | Seg Latency(ms) | Pose Latency(ms) | CPU(%) | MEM(%) | Context Switch |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 43.26 | 49.01 | 39.61 | 11.77 | 12.96 | 4,493,317 |

## 사용 모델

| 모델 | 태스크 | 입력 | 출력 수 |
|---|---|---|---|
| yolov8s_h8l | Object Detection | 640x640x3 | 1 |
| yolov5n_seg_h8l | Segmentation | 640x640x3 | 4 |
| yolov8s_pose_h8l | Pose Estimation | 640x640x3 | 9 |

## Scheduler 파라미터 설명

| 파라미터 | 설명 |
|---|---|
| **Batch-size** | 한 번에 처리할 프레임 수. 클수록 처리량 증가, 최대 63 |
| **Scheduling scheme** | Round Robin 고정 (현재 유일 지원 방식) |
| **Threshold** | 스케줄링 실행 전 최소 요청 누적 수. 0이면 즉시 실행 |

