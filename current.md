# Project Context Handoff Document

> 작성 2026-09-02 · 대상: 후속 세션 AI · 위치: 저장소 루트 `current.md`
> (`memory/` 는 `.gitignore` 대상이므로 인계 문서를 그쪽에 두지 말 것 — §4-D 참조)

---

## 1. Project Overview & Stakeholder Requirements

- **프로젝트 핵심 목표**
  엣지 NPU(Hailo-8L)에서 다중 모델 동시 추론 시 HailoRT Model Scheduler(ROUND_ROBIN)의
  거동을 측정. 최종 산출물은 NPU Scheduling 논문 / SCIE 저널.

- **이번 과제의 목적 (교수님 발언, Slack)**
  > "이게 일반적으로 yolo에서만 발생하는 문제가 아니라 다른 작업에도 동일하게 발생한다는
  > **일반화의 목적**이 있기 때문에 **새로운 발견을 하지 않아도 무방**할것 같아요.
  > 우선 **2번**으로 진행해봅시다."

  → 결론은 "새 발견"이 아니라 **기존 YOLOv8 실험 결과가 다른 Task/구조에서도 재현되는가**.
  → 시나리오는 **SMART TRAFFIC 2번 확정**. 시나리오 3(ImageNet+GTSRB) 제안 불필요.

- **역할 분담 (교수님 지시)**

  | 담당 | 워크로드 |
  |---|---|
  | 전정민(본인) | SMART TRAFFIC 시나리오 2 |
  | 허민 | SERVICE ROBOT |

  (순서상 추정. 조교님 확인 미완)

- **조교님(김민주) 지시 요약**
  1. 대상 3모델: `ssd_mobilenet_v1`, `deeplab_v3_mobilenet_v2_wo_dilation`, `mobilenet_v2_1.0`
  2. Hailo Model Zoo 재학습 가이드 + `workload2.xlsx` 참고하여 재학습 및 NPU 컴파일
  3. **파라미터 미조정(Default) 상태**로 추론 성능 측정
  4. 화요일(2026-09-01)까지 노션 정리 → **1일 연장 요청 상태**

- **작업 규칙 (필수)**
  - **"노션에 정리해줘" = Notion MCP 로 페이지 생성 금지.** 채팅에 마크다운 텍스트로만 출력.
    사용자가 직접 붙여넣음. (근거: `memory/notion_workflow.md`, 2026-08-05 정정 이력)
  - 측정 보드는 **Hailo-8L 단독** 확정.

- **워크로드 정의 (`workload2.xlsx` / Scenario 시트 / SMART TRAFFIC 2)**

  | Task | 모델 | 목표 데이터셋 | 입력 | 출력 | Params | 공식 FPS(8L,b1) |
  |---|---|---|---|---|---|---|
  | 차량 검출 | ssd_mobilenet_v1 | COCO | 300x300 | 90x8x1 | 6.79M | 356 |
  | 차선/도로 분할 | deeplab_v3_mnv2_wo_dilation | **Cityscapes** | 513x513 | 513x513x1 | 2.10M | 60.6 |
  | 표지판 분류 | mobilenet_v2_1.0 | **GTSRB** | 224x224 | 1001 | 3.49M | 1738 |

  **중요**: 엑셀의 `dataset` 칸은 **재학습 목표**이지 현재 상태가 아님.
  Model Zoo 사전학습은 deeplab=PASCAL VOC, mnv2=ImageNet.
  근거: 같은 시트의 "고려할 부분 — Cityscapes, GTSRB 재학습 및 컴파일 필요",
  그리고 mnv2 행의 output 이 43 이 아니라 **1001**(ImageNet 값)로 적혀 있음.
  FPS 컬럼(356/60.6/1738)은 **Hailo-8L 공식 수치와 정확히 일치** → 엑셀은 8L 기준 표.

---

## 2. Technical Architecture & Core Guidelines

### 2-1. 보드 / 계정

**Hailo-8L 계정 ID 는 `npu-rpi1`** (구 `rpi1`, 2026-08-20 이관).

```bash
ssh npu-rpi1@155.230.16.157 -p 40021    # 홈: /home/npu-rpi1/
```

| 별칭 | NPU | HailoRT | 비고 |
|---|---|---|---|
| **npu-rpi1** | Hailo-8L | 4.23.0 | 주 실험 보드. **HRTT 가능** |
| npu-rpi2 | Hailo-8L | — | **조교님 기기** (우리 것 아님) |
| npu-rpi5 | Hailo-10H | 5.3.0 | HRTT **구조적 불가** |
| rpi4 | Hailo-8 | — | 2026-08-19 이후 사용 중단 |

PCIe 전 보드 **Gen3 x1**. 공식 Model Zoo FPS 는 Gen3 **x2** 기준 → 대역폭 절반.

**미해결 부채**: 저장소에 `rpi1@` 15파일 / `/home/rpi1/` 18파일이 잔존.
특히 `hailo_8L/infer_scheduler_noppt.cpp`, `infer_scheduler_compare.cpp`,
`parse_npu_log.py`, `scripts/run_det_v8s_vs_v5npu_*.sh` 는 실행 시 HEF 로드 실패.
(문서만 먼저 치환, 소스는 실기 재빌드 검증과 함께 처리 권장)

### 2-2. 기존 Hailo-8L 추론 구조 준수

`mz3_sched_bench.cpp` 는 `hailo_8L/model_runner.hpp` 구조를 이식한 것.

**이미 일치하는 항목 (변경 금지)**

- writer / reader 스레드 producer-consumer 쌍
- `latency = mean(deq_ts[i] - enq_ts[i])`, `enq_ts` 는 `write()` **직전** 기록
  → **`write()` 블로킹 대기가 latency 에 포함됨**. HRTT 의 device latency 와 정의가 다름
  (2026-08-08 기록: 최대 50배 차이 관측)
- `HAILO_TIMEOUT` 발생 시 `do-while` 재시도 (프레임 유실 방지)
- 프레임별 전처리 (한꺼번에 X)
- ctx switch = writer + reader 증가분 합 (`/proc/thread-self/status`)
- 입력 vstream: `HAILO_FORMAT_TYPE_AUTO`, timeout **300000ms**, `HAILO_DEFAULT_VSTREAM_QUEUE_SIZE`
- 출력 vstream: **`AUTO`** — `model_setup.hpp` 주석 근거
  ("ENABLE_POSTPROCESS=0 이면 조교님 코드(AUTO)와 동일 조건으로 되돌린다")

**2026-09-01 정렬 작업 (완료)**

- `total_time_s` 정의를 원본과 통일: 함수진입~join → **첫 enqueue ~ 마지막 dequeue**
- `max_latency_ms` 추가 (host 실측. 8L 원본은 HRTT 에서 채우나 10H 방침 준용)
- CSV 스키마를 `csv_writer.hpp` 와 동일 정렬 (det/seg/pose → ssd/deeplab/mnv2), **24 → 57컬럼**

**차이가 남아 있는 항목 (의도적)**

- 전처리: 원본 letterbox → MZ3 는 **단순 resize**(Model Zoo 전처리 규약). NPU latency 영향 없음
- 후처리: 원본 `decode_det/pose/seg` → MZ3 는 **없음**(세 모델 모두 칩 처리 또는 그래프 내장)

### 2-3. 파라미터 API 처리

**"Default" = setter 를 호출하지 않는 것.** `0` 을 넣는 것과 **다름**.

```cpp
// Default 아님
network_group->set_scheduler_priority(0);
// Default — 코드 자체가 없음
```

- `set_scheduler_threshold` / `set_scheduler_timeout` / `set_scheduler_priority` **전부 미호출**
- `create_configure_params()` 결과의 `batch_size` / `power_mode` **미수정**
- 결과 HailoRT 기본값: **batch=0(auto) / threshold=1 / timeout=0 / priority=16(NORMAL) / ROUND_ROBIN**
- CSV 에는 이 기본값을 **숫자로 기록** (값이 고정이어도 남겨야 후속 파라미터 스윕과 같은 표에 놓임)

**예외 1개 (보고서에 반드시 명시)**: input/output vstream 의 **host 측 timeout 만 300000ms** 로 확대.
스케줄링 정책이 아니라 starvation 시 프레임 유실을 막는 안전장치. 기존 전 실험 공통 관례.

**컴파일 측 Default 도 유지**: `--performance` 금지, `resources_param`/`performance_param` 추가 금지,
`model_optimization_flavor(compression_level=0)`.

### 2-4. 다중 조합 추론 (7조건)

모델 N개 → 조합 `2^N - 1`. N=3 → **7조건**.

```
CONDITIONS = ssd / deeplab / mnv2 / ssd_deeplab / ssd_mnv2 / deeplab_mnv2 / ssd_deeplab_mnv2
REPEAT     = 3
input_fps  = 0 (무제한) 및 30 (제한)  -> 2세트
FRAMES     = 673 (sampled_val2017 전량)
총 7 x 3 x 2 = 42런 (약 15분)
```

**데이터 흐름**

```
run_mz3_retrained_sweep.sh
 |- MNV2_HEF 로 mnv2 HEF 선택 -> resources_retrained/mobilenet_v2_1.0.hef 로 복사
 |- parse-hef 로 3모델 확인 -> read -p 대기 (nohup 시 이 줄 제거 필요)
 |- export HAILO_MONITOR=1, HAILO_TRACE=scheduler, BOUNDED_DUMP=30
 `- 조건 루프 x 반복 루프
      |- npu_log.txt 비움 -> hailo_utilization.py 백그라운드 기동 (venv python)
      |- mz3_sched_bench --ssd/--deeplab/--mnv2 <0|1> --fps N --res <dir> --csv <path>
      |    `- 모델당 writer/reader 스레드, VDevice 1개 공유, ROUND_ROBIN
      |- 모니터 kill -> NPU>0 샘플만 평균 -> npu_percent_*.csv 에 append
      `- traces_retrained*/hailort_<ts>.hrtt 자동 생성
```

**환경변수 스위치**

| 변수 | 효과 |
|---|---|
| `SKIP_DEEPLAB=1` | deeplab 포함 4조건 제외 → 3조건(9런/세트) |
| `MNV2_HEF=<파일명>` | mnv2 HEF 교체. `_sc.hef`→접미사 `_sc`, `mobilenet_v2_1.0.hef`→`_orig` |

**비활성 모델은 CSV 에 `NaN`** (원본 `csv_writer.hpp` 관례).

---

## 3. Current Progress

### 3-1. 완료된 측정

| 데이터셋 | 조건 | HEF | 상태 |
|---|---|---|---|
| 사전학습 42런 | 7조건 x 3회 x fps 2종 | Model Zoo 공식 3종 | 완료. CSV + **HRTT 42개 온전** |
| 재학습 18런 | 3조건(deeplab 제외) x 3회 x fps 2종 | mnv2 만 `_gtsrb.hef` | 완료. CSV + npu_percent, **HRTT 손상** |

**사전학습 fps=30 (ms)**

| 조건 | ssd | deeplab | mnv2 | CPU% |
|---|---|---|---|---|
| 단독 | 8.18 | 17.75 | 2.58 | 6.3 / 10.2 / 6.5 |
| ssd+deeplab | 19.88 | 30.23 | — | 20.8 |
| ssd+mnv2 | 22.47 | — | 9.27 | 16.1 |
| deeplab+mnv2 | — | 25.55 | 8.78 | 18.5 |
| 3개 전부 | 41.90 | 37.87 | 37.28 | 30.3 |

슬로우다운(3개 전부): ssd **5.12x** / deeplab **2.13x** / mnv2 **14.45x**

**사전학습 fps=0**: 3모델 동시 시 세 모델 FPS 가 **42.4 로 수렴** (단독 195.3 / 58.0 / 213.3).
단독 latency 는 ssd 8.11 / deeplab 195.42 / mnv2 2.45.
※ fps=0 은 큐 포화라 latency 에 대기시간 포함. **throughput 은 FPS 로 판단할 것.**

**HRTT (사전학습)**

- fps=30 2모델 조합 **약 60 switch/s = 30fps x 2모델 → 프레임마다 라운드로빈** (실측 확인)
- `over_threshold` 와 `over_timeout` 이 **항상 동일**, `idle` 항상 0
  → 기본값 조건에서 전환 사유 분해 무의미
- `activate_core_op` duration **전 조건 0.0000ms**

**재학습 vs 사전학습 (mnv2 교체)**

| 조건 | 모델 | 사전학습 | 재학습 | 차이 |
|---|---|---|---|---|
| fps=30 ssd 단독 | ssd | 8.18 | 7.98 | -2.4% |
| fps=30 mnv2 단독 | mnv2 | 2.58 | 3.00 | +16.4% |
| fps=30 ssd+mnv2 | mnv2 | 9.27 | 6.01 | -35.2% |
| **fps=0 mnv2 단독** | mnv2 | 2.45 | **5.23** | **+113%** |
| **fps=0 ssd+mnv2** | ssd | 31.12 | **70.14** | **+125%** |
| **fps=0 ssd+mnv2** | mnv2 | 32.66 | **75.61** | **+132%** |

### 3-2. parse-hef 실측

| HEF | 컨텍스트 | 출력 |
|---|---|---|
| `ssd_mobilenet_v1` | **1 Single** | FLOAT32, HAILO NMS BY CLASS(90) — **on-chip NMS 확정** |
| `mobilenet_v2_1.0` (ImageNet) | **1 Single** | — |
| `mobilenet_v2_1_0_gtsrb` (재학습) | **2 Multi** | UINT8, NC(43) |
| `deeplab_v3_..._wo_dilation` | **미측정** | — |

### 3-3. 재학습

- **ssd_mobilenet_v1**: 재학습 **불필요** (목표=COCO=Model Zoo 사전학습셋). 공식 HEF 사용
- **mobilenet_v2_1.0 → GTSRB 43클래스**: 완료. float **94.14%** (테스트셋 12,630장).
  `.alls` 가 Hailo 공식과 100% 일치. **HEF(int8) 정확도 미측정**
- **deeplab → Cityscapes 19클래스**: 학습 완료, **HEF 컴파일 실패 (4-A)**
  - 1차: ASPP 헤드 → 11.7M params (공식 2.10M 의 **5.6배**) → 폐기
  - 2차: `--head simple`(1x1 conv) → epoch 57/60, **mIoU 47.72**
  - ONNX export 성공, 출력 `[1,1,513,513]` 확인

### 3-4. 기술 스택 / 환경

| 위치 | 구성 |
|---|---|
| npu-rpi1 (RPi5) | Hailo-8L, HailoRT 4.23.0, `~/mz3_exp/`, `~/hailo_platform_venv` (protobuf 7.35.0) |
| 노트북 WSL2 Ubuntu 22.04 | **DFC 3.33.0** (`~/hailo_venv`), 저장소 `/mnt/c/Users/sset0/jungmin-claude/StudentExperiment/NPUscheduler` |
| 데탑 Windows | **RTX 5060 (Blackwell sm_120, 8GB)**, PyTorch **2.11.0+cu128**, clone `NPUscheduler-git`, `C:\datasets\Cityscapes` |

**버전 제약 (중요)**

- Model Zoo **master(v5.4.0)의 3모델 yaml 에서 `supported_hw_arch` 에 hailo8/8l 이 빠짐.**
  Hailo-8L 컴파일은 **v2.19.0 이하 + DFC 3.3x**
- **DFC 는 Linux x86 wheel 만 배포** (Hailo 공식 "ARM 이식 계획 없음"). RPi 컴파일 불가
- Hailo 공식 재학습 Docker 에 **SSD/DeepLab/MobileNetV2 없음**
  (지원: YOLOv3/4/5/8/X, DAMO-YOLO, NanoDet, CenterPose, MSPN, FCN, YOLACT, YOLOv8_seg, ArcFace)
  → 학습은 원 프레임워크 직접, 컴파일만 Model Zoo yaml/alls 재사용
- TF `research/deeplab` 은 **TF1 전용**. NGC TF1 최종 컨테이너(23.03-tf1-py3)가 CUDA 12.1/**Hopper까지**
  → **Blackwell(sm_120) 미지원** → PyTorch 재구현 불가피

---

## 4. Current Issues & Roadblocks

### 4-A. deeplab HEF 컴파일 실패 (최우선 블로커)

**정확한 지점**: `compile_hailo.py` `[4/4] 컴파일` 단계. 파티셔너는 3 contexts 를 찾고 8.4% 개선까지
성공했으나 microcode 할당에서 실패.

```
[error] Mapping Failed (allocation time: 10m 11s)
microcode exceeded size for resize1_sd0 (cluster_index=0, layer_index=0)
[error] BackendAllocatorException: Compilation failed
```

**원인**: 백본 출력 `17x17x19`(OS=32)를 **한 번에 513x513x19(=500만 원소)로 bilinear 업샘플**.
공식 모델은 이 resize 를 그래프 레이어가 아니라 **DFC 의 device pre/post layer** 로 처리
(`yaml: device_pre_post_layers: {bilinear: true, argmax: true}`). PyTorch `F.interpolate` 가
그래프 안에 `Resize` 노드로 들어가면서 microcode 한계 초과.

**시도 이력**

| 시도 | 결과 |
|---|---|
| ASPP 헤드로 학습 | 파라미터 11.7M → 공식 5.6배, latency 비교 불가 → 폐기 |
| `--head simple`(1x1 conv) 재학습 | 2.25M 근접(**미검증**), mIoU 47.72, ONNX OK |
| `--start-node`/`--end-node` 생략 컴파일 | 파싱은 통과, **resize 에서 실패** |

**제안된 다음 수정 (미적용)**: `argmax` 를 업샘플 **앞으로** 이동 → resize 가 19채널→1채널(비용 1/19).
출력 shape `513x513x1` 과 D2H 전송량은 공식과 동일 유지.

```python
# train_deeplab_cityscapes.py :: DeepLabV3MNv2.forward, export 분기만 교체
if self.export:
    idx = torch.argmax(logits, dim=1, keepdim=True).to(torch.float32)
    idx = F.interpolate(idx, size=(IMG, IMG), mode="nearest")
    return idx.to(torch.int32)
```

> **미검증 위험**: 이 구조는 argmax 가 **그래프 중간**에 놓임. 공식 모델은 argmax 가 **최종 노드**.
> DFC 가 중간 argmax 를 파싱 못 하면 `[1/4]` 에서 실패. **컴파일 1회 = 약 10분**이라 시행착오 비용 큼.
> **플랜 B 를 먼저 준비할 것**: 그래프를 `17x17x19` logits 에서 끊고 `.alls` 에
> `resize(<layer>, resize_shapes=[513,513], resize_method=bilinear, engine=nn_core)` +
> `logits_layer(...)` 로 넘기는 방식 (= 공식 device_pre_post_layers 경로).

### 4-B. HRTT 데이터 손상 (재학습 18런)

`HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=N` 은 "N초 후 덤프"가 아니라
**"최근 N초만 담는 링버퍼"**. 실측 확정:

| 설정 | 결과 |
|---|---|
| N=30 (사전학습 42런) | 런 전체 기록. 42파일, D2H 673/1346 전량 (정상) |
| N=3 (재학습 18런) | 전 파일 **2.94초로 잘림**, 31파일(런당 다수), D2H 335~499만 (손상) |

스크립트는 **30으로 되돌림 완료**. 재학습 18런은 **재수집 필요**.
(2026-08-08 rpi4 기록 "30이면 짧은 런 트레이스 안 생김"은 npu-rpi1 에서 재현되지 않음 — 원인 미검증)

### 4-C. 논리적 결함 — 일반화 주장

**보고서 초안의 "무게비 = 슬로우다운 편차 일치" 근거가 YOLOv8 에서 무너짐.**

- YOLOv8 무게비 1.45 = seg(470.7)/pose(323.9), 슬로우다운 편차 1.45 = det(3.81)/seg(2.62)
  → **분자·분모의 주체가 어긋났는데 값만 우연히 일치.** det 는 무게 중간인데 슬로우다운 최대
- MZ3 6.88 vs 6.78 은 주체가 정확히 역대응 → **진짜 성립**
- 참고: Sigma/L 모델이 성립하면 "편차 = 무게비"는 **항등식**

**대신 발견된 패턴 (미검증, 가치 높음)** — 두 워크로드 모두 **on-chip NMS 모델만 예측에서 크게 이탈**:

| 워크로드 | 모델 | 실측/예측 | 후처리 |
|---|---|---|---|
| YOLOv8 | seg / pose | 1.04 / 1.03 | 없음 |
| YOLOv8 | **det** | **1.25** | HEF 내장 NMS |
| MZ3 | mnv2 / deeplab | 1.31 / 1.33 | 없음 / 칩 |
| MZ3 | **ssd** | **1.47** | on-chip NMS |

### 4-D. 인프라 / 데이터 정합성

| 문제 | 상세 |
|---|---|
| **`memory/` 가 `.gitignore`** | 가이드 문서는 `hailo_8L/docs/` 로 복사해 **해결(2026-09-02)**. 다만 `memory/MEMORY.md` 인덱스 수정분은 여전히 커밋 안 됨 — 인덱스는 저장소 공유 대상이 아니므로 영향 없음 |
| **CSV 스키마 불일치** | 사전학습 42런 = 24컬럼(구), 재학습 18런 = 57컬럼(신). **병합 불가**, 변환 스크립트 없음 |
| **npu_percent 미병합** | `npu_percent_*.csv` 별도 파일로만 존재. 본 CSV 컬럼은 NaN. 보고서 수치는 수동 계산이라 **재현 불가** |
| **npu_percent 정의 불일치** | 사전학습 세트는 아예 미측정. 두 세트 NPU% 직접 비교 불가 |
| **npu_percent 샘플 부족** | fps=0 은 5~24샘플(ssd 단독 3.4초에 5샘플). fps=30 은 72샘플. 신뢰도가 다른데 동일 표기 |
| **HRTT 매핑 미검증** | `deeplab_mnv2_ssd`(fps=30)만 파일 6개·dur 19.17s·switch/s 32.0. **3모델이 2모델(60)보다 전환이 적을 수 없음** → 데이터 손실 징후. 조건별 파일 수 합이 42인지 미확인 |
| **분산 미제시** | 전 표가 3회 평균만. -2.4%/+16.4% 가 유의한지 판단 불가. "재현성 확인" 주장의 근거 부족 |
| **조건 등가성 미검증** | YOLOv8 비교군은 `priority=0`/`batch=1` **명시 설정**, MZ3 는 setter 미호출(16/auto). **`batch=auto` 의 실제값 미확인** |
| mnv2 int8 정확도 미측정 | 94.14% 는 float. `eval_gtsrb_hef.py` 작성 완료·실행 안 됨 |
| deeplab 파라미터 미검증 | 2.25M 은 계산 추정. 학습 로그의 `파라미터=?M` 미확인 |
| mIoU 재측정 경로 없음 | argmax 순서 변경 시 `--eval-only`(학습 경로)와 export 모델이 달라짐 |
| 조건명 표기 불일치 | 보고서 `ssd+mnv2` vs HRTT 파서 `mnv2_ssd`(알파벳순). 병합 시 매핑 실수 위험 |
| 실험 폴더 혼재 | `2026-08-31_mz3_default_exp1/` 에 9/1~9/2 재학습 산출물이 누적. 조건이 다른 별개 실험이므로 분리 권장 |

---

## 5. Next Action Items

### 즉시 (각 1분 이내, 후속 분석의 전제)

- [ ] **1. deeplab HEF 컨텍스트 수 측정** — 5절 해석과 오버헤드 계수 논의 전체가 이 값 위에 서 있음.
      현재 "multi-context 3"은 `mz3_sched_bench.cpp` **주석의 추정**일 뿐.
      ```bash
      hailortcli parse-hef ~/mz3_exp/resources/deeplab_v3_mobilenet_v2_wo_dilation.hef | grep -i context
      ```

- [x] **2. `memory/` gitignore 대응 — 완료(2026-09-02).**
      `hailo_8L/docs/2026-08-31_mz3_retrain_baseline_guide.md` 로 복사해 git 추적 가능 상태로 만듦.
      `memory/` 원본도 남아 있으나 커밋되지 않으므로 **docs 쪽을 정본으로 사용할 것**.
      두 곳을 동시에 수정하면 갈라지므로, 앞으로 갱신은 docs 쪽만 할 것.

- [ ] **3. `batch=auto` 실제값 확인** — YOLOv8(batch=1)과의 조건 등가성 근거.

### 코드 수정

- [ ] **4. deeplab 컴파일 재시도** (4-A). argmax 선행안 적용 → export → `[1/4] 파싱` 통과 여부를
      **먼저** 확인(10분 낭비 방지). 실패 시 즉시 플랜 B(`.alls` resize/logits_layer)로 전환.

- [ ] **5. `--eval-as-export` 플래그 추가** — export 경로와 동일한 forward 로 mIoU 재측정.
      argmax 순서 변경 후 47.72 는 배포 모델 값이 아님.

- [ ] **6. `fill_hrtt_columns_mz3.py` 작성** — `tools/hrtt/parse_hrtt.py::compute_metrics()` 재사용.
      기존 `hailo_8L/scripts/fill_hrtt_columns_*.py` 9종이 패턴 참고용.
      `.hrtt` 파일명에 조건 라벨이 없으므로 **`added_core_op` 이름 + run_time 으로 조건 식별**
      (파일명·시각 순 1:1 매핑 사용 금지 — 4-D).

- [ ] **7. npu_percent 병합 스크립트** + **구스키마 → 신스키마 CSV 변환 스크립트**.

### 재측정

- [ ] **8. 재학습 18런 재수집** (`BOUNDED_DUMP=30`) — HRTT 온전 확보. 약 6분.
- [ ] **9. `MNV2_HEF=mobilenet_v2_1.0_gtsrb_sc.hef` 로 18런** — 같은 가중치·1 context.
      "재학습 효과" vs "컨텍스트 전환 비용" 분리. 약 6분.
- [ ] **10. deeplab HEF 확보 후 7조건 42런** (`SKIP_DEEPLAB` 없이). 약 15분.
- [ ] **11. mnv2 int8 정확도** — `eval_gtsrb_hef.py` 실행 (GTSRB 테스트셋 88MB scp 필요).

### 보고서 수정

- [ ] **12. 일반화 결론 문장 교체** (4-C). "두 워크로드에서 성립" → "MZ3 성립, YOLOv8 은 det 이탈".
      **on-chip NMS 모델만 이탈하는 패턴**을 새 절로 추가.
- [ ] **13. 조건 대응표 추가** — YOLOv8 과 MZ3 의 설정 방식 차이 명시(priority 0 vs 16, batch 1 vs auto).
- [ ] **14. 표준편차 열 추가** — CSV 원본에 3행이 다 있으므로 재계산만 하면 됨.
- [ ] **15. 조교님 메시지의 인과 단정 수정** — "컨텍스트 전환 비용으로" → "컨텍스트 수가 다른 HEF 에서".
      노션 9-3절("인과 미확정")과 상충 중.
- [ ] **16. fps=0 각주 추가** — latency(195.4ms)와 FPS(58=17.2ms)가 모순돼 보이는 이유.

### 논증 보강 (문서에 추가할 것)

- **출력 크기는 반대 방향을 가리킴**: 재학습 mnv2 출력 `NC(43)`, 원본 `NC(1001)` → D2H **23배 작음**.
  PCIe Gen3 x1 병목 환경에서 **더 빨라야 정상인데 2배 느려짐** → 전송량 외 요인(컨텍스트)이 지배적임을
  간접 지지. 현재 이 논증이 빠져 있음.

---

## 부록 A. 핵심 파일 위치

| 용도 | 경로 |
|---|---|
| 벤치 소스 | `hailo_8L/experiments/2026-08-31_mz3_default_exp1/src/mz3_sched_bench.cpp` |
| 스윕 스크립트 | 같은 폴더 `run_mz3_retrained_sweep.sh` (SKIP_DEEPLAB / MNV2_HEF 지원) |
| 사전학습 결과 | 같은 폴더 `csv/results_mz3_default{,_fps30}.csv` (구 24컬럼) |
| deeplab 학습 | `hailo_8L/experiments/2026-08-31_mz3_retrain/src/train_deeplab_cityscapes.py` |
| HEF 컴파일 | 같은 폴더 `compile_hailo.py`, `deeplab.alls` |
| HEF 정확도 | 같은 폴더 `eval_gtsrb_hef.py` (미실행) |
| 상세 기술 가이드 | `hailo_8L/docs/2026-08-31_mz3_retrain_baseline_guide.md` (git 추적됨. `memory/` 원본은 gitignore 대상) |
| HRTT 공용 파서 | `tools/hrtt/parse_hrtt.py`, `tools/hrtt/profiler_pb2.py` |
| 프로젝트 지침 | `CLAUDE.md` (보드/계정/버전 제약) |
| 과거 전체 이력 | `PROJECT_SUMMARY.md` (2026-08-19 기준, 계정명 일부 구버전) |
| 10H 측정 한계 | `hailo_10h/experiments/2026-08-24_det_seg_pose_workload_exp1/HRTT_ON_HAILO10H.md` |
| 워크로드 정의 | `workload2.xlsx` (Scenario 시트) |

## 부록 B. 이번 세션에서 발생한 오판 (반복 방지용)

| 오판 | 실제 | 대가 |
|---|---|---|
| `BOUNDED_DUMP` 를 30→3 으로 낮춤 (8/8 rpi4 기록을 잘못 해석) | N 은 링버퍼 길이. 런보다 커야 함 | 재학습 18런 HRTT 손실 → 재수집 |
| DeepLabV3 이니 ASPP 가 있어야 한다고 가정 | TF `model_variant="mobilenet_v2"` 는 atrous_rates 미지정 시 ASPP 없음 | deeplab 재학습 1회 낭비(약 40분) |
| activation duration 확인 전 "컨텍스트 전환 비용" 인과 단정 | HRTT activation = 0.0000ms. 인과 미확정 | 조교님 메시지에 과단정 문장 잔존 |
| 42런 트레이스가 "0개일 것"이라 예측 | 실제 42개 온전 | 불필요한 우려 |
| PyTorch `F.interpolate` 를 그래프에 두고 컴파일 시도 | 공식은 DFC device pre/post layer 로 처리 | 컴파일 10분 낭비, 미해결 |
