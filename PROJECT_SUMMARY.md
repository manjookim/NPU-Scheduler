# NPU Scheduler 실험 프로젝트 — 전체 요약 (AI 인수인계용)

이 문서는 다른 기기(스마트폰 등)에서 다른 AI가 이 프로젝트를 이어서 도와줄 때
빠르게 맥락을 파악할 수 있도록 정리한 문서입니다. 최종 갱신: 2026-08-02.

---

## 1. 프로젝트 목적

Raspberry Pi 위에 장착된 Hailo NPU(**Hailo-8L**, **Hailo-8** 두 보드)에서
**Detection / Segmentation / Pose** 세 개의 YOLOv8 계열 딥러닝 모델을 **HailoRT Model
Scheduler(ROUND_ROBIN)로 동시에** 추론시키면서, 모델별로 독립 설정 가능한 스케줄러
파라미터

- **priority** (0~31, 클수록 우선)
- **threshold** (스케줄 자격을 얻기 위한 최소 큐 누적 프레임 수)
- **timeout** (threshold 미달이어도 강제로 스케줄 자격을 주는 대기시간, ms)
- **batch_size** (네트워크그룹 입출력 큐 크기)

가 지연시간(latency)·처리량·NPU/CPU/메모리 사용률에 미치는 영향을 측정하는
대학교 과제/연구 실험. 조교님(김민주)이 요청사항을 주면 실행하고 결과를 정리해
보고하는 방식으로 진행 중.

---

## 2. 하드웨어 / 환경

| 항목 | Hailo-8L (rpi1) | Hailo-8 (rpi4) |
|---|---|---|
| 보드 | Raspberry Pi 5 | Raspberry Pi 5 |
| NPU | Hailo-8L | Hailo-8 |
| 작업 디렉토리(보드) | `~/hailo_cpp_test/` | `~/hailo_cpp_test/` |
| HEF 경로 | `/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef` 등 | `/home/rpi4/hailo_cpp_test/resources/yolov8s.hef` 등 |
| 이미지 데이터셋 | `/home/rpi1/datasets/sampled_val2017/` | `/home/rpi4/hailo_cpp_test/datasets/sampled_val2017/` |
| 데이터셋 출처 | COCO val2017 샘플(최대 600~673장) | 동일 |
| 세 모델 아키텍처 | Detection=YOLOv8, Segmentation=YOLOv8-seg, Pose=YOLOv8-pose (모두 실제 정확한 서브 버전은 미확인, 코드상 decode 방식으로 추정) | 동일 |
| 빌드 | `g++ -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17` | 동일 |
| 원격 접속 | `ssh rpi1@155.230.16.157 -p 40021`(포트/계정은 보드별로 다를 수 있음, PROJECT_HANDOFF.md 참고) | 유사 |

로컬 저장소: `C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler`
(git 저장소, GitHub: `github.com/manjookim/NPU-Scheduler`, branch `main`)

---

## 3. 코드 구조

```
NPUscheduler/
├── README.md, PROJECT_HANDOFF.md   — 초기 프로젝트 설명/인수인계 문서(구버전, 8L 위주)
├── hailo_8L/
│   ├── infer_scheduler.cpp         — 핵심 C++ 추론 프로그램 (Hailo-8L용)
│   ├── postprocess_8l.hpp          — 후처리(Det NMS 파싱 / Seg,Pose DFL디코딩+NMS+마스크)
│   ├── hailo_utilization.py        — 백그라운드 NPU 사용률 샘플링
│   ├── scripts/                    — 실험 자동화 셸스크립트 모음(아래 4번 참고)
│   ├── docs/, experiments/         — 실험별 결과 폴더(csv/traces/html)
├── hailo_8/
│   ├── infer_scheduler_hailo8.cpp  — Hailo-8L 버전과 거의 동일 구조(HEF경로/IMG_DIR만 다름)
│   ├── postprocess_hailo8.hpp
│   ├── hailo_utilization_hailo8.py
│   ├── scripts/, experiments/
```

### infer_scheduler.cpp / infer_scheduler_hailo8.cpp 핵심 구조 (2026-08-02 리팩토링 후)
과거엔 모든 로직이 `main()` 안에 몰려있었으나, 기능별로 헤더 파일을 분리함
(파일명은 동일하게 유지해 실험 자동화 스크립트의 sed 편집이 그대로 동작):

- `model_types.hpp` — ModelKind/ModelConfig/ModelResult/OutRole/OutMeta 등 데이터 구조체
- `sys_monitor.hpp` — CPU/메모리/컨텍스트 스위치 측정 (`/proc/stat`, `/proc/meminfo`, `/proc/thread-self/status`)
- `image_utils.hpp` — letterbox 전처리, 이미지 파일 목록 로드, 타임스탬프 유틸
- `output_classify.hpp` — output vstream을 채널수 기준으로 역할 분류(box/score/kpts/cls/coeff/proto)
- `model_setup.hpp` — HEF 로드→configure→스케줄러 파라미터 설정, vstream 생성
- `model_runner.hpp` — 모델별 writer/reader 스레드 (1장씩 전처리→추론→후처리, producer/consumer)
- `csv_writer.hpp` — 결과 CSV 저장
- `infer_scheduler.cpp`(메인) — `#define` 파라미터 블록 + `main()`만 남김(오케스트레이션 역할만)

### 현재 소스 코드 아키텍처의 핵심 특징
1. **1장씩 전처리→추론→후처리** (한꺼번에 673장 전처리 X). writer 스레드가 매 프레임
   `imread → letterbox → BGR2RGB → write()`, reader 스레드가 `read() → 후처리 디코딩`.
2. 세 모델(Det/Seg/Pose) output vstream 포맷은 **FLOAT32로 통일**.
3. 후처리:
   - Detection: HEF에 이미 baked-in된 on-chip NMS 결과(`[batch,num_classes,5,num_proposals]`)를
     `decode_det()`가 파싱 + letterbox unpad 좌표복원 + COCO id 매핑.
   - Segmentation/Pose: raw tensor를 host C++에서 직접 DFL 디코딩 + NMS + (Seg는 prototype
     mask ×coefficient 행렬곱으로 마스크 복원)까지 수행 — **NPU가 아니라 항상 host CPU에서 처리됨**
     (Hailo Dataflow Compiler 문서 확인 결과, Seg/Pose 모두 on-chip NMS 지원 대상 아키텍처가 아님 — 6번 항목 참고).
4. CSV에 전처리/추론(latency)/후처리 시간을 모델별로 각각 독립 기록.
5. 진단용 매크로(`ENABLE_POSTPROCESS`, `DEBUG_WRITE_TIMING`)로 후처리 on/off, 큐 크기 진단 가능.

---

## 4. 실험 이력 (연대순 요약)

| 날짜/폴더 | 보드 | 내용 | 조건 | 상태 |
|---|---|---|---|---|
| `hailo_8L/experiments/2026-07-10_scheduler_param_verification` | 8L | 스케줄러 파라미터(threshold/timeout/priority)가 실제 적용되는지 HRTT `core_op_set_value` 트레이스로 검증 | 1회성 확인 실험 | 완료 |
| (초기 63개 실험, README.md 기준) | 8L | Single(3)+2모델조합(27)+3모델조합(27)=63개, priority(0/15/31) 조합 | batch=1 등 초기값 | 완료(구버전 코드) |
| `hailo_8L/experiments/2026-07-26_batch_priority_exp1` | 8L | priority × batch_size 전체 조합 실험 | 여러 조합 × 3회 반복, 252 조건 | 완료, xlsx 통합(전체+3회평균 시트) |
| Hailo-8 배치사이즈 스윕 실험 (`hailo_8/experiments/`) | 8 | batch(1/16/32/48/63) × 단일/2조합/3조합 | 3회 반복 | 완료, xlsx 통합 |
| **[핵심 아키텍처 변경, 07-28]** | 8→8L 포팅 | Hailo-8에 있던 후처리(decode_det/pose/seg) + FLOAT32 출력 통일 + 프레임별 전처리 구조를 Hailo-8L에도 포팅 (8L엔 원래 후처리 로직이 전혀 없었음) | — | 완료, 실기 검증됨 |
| `hailo_8L/experiments/2026-07-28_default_workload_exp1` | 8L | 조교님 요청: "메모리 사용률이 이상해서" — batch=1/threshold=1/timeout=0/priority=16(SDK 기본값) 고정, single(3)+2조합(3)+3조합(1)=7조건×3회=21회 | default 파라미터 workload 실험 | 완료 |
| **큐사이즈 실측 검증 (07-29)** | 8L | batch_size가 커지면 vstream 큐(버퍼링 여유)도 커지는지 write()블로킹시간 + HailoRT 공식 `get_queue_size_accumulators()` API 두 방법으로 실측 | batch=1 vs batch=63 비교 | 완료 — "커진다"로 결론(단, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE 상수 자체는 고정값=2, 실제로 커지는 건 네트워크그룹 자체의 batch_size 파라미터) |
| `hailo_8L/experiments/2026-07-29_batch_singlemodel_pp_exp1` | 8L만 (8은 사용자가 "안해도돼"라고 명시적으로 스킵) | batch{1,17,32,63} × 싱글모델만(Det/Seg/Pose 각각 단독) × 후처리 ON/OFF × 3회 반복 = 72회 | 결과: `csv/results_pp_on.csv`, `results_pp_off.csv` (각 36행), traces/html을 pp_on/pp_off 폴더로 분리 정리 완료(조교님껜 pp_off만 전달 예정) | 완료 |

### 미실행/보류
- Hailo-8 쪽 batch{1,17,32,63}×pp on/off 실험 스크립트는 준비됐지만(`hailo_8/scripts/auto_experiment_batch_singlemodel_pp.sh`) 사용자가 "8은 안해도돼"라고 해서 **실행 안 함**.

---

## 5. 핵심 발견/제약사항 모음

- **threshold ≤ batch_size 필수**: 초과 시 `set_scheduler_threshold`가 `HAILO_INVALID_ARGUMENT`로 실패, threshold는 기본값(1)로 남음.
- **timeout=0이면 threshold가 사실상 무력화**됨(1프레임만 있어도 즉시 활성화 자격 얻음). threshold 효과를 보려면 timeout>0 필요.
- **입력을 한꺼번에(최대속도) 밀어넣으면 큐가 항상 포화 상태**가 되어 threshold/timeout 효과 관측 어려움 → `INPUT_FPS`로 입력 속도 제한 가능.
- **priority 차이 15 이상이면 낮은 priority 모델이 starvation**(NPU 스케줄링 기회를 거의 못 받음).
- **단일 모델만 써도 NPU 사용률이 ~99~100%로 찍히는 이유**: writer가 쉬지 않고 계속 write하는 구조라(INPUT_FPS=0) 큐가 항상 채워져 있어서 — 다중 모델 경쟁과 무관.
- **batch_size는 "네트워크그룹 입출력 큐 크기"** 역할도 함 — HailoRT VStreams API의 `queue_size` 파라미터(고정값 2, `HAILO_DEFAULT_VSTREAM_QUEUE_SIZE`)와는 별개의, network group configure 시점의 `batch_size`가 실질적인 버퍼링 여유를 결정한다는 것을 실측으로 확인함.
- **후처리(특히 Segmentation)가 무거우면 "측정된 latency" 자체가 늘어남**: reader 스레드가 `read()→후처리→다음read()` 순차 구조라, 후처리가 느리면 큐 배수가 늦어져 다음 프레임의 enqueue-dequeue 간격(=latency로 측정하는 값)이 밀림. 후처리 자체 시간이 latency 이후에 측정되더라도 간접적으로 영향을 줌 (pp_on 765ms vs pp_off 477ms, batch=1 Seg 예시로 확인).
- **HailoRT 최대 batch_size = 63** (64 아님).
- **avg_preprocess_ms / postprocess_ms / total_time_ms**: 07-28 이전 실험은 "전처리 시간"이 모델 공유값(한꺼번에 전처리)이었고, 07-28 이후부터 모델별 독립 측정으로 바뀜 — 옛 데이터와 신 데이터를 비교할 때 이 컬럼 의미가 다르므로 주의.

---

## 6. NPU(neural core)에서 후처리가 가능한가? (조교님 최신 질문, 조사 완료, 답변 대기중)

Hailo Dataflow Compiler(모델을 HEF로 컴파일하는 도구)의 `nms_postprocess(engine=nn_core)`
모델 스크립트 명령어로 **컴파일 시점**에 NMS를 NPU(neural core)에서 돌게 만들 수 있는
기능이 실제로 존재함. 단, **아키텍처별 지원 제약이 있음**(DFC User Guide 및 Hailo 공식
GitHub `hailo_model_zoo` 설정파일로 확인):

| 모델 | 지원 엔진 | 근거 |
|---|---|---|
| SSD / CenterNet | neural core 지원 | DFC User Guide 69p |
| YOLOv5 (detection) | neural core 일부(bbox decoding+score threshold만), IoU필터링은 CPU | DFC User Guide 69p, 84p |
| **YOLOv8 (우리 Det)** | **CPU 전용, neural core 미지원** | DFC User Guide 69p + 공식 GitHub `hailo_model_zoo/.../yolov8n.alls`에 `engine=cpu` 하드코딩 확인 |
| **YOLOv5-SEG류 (우리 Seg 추정)** | **CPU 전용** | DFC User Guide 72p "The post-processing runs on the CPU" 명시 |
| **Pose** | **이 기능의 대상 아키텍처 목록에 아예 없음**(`NMSMetaArchitectures` enum에 Pose 계열 없음, DFC User Guide 173p) | 애초에 NPU 처리 옵션 자체가 없음, host 처리가 유일한 방법 |

**결론(조교님께 전달할 답변 초안)**: NPU에서 후처리를 처리하는 기능 자체는 존재하지만
(DFC의 `nms_postprocess(engine=nn_core)`), **우리가 쓰는 모델 아키텍처(YOLOv8 Det,
YOLOv5-SEG류 Seg)는 이 기능의 지원 대상이 아니라서 실제로는 불가능**하고, Pose는 애초에
이 기능과 무관한 영역. 그리고 이건 **런타임 C++ 코드로 바꿀 수 있는 게 아니라 HEF
컴파일 시점(Dataflow Compiler)에 결정되는 값**이라, 설령 지원되는 아키텍처였어도
`infer_scheduler.cpp`를 고치는 게 아니라 원본 모델을 재컴파일해야 하는 작업임.

현재 **조교님께 질문을 드려놓고 답변 대기 중** (2026-08-02 기준).

---

## 7. 현재 진행 중 / 대기 중인 작업

- [대기] 조교님 답변 대기 중 (NPU 후처리 가능 여부 질문에 대한 회신)
- [완료, 08-02] 코드 리팩토링: `infer_scheduler.cpp`/`infer_scheduler_hailo8.cpp`를 기능별 헤더 파일로 분리 (스파게티 코드 개선 요청 처리)
- [미확인] `git push origin main:master`를 사용자가 로컬에서 실제로 실행했는지 마지막 확인 안 됨 (샌드박스에는 GitHub 인증정보가 없어 직접 push 불가, 사용자가 본인 PC에서 실행해야 함)
- [보류] Hailo-8 쪽 batch{1,17,32,63}×pp on/off 실험(스크립트는 준비됨, 미실행)
- [완료] 7월 연구실 활동일지(매주 화/목) 초안 작성해 채팅으로 전달함

---

## 8. 파일 위치 빠른 참고

- 실험 결과 통합 엑셀: `hailo_8L/experiments/2026-07-26_batch_priority_exp1/results_batch_priority_combined*.xlsx`, `hailo_8/experiments/results_batch_sweep_combined*.xlsx`
- 최신 batch×pp 실험: `hailo_8L/experiments/2026-07-29_batch_singlemodel_pp_exp1/{csv,traces,html}` (pp_on/pp_off 하위폴더로 분리됨)
- Notion 정리 페이지: "7/28 NPU 스케줄러 실험 작업 정리" (workspace 최상위, 프로젝트 홈 없음 — 독립 페이지)
- GitHub: `github.com/manjookim/NPU-Scheduler`, branch `main` (로컬 기준 여러 커밋 앞서 있고 push 상태 불확실)
