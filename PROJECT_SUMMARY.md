# NPU Scheduler 실험 프로젝트 — 전체 요약 (AI 인수인계용)

이 문서는 다른 기기(스마트폰 등)에서 다른 AI가 이 프로젝트를 이어서 도와줄 때
빠르게 맥락을 파악할 수 있도록 정리한 문서입니다. 최종 갱신: 2026-08-19.

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
| `hailo_8L/experiments/2026-08-04_batch1_pri0_fps30_exp1` | 8L | batch=1/threshold=1/timeout=0/priority=0 고정 + INPUT_FPS=30, single(3)+2조합(3)+3조합(1)=7조건×3회=21회, 후처리 ON(측정함) | 결과: `csv/results_batch1_pri0_fps30.csv` + combined xlsx | 완료 |
| ~~`hailo_8L/experiments/*_default_pri0_ppoff_exp1`~~ (사용 중단) | 8L | INPUT_FPS=0 버전 — 사용자가 실행 도중 "fps 30으로 해야돼"라고 정정하여 폐기. RPi에서 잠깐 백그라운드로 돌았을 수 있음(kill 후 결과 폴더는 안 써도 됨) | 스크립트: `auto_experiment_default_pri0_ppoff.sh`(사용 중단 주석 추가함) | 중단 — 아래 fps30 버전으로 대체 |
| `hailo_8L/experiments/2026-08-05_default_pri0_fps30_ppoff_exp1` | 8L | batch=1/threshold=1/timeout=0/priority=0(전부 default) 고정, INPUT_FPS=30(2026-08-04_batch1_pri0_fps30_exp1과 동일), **ENABLE_POSTPROCESS=0 매크로 토글로 후처리 시간 미측정**, single(3)+2조합(3)+3조합(1)=7조건×3회=21회 | 완료, HRTT까지 채움. `csv/results_default_pri0_fps30_ppoff.csv`. pp ON(08-04) 대비 비교 결과: Segmentation 단독일 때만 latency -38.7%(768→471ms, reader 스레드가 read→후처리→다음read 순차구조라 무거운 Seg 후처리가 다음 프레임 대기시간까지 밀어서 생기던 왜곡이 사라짐), Detection/Pose 단독은 거의 무변화(0%), 다중조합은 -1~-5% 수준(NPU 스케줄링 경합이 지배적) | 완료 |
| `hailo_8L/experiments/{실행일}_default_pri0_fps30_noppt_exp1` (스크립트만 준비, 실행 대기) | 8L | 조교님이 "ENABLE_POSTPROCESS 매크로 토글 말고 아예 후처리 시간 측정 코드 자체를 없앤 버전"을 요청 — batch=1/threshold=1/timeout=0/priority=0/INPUT_FPS=30 동일 조건, single(3)+2조합(3)+3조합(1)=7조건×3회=21회 | 매크로 토글이 아니라 완전히 별도 파일로 분리: `hailo_8L/infer_scheduler_noppt.cpp`(infer_scheduler.cpp와 diff는 model_runner include와 ENABLE_POSTPROCESS 매크로 유무뿐) + `hailo_8L/model_runner_noppt.hpp`(pp_t0/pp_t1 타이머와 decode_det/pose/seg 호출 자체를 삭제, `result.avg_postprocess_ms`를 아예 대입하지 않아 ModelResult 기본값 -1 유지 → csv_writer.hpp 관례대로 CSV엔 항상 NaN). 실행 스크립트: `hailo_8L/scripts/auto_experiment_default_pri0_fps30_noppt.sh`, 결과처리: `hailo_8L/scripts/fill_hrtt_columns_default_pri0_fps30_noppt.py`. **보드에서 실기 실행 전 — scp 후 사용자가 RPi에서 nohup으로 직접 실행 필요** | 스크립트/코드 준비 완료, 실행 대기 |

### 미실행/보류
- Hailo-8 쪽 batch{1,17,32,63}×pp on/off 실험 스크립트는 준비됐지만(`hailo_8/scripts/auto_experiment_batch_singlemodel_pp.sh`) 사용자가 "8은 안해도돼"라고 해서 **실행 안 함**.
- `hailo_8L/scripts/auto_experiment_default_pri0_fps30_ppoff.sh` — 위 표 참고, 코드/스크립트만 준비됨, 실기 실행 안 함(샌드박스에 RPi SSH 접근 권한 없음, 사용자가 직접 scp+ssh로 실행해야 함).

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
- **[08-07] rpi4에서 `hailo/hailort.hpp`가 표준 경로(`/usr/include`)가 아니라
  `/home/rpi4/hm/monitor/hailort/hailort/libhailort/include/`에 있음** — 실기에서
  `find / -name hailort.hpp`로 확인. 이 경로를 `-I`로 명시하지 않고 빌드하면 include를
  못 찾아 대량 연쇄 컴파일 에러가 남(사용자가 "코드 오류 1000줄"로 보고한 원인). `.so`는
  `/usr/lib/libhailort.so`로 ldconfig 캐시에 정상 등록돼 있어 `-L`은 불필요, `-I`만 추가하면
  해결됨. `hailo_8/scripts/run_v5det_seg_pose.sh` / `run_v5det_cpu_seg_pose.sh` /
  `run_yolov5_nms_core.sh` 세 개 다 `HAILORT_INCLUDE` 변수 추가해 고쳐둠. **원본
  `infer_scheduler_hailo8.cpp`를 위한 기존 자동화 스크립트(`auto_experiment_*.sh` 등)는
  이 문제를 언제 겪었는지 불명 — 만약 그쪽도 같은 증상이면 동일하게 `-I` 추가 필요.**

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

**[2026-08-06 후속 조사 완료]** YOLOv5는 `engine=nn_core`(SSD/Centernet과 함께 지원됨) 외에
YOLOv5 전용 `engine=auto`(bbox 디코딩+score threshold는 neural core, IoU 필터링만 CPU)도
지원한다는 걸 Hailo 공식 문서(DFC User Guide Model Script 레퍼런스, RidgeRun 미러 페이지로 확인)
와 hailo_model_zoo 공식 저장소(github.com/hailo-ai/hailo_model_zoo)를 재조사해 확인함.
중요한 건 **직접 재컴파일할 필요 없이, hailo_model_zoo가 이미 컴파일해 공개 배포 중인
HEF가 있다는 것** — `yolov5xs_wo_spp_nms_core.hef` (hailo8 타겟, 입력 512x512x3):
`https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov5xs_wo_spp_nms_core.hef`
CHANGELOG.rst엔 "contains bbox decoding and confidence thresholding on Hailo-8"로 설명돼
있어 최소 bbox 디코딩+score threshold는 확실히 칩에서 돈다(nn_core/auto 중 정확히 어느 쪽인지는
`hailortcli parse-hef`로 실기 재확인 필요). `hailo_8/infer_yolov5_hailo8.cpp`(신규, 이 모델
전용 단독 벤치마크 프로그램)로 batch=1/threshold=1/timeout=0/priority=0 조건에서 타이밍
측정까지 코드는 준비됨 — RPi(rpi4)에 HEF를 받아 실행하는 것만 남음
(`hailo_8/scripts/run_yolov5_nms_core.sh` 참고). YOLOv8 Det/Seg 쪽 결론(CPU 전용)은 변동 없음.

---

## 6-1. CSV 스키마 변경 (2026-08-06)

`total_time_ms_nopp_det/seg/pose` 컬럼 3개를 `csv_writer.hpp`에 추가(`total_time_ms_pose` 바로 뒤,
맨 끝). 후처리 측정 여부와 무관하게 항상 `avg_preprocess_ms_* + *_latency_ms`만 더한 값(=후처리
제외 전체시간) — 기존 `total_time_ms_*`(후처리 측정됐으면 포함)와 같은 행에서 나란히 비교 가능.
(중간에 "후처리 측정 여부" boolean 플래그 컬럼을 먼저 추가했었는데, 사용자가 원한 건 그게 아니라
이 숫자 컬럼이라고 정정하여 플래그는 제거하고 이걸로 교체함.) `model_types.hpp`/`model_runner.hpp`/
`model_runner_noppt.hpp`는 변경 없음(csv_writer.hpp 계산만으로 충분). 기존 결과 2건도 소급 패치함
(`results_batch1_pri0_fps30.csv`+`_avg.csv`, `results_default_pri0_fps30_ppoff.csv`). 단
`results_batch1_pri0_fps30_combined.xlsx`는 CSV 패치 전 버전이라 컬럼이 안 맞음 — 재생성 필요.

## 7. 현재 진행 중 / 대기 중인 작업

- [대기→일부 해소, 08-06] NPU 후처리 가능 여부: YOLOv5는 가능(engine=nn_core/auto), 공식 배포
  HEF(`yolov5xs_wo_spp_nms_core.hef`)도 찾음 — §6 참고. Segmentation/Pose는 재확인해도 NPU측
  후처리 옵션이 아예 없음(CPU 전용) — 공식 문서 원문(`nms_postprocess()` 3개 엔진 모드 목록)과
  hailo_model_zoo 실제 배포 파일 둘 다로 재검증함.
  - `hailo_8/infer_yolov5_hailo8.cpp` — YOLOv5(NPU) 단독 벤치마크(비교용 기준선)
  - `hailo_8/infer_scheduler_hailo8_v5det.cpp` — **Detection만 YOLOv5(NPU 후처리)로 교체,
    Segmentation/Pose는 기존 YOLOv8(CPU 후처리) 그대로 유지한 3모델 동시 실행 버전**
    (batch=1/threshold=1/timeout=0/priority=0 통일). 원본 `infer_scheduler_hailo8.cpp`는
    자동화 스크립트가 sed로 편집하므로 손대지 않고 별도 사본으로 만듦.
  코드 준비 완료, **실기 실행(RPi에 HEF 다운로드 + 빌드 + 실행)만 남음** — 샌드박스는 RPi 접근
  권한이 없어 실행 불가, 사용자가 `hailo_8/scripts/run_v5det_seg_pose.sh`(3모델) 또는
  `run_yolov5_nms_core.sh`(YOLOv5 단독) 직접 실행 필요.
- [완료(코드), 08-07] 사용자가 "3개 모델 다 YOLOv5로 바꾸고, Detection은 NPU 후처리 1회 +
  CPU 후처리 1회로 비교하고 싶다"고 요청. hailo_model_zoo 공식 문서 재조사 후 범위를
  사용자와 함께 다음과 같이 확정함:
  - **Detection**: YOLOv5로 전환 확정. NPU(`yolov5xs_wo_spp_nms_core.hef`, engine=nn_core/auto)와
    CPU(`yolov5xs_wo_spp.hef`, engine=cpu) 두 HEF가 **같은 아키텍처·같은 입력크기(512x512x3)**로
    공식 배포되어 있어(`docs/public_models/HAILO8/HAILO8_object_detection.rst`) 후처리 엔진
    차이만 격리해서 비교하기 좋은 쌍. 출력이 둘 다 HAILO_NMS_BY_CLASS 포맷이라 기존
    `decode_det()` 그대로 재사용(로직 변경 없음).
  - **Segmentation**: YOLOv5-seg(`yolov5s_seg.hef` 등)는 공식 모델이 존재하긴 하나 박스 디코딩이
    앵커 기반이라(YOLOv8의 DFL 기반 `decode_seg()`와 수학이 다름) 새 디코딩 로직을 짜야 하고,
    실기 없이는 정확도 검증이 불가능함. NPU 후처리 옵션 자체가 Seg엔 없어서(v5든 v8이든 무관)
    바꿔도 "NPU 후처리 가능 여부"라는 실험 결론에는 영향이 없음 → **사용자가 이번 범위에서
    제외하기로 결정**(YOLOv8-seg CPU 후처리 그대로 유지).
  - **Pose**: YOLOv5 계열 자체가 공식 Model Zoo에 없음(포즈 추정은 YOLOv8부터 추가된 기능이라
    "YOLOv5-Pose" 공식 모델이 아예 존재하지 않음) — 어떤 방법으로도 대체 불가, YOLOv8-pose 유지.
  - 신규 파일: `hailo_8/infer_scheduler_hailo8_v5det_cpu.cpp`(Detection=YOLOv5 **CPU** 후처리,
    `infer_scheduler_hailo8_v5det.cpp`와 DET_HEF/모델명 라벨 외 전부 동일 — diff로 확인함),
    `hailo_8/scripts/run_v5det_cpu_seg_pose.sh`(CPU판 실행 스크립트, `run_v5det_seg_pose.sh`와
    짝), `hailo_8/scripts/run_v5det_npu_vs_cpu_compare.sh`(두 스크립트를 순서대로 한 번에
    실행하는 래퍼 — NPU 실행 후 CPU 실행).
  코드/스크립트 준비 완료.
- [08-07 후속] 사용자가 실험 설계를 **det / det+seg / det+pose / det+seg+pose 4조합** 비교로
  확정(기존 README.md의 single/2조합/3조합 관례와 동일한 방식, Detection은 축이라 항상 켜둠).
  `run_v5det_seg_pose.sh`/`run_v5det_cpu_seg_pose.sh` 둘 다 4조합 스윕으로 재작성함 —
  `infer_scheduler_hailo8_v5det[_cpu].cpp`의 `#define USE_SEG`/`USE_POSE` 두 줄을 조합마다
  sed로 토글 → 재빌드 → 3회 반복 실행, 을 4번 반복(조합당 3회 × 4조합 = 12회, NPU+CPU 합쳐
  총 24회). 결과는 조합 구분 없이 같은 CSV 하나에 계속 쌓이고(`csv_writer.hpp`의
  `use_det/use_seg/use_pose` 컬럼으로 조합 식별), 스크립트가 매 조합마다 그 컬럼 기준으로
  이미 몇 개 모였는지 세서 부족한 만큼만 더 실행 — 크래시로 중단돼도 재실행하면 이어서 진행됨.
  - 결과 CSV 경로 고정: NPU=`hailo_8/experiments/2026-08-07_v5det_seg_pose_exp1/csv/results_v5det_seg_pose.csv`,
    CPU=`hailo_8/experiments/2026-08-07_v5det_cpu_seg_pose_exp1/csv/results_v5det_cpu_seg_pose.csv`
    (날짜 폴더를 스크립트에 고정 하드코딩 — 재실행 날짜가 달라져도 같은 CSV에 누적되게 함).
- [08-07 실기 실행 이슈 — 둘 다 해결·완화됨]
  1. rpi4의 `hailort.hpp`가 표준 경로가 아니라 `/home/rpi4/hm/monitor/hailort/hailort/libhailort/include/`에
     있어 `-I` 없이 빌드하면 대량 연쇄 컴파일 에러 → 세 스크립트에 `HAILORT_INCLUDE` 변수 추가로 해결.
  2. `infer_scheduler_hailo8_v5det.cpp` 16번째 줄 주석("BATCH_*/THRESHOLD_*/...") 안에서 우연히
     형성된 `*/`가 `/** */` 블록 주석을 조기 종료시켜 그 뒤 설명 문장이 코드로 파싱되며 또 한 번
     대량 에러 → 주석 문구를 슬래시 없는 형태로 수정해 해결(08-06 준비 시 실기 컴파일 테스트를
     한 번도 안 해서 지금까지 안 걸렸던 잠재 버그).
  3. `QUESTION_FOR_TA.md`의 `hailo_pci` 드라이버 크래시(`find_vma` 레이스)가 실제로 재발함
     (run_id=1 완료 후 run_id=2 시작 시점) — 단 이번엔 `hailortcli fw-control identify`가
     정상 응답해 재부팅 없이 복구됨. 이후 재시도 중 또 한 번, 이번엔 커널 크래시가 아니라
     **프로세스 종료(정리) 시점 세그폴트**가 발생(`run_id=1` 결과는 CSV에 이미 정상 저장된 뒤
     죽음) — 두 경우 다 "결과가 이미 저장된 뒤" 죽는 패턴이라, 스크립트를 "CSV에 새 줄이
     실제로 늘었는지"로 성공 여부를 판단하고 실패 시 같은 조건으로 자동 재시도하도록
     고침(`set -e`가 전체를 멈추지 않게 개별 실행만 `||`로 감쌈). 세그폴트 자체의 근본 원인
     (아마 VDevice/네트워크그룹 정리 순서 관련 HailoRT 바인딩 이슈로 추정)은 별도로 디버깅
     안 됨 — 데이터가 이미 저장된 뒤라 실험 결과에는 영향 없다고 판단하고 재시도 방식으로 우회.
  - 실행 상태: NPU 조건 `det+seg+pose` 조합만 부분 진행 중(크래시로 여러 번 중단·재시도),
    나머지 3조합(NPU) + 4조합 전체(CPU)는 미실행. 사용자가 `run_v5det_npu_vs_cpu_compare.sh`
    (또는 개별 스크립트) 재실행해서 이어가는 중.
- [완료, 08-02] 코드 리팩토링: `infer_scheduler.cpp`/`infer_scheduler_hailo8.cpp`를 기능별 헤더 파일로 분리 (스파게티 코드 개선 요청 처리)
- [미확인] `git push origin main:master`를 사용자가 로컬에서 실제로 실행했는지 마지막 확인 안 됨 (샌드박스에는 GitHub 인증정보가 없어 직접 push 불가, 사용자가 본인 PC에서 실행해야 함)
- [보류] Hailo-8 쪽 batch{1,17,32,63}×pp on/off 실험(스크립트는 준비됨, 미실행)
- [완료] 7월 연구실 활동일지(매주 화/목) 초안 작성해 채팅으로 전달함

---

## 7-1. [2026-08-08] v5det NPU-vs-CPU 실험 리뷰 + 버그 수정

사용자가 "det만 NPU 후처리 실험이 제대로 안 된 것 같다"고 해서 코드/CSV/xlsx/.hrtt를
직접 뜯어본 결과와 그 자리에서 고친 내용.

**확인된 현상(버그 아님, 재현됨)**: det 단독 latency가 NPU 후처리 조건(152.5~152.9ms)이
CPU 후처리 조건(4.6ms)보다 33배 느림. 조합이 늘수록 격차는 줄어듦(det+seg 4.6배 →
det+pose 1.5배 → det+seg+pose 1.25배). cpu_percent도 CPU 조건이 더 높아서(55.9% vs
22.7%) "CPU 조건은 host에서 NMS를 도는" 가설과 방향은 맞음. 다만 두 HEF가 정말 후처리
엔진만 다른 동일 아키텍처인지 `hailortcli parse-hef`로 실측 diff는 아직 안 해봤고,
decode_det()가 실제로 유효한 검출을 만드는지(개수/좌표) 정확도 검증도 프로젝트 전체에서
한 번도 없었음 — 이 두 가지가 확인되기 전까진 "NPU 후처리가 진짜로 느리다"를 확정할 수
없음. 다음 세션에서 우선순위로 확인할 것.

**확인된 버그 3가지, 전부 수정함**:

1. **`HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=90` 오타성 버그** — `run_v5det_seg_pose.sh`/
   `run_v5det_cpu_seg_pose.sh`만 90을 쓰고 있었고(프로젝트 내 다른 모든 스크립트는 30,
   `hailo_8L/PROJECT_HANDOFF.md`에 "dump 값 크거나 없으면 fps 저하"라고 명시돼 있었음),
   그 결과 90초보다 일찍 끝나는 조합(det/det+seg/det+pose)은 `.hrtt` 파일이 아예 안 만들어짐
   — 실제로 `2026-08-07_v5det_seg_pose_hrtt_exp1/traces/`엔 det+seg+pose(유일하게 90초를
   넘긴 조합) 트레이스 3개만 있고 나머지 9번의 실행분 트레이스는 하나도 없었음. 90초 넘게
   걸리는 조합에서도 latency 자체를 부풀렸을 가능성 있음(문서화된 부작용). 두 스크립트
   모두 30으로 되돌림.
2. **`run_combo()`가 CSV 행 수만 보고 "완료" 판정** — `.hrtt`가 하나도 안 생겨도 CSV에
   REPEATS(3)개만 쌓이면 다음 조합으로 넘어갔음(1번 버그의 원인이자 결과). 이제
   `count_traces()`를 추가해서 CSV 행 수와 트레이스 개수 둘 다 REPEATS에 도달해야
   완료로 치고, 트레이스가 부족하면 max_attempts까지 자동으로 더 재시도하도록 고침.
3. **`npu_percent`가 항상 NaN** — `hailo_utilization_hailo8.py`(백그라운드 폴링, ProtoMon
   기반)가 이 실험에서 샘플을 하나도 못 얻었던 것으로 보임(venv/의존성 문제로 추정, 실기에서
   직접 확인 필요). `.hrtt` 트레이스 자체엔 이미 모델별 `device_usage_pct`(해당 core_op가
   활성이었던 시간 비율)가 있어서, 새로 만든 `fill_hrtt_columns_v5det.py`가 활성 모델들의
   `device_usage_pct` 합을 npu_percent 대신 채우도록 함(외부 폴링 실패해도 항상 채워짐).
   실제로 det+seg+pose 트레이스 3개에 바로 적용해봤더니 npu_percent≈99.998%로 채워짐
   (3모델이 라운드로빈으로 계속 NPU를 점유하고 있었다는 뜻, 말이 됨).

**의외의 발견**: `.hrtt` 트레이스 기반 순수 device latency(H2D→D2H, det≈26ms)가 앱이 자체
측정한 `det_latency_ms`(≈1316ms, det+seg+pose 조합)와 50배 차이남. `det_latency_ms`는
`inputs[0].write()`가 큐가 찰 때까지 블로킹 대기하는 시간까지 포함한 "앱 관측 왕복시간"이고
hrtt의 avg_latency는 순수 디바이스측 시간이라 정의가 다름 — 어느 쪽이 틀린 게 아니라 서로
다른 걸 재는 지표. `fill_hrtt_columns_v5det.py`는 이 hrtt측 값을
avg_latency_det/seg/pose 컬럼에 채우고 기존 det_latency_ms는 안 건드림(둘 다 남겨서
비교 가능하게). det+seg+pose 조합의 700~1300ms대 숫자는 1번 버그(BOUNDED_DUMP=90)의
영향을 받았을 가능성이 있어 재수집 권장.

**신규 파일**: `hailo_8/scripts/fill_hrtt_columns_v5det.py` — hailo_8L의
`fill_hrtt_columns_batch1_pri0_fps30.py` 패턴을 hailo_8 v5det 실험용으로 이식.
`tools/hrtt/parse_hrtt.py`의 `compute_metrics()`를 그대로 재사용(보드/실험 무관 공용 파서).
`hailo_8/experiments/2026-08-07_v5det_seg_pose_hrtt_exp1/csv/results_v5det_seg_pose.csv`의
det+seg+pose 3행(run_id=10/11/12)엔 이미 적용 완료.

**데이터 정리**: 레포 루트에 잘못 위치해 있던(스크립트/문서가 가리키는
`hailo_8/experiments/` 경로가 아니라 리포 최상위에 있었음 — 아마 예전 스크립트 버전을
잘못된 작업 디렉토리에서 실행한 결과로 추정) "모니터링 없이 수집한 깨끗한" 12+12행짜리
CSV 두 개를:
  - `2026-08-07_v5det_seg_pose_exp1/csv/results_v5det_seg_pose.csv` →
    `hailo_8/experiments/2026-08-07_v5det_seg_pose_nomonitor_exp1/csv/`로 이동
  - `2026-08-07_v5det_cpu_seg_pose_exp1/csv/results_v5det_cpu_seg_pose.csv` →
    `hailo_8/experiments/2026-08-07_v5det_cpu_seg_pose_nomonitor_exp1/csv/`로 이동
"nomonitor" 접미사로 구분한 이유: 이 데이터는 HAILO_TRACE/HAILO_MONITOR가 전혀 안 켜진
상태로 수집돼서(위 버그들의 영향이 없음) det 단독 152.9ms/4.6ms 비교는 신뢰할 수 있지만,
`_hrtt_exp1`(모니터링 켠 채로 수집, 위 버그들 영향 받음) 데이터와 절대 같은 표에 섞으면
안 됨 — 측정 조건 자체가 다름. 레포 루트에 남아있던 빈 폴더 2개
(`2026-08-07_v5det_seg_pose_exp1/`, `2026-08-07_v5det_cpu_seg_pose_exp1/`)는 워크스페이스
폴더 삭제 제한 때문에 못 지웠음 — 안에는 아무 내용 없음, 무시해도 됨.

**[2026-08-08 실기 1차 실행 결과 — 추가 버그 2개 발견, 수정함]**
사용자가 실제로 rpi4에서 두 스크립트를 돌려본 결과:
- parse-hef 확인 + 첫 프레임 디버그 로그로 NPU 후처리 조건의 decode_det()가 실제로 유효한
  검출(8개, coco_id=1, 좌표 정상)을 만든다는 것 확인함. det 단독 latency도 152.73ms로
  또 재현(네 번째 독립 확인) — 진짜 결정론적 값으로 봐도 됨.
- **det 단독처럼 짧게 끝나는 조합(~9.6초)은 BOUNDED_DUMP=30으로 고쳤는데도 여전히 트레이스가
  하나도 안 생김**(실기에서 15번 시도, CSV는 15행 쌓였는데 트레이스 0/3). 반면 90초 넘게
  걸리는 det_seg_pose는 3/3 계속 성공, det_seg/det_pose(중간 길이)는 6번 중 3번(50%) 성공.
  → "N초가 지나야 덤프가 트리거된다"는 가설과 정확히 들어맞음(짧을수록 실패율 100%에
  가까움). 가장 짧은 조합(det 단독, ~9.6초)보다 확실히 작아야 해서 30 → **3**으로 다시 낮춤.
  이 값이 "그 시점까지의 전체 트레이스를 덤프"인지 "최근 N초만 남기는 링버퍼"인지는
  Hailo 쪽 공식 문서를 못 찾아서(웹 검색해봤지만 이 환경변수에 대한 공개 문서 없음)
  확실친 않음 — 다음 실행 후 det 단독 트레이스의 hrtt_run_time_sec가 진짜 실행시간(~9.6초)과
  비슷하게 나오는지로 검증할 것. 만약 3초 근처로 뚝 잘려 나오면 "최근 N초만" 방식이 맞다는
  뜻이라 다른 전략이 필요함(예: 조합별로 다른 값을 주거나, 짧은 조합은 아예 이 방식을 포기).
- **`fill_hrtt_columns_v5det.py`가 rpi4에서 protobuf `ImportError: cannot import name
  'runtime_version'`로 실패함** — 보드 시스템 python3의 `google-protobuf` 패키지가
  `profiler_pb2.py`를 생성한 protoc 버전(6.33.5)보다 오래됨. 두 실행 스크립트의
  자동 호출부를 `~/hailo_platform_venv`(있으면 더 최신 protobuf일 수 있음)로 먼저
  시도하고 실패하면 시스템 python3로 폴백하도록 고침. 그래도 실패하면
  `pip3 install --upgrade protobuf --break-system-packages`가 다음 시도 대상
  (아직 실기에서 검증 안 됨 — 다음 세션에서 확인 필요).

**다음 세션 우선순위**:
1. `hailortcli parse-hef`로 `yolov5xs_wo_spp_nms_core.hef` vs `yolov5xs_wo_spp.hef` 레이어/
   출력 shape diff — 진짜 "후처리 엔진만 다른 동일 모델"인지 확인.
2. `model_runner.hpp`의 `i==0` 디버그 프린트(검출 개수/좌표)를 몇 프레임 더 찍어서 두
   HEF 다 유효한 검출을 만드는지 확인 — 지금까지 정확도 검증이 한 번도 없었음.
3. 수정한 스크립트로 CPU 조건 `_hrtt_exp1`를 새로 수집(아직 한 번도 안 돌림) +
   NPU 조건의 det/det+seg/det+pose 트레이스 재수집(버그 때문에 빠져 있었음).
4. rpi4에서 `source ~/hailo_platform_venv/bin/activate && python3 hailo_utilization_hailo8.py`
   수동 실행해서 npu_percent 폴링이 왜 안 되는지 직접 에러 확인(지금은 hrtt 기반 대체값으로
   우회했지만 근본 원인은 안 고쳐짐).

---

## 7-2. [2026-08-19] Det 단일모델 v8s(CPU 후처리) vs v5-nms_core(NPU 후처리) FPS=60 실험 준비

사용자 요청: Det 모델만 대상으로 HEF 2종(`yolov8s.hef`=CPU 후처리, `yolov5xs_wo_spp_nms_core.hef`
=NPU 후처리)을 비교. **단일 모델 추론만**(Seg/Pose 없음), 조건당 3회 반복 후 평균, 측정 항목은
전체 추론시간 + 전처리/추론/후처리 각각의 시간. 후처리 담당 프로세서(NPU/CPU)와 그로 인해
불가피하게 달라지는 모델 버전(v8s/v5, img_size 640/512) 외에는 전부 동일 환경, **INPUT_FPS=60**
기준으로 통일해서 돌리고 싶다는 요청.

(참고: 리포 루트의 `results_singlemodel_fps5to25.xlsx`(오늘 날짜로 갱신된 최신 파일)를 보면
직전에는 Hailo-8L(rpi1)에서 `infer_scheduler.cpp` 기반 단일모델 프레임워크로 Det/Seg/Pose 각각
단독을 INPUT_FPS={5,10,15,20,25}로 스윕한 실험을 진행했었음 — 이번 요청은 그 다음 단계로,
Hailo-8(rpi4) 쪽에서 이미 준비돼 있던 v5det NPU-후처리 비교 코드를 Det 단독 케이스로 좁혀
FPS=60에서 재실행하는 것.)

**이미 존재하던 코드로 대부분 해결됨**: `hailo_8/infer_yolov5_hailo8.cpp`가 2026-08-06에 이미
`USE_CPU_BASELINE_INSTEAD` 매크로로 정확히 이 비교(0=YOLOv5 NPU 후처리, 1=YOLOv8s CPU 후처리)를
할 수 있도록 만들어져 있었음(Det 단독, chrono 타이머로 전처리/추론(latency)/후처리/전체시간을
직접 측정해 1행 CSV로 저장) — 실기 실행만 한 번도 안 된 상태였음.

**이번에 추가/수정한 것**:
1. `hailo_8/infer_yolov5_hailo8.cpp`: `INPUT_FPS` 기본값 `0`(무제한) → `60`으로 변경(주석도
   갱신). 로직 변경 없음 — 모델 1개 단독이라 스케줄링 경합 자체는 없지만, "동일 환경" 정의를
   맞추기 위해 입력 속도를 두 조건 다 60fps로 통제.
2. `hailo_8/scripts/run_det_v8s_vs_v5npu_fps60.sh` (신규): `USE_CPU_BASELINE_INSTEAD`와
   `INPUT_FPS`를 sed로 토글해가며 v5(NPU)/v8s(CPU) 두 조건을 각각 3회씩 자동 빌드·실행(총
   6회), 같은 CSV에 순차 저장(run_id 1~3=v5 NPU, 4~6=v8s CPU). `yolov5xs_wo_spp_nms_core.hef`가
   없으면 공식 hailo_model_zoo S3에서 자동 다운로드. HAILORT_INCLUDE `-I` 플래그 포함(rpi4
   경로 이슈, §4 참고). hailo_pci 드라이버의 간헐적 크래시(QUESTION_FOR_TA.md)에 대비해 실행
   직후 CSV 행 수 증가를 확인하고 안 늘면 자동 재시도(최대 5회)하는 로직 포함. 결과 경로:
   `hailo_8/experiments/{실행일}_det_v8s_vs_v5npu_fps60_exp1/csv/results_det_v8s_vs_v5npu_fps60.csv`.
3. `hailo_8/scripts/make_avg_csv_det_v8s_vs_v5npu.py` (신규): 6행 raw CSV를 `hef_name` 기준
   2그룹(v5-NPU/v8s-CPU)으로 묶어 3회 평균 낸 `_avg.csv` 생성(`hailo_8L/scripts/make_avg_csv.py`
   와 동일 패턴, 그룹핑 키만 단일 모델용으로 단순화). 위 스크립트가 실행 끝에 자동 호출함.
4. `hailo_8/scripts/build_xlsx_det_v8s_vs_v5npu.py` (신규): raw CSV + avg CSV를 컬럼설명/
   전체(6행)/조건별_3회평균(2행) 3개 시트짜리 xlsx로 합침(`results_singlemodel_fps5to25.xlsx`와
   동일한 시트 구성 관례). RPi가 아니라 결과를 scp로 받은 뒤 PC(또는 openpyxl 있는 아무 환경)
   에서 실행하는 용도. 더미 데이터로 동작 검증 완료(3개 스크립트 모두 문법/실행 테스트 통과).

**아직 안 된 것 — 실기 실행 필요**: 샌드박스는 RPi(rpi4) SSH 접근 권한이 없어 직접 실행 불가.
사용자가 `hailo_8/infer_yolov5_hailo8.cpp` + `hailo_8/scripts/run_det_v8s_vs_v5npu_fps60.sh` +
`hailo_8/scripts/make_avg_csv_det_v8s_vs_v5npu.py`(+ 필요시 `postprocess_hailo8.hpp`,
`model_types.hpp` 등 기존 헤더들 — 이미 rpi4에 있을 가능성 높음, 없으면 같이 scp)를 rpi4에
scp한 뒤 `./run_det_v8s_vs_v5npu_fps60.sh` 직접 실행해야 함. 결과 CSV를 PC로 받아온 뒤
`build_xlsx_det_v8s_vs_v5npu.py`로 최종 xlsx까지 만들 수 있음(원하면 다음 세션에서 CSV를
주면 바로 만들어 줄 수 있음).

---

## 7-3. [2026-08-19, 같은 세션 내 정정] "Hailo-8은 안 함, 앞으로는 Hailo-8L에서만 실험"

사용자가 위 7-2절 작업 직후 "앞으로 8L에서 실험할거야 8에서 안해"라고 정정함 — 이후 재차
"이 실험을 8말고 rpi1인 Hailo-8L에서 실험해보고 싶어"로 확인. 즉 위 7-2절의 Hailo-8(rpi4)용
산출물은 **더 이상 실행 대상이 아니고, 이 실험을 Hailo-8L(rpi1)로 이관**했다. (7-2절 파일들은
참고용으로 남겨두되, 실기 실행은 아래 8L 버전으로 진행할 것.)

**포팅하며 새로 확인/작업한 것**:
1. **HAILO8L용 nms_core HEF 존재 확인**: 공식 hailo_model_zoo
   (`docs/public_models/HAILO8L/HAILO8L_object_detection.rst`, `raw.githubusercontent.com`으로
   원문 확인)에 Hailo-8과 동일한 이름·해상도(512x512x3)의 `yolov5xs_wo_spp_nms_core.hef`가
   `hailo8l` 타겟으로 따로 컴파일되어 배포 중임을 확인:
   `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/yolov5xs_wo_spp_nms_core.hef`
   (샌드박스 네트워크에서 이 S3 도메인 직접 다운로드는 막혀 있어 curl로 실물 검증은 못 했음 —
   공식 문서 등재 사실만 근거로 함, RPi에서 스크립트가 wget + `hailortcli parse-hef`로
   Architecture=HAILO8L 확인까지 하도록 만들어둠). 공식 문서 기준 참고 성능(batch=1): CPU
   후처리(`yolov5xs_wo_spp`)=206 FPS, NPU 후처리(`yolov5xs_wo_spp_nms_core`)=57.2 FPS — Hailo-8
   에서 이미 관측한 "NPU 후처리가 오히려 느림" 경향과 방향이 일치.
2. **hailo_8L 프레임워크에 img_size 파라미터 이식**: `hailo_8L/model_types.hpp`(ModelConfig에
   `int img_size = 640` 필드 추가)와 `hailo_8L/model_runner.hpp`(run_model_async에 `int
   img_size = 640` 파라미터 추가, letterbox/zeros 폴백/decode_det 호출에서 하드코딩된 640 대신
   사용)를 hailo_8 쪽에 이미 있던 동일 변경사항 그대로 포팅함. 둘 다 기본값 640이라 기존
   `infer_scheduler.cpp`/`infer_scheduler_noppt.cpp`(3모델 프레임워크) 동작에는 영향 없음 —
   하위 호환 확인됨(`image_utils.hpp::letterbox()`와 `postprocess_8l.hpp::decode_det()`는 원래도
   이미 `img_size`/`model_input_size` 파라미터를 갖고 있어서 손댈 필요 없었음,
   `output_classify.hpp::classify_outputs()`도 이미 img_size 파라미터 보유).
3. **`hailo_8L/infer_yolov5_hailo8l.cpp`** (신규): `hailo_8/infer_yolov5_hailo8.cpp`를 그대로
   포팅. `postprocess_8l.hpp` 사용, HEF 경로를 8L 관례(`/home/rpi1/hailo-rpi5-examples/
   resources/yolov5xs_wo_spp_nms_core.hef`, `.../yolov8s_h8l.hef`)로, `IMG_DIR`을
   `/home/rpi1/datasets/sampled_val2017/`로 교체. `USE_CPU_BASELINE_INSTEAD` 매크로로 두
   HEF를 스위칭하는 로직/CSV 스키마는 100% 동일(두 보드 결과를 나중에 같은 컬럼으로 비교
   가능). `INPUT_FPS=60` 기본 반영. 8L은 rpi4의 `HAILORT_INCLUDE` 비표준 경로 이슈가 없어
   빌드 커맨드에 `-I` 불필요.
4. **`hailo_8L/scripts/run_det_v8s_vs_v5npu_fps60.sh`** (신규): `hailo_8L/scripts/*.sh`의 기존
   관례(`cd ~/hailo_cpp_test` 절대경로, `postprocess_8l.hpp` 존재 확인 등)를 따름.
   `USE_CPU_BASELINE_INSTEAD`를 sed로 토글해 v5(NPU)/v8s_h8l(CPU) 각 3회씩 자동 빌드·실행(총
   6회, run_id 1~3=v5 NPU, 4~6=v8s CPU), 같은 CSV에 순차 저장. HEF 없으면 hailo8l 타겟 S3에서
   자동 다운로드 + `hailortcli parse-hef`로 Architecture 확인 프롬프트 포함. 8L은 rpi4의
   `hailo_pci` find_vma 크래시 이슈가 보고된 적 없는 보드라(그 버그는 rpi4의 커널 6.12.x
   계열 한정) 재시도 로직은 최소화(3회)만 남김. 결과 경로:
   `hailo_8L/experiments/{실행일}_det_v8s_vs_v5npu_fps60_exp1/csv/results_det_v8s_vs_v5npu_fps60.csv`.
5. **`hailo_8L/scripts/make_avg_csv_det_v8s_vs_v5npu.py`**, **`hailo_8L/scripts/
   build_xlsx_det_v8s_vs_v5npu.py`** (신규): hailo_8 버전과 CSV 스키마가 완전히 동일해서
   로직은 그대로, 주석/설명만 8L(v8s_h8l, hailo8l 타겟)에 맞게 갱신. 3개 스크립트 모두
   더미 데이터로 평균 계산 + xlsx 3시트(컬럼설명/전체(6행)/조건별_3회평균(2행)) 생성까지
   재검증 완료.

**아직 안 된 것 — 실기 실행 필요**: 샌드박스는 RPi(rpi1) SSH 접근 권한이 없어 직접 실행
불가. 사용자가 `hailo_8L/infer_yolov5_hailo8l.cpp`(및 이번에 img_size 필드/파라미터가 추가된
`hailo_8L/model_types.hpp`, `hailo_8L/model_runner.hpp` — 기존 버전을 덮어써야 함, 3모델
실험 파일들과 헤더를 공유하므로 반드시 최신본으로 교체할 것) + `hailo_8L/scripts/
run_det_v8s_vs_v5npu_fps60.sh` + `hailo_8L/scripts/make_avg_csv_det_v8s_vs_v5npu.py`(+
`postprocess_8l.hpp`, `image_utils.hpp`, `output_classify.hpp`, `sys_monitor.hpp`,
`model_setup.hpp` 등 나머지 헤더 — 이미 rpi1에 있는 버전과 호환되지만, model_types.hpp/
model_runner.hpp는 이번에 바뀐 버전으로 재scp 필요)를 rpi1(`~/hailo_cpp_test/`)에 scp한 뒤
`./run_det_v8s_vs_v5npu_fps60.sh` 직접 실행해야 함. 결과 CSV를 PC로 받아온 뒤
`build_xlsx_det_v8s_vs_v5npu.py`로 최종 xlsx까지 만들 수 있음(CSV를 주면 다음 세션에서 바로
만들어 줄 수 있음).

---

## 8. 파일 위치 빠른 참고

- 실험 결과 통합 엑셀: `hailo_8L/experiments/2026-07-26_batch_priority_exp1/results_batch_priority_combined*.xlsx`, `hailo_8/experiments/results_batch_sweep_combined*.xlsx`
- 최신 batch×pp 실험: `hailo_8L/experiments/2026-07-29_batch_singlemodel_pp_exp1/{csv,traces,html}` (pp_on/pp_off 하위폴더로 분리됨)
- Notion 정리 페이지: "7/28 NPU 스케줄러 실험 작업 정리" (workspace 최상위, 프로젝트 홈 없음 — 독립 페이지)
- GitHub: `github.com/manjookim/NPU-Scheduler`, branch `main` (로컬 기준 여러 커밋 앞서 있고 push 상태 불확실)
