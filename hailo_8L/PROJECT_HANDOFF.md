# NPU 스케줄러 실험 — 작업 인수인계 문서

다른 AI가 이어받을 수 있도록 프로젝트 목적·환경·현재 상태·다음 할 일을 정리한 문서.

---

## 1. 프로젝트 목적

Raspberry Pi 5 + **Hailo-8L NPU** 환경에서 Detection / Segmentation / Pose 3개 딥러닝 모델을
**동시에(하나의 NPU 공유)** 추론할 때, **HailoRT 스케줄러 파라미터**가
지연시간(latency)·스케줄링 동작에 미치는 영향을 측정하는 실험.

측정·조절 대상 파라미터:
- **priority** (우선순위): 0 / 15 / 31
- **threshold** (활성화 임계 프레임 수)
- **timeout** (최대 대기 시간, ms)
- **batch_size**

## 2. 실험 환경

- **하드웨어**: Raspberry Pi 5 + Hailo-8L NPU
- **HailoRT 버전**: v4.23.0 (RPi 설치본. 웹 문서 최신과 API 시그니처가 다를 수 있음 → **RPi 빌드 에러가 정답**)
- **모델(HEF)** — RPi 경로:
  - Detection: `/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef`
  - Segmentation: `/home/rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef`
  - Pose: `/home/rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef`
- **스케줄러**: `HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN`
- **데이터셋**: COCO val2017 샘플 673장 (`sampled_val2017/`) — 단, 스케줄러 검증엔 더미 버퍼로 충분

## 3. 코드 (infer_scheduler.cpp) — C++ API

HailoRT **C++ API**(`hailo/hailort.hpp`, `hailort::` 네임스페이스) 사용.
공식 예제 `hailort/libhailort/examples/cpp/switch_network_groups_example/switch_network_groups_example.cpp` 형태를 따름.

핵심 흐름:
1. `VDevice::create(params)` — params.scheduling_algorithm = ROUND_ROBIN
2. 모델별: `Hef::create()` → `vdevice->create_configure_params(hef)` (batch_size 설정) → `vdevice->configure(hef, cfg)` → network_group 획득
3. **스케줄러 파라미터 설정** (모델별, network_group 메서드):
   ```cpp
   network_group->set_scheduler_threshold((uint32_t)threshold);           // 기본 1
   network_group->set_scheduler_timeout(std::chrono::milliseconds(TIMEOUT_MS)); // 기본 0
   network_group->set_scheduler_priority((uint8_t)priority);              // 기본 16(NORMAL)
   ```
   반환값이 `HAILO_SUCCESS`면 적용됨 → `[적용확인]` 로그로 출력(getter가 없어 반환 status로 확인)
4. `VStreamsBuilder::create_vstreams(*ng, {}, HAILO_FORMAT_TYPE_AUTO)` — 입출력 vstream 한 번에
5. 모델별 write/read 스레드로 추론

현재 파라미터 값(직접 #define 수정):
```
BATCH_SIZE=1, THRESHOLD(DET/SEG/POSE)=1, TIMEOUT_MS=100, PRIORITY(DET/SEG/POSE)=0
```

## 4. 빌드 & 실행 (RPi)

```bash
# 빌드 (C++17, OpenCV 미사용)
g++ infer_scheduler.cpp -o infer_scheduler -lhailort -lpthread -std=c++17

# HRTT 트레이스 생성 설정 (추론 도는 터미널에서)
export HAILO_TRACE=scheduler
export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30   # dump 값 크거나 없으면 fps 저하
export HAILO_TRACE_PATH=/path/to/traces
export HAILO_MONITOR=1
./infer_scheduler
```

## 5. HRTT → HTML 변환 (PC/WSL, DFC 설치 환경)

```bash
source ~/hailo_venv/bin/activate     # Dataflow Compiler / hailo CLI 설치된 환경
hailo runtime-profiler <파일>.hrtt   # runtime_report.html 생성
```
- **HRTT 생성 = RPi(HailoRT 런타임)** / **변환 = PC(DFC/hailo CLI)**

## 6. 파라미터 적용 검증 방법

HailoRT엔 값을 되읽는 **getter가 없음**. 그래서 적용 확인은 두 가지로:
1. **setter 반환 status** — `set_scheduler_*`가 `HAILO_SUCCESS` 반환 = 적용됨 (실행 로그 `[적용확인]`)
2. **HRTT의 `core_op_set_value` 트레이스 이벤트** — HailoRT가 실행 중 실제 적용값을 기록.
   HRTT(protobuf) 파싱해 threshold/timeout/priority 읽으면 됨.
   (거부된 설정은 이벤트가 안 남음 → 미적용으로 확인 가능)
- **주의**: 프로파일러 HTML의 "Network Parameters" 칸은 값이 안 뜰 수 있음(UI 한계).
  단, **timeout 등을 기본값(0)이 아닌 값**으로 주면 표시됨.

## 7. 핵심 발견 (중요)

- **threshold 정의**: "입력 큐에 threshold개 프레임이 쌓이면(또는 timeout 경과 + 1프레임) 네트워크 활성화."
  즉 큐 깊이 기반. → **큐가 안 쌓이면 threshold 무의미.**
- **timeout=0이면 threshold 무력화**: 1프레임만 있으면 즉시 활성화 → threshold 도달 기다리지 않음.
  threshold 실험엔 **timeout > 0 필수**.
- **batch_size = burst size**: 한 번 활성화 시 처리하는 프레임 수는 threshold가 아니라 batch_size가 결정.
- **오프라인 최대속도 배치 추론에선 threshold/timeout 효과가 거의 안 나타남**:
  프레임을 최대한 몰아넣으면 큐가 항상 포화(saturation) → 모든 모델이 항상 활성화 자격 충족 → threshold 무의미.
  효과를 보려면 입력 속도를 NPU 처리량 근처로 제한(실시간 스트리밍 모사)해야 함.
- **priority는 정상 작동**: priority 차이가 15 이상이면 낮은 모델이 starvation(굶음).

## 8. 지금 하려는 작업 / 다음 할 일

- **파라미터(threshold/timeout/priority)가 실제 적용되는지 검증** + 추론 정상 동작 확인이 최우선.
- 현재: priority=0, threshold=1, timeout=100, batch=1로 1회 실행 → HRTT 생성 → HTML 변환 →
  `core_op_set_value`로 적용값(threshold=1, timeout=100, priority=0) 확인.
- 이후: 값을 바꿔가며(특히 timeout>0 + 입력 속도 제한) threshold 효과 관측 시도.

## 9. 참고 자료

- HailoRT GitHub (v4.23.0): `github.com/hailo-ai/hailort` tree `v4.23.0`
  - C++ 스케줄러 예제: `hailort/libhailort/examples/cpp/switch_network_groups_example/`
- HailoRT User Guide 6.9 Model Scheduler / 6.10 Optimizations — threshold/timeout/priority 정의
- C++ 스케줄러 API 시그니처:
  - `set_scheduler_threshold(uint32_t threshold, const std::string &network_name = "")`
  - `set_scheduler_timeout(const std::chrono::milliseconds &timeout)`
  - `set_scheduler_priority(uint8_t priority)`
