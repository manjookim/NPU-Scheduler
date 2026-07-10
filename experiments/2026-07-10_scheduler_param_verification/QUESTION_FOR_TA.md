# 스케줄러 파라미터(threshold/timeout) 적용 검증 관련 질문

## 배경

Hailo-8L(Raspberry Pi 5)에서 Detection/Segmentation/Pose 3개 모델을 동시에 추론하면서, 모델별로 `priority`/`threshold`/`timeout`/`batch_size`를 독립적으로 설정해 벤치마킹하는 작업을 진행 중입니다. HailoRT C++ API로 구현했고 (`infer_scheduler.cpp`), val2017 데이터셋(최대 600장)을 실제 입력으로 사용합니다.

이 폴더에는 아래 두 파일의 **작업 당시 스냅샷**을 넣어뒀습니다 (계속 수정되는 최신 버전은 저장소 루트의 `infer_scheduler.cpp`, `tools/hrtt/verify_params.py`에 있습니다).

- [`infer_scheduler.cpp`](./infer_scheduler.cpp) — 3개 모델 동시 추론 + 모델별 스케줄러 파라미터 설정 코드
- [`verify_params.py`](./verify_params.py) — HRTT 트레이스에서 실제 적용된 파라미터 값을 확인하는 스크립트 (실행하려면 `tools/hrtt/profiler_pb2.py`가 같은 폴더에 필요합니다)

## 진행 중 확인하고 싶은 것 두 가지

### 1. threshold ≤ batch_size 제약이 공식 문서화된 내용인가요?

`set_scheduler_threshold()`로 threshold를 batch_size보다 크게 설정하면 `HAILO_INVALID_ARGUMENT`로 실패합니다.

```
[HailoRT] [error] CHECK failed - Threshold must be equal or lower than the maximum batch size!
```

`hailo-ai/hailort`(hailo8 브랜치) `network_group.hpp`의 `set_scheduler_threshold` 문서 주석에는 이 제약이 언급돼 있지 않아서, 실기에서 에러 메시지를 보고서야 알게 됐습니다. 저희가 놓친 공식 문서가 있는지, 아니면 저희처럼 실기로만 확인 가능한 부분인지 궁금합니다.

### 2. "성공으로 반환됐지만 실제로는 적용 안 된" 경우가 있을 수 있는지

setter(`set_scheduler_threshold`/`timeout`/`priority`)에 getter가 없어서, HailoRT 실행 시 `HAILO_TRACE=scheduler`로 얻은 `.hrtt`의 `core_op_set_value` 트레이스 이벤트를 파싱해 실제 적용값을 재확인하는 방식을 쓰고 있습니다 (`verify_params.py`).

이 방식이 신뢰할 만한지 알아보려고, 일부러 Segmentation의 threshold를 batch_size보다 크게(99) 설정해 **거부되는 케이스**를 만들고 트레이스를 확인했습니다.

- 결과: 거부된 threshold는 `core_op_set_value`에 아예 나타나지 않음 (요청값 99가 echo되지도 않고, 그냥 없음). 같은 모델의 다른 성공한 파라미터(timeout, priority)는 정상적으로 트레이스에 남음.
- 이를 근거로 "이 트레이스는 요청값을 그대로 기록하는 게 아니라 실제 스케줄러 내부 상태를 반영한다"고 판단하고 있는데, 저희가 확인한 건 **명시적으로 거부된(에러 반환) 케이스** 하나뿐입니다.
- 궁금한 점: setter가 `HAILO_SUCCESS`를 반환했는데도 내부적으로는 요청과 다른 값(예: default로 조용히 대체)으로 처리되는 경우가 이론적으로/실제로 존재할 수 있을까요? 그런 사례를 만드는 방법이나, 더 확실한 검증 방법(예: 공식적으로 지원되는 파라미터 조회 방법)이 있다면 조언 부탁드립니다.

## 환경

- HailoRT v4.23.0, Hailo-8L (Raspberry Pi 5)
- 참고한 공식 자료: `github.com/hailo-ai/hailort` `hailo8` 브랜치의 `hailort/libhailort/include/hailo/network_group.hpp`, `hailort/libhailort/examples/cpp/switch_network_groups_example/switch_network_groups_example.cpp`
