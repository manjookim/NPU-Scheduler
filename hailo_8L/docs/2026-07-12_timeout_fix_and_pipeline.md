# NPU 스케줄러 실험 — timeout hang 해결 & 결과 파이프라인 (2026-07-12)

batch × priority 전체조합 실험(252조건 × 3회 = 756회) 도중 발생한 hang을 잡고, 완주 후 HRTT → HTML/CSV까지 정리한 작업 기록.

---

## 1. 문제 발견 — 우선순위 다른 다중모델에서 hang

- 실험 초반은 정상. 그런데 **우선순위가 다른 2모델 조건**(예: Det=0, Seg=15)에서 멈춤.
- 우선순위가 **같을 때(0/0)**: 정상 완료(~2분/조건).
- 우선순위가 **다를 때**: 7분 넘게 진행 없음.
- 진단 결과: `NPU 0%`, `infer_scheduler` 프로세스가 31분째 CPU를 0.7%만 쓰며 블록 → **완전 hang 확정**.

## 2. 원인 규명 (2단계)

**1차 원인 — threshold=0**
- 우선순위가 다를 때, threshold=0이면 우선순위 높은 모델이 **큐가 비어도 계속 실행 자격을 유지** → 낮은 모델에게 NPU를 영영 안 넘김 → 영구 starvation.
- threshold를 **1(HailoRT 기본값)**로 바꾸니 hang은 사라짐. 큐가 비면 모델이 자격을 반납하기 때문.

**2차(근본) 원인 — vstream 기본 timeout 10초**
- threshold=1로 고친 뒤엔 대신 `HAILO_TIMEOUT` 에러 발생. 낮은 우선순위 모델이 굶어 write가 실패 → 프레임 유실.
- 근본 원인: **입력 vstream 기본 timeout이 10초**. 낮은 우선순위 모델이 10초 넘게 스케줄을 못 받으면 write가 HAILO_TIMEOUT → 내부 파이프라인 스레드가 죽어 프레임 유실(재시도로도 복구 불가).
- 공식 헤더(`hailort v4.23.0 / network_group.hpp`) 확인해 정확한 API 파악.

## 3. 수정 (infer_scheduler.cpp, 3가지)

1. **threshold 0 → 1** (HailoRT 기본값). 높은 우선순위 모델이 큐가 비면 자격을 반납 → 낮은 모델이 이어받음.
2. **vstream timeout을 5분(300000ms)으로 확대**. `make_input/output_vstream_params` + `create_input/output_vstreams` 사용 → 굶는 동안 조기 timeout 방지.
3. **write/read에 HAILO_TIMEOUT 재시도 로직 추가** (안전망). 유실 없이 대기 후 성공.

## 4. 검증 (수정 후 재컴파일 테스트)

| 조건 | 결과 |
|---|---|
| 2모델 Det=0 / Seg=15 | 에러 없음, 둘 다 완료, 종료코드 0. Det total_time **51s ≫** Seg 29s |
| 3모델 Det=0 / Seg=31 / Pose=31 | 에러 없음, 셋 다 완료, 종료코드 0. Det **75s ≫** Seg·Pose 각 ~53s |

- HAILO_TIMEOUT 에러 0개, 프레임 유실 없음.
- **starvation이 `total_time`에 정상 반영** — 우선순위 낮은 모델이 더 오래 걸리는 게 데이터로 관측됨(이게 관측 목표였음).

## 5. 전체 실험 완주

- 수정본으로 **처음부터 재시작** (threshold 값이 바뀌어 데이터 일관성 위해 전량 재실행).
- **756회 전체 hang 없이 완료.**
- batch 값: **1 / 10 / 50 / 63** (HailoRT batch_size 상한 63 때문에 100 대신 63 사용).
- 결과: 사용모델수(1/2/3) × batch(4종) = **12개 CSV로 분리 저장**.

## 6. HRTT → HTML 변환 & CSV 완성

- HRTT 트레이스 756개를 tar로 묶어 PC로 다운로드.
- WSL에서 `hailo runtime-profiler`로 **756개 전부 HTML 변환** (고정 출력명 `runtime_report.html`을 입력 파일명으로 바꿔 `html/`에 정리).
- HRTT 전용 지표를 **protobuf 직접 파싱**해 CSV의 NaN 자리에 채움 (`scripts/fill_hrtt_columns.py`):
  - 전역: `switches_per_s`, `idle_time_pct`
  - 모델별: `avg_fps`, `avg_latency`, `max_latency`, `activation`
  - **756개 전부 매칭 성공** (실패 0). 전역 지표는 전 행에 채워짐.
- 채워진 CSV로 **12시트 xlsx** 재생성.

## 7. 파일 정리 (드라이브 업로드용)

- `hrtt/batch_priority/` 와 `html/batch_priority/` 를 각각 **batch별 폴더(b1 / b10 / b50 / b63, 각 189개)**로 정리. 총 756개.

## 부록 — main 브랜치 threshold/timeout 적용 여부 검증

- 의문: 예전 main 브랜치 실험에서 threshold/timeout이 실제 적용됐나?
- `prithr` 폴더 HRTT 974개를 `core_op_set_value` 트레이스로 전수 검증:
  - **threshold: 적용됨** — 의도값과 일치 2671건(15·31 포함), 불일치는 손상된 트레이스 2개뿐.
  - **timeout: 0으로 적용** — 값은 반영됐으나 0이라 스케줄 효과는 없음.
- 결론: threshold 15·31이 실제 적용됐다는 건 그 실험이 batch=1이 아니라 **batch ≥ 31**로 돌았다는 뜻. (코드 추정보다 트레이스 실측이 정답.)

## 핵심 교훈

- **threshold=0은 위험** — 우선순위 다른 다중모델에서 starvation hang 유발. 기본값 1을 쓸 것.
- **최대 처리량(INPUT_FPS=0) + 우선순위 격차** 조합에선 낮은 모델이 오래 굶으므로 **vstream timeout을 충분히 크게** 줘야 함(기본 10초로는 부족).
- **threshold ≤ batch_size**일 때만 set 성공. 초과 시 거부되어 기본값(1) 유지.
- cpp의 `*_latency_ms`(end-to-end, 큐 포함)와 HRTT의 `avg_latency`(순수 추론, ~48ms)는 **다른 지표** — 비교 시 혼동 주의.
