# Hailo-8L → Hailo-10H 구조 동등 이식본 (infer_scheduler_h10)

`hailo_8L/infer_scheduler.cpp`의 **파이프라인 구조·데이터 흐름·측정 정의·CSV 스키마를 그대로 유지**한 채
Hailo-10H(HailoRT 5.x)에서 실행하기 위한 재구현이다. 목적은 성능 향상이 아니라 **공정 비교**다.

## 현재 실행 조건 (2026-08-31)

| 항목 | 상태 |
|---|---|
| 후처리(디코딩/NMS) | **코드에서 완전히 제거** — 디코딩 코드, `output_classify.hpp`, `OutRole`/`OutMeta` 모두 삭제 |
| 출력 포맷 | **항상 `HAILO_FORMAT_TYPE_AUTO`** — FLOAT32/NHWC 강제 경로 자체를 삭제. HailoRT 내부 역양자화·재배열 비용이 latency에 안 섞인다 |
| 스케줄러 파라미터 API | **주석 처리** — `set_scheduler_threshold` / `set_scheduler_timeout` / `set_scheduler_priority` 미호출. HailoRT 기본값(1 / 0 ms / 16=NORMAL)으로 동작 |
| 백엔드 | **`--api infer` (기본값)** — H10은 VStreams 경로 미지원(`HAILO_NOT_IMPLEMENTED(7)`, 실측) |
| batch | 실제로 적용됨 (configure 시점 인자, 기본 1) |
| CSV | 8L과 동일한 **49컬럼 유지**. `postprocess_ms_*`는 NaN |

이 조건은 8L의 **B/D조건**(`ENABLE_POSTPROCESS=0`, `FORCE_OUTPUT_FLOAT32=0`)과 같다.
8L은 priority를 15로 줬지만 세 모델이 전부 같은 값이라 H10 기본값 16과 스케줄링 결과가 동일하다(Round-Robin) —
**setter 주석 처리로 8L 대비 스케줄링 동작이 달라지지 않는다.**

되살리려면: `model_setup.hpp`의 `apply_scheduler_params()` 안 주석 3줄을 풀고,
`infer_scheduler_h10.cpp`의 `THRESHOLD_*` / `TIMEOUT_*_MS` / `PRIORITY_*`를 원하는 값으로 바꿔 재컴파일한다.
(CSV의 해당 컬럼은 "실제 적용값"을 적어야 하므로, 주석을 푸는 순간 `#define` 값이 곧 기록값이 된다.)

## 파일 구성 (8L과 1:1 대응)

| 이 폴더 | 대응하는 8L 파일 | 역할 |
|---|---|---|
| `infer_scheduler_h10.cpp` | `infer_scheduler.cpp` | 파라미터 `#define` 블록 + main 오케스트레이션 |
| `model_setup.hpp` | `model_setup.hpp` | HEF 로드 → configure → (스케줄러 setter 주석) → 실행 자원 생성 |
| `model_runner.hpp` | `model_runner.hpp` | writer/reader 스레드 쌍으로 비동기 추론 + 측정 |
| `csv_writer.hpp` | `csv_writer.hpp` | **49컬럼 동일 스키마** CSV append |
| `model_types.hpp` | `model_types.hpp` | 공용 데이터 타입 |
| `sys_monitor.hpp` | `sys_monitor.hpp` | CPU/MEM/ctx-switch (`/proc`) — 로직 동일 |
| `image_utils.hpp` | `image_utils.hpp` | letterbox 전처리 — 동일 |
| `npu_monitor.hpp` | (8L은 `tools/monitoring/*.py`) | `hailortcli monitor` 파싱 → `npu_percent` |
| — | `output_classify.hpp`, `postprocess_8l.hpp` | **후처리 제거로 삭제** |
| `build.sh` / `run_workload_sweep_h10.sh` / `run_inflight_sweep.sh` | `scripts/auto_experiment_*.sh` | 빌드·스윕 |

## 빌드 & 실행

```bash
chmod +x build.sh run_workload_sweep_h10.sh run_inflight_sweep.sh
./build.sh

# H10 기본 실행 (infer 백엔드, 슬롯 12개가 기본값)
./infer_scheduler_h10 1 csv/results.csv --models det,seg,pose

# 깊이를 바꿔가며
./infer_scheduler_h10 1 csv/results.csv --models det,seg,pose --api infer --inflight 8

# 8L default_workload와 같은 7조건 x 3회
./run_workload_sweep_h10.sh

# in-flight 깊이 스윕 (H10 전용 진단)
MODELS=det ./run_inflight_sweep.sh
```

옵션: `--api {vstreams|infer}`, `--inflight N`, `--models det,seg,pose`, `--batch N|N,N,N`,
`--num_images N`, `--img_dir PATH`, `--csv PATH`, `--run_id N`, `--no_npu_monitor`.

`--threshold` / `--timeout_ms` / `--priority`는 **없다**. setter가 주석 처리된 상태라 값을 받아도
적용되지 않고 CSV만 거짓으로 만들기 때문이며, 실수로 넣으면 에러 메시지와 함께 종료한다.

인자를 하나도 주지 않으면 소스의 `#define` 값 그대로 동작한다 — 8L처럼 `sed`로 `#define`을
패치해 재컴파일하는 방식도 그대로 쓸 수 있다.

## 백엔드 — H10에서는 `--api infer` 하나뿐이다

> **실측 (2026-08-31, npu-rpi5 / HailoRT 5.3.0)**
> `vdevice->create_configure_params(hef)` 가 `HAILO_NOT_IMPLEMENTED(7)` 로 실패하며
> HailoRT가 직접 안내한다:
> `"Not supported. Did you try calling create_configure_params on H10? If so, use InferModel instead"`
> → **Hailo-10H에는 VStreams / ConfiguredNetworkGroup 경로 자체가 없다.**

| | `--api infer` (H10 기본값) | `--api vstreams` |
|---|---|---|
| API | `InferModel` / `run_async` / 완료 콜백 | `create_input_vstreams` / `write` / `read` |
| H10 | **동작** | **미지원** (`HAILO_NOT_IMPLEMENTED(7)`) |
| 8L과의 관계 | 구조 동등(writer/reader 유지) + 깊이 통제 | 코드 1:1 |
| in-flight 깊이 | `--inflight D`로 명시 지정 | HailoRT가 결정 (8L 실측 ≈ 11) |
| 용도 | H10 실험 전부 | 8L 대조용 (이 파일을 8L에서 빌드할 때) |

따라서 **8L과의 구조적 동등성은 `--api infer --inflight 12` 로 확보한다.** 8L vstream의
실측 in-flight가 약 11프레임이었으므로 슬롯 12개가 가장 가까운 조건이다.
두 백엔드는 `finalize_result()`를 공유하므로 **latency/total_time/FPS의 정의가 완전히 같다.**

## 측정 정의 (8L과 동일하게 고정)

```
enq_ts[i] : 프레임을 파이프라인에 넘기기 직전 (write 호출 직전 / 슬롯 확보 직전)
deq_ts[i] : 그 프레임의 모든 출력을 다 받은 시각
det_latency_ms   = mean(deq_ts - enq_ts)          ← 큐 대기 포함(sojourn)
total_time_*_s   = 마지막 deq_ts - 첫 enq_ts
total_time_ms_*  = 전처리 + latency                (후처리 제거로 후처리항 = 0)
```

`run_async` 직후 `wait()`하면 in-flight가 1이 되어 **서비스시간**을 재게 되고, 같은 이름의
컬럼에 10배 작은 값이 들어간다. 이 이식본이 막으려는 것이 정확히 그 상황이다.
실행 로그 끝의 `[in-flight 추정]` 줄(= Little의 법칙 `L = FPS × latency`)이 8L의 11.0~11.2와
비슷하게 나오면 파이프라인 깊이가 맞춰진 것이다.

## CSV 스키마

`csv_writer.hpp`의 `HEADER`는 `hailo_8L/csv_writer.hpp`의 `HEADER`와 **바이트 단위로 동일**하다(49컬럼).

| 컬럼 | H10에서의 처리 |
|---|---|
| `switches_per_s`, `idle_time_pct`, `activation_det/seg/pose` | **항상 `NaN`** — H10은 `.hrtt`가 0바이트라 원리적으로 측정 불가 (`HRTT_ON_HAILO10H.md` §7) |
| `avg_latency_det/seg/pose` | **항상 `NaN`** — `det_latency_ms`와 정의가 겹친다. 호스트 값으로 채우면 "HRTT 유래"와 "호스트 실측"이 한 파일에서 구분 불가능해지므로 의도적으로 비운다 |
| `postprocess_ms_det/seg/pose` | **항상 `NaN`** — 후처리를 코드에서 제거했다. 이 때문에 `total_time_ms_*`와 `total_time_ms_nopp_*`가 같은 값이 된다(8L 계산 규약 그대로) |
| `threshold_*`, `timeout_ms`, `priority_*` | 스케줄러 setter 주석 처리 상태이므로 **HailoRT 기본값(1 / 0 / 16)** 이 기록된다 = 실제 적용값 |
| `batch` | 실제 적용값 |
| `avg_fps_*`, `max_latency_*` | 호스트 실측으로 채움 (8L은 HRTT가 채우던 값). `FILL_HOST_DERIVED 0`으로 빌드하면 8L 런타임 출력과 똑같이 `NaN` |
| `npu_percent` | `hailortcli monitor`의 NNC% 평균(NNC>0 샘플만 — 8L `parse_npu_log.py`와 같은 정의). `--no_npu_monitor`면 `NaN` |
| 나머지 | 8L과 같은 방식으로 실측 |

결측은 빈 칸이 아니라 문자열 `"NaN"`으로 쓴다 — 8L `dtos()`와 같은 규약이고
`pandas.read_csv`가 기본으로 결측 처리하며, 8L 파일과 바이트 비교도 가능하다.
기존 파일에 append할 때 헤더가 다르면 **행을 쓰지 않고 중단**한다(스키마 오염 방지).

H10 전용 값이 필요하면 `CSV_EXTENSION_COLUMNS 1`로 빌드한다 — `backend`, `inflight_depth`,
`n_images_*`, `service_latency_ms_*`, `queue_wait_ms_*`가 **맨 뒤에만** 붙으므로
`df.iloc[:, :49]`로 자르면 8L과 그대로 비교된다.

## 비교 전 체크리스트

- [ ] `NUM_IMAGES`를 8L과 동일하게 (기존 8L 실험은 600장)
- [ ] 비교 대상 8L CSV가 **B조건 또는 D조건**인지 (후처리 off + AUTO 포맷 — 지금 이 코드와 같은 조건)
- [ ] `[스케줄러]` 로그에 setter 미호출이 찍히는지 (주석을 푼 뒤라면 `[적용확인]`이 전부 `OK`인지)
- [ ] ULTRA_PERFORMANCE 폴백 경고가 떴는지 (떴다면 8L과 전력모드 조건이 달라진 것)
- [ ] `[in-flight 추정]`이 8L(≈11)과 비슷한지 (`--inflight 12` 기준)

## 빌드가 막힐 수 있는 지점 (설치본 헤더 기준으로 확인할 것)

HailoRT 5.3.0 설치본의 `hailo/infer_model.hpp` / `hailo/network_group.hpp`와 대조해 볼 항목:

- `wait_for_async_ready(std::chrono::milliseconds)` — 두 번째 인자 `frames_count`가 필수인 빌드면 `, 1`을 추가.
- `vdevice->create_configure_params(hef)` — H10에서 `HAILO_NOT_IMPLEMENTED(7)`로 **확정 실패**(실측).
  `configure_models()`가 힌트를 출력하고 종료한다. H10에서는 항상 `--api infer`를 쓴다.
- `hailort::Expected<T>`는 이동 대입이 `= delete`다(`expected.hpp:393`). `exp = f()`로 재대입하면
  컴파일 에러가 난다 — 재시도 경로마다 새 `Expected`를 받아야 한다.
- `power_mode = HAILO_POWER_MODE_ULTRA_PERFORMANCE` — H10이 거부하면 자동으로 PERFORMANCE 폴백하고
  경고를 찍는다. 폴백이 걸렸다면 8L과 전력모드 조건이 달라진 것이므로 실험 노트에 남길 것.

---

## 8L → 10H API 변경 요약

1. **VDevice**: 생성 코드(`hailo_init_vdevice_params` + `ROUND_ROBIN`)는 동일하지만, H10은 DISCRETE로 분류되어 `VDevice::create()`가 로컬 `VDeviceBase`가 아니라 **`VDeviceHrpcClient`(RPC 클라이언트)** 를 돌려준다 — 스케줄러·VDMA 실체는 디바이스 안 `hailort_server`에 있고, 호스트에는 껍데기만 남아 `.hrtt`가 0바이트가 된다.
2. **스케줄러**: `set_scheduler_threshold/timeout/priority`는 8L과 이름·시그니처가 같고 H10에서도 RPC로 전달되지만 **현재 주석 처리 상태**라 HailoRT 기본값(1 / 0 ms / 16)으로 돈다. 주석을 풀더라도 H10은 설정값 read-back이 불가하므로 `[적용확인]` 로그가 유일한 검증 수단이고, `threshold`는 "동시에 큐에 쌓일 수 있는 요청 수"(vstreams=batch, infer=`--inflight`) 이하여야 한다.
3. **실행 자원**: VStreams(`configure` → `make_*_vstream_params` → `create_*_vstreams`) 경로는 **H10에서 제거됐다** — `create_configure_params`가 `HAILO_NOT_IMPLEMENTED(7)`로 실패하고 HailoRT가 "use InferModel instead"라고 안내한다(실측). 대체 경로는 `create_infer_model` → `set_batch_size` → `configure` → `create_bindings`인데, `Bindings`와 버퍼가 **프레임 1개당 1세트**라 파이프라이닝하려면 슬롯을 D개 만들어 돌려 써야 한다(1개만 두고 `run_async` 직후 `wait`하면 동기 추론이 된다). 참고로 `hailort::Expected<T>`는 이동 대입이 `= delete`라 `exp = f()` 형태의 재대입이 불가능하다.
4. **출력 포맷·모니터링**: 출력 타입/순서 지정 API가 `make_output_vstream_params(...)`에서 `infer_model->output(name)->set_format_type/set_format_order(...)`로 옮겨갔지만, 후처리를 제거해 둘 다 쓰지 않고 AUTO로 받는다. `HAILO_MONITOR=1` → `/tmp/hmon_files` 경로는 H10에 없어 `npu_percent`는 `hailortcli monitor` 출력을 파싱해 채운다.
