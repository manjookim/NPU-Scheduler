# Hailo-10H 트레이싱·측정 정리

작성 2026-08-27 · 검증 환경: npu-rpi5 (Raspberry Pi 5 + Hailo-10H M.2), HailoRT 5.3.0,
hailort-pcie-driver 5.3.0, PCIe Gen3 x1

> **결론** — Hailo-8L에서 쓰던 `HAILO_TRACE=scheduler` → `.hrtt` → `hailo runtime-profiler`
> 워크플로우는 Hailo-10H에서 **원리적으로 동작하지 않는다.** 버그·설정 오류·API 선택 문제가 아니라
> **프로세스 경계 문제**이며, Perfetto 대안과 `hailortcli` 전 명령을 모두 확인했지만 스케줄러 내부
> 지표를 얻는 경로는 존재하지 않는다. 대신 host_trace(직접 계측) + `hailortcli monitor`로
> 성능 지표는 모두 측정 가능하다.

---

## 1. 구조 차이

**Hailo-8L** — HailoRT 추론 스택 전체가 우리 앱과 같은 프로세스 안에 있다.

```
[호스트 프로세스]  우리 앱 → libhailort → VDevice / 스케줄러 / VDMA   ← TRACE() 발생
                                              ↓ PCIe
[Hailo-8L 칩]      연산만 수행
```

**Hailo-10H** — 모듈 안에 자체 Linux(Yocto 5.15)가 돌고, 추론 스택은 그 안의 `hailort_server`에 있다.

```
[호스트 프로세스]  우리 앱 → libhailort → VDeviceHrpcClient   ← RPC 껍데기, TRACE() 0개
                                              ↓ PCIe (RPC)
[Hailo-10H 모듈]   hailort_server → VDevice / 스케줄러 / VDMA  ← TRACE()가 여기서 발생
```

호스트가 부팅 시 커널·루트파일시스템을 PCIe로 올려주면 디바이스가 자기 Linux를 부팅해 서버를 띄운다.

---

## 2. `.hrtt`가 0바이트인 원인

`.hrtt`를 쓰는 `SchedulerProfilerHandler`는 **자기 프로세스 안에서 발생한 이벤트만** 수집한다.
H10에서는 호스트 프로세스에서 발생하는 이벤트가 사실상 없다. 파일 헤더를 만드는
`InitProfilerProtoTrace`조차 `VDeviceBase::create()` 안에만 있는데, H10에서는 `VDeviceBase`가
호스트에 생성되지 않는다 → **빈 protobuf 직렬화 = 0바이트**.

### 소스 근거 (hailo-ai/hailort, master)

| 위치 | 내용 |
|---|---|
| `vdevice/vdevice.cpp:243` | 디바이스가 `DISCRETE`(=H10)면 `VDeviceHrpcClient` 반환. H8/8L은 `VDeviceHandle` |
| `*_hrpc_client.cpp` 3종 | 호스트측 RPC 클라이언트 — `TRACE(` 호출 **0개** |
| `scheduler/scheduler.cpp`(11곳), `vdma/vdma_config_core_op.cpp`, `net_flow/pipeline/*` | 실제 TRACE 지점 — 전부 디바이스측 실행 코드 |
| `vdevice/vdevice.cpp:350~408` | `InitProfilerProtoTrace` 등 5개가 모두 `VDeviceBase::create()` 내부 |
| `hef/hef.cpp:630` | `HefLoadedTrace` — 호스트에서 발생 가능한 **유일한** 이벤트 |

### 실기 근거

| 확인 | 결과 |
|---|---|
| `hailortcli logs runtime` | `HailoRT-Server`가 **디바이스 내부**(hostname `hailo10`, Linux 5.15 yocto)에서 구동 중 |
| `ls /lib/firmware/hailo/hailo10h/` | `fitImage`(디바이스 커널) + `image-fs`(ext2 루트fs 100MB) |
| `image-fs` 내부 | `/etc/init.d/hailort_server.sh` 존재 |
| `strings libhailort.so.5.3.0` | `vdevice_hrpc_client.cpp`, `VDeviceHrpcClient` 포함 |

### 관측 현상이 모두 이 하나로 설명됨

| 관측 | 설명 |
|---|---|
| 우리 앱 → 0바이트 | 호스트 이벤트 0개 → 빈 protobuf |
| `hailortcli benchmark` → 129바이트 | 호스트에서 HEF를 파일 경로로 파싱해 `HefLoadedTrace` 1개만 기록 |
| strace에 `open`→`close`, `write` 없음 | 쓸 바이트가 0이라 write 자체가 발생 안 함 |
| VStreams API로 재작성해도 0바이트 | API 문제가 아니라 프로세스 경계 문제 |
| `HAILO_TRACE=all/api/1` → 파일도 안 생김 | `scheduler` 외 값은 무효 |

---

## 3. Perfetto 대안 검토 — 4가지 경로 모두 막힘

| 접근 | 결과 | 근거 |
|---|---|---|
| HailoRT 내장 Perfetto 트레이서 (`HAILO_PERFETTO_TRACE=1`) | 배포 빌드에 미포함 + 있어도 무의미 | `hailort/CMakeLists.txt:14`에서 `ENABLE_PERFETTO` 기본 **OFF**. `HAVE_PERFETTO` 없으면 트레이싱 매크로가 빈 정의. 설치본 strings에 perfetto 문자열 2개뿐(SDK 미링크). 켜도 이벤트 지점은 디바이스측 |
| `hailo-ai/Hailo-SoC-Profiler` 레포 | Hailo 전용 도구가 아님 | `google/perfetto`의 **단순 fork**(fork=true, 기본 브랜치 `1.12.1`, 스톡 README, star 1). 전용 실행 파일·NPU 데이터 소스 없음. Raspberry Pi OS apt에 `perfetto` 패키지가 이미 존재 |
| 호스트 ftrace/Perfetto 시스템 트레이싱 | NPU 정보 없음 | `sudo ls /sys/kernel/tracing/events \| grep hailo` → **출력 없음**. `hailo1x_pci` 드라이버가 ftrace 트레이스포인트를 하나도 정의하지 않음 |
| 디바이스 안에서 트레이스 활성화 | 파일 회수 불가 | 디바이스 `/etc/fstab`이 `/dev/root / ro`(루트fs 읽기 전용). 호스트↔디바이스 공유 FS·네트워크 없음. 유일한 통로 `logs runtime`은 작은 링 버퍼. 부팅 이미지 수정은 실패 시 NPU 미부팅 위험 |

호스트 ftrace로 얻을 수 있는 것은 커널 일반 이벤트(스레드 스케줄링, ioctl, irq)뿐이다.

---

## 4. `hailortcli` 전수 조사 (H10 실측)

| 명령 | H10에서 주는 것 |
|---|---|
| `monitor` | NNC 가동률 / 디바이스 CPU / RAM / 온도 / 전압 — **디바이스 전체 합계만** |
| `monitor -v` | help에 **"H15 only"** 명시 |
| `run2 -j <json>` | 네트워크별 **FPS만**. 스케줄러 파라미터 지정은 가능 |
| `run2 --measure-latency` | **동작함** → `hw_latency` 5.72 ms / `overall_latency` 17.55 ms. **단일 모델 실행 시에만** |
| `run2 collect-runtime-data` | **실패** — `Operation 59 is not allowed when FW version in not supported` → `HAILO_UNSUPPORTED_FW_VERSION(16)`. H10 펌웨어가 FW action list 다운로드 opcode 미지원 |
| `benchmark` | FPS + 온도만. latency·power 항목 없음. `--csv`는 exit 0인데 **파일 미생성**(미문서화 동작) |
| `logs runtime` | 디바이스 syslog(텍스트) — 스케줄러 이벤트 없음 |
| `logs nnc` | 바이너리 덤프, 파싱 불가 |
| `logs system_control` | SCU 부팅·전원·클럭 로그 |
| `fw-control` | `identify`만 |
| `measure-power` | 미지원 — `HAILO_OPEN_FILE_FAILURE(13)` (보드에 INA 센서 없음) |
| `parse-hef` | HEF 정적 정보 (컨텍스트 개수 등) |
| `scan` | 디바이스 목록 |

**핵심**: `hailortcli`도 우리 앱과 똑같은 RPC 클라이언트다. 같은 `libhailort`를 링크해 디바이스 안
서버에 붙을 뿐이라 특별한 통로가 없다. 명령을 바꿔 써도 달라지지 않는다.

### `collect-runtime-data` 사용 시 주의

단독으로 `collect-runtime-data <hef>`를 주면 `Only one model is allowed`가 뜬다. 이 빌드에서
서브커맨드가 HEF를 네트워크로 등록하지 못하는 별도 버그로 보이며, `set-net`을 같이 줘야 실제
실행까지 진행된다(그리고 위 FW 미지원 에러로 끝난다):

```bash
hailortcli run2 -t 5 --mode raw_async \
  set-net resources/yolov8s.hef \
  collect-runtime-data resources/yolov8s.hef --output-path /tmp/rt.json
```

---

## 5. H10에서 얻을 수 있는 값 전체

### 5-1. 우리 벤치가 직접 재는 값 (CSV + 터미널)

[2026-08-31] CSV 컬럼을 `hailo_8L/csv_writer.hpp`와 같은 구성으로 맞췄다. HRTT에서만 나오는
`switches_per_s` / `idle_time_pct` / `activation_*` 세 종류만 빼고 나머지는 전부 채운다.
8L에서 HRTT가 채워주던 `avg_fps_*` / `max_latency_*`는 호스트 실측으로 대체한다.

| 컬럼 | 단위 | 정의 | 8L에서의 출처 |
|---|---|---|---|
| `label`, `run_id`, `use_det/seg/pose` | | 실행 구성 | 동일 |
| `batch`, `threshold_*`, `timeout_ms`, `priority_*` | | 적용된 스케줄러 파라미터. 옵션을 안 주면 HailoRT 기본값(0 / 1 / 0 / 16)을 그대로 적는다 | 동일(8L은 `#define`) |
| `det/seg/pose_latency_ms` | ms | `run_async()` 제출 → `wait()` 완료. **전처리 제외** | 동일(호스트 실측) |
| `cpu_percent`, `mem_percent` | % | 호스트 시스템 전체 (`/proc/stat` 300 ms 샘플 평균, `/proc/meminfo`) | 동일 |
| `voluntary/nonvoluntary_ctx_switches` | 회 | 각 모델 워커 스레드의 증가분 합 (`/proc/thread-self/status`) | 동일 |
| `npu_percent` | % | 디바이스 NNC 가동률. **NNC>0인 샘플만** 평균(8L `parse_npu_log.py`와 같은 정의) | 8L은 `HAILO_MONITOR=1` → `/tmp/hmon_files`. **H10은 이 경로가 없어 `hailortcli monitor`를 자식 프로세스로 띄워 파싱**(아래 참고) |
| `run_time_s` | 초 | 워커 전체 구간 wall-time | 동일 |
| `*_frame_count` | 개 | 모델별 성공 프레임 수 | H10 추가 |
| `avg_fps_*` | 프레임/초 | 프레임 수 ÷ 워커 전체 시간. **전처리 포함** | 8L은 HRTT → **H10은 호스트 실측** |
| `max_latency_*` | ms | 프레임별 latency의 최댓값 | 8L은 HRTT → **H10은 호스트 실측** |
| `total_time_*_s` | 초 | 모델별 워커 루프 총 시간 | 동일 |
| `avg_preprocess_ms_*` | ms | imread + cvtColor + resize + memcpy 장당 평균 | 동일 |
| `postprocess_ms_*` | ms | **NaN** — 이 벤치는 디코딩/NMS를 하지 않는다 | 8L은 `postprocess_8l.hpp`로 실측 |
| `total_time_ms_*` | ms | 전처리 + latency + 후처리(측정한 경우만). 후처리가 없으므로 `_nopp`와 같은 값 | 동일 규약 |
| `total_time_ms_nopp_*` | ms | 전처리 + latency | 동일 |

`fps ≠ 1000/avg_latency`인 이유: 전처리가 latency에는 빠지고 FPS에는 들어간다.

#### npu_percent를 8L 방식으로 못 받는 이유 (2026-08-31 실측)

8L은 `HAILO_MONITOR=1`을 주면 libhailort가 `/tmp/hmon_files`에 ProtoMon 파일을 떨궈서
`tools/monitoring/hailo_utilization.py`가 그걸 읽었다. H10에서 같은 환경변수로 돌려보면
**`/tmp/hmon_files` 디렉토리 자체가 생기지 않는다** — 스케줄러가 디바이스 안에 있으니
호스트 libhailort에는 쓸 상태가 없다(§1~2와 같은 원인).
대신 `hailortcli monitor`가 디바이스에 직접 물어본 NNC 가동률을 약 1 Hz로 뿌리므로,
벤치가 이 명령을 자식 프로세스로 띄워 파이프로 읽는다(`src/npu_monitor.hpp`).
우리 프로세스가 VDevice를 잡고 있어도 동시 실행에 문제 없음(양쪽 순서 모두 실측 확인).
별도 모니터 터미널을 띄울 필요가 없고, `--no_npu_monitor`로 끄면 `npu_percent`는 NaN이 된다.


### 5-2. host_trace 타임라인 (프레임 단위 원자료)

`traces/<label>_run<N>_events.csv` + `_trace.json`

| 값 | 정의 |
|---|---|
| `prep` 구간 | imread → cvtColor → resize → memcpy |
| `infer` 구간 | 프레임별 추론 왕복 |
| `enqueue_ms` | `run_async()` 반환까지 걸린 시간 = **디바이스 큐 backpressure** |
| `ok` | 프레임 성공 여부 |
| `epoch_wall_clock_ns` | 시간 원점 — 다른 로그와 시각 정합용 |

파생 가능: 지연시간 분포(p50/p95/p99·최대), 시간에 따른 변화, **동시 in-flight 요청 수 분포**,
모델 간 처리 순서·간격, 전처리 대 추론 비율, 타임라인 시각화(`chrome://tracing`, `ui.perfetto.dev`).

### 5-3. 디바이스 측 (`hailortcli monitor`)

NNC Utilization(%), Device CPU(%), RAM(% + MB), On Die Temperature(°C), On Die Voltage(mV),
Device ID, Architecture. 디바이스가 스스로 측정해 보고하므로 호스트 측정과 독립적인 교차 검증이 된다.

### 5-4. 기준선·비교용

| 값 | 획득 |
|---|---|
| 단일 모델 하드웨어 FPS | `benchmark <hef>` → yolov8s = 166.66 FPS |
| 단일 모델 hw / overall latency | `run2 --measure-latency --measure-overall-latency` |
| 다모델 동시 실행 네트워크별 FPS | `run2 -j out.json` (랜덤 입력) |

### 5-5. 정적 정보

`parse-hef` → **HW 컨텍스트 개수**(`Multi Context - Number of contexts: 3`), 호환 아키텍처,
HEF 컴파일러(DFC) 버전, 입출력 스트림 이름·타입·shape, 내장 NMS 파라미터.
`fw-control identify` → FW 버전·아키텍처. `scan` → 디바이스 목록.

### 5-6. 호스트 시스템

CPU 사용률·주파수·거버너, 메모리, 온도, PCIe 링크 상태(`lspci -vv` → `LnkSta: 8GT/s, x1`),
커널 ftrace 일반 이벤트.

---

## 6. HRTT 값 ↔ `hailortcli` 대응 (얻어지는 것만)

| HRTT 값 | hailortcli | 제약 |
|---|---|---|
| 모델별 평균 지연시간 | `run2 --measure-latency` → `hw_latency` / `overall_latency` | **단일 모델만** |
| 모델별 FPS | `run2 -j` | 랜덤 입력, 전처리 없음 |
| run time | `run2 -t` → JSON `time_to_run` | |
| 네트워크 개수·이름 | `run2 -j`, `parse-hef` | |
| HEF 이름 | `run2 -j`, `parse-hef` | |
| DFC 버전 | `parse-hef` → `HEF Compiler Version` | |
| HailoRT / FW 버전 | `fw-control identify`, `--version` | |
| device_id / device_arch | `scan`, `fw-control identify`, `monitor` | |
| 디바이스 점유율 | `monitor` NNC Utilization % | **근사** — 전체 합계라 단독 실행 때만 |
| idle 비율 | `monitor` NNC 0% 구간 | **근사** — HRTT 정의와 다름 |

---

## 7. 끝내 얻을 수 없는 값

- 컨텍스트 스위치 횟수 / 초
- 전환 사유 분해 (threshold 초과 / timeout 초과 / idle 전환)
- 모델별 activation·deactivation 소요 시간
- **동시 실행 시** 모델별 디바이스 점유율
- 프레임별 H2D/D2H 큐 입출력 타임스탬프 (큐 대기시간 vs 실제 연산시간 분리)
- 스트림별 큐 깊이
- 적용된 스케줄러 파라미터의 **확인**(설정은 가능, 읽기 불가)
- 실제 적용된 dynamic batch size
- 모델별 최대 지연시간 (CLI 기준. host_trace로는 계산 가능)
- HEF md5, 호스트 OS/CPU/RAM·PCIe gen/lanes 정보 (HRTT top header 항목)
- **전력 소비** — 보드에 INA 센서 없음

---

## 8. 실험 변수로 설정 가능한 것 (H10에서도 가능)

| 항목 | 방법 |
|---|---|
| 스케줄러 priority / threshold / timeout | 앱 API `set_scheduler_priority/threshold/timeout` — `ConfiguredInferModelHrpcClient`에 구현되어 있어 RPC로 디바이스 스케줄러에 전달됨. CLI는 `run2 set-net --scheduler-*` |
| batch size | 동일 |
| 모델 조합 / 프레임 수 / 데이터셋 | `--models`, `--num_images`, `--img_dir` |

따라서 **파라미터 스윕 실험 자체는 H10에서 가능하다.** 다만 관측되는 것은 *결과*(지연시간·FPS·
NNC 가동률)이고, *메커니즘*(몇 번·왜 전환했는지)은 볼 수 없다. H8L에서는 둘 다 가능했다.

---

## 9. host_trace 사용법

추가 파일 `src/host_trace.hpp`, 계측 대상 `src/hailo10h_sched_bench.cpp`.
워커 스레드는 자기 버퍼에만 기록하고 종료 후 메인 스레드가 한 번에 flush하므로 측정값에 영향 없음.

```bash
# 빌드 (npu-rpi5)
cd ~/hailo10h_sched_exp1
g++ src/hailo10h_sched_bench.cpp -o hailo10h_sched_bench -I src \
    $(pkg-config --cflags --libs opencv4) -lhailort -lpthread -std=c++17

# 실행 (타임라인·npu_percent 수집 모두 기본 ON)
#   --no_trace        타임라인 끄기,  --trace_dir  타임라인 위치
#   --no_npu_monitor  hailortcli monitor 자식 프로세스를 안 띄움(npu_percent=NaN)
./hailo10h_sched_bench --models det,seg,pose --label det_seg_pose --run_id 1 \
    --csv csv/results.csv --img_dir $HOME/datasets/sampled_val2017 --trace_dir traces

# 스케줄러 파라미터 스윕 (옵션을 안 주면 setter 자체를 호출하지 않음 = HailoRT 기본 동작)
#   --threshold / --priority 는 값 1개(세 모델 공통) 또는 det,seg,pose 순서로 3개
./hailo10h_sched_bench --models det,pose --label prio_test --run_id 1 \
    --csv csv/results.csv --img_dir $HOME/datasets/sampled_val2017 \
    --batch 1 --threshold 2 --timeout_ms 50 --priority 16,24,16
```

파라미터가 실제로 디바이스에 먹는지 확인한 실측(2026-08-31, det+pose 각 200장):
기본값 28.6 ms / NNC 67.5 % → `--threshold 2 --timeout_ms 50`을 주면 94.2 ms / NNC 13.0 %.
§8의 "H10에서도 파라미터 스윕은 가능하다"가 결과로도 확인된다.

### 스모크 검증 (2026-08-27, det+seg+pose 각 60장)

```
det : avg_latency 26.65 ms, enqueue 0.130 ms, prep 4.80 ms
seg : avg_latency 26.30 ms, enqueue 0.159 ms, prep 4.85 ms
pose: avg_latency 26.47 ms, enqueue 0.148 ms, prep 4.80 ms

동시 in-flight 요청 수 점유율 (전체 1878.8 ms)
  3개 : 1025.9 ms (54.6%)   2개 : 834.8 ms (44.4%)   1개 : 18.1 ms (1.0%)
```
같은 실행에서 `.hrtt`는 0바이트, host_trace는 events.csv 17.6KB / trace.json 50KB.
150장 실행 중 `hailortcli monitor` 동시 측정: **NNC 68~71%**, 디바이스 CPU 42%, RAM 788/7221 MB,
온도 47~48 °C (유휴 시 NNC 0%).

샘플은 `traces_sample/`에 있다 (실험 결과가 아니라 포맷 참고용).

---

## 10. 지표 출처 구분 (누가 만든 값인가)

| 구분 | 내용 |
|---|---|
| HailoRT가 직접 계측 | `.hrtt`의 원시 타임스탬프·`activate_core_op.duration`·전환 사유 플래그·`queue_size`·스케줄러 파라미터 |
| Hailo 공식 툴 계산 | `hailo runtime-profiler` HTML 화면값 (계산식 비공개) — `add_hrtt_columns.py`는 빈 열만 만들고 사람이 화면 보고 입력 |
| 우리 코드 | `tools/hrtt/parse_hrtt.py`, `parse_metrics_final.py` — protobuf를 직접 파싱해 **우리가 정의한 공식**으로 계산. `parse_metrics_final.py`는 docstring에 "화면값에 맞춰 보정한 공식"이라고 명시(리버스 엔지니어링) |
| H10용 신규 우리 코드 | `host_trace.hpp` — `chrono`로 직접 측정 |
| Hailo 공식 (디바이스 측정) | `monitor`의 NNC%·온도, `benchmark`/`run2`의 FPS·latency |

### 우리 파서 사용 시 주의 (H8L 데이터 재분석 시)

1. `activation_ms`를 두 스크립트가 다르게 계산한다 — `parse_hrtt.py`는 **평균**, `parse_metrics_final.py`는
   **합계**. HTML 화면값과 일치하는 건 합계 쪽.
2. `idle_time_pct`도 정의가 두 개다 — `parse_hrtt.py`는 `(첫 activation 전 구간)/전체`,
   `parse_metrics_final.py`는 `(run − busy 합집합)/run`.
3. 전환 사유 3개 플래그는 배타적이지 않아 퍼센트 합이 100%가 아닐 수 있다.

---

## 11. 검증 명령어

```bash
# 디바이스 내부 Linux / HailoRT-Server 로그
hailortcli logs runtime

# 디바이스용 커널 + 루트파일시스템
ls /lib/firmware/hailo/hailo10h/
file /lib/firmware/hailo/hailo10h/image-fs
/sbin/debugfs -R 'cat /etc/fstab' /lib/firmware/hailo/hailo10h/image-fs

# 호스트 라이브러리의 RPC 경로 / Perfetto 미링크
strings /lib/libhailort.so.5.3.0 | grep -i 'hrpc\|perfetto'

# 드라이버 ftrace 트레이스포인트 부재
sudo ls /sys/kernel/tracing/events | grep -i 'hailo\|h1x'

# NPU 가동률 (TUI — 파일로 남길 땐 ANSI 이스케이프 제거)
hailortcli monitor
hailortcli monitor 2>&1 | sed -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' | grep -a HAILO10H

# 단일 모델 hw latency
hailortcli run2 -t 5 --measure-latency --measure-overall-latency -j out.json \
  set-net resources/yolov8s.hef

# 다모델 동시 FPS
hailortcli run2 -t 5 -j out.json set-net a.hef set-net b.hef
```

---

## 12. 결론 — 실험 설계 방향

이 실험(`2026-08-24_det_seg_pose_workload_exp1`)은 스케줄러 파라미터를 기본값으로 고정하고
워크로드 개수에 따른 지연시간/FPS 스케일링을 보는 것이므로, **HRTT 없이 성립한다.**
스케줄러 내부 동작(전환 횟수·사유·모델별 점유율) 분석이 필요하면 Hailo-8L에서 수행하고,
Hailo-10H에서는 성능 비교와 파라미터 스윕의 *결과* 측정에 집중한다.
