# NPU Scheduler Experiment

Hailo NPU(Raspberry Pi 5)에서 Detection / Segmentation / Pose 모델을 동시에 스케줄링할 때 priority 설정이 latency 및 NPU 활용률에 미치는 영향을 측정하는 실험 프로젝트.

---

## 실험 구성 (총 63개)

| 구분 | 내용 | 개수 |
|---|---|---|
| Single | Det / Seg / Pose 각각 × priority (0, 15, 31) | 9개 |
| Multi (2모델) | Det+Seg / Det+Pose / Seg+Pose × priority 조합 | 27개 |
| Triple (3모델) | Det+Seg+Pose × priority 조합 | 27개 |

---

## 모델 파일 (Raspberry Pi 5)

```
/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef       # Detection
/home/rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef        # Segmentation
/home/rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef   # Pose
```

---

## 파일 구조

```
NPUscheduler/
├── infer_scheduler.cpp          # 핵심 추론 코드 (C++)
├── auto_experiment_all.sh       # 63개 실험 자동 실행 스크립트
├── param_range_experiment.sh    # threshold/timeout 유효 범위 탐색 (2차 실험)
├── hailo_utilization.py         # NPU 사용률 측정 (백그라운드 실행)
├── parse_npu_log.py             # NPU 로그 파싱 → CSV npu_percent 컬럼 추가
├── parse_hrtt.py                # HRTT → CSV 자동 파싱
├── add_hrtt_columns.py          # HRTT 파싱 결과를 CSV에 병합
├── convert_hrtt_to_html.sh      # HRTT → HTML 변환 (WSL 환경)
├── profiler.proto               # HailoRT 프로파일러 protobuf 스키마
├── profiler_pb2.py              # protobuf 컴파일 결과
├── scheduler_mon_pb2.py         # 스케줄러 모니터링 protobuf
├── results_all.csv              # 1차 실험 결과 (63개)
├── results_with_hrtt.csv        # 수동 입력 HRTT 분석 결과
└── results_with_hrtt_parsed.csv # 자동 파싱 HRTT 분석 결과
```

---

## CSV 컬럼 설명 (`results_all.csv`)

| 컬럼 | 설명 |
|---|---|
| use_det / use_seg / use_pose | 모델 활성화 여부 (0 or 1) |
| batch | 배치 크기 |
| threshold | 스케줄러 threshold (프레임 수) |
| timeout_ms | 스케줄러 timeout (ms, 0=비활성화) |
| priority_det / seg / pose | 각 모델의 priority (0~31, None=비활성) |
| det_latency_ms / seg / pose | 평균 추론 latency (ms, None=비활성) |
| cpu_percent | CPU 사용률 (%) |
| mem_percent | 메모리 사용률 (%) |
| context_switches | 컨텍스트 스위치 횟수 |
| npu_percent | NPU 사용률 (%) |

비활성 모델의 priority 및 latency는 `None`으로 기록됨.

---

## HRTT 파일명 규칙

```
{USE_D}D-{USE_S}S-{USE_P}P_{PRI_D}PD-{PRI_S}PS-{PRI_P}PP_{TIMESTAMP}.hrtt

예) 1D-1S-1P_0PD-15PS-31PP_2026-06-05_19-11-51.hrtt
    → Det+Seg+Pose 동시 실행, priority: Det=0, Seg=15, Pose=31
```

---

## CSV 데이터 수집 방식

### Latency 측정 (C++ / `infer_scheduler.cpp`)
HailoRT API가 아닌 **C++ 표준 타이머**로 wall-clock 시간 측정.

```cpp
auto start = std::chrono::high_resolution_clock::now();
hailo_vstream_write_raw_buffer(input_vstreams[0], frame, size);
hailo_vstream_read_raw_buffer(output_vstreams[i], buf, size);
auto end = std::chrono::high_resolution_clock::now();
// latency = (end - start) / BATCH_SIZE
```

starvation 발생 시 `hailo_vstream_read_raw_buffer`가 블로킹되어 **대기 시간 포함 값**이 기록됨. 리턴값은 무시됨.

### CPU / Memory / Context Switch (C++ / Linux proc)
```cpp
/proc/stat      // CPU 사용률, context switch 횟수
/proc/meminfo   // 메모리 사용률
```

### NPU 사용률 (Python)
- `hailo_utilization.py`: 추론 중 백그라운드에서 NPU 사용률 샘플링
- `parse_npu_log.py`: 샘플링 결과를 파싱하여 CSV `npu_percent` 컬럼에 기록

---

## 실험 실행 방법

### 1. RPi에 파일 전송 (PC에서)
```bash
scp -P 40021 infer_scheduler.cpp hailo_utilization.py parse_npu_log.py \
    scheduler_mon_pb2.py auto_experiment_all.sh \
    rpi1@155.230.16.157:~/hailo_cpp_test/
```

### 2. RPi 접속 및 실험 실행
```bash
ssh rpi1@155.230.16.157 -p 40021
cd ~/hailo_cpp_test
chmod +x auto_experiment_all.sh
nohup ./auto_experiment_all.sh > experiment_log.txt 2>&1 &
```

### 3. 결과 다운로드 (PC에서)
```bash
scp -P 40021 rpi1@155.230.16.157:~/hailo_cpp_test/results_all.csv .
scp -P 40021 "rpi1@155.230.16.157:~/hailo_cpp_test/traces/*.hrtt" ./hrtt/
```

### 4. HRTT → HTML 변환 (WSL 환경)
```bash
source ~/hailo_venv/bin/activate
for f in hrtt/*.hrtt; do
    name=$(basename "$f" .hrtt)
    hailo runtime-profiler "$f"
    mv runtime_report.html "html/${name}.html"
done
```

---

## 주요 발견 사항

### Priority Starvation
`TIMEOUT_MS=0` (스케줄러 timeout 비활성화) 상태에서 priority 차이가 클수록 낮은 priority 모델이 NPU 스케줄링 기회를 거의 받지 못함.

| Det | Seg | Pose | Det Device Usage | Det FPS |
|---|---|---|---|---|
| 0 | 15 | 31 | 0.13% | 0.03 |
| 0 | 0  | 0  | ~33%  | ~8.3 |

- `TIMEOUT_MS > 0` 설정 시 강제 선점으로 starvation 방지 가능
- HRTT 타임라인에서 starved 모델의 activation이 줌아웃 시 비가시 (38ms / 29min = 0.002% 폭)

### Latency 왜곡
Starvation 케이스에서 CSV latency는 **실제 추론 시간 + NPU 대기 시간**이 합산되어 기록됨.

### 스케줄러 파라미터 타입
- `hailo_set_scheduler_threshold()` / `hailo_set_scheduler_timeout()` 인자 타입: `uint32_t`
- 범위: [0, 4,294,967,295]

---

## 환경

- Hardware: Raspberry Pi 5 + Hailo-8L NPU
- HailoRT: ROUND_ROBIN scheduling algorithm
- Dataset: COCO val2017 (`/home/rpi1/datasets/coco/val2017/`)
- Build: `g++` with HailoRT / OpenCV libraries
