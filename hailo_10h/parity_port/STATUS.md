# H10 vs 8L 비교 실험 — 현황과 다음 할 일

최종 갱신 2026-09-01 · `hailo_10h/parity_port/`

---

## 1. 결론 세 줄

1. **처음 보였던 "10배 차이"는 착시였다.** H10 코드가 `run_async` 직후 `wait()`해서 in-flight가 1이었고, 그래서 8L과 **다른 물리량**을 재고 있었다.
2. **슬롯 풀(in-flight 12)로 고치니 39.9 → 166.0 FPS.** 파이프라인 깊이가 8L(11.23)과 11.60으로 맞았고, 이제 두 보드를 같은 표에 놓을 수 있다.
3. **남은 차이는 하드웨어 체급이다.** 처리량 비 3.23배가 순수 연산 서비스시간 비 3.41배의 95%까지 설명된다.

---

## 2. 확정 수치 (600장, 후처리 없음, 출력 AUTO)

| 조건 | 8L FPS | 8L latency | H10 FPS | H10 latency | 배율 |
|---|---|---|---|---|---|
| det 단독 | 51.33 | 218.9 ms | **166.00** | 69.9 ms | **3.23×** |
| seg 단독 | 36.81 | 281.8 ms | 124.40 | 97.8 ms | 3.38× |
| pose 단독 | 47.81 | 213.3 ms | *미측정* | — | — |
| 3모델 (모델당) | 11.44 | 964.1 ms | 32.82 | 385.9 ms | 2.87× |
| 3모델 (합계) | 34.32 | — | 98.46 | — | 2.87× |

**H10 det 단독 상세** (600장 × 3회)

| 지표 | 값 | 해석 |
|---|---|---|
| in-flight | 11.60 | 8L 11.23 대비 1.03× → **비교 조건 성립** |
| `npu_percent` | 91.6% | NNC 포화 = 정상 |
| 실측 ÷ `benchmark` | 166.00 ÷ 166.66 = **99.6%** | 런타임 손실 없음 |
| 전처리 | 4.55 ms | 주기 6.02 ms의 76% |
| 큐대기 | 1.17 ms | **호스트에 남은 여유** — 0이 되면 호스트가 병목 |
| `hw_latency` | 5.72 ms | `run2 --measure-latency` |

**모델이 늘수록 배율이 3.23 → 2.87로 떨어진다.** `npu_percent`도 91.6 → 75.9%로 같이 떨어짐. 컨텍스트 전환 증가로 보이지만 **H10에서는 계측 불가**(`.hrtt` 0바이트).

---

## 3. 코드 작업 요약

### 성능을 바꾼 것 — 딱 하나

**슬롯 풀 도입.** `Bindings`+입출력 버퍼를 모델당 1세트 → 12세트로. `run_async` 후 `detach()`하고 완료는 콜백으로 받는다.

> **스레드가 늘어난 게 아니다.** 입력 스레드는 여전히 1개. H10의 `InferModel`은 우리 버퍼를 **복사하지 않고 DMA로 직접 읽어가서**, 완료 전에 덮어쓰면 안 되기 때문에 버퍼가 12벌 필요한 것이다. (8L의 `write()`는 복사해 가므로 버퍼 1개로 충분했다.)

### 구조상 불가피했던 것

**H10에는 VStreams / `ConfiguredNetworkGroup` 경로가 없다.** `create_configure_params()`가 `HAILO_NOT_IMPLEMENTED(7)`로 실패하며 HailoRT가 *"use InferModel instead"* 라고 직접 안내한다. 그래서 8L이 HailoRT에게 맡겼던 파이프라이닝을 우리가 손으로 만들어야 했다.

### 8L 로직 복원 — 완료

| 항목 | 조치 |
|---|---|
| 후처리 구간 계측 | 8L은 `ENABLE_POSTPROCESS=0`에서도 ~1e-4 ms를 기록(NaN 아님). 디코딩 코드 없이 계측만 복원 |
| `get_image_files()` | 8L 원본(`.jpg`/`.JPG` 부분 문자열)으로 되돌림 + 호출부에서 `img_dir` 슬래시 보정 |
| `cpu_end` 위치 | 8L처럼 "장당 전체시간" 루프 뒤로 이동 |
| `avg_total_time_ms` | `전처리 + latency + 후처리` (8L 식) |
| 측정 지점 | `enq_ts`는 8L의 `write()` 직전 = H10의 슬롯 확보 직전 |

### 발견·수정한 버그

- **슬롯 이중 반납** — `run_async` 실패 시 writer와 reader가 같은 슬롯을 두 번 반납 → 두 프레임이 같은 버퍼를 공유 → **조용히 틀린 결과**. `slot_of[i] = -1`로 수정
- **스윕 CSV에 in-flight 구분 불가** — 49컬럼에 D 칸이 없어 파일명으로 분리
- **`service` 주석 오류** — "큐 대기 제외"가 아니라 **디바이스 큐 대기 포함**(D × 1/FPS에 비례)

### 종결된 의문

**`power_mode ULTRA_PERFORMANCE`는 H10에 없는 개념이다.** Hailo 스태프 답변: *"this flag is not relevant for Hailo-10H (it only applies to Hailo-8)."* → 조건 불일치가 아니라 대응 설정 자체가 부재. **이로써 8L 대비 미확인 조건 차이는 남아 있지 않다.**

---

## 4. 다음 실험 — 우선순위 순

### ① 스윕 완주 (필수)

SSH가 끊겨 `pose 단독`과 쌍 조합 3개가 빠져 있다. 표의 빈칸이 그것.

```bash
ssh -o ServerAliveInterval=30 npu-rpi5@155.230.16.157 -p 40020
cd ~/hailo10h_sched_exp1/parity_port
./build.sh                                    # 8L 로직 복원분 반영 필요
mv csv/results_h10_*.csv csv/old_$(date +%m%d)/ 2>/dev/null
nohup ./run_workload_sweep_h10.sh > logs/sweep_full.log 2>&1 &
tail -f logs/sweep_full.log                   # Ctrl-C 해도 계속 돔
```

### ② in-flight 깊이 스윕 (방어 논리)

`--inflight 12`는 8L의 11을 보고 맞춘 값이라 "유리한 값을 골랐다"는 지적이 가능하다. **곡선을 제시하면 12가 특별한 값이 아니라 포화 구간이라는 게 보인다.**

```bash
MODELS=det          ./run_inflight_sweep.sh   # D = 1,2,4,8,12,16
MODELS=det,seg,pose ./run_inflight_sweep.sh
```

각 로그의 `[in-flight 추정]`이 실제로 D만큼 찼는지 확인할 것.

### ③ 8L 순수 HW 상한 실측 (가장 큰 구멍)

표의 8L 서비스시간 **19.5 ms는 FPS에서 역산한 값**이다. 이게 실측되어야 "3.41배"가 확정된다.

```bash
# npu-rpi1 (포트 40021, 계정 npu-rpi1)
hailortcli benchmark resources/yolov8s_h8l.hef
hailortcli run2 -t 10 --measure-latency --measure-overall-latency \
  set-net resources/yolov8s_h8l.hef
```

판정: 각 보드에서 `실측 FPS ÷ benchmark FPS ≥ 0.9`면 그 보드의 런타임은 문제 없음.

### ④ HEF 컨텍스트 개수 비교 (TOPS로 설명 안 되는 2배의 정체)

TOPS는 1.54배 차이인데 서비스시간은 3.41배 차이다. **초과분의 유력 원인은 컨텍스트 교체** — 8L은 DRAM-free라 모델이 SRAM에 안 들어가면 프레임마다 가중치를 갈아끼운다.

```bash
hailortcli parse-hef resources/yolov8s.hef | grep -i -A2 context   # H10 = 3 contexts
hailortcli parse-hef resources/yolov8s_h8l.hef | grep -i -A2 context
```

8L 쪽이 더 많으면 → 격차의 초과분은 "연산기가 적어서"가 아니라 **"모델이 칩에 안 들어가서"**.

### ⑤ RPi5 전원·스로틀 확인 (166 FPS의 신뢰도)

Hailo 스태프가 *전력 부족 시 모듈이 스스로 reduced-performance mode로 내려간다*고 언급했다. 지금의 166 FPS가 저전력 상태에서 나온 값일 가능성을 배제해야 한다.

```bash
vcgencmd get_throttled                  # 0x0 이어야 정상
dmesg | grep -i -E 'under.?volt|throttl'
hailortcli monitor | head -20           # 온도·전압
```

`benchmark`도 같은 조건에서 돌기 때문에 **99.6% 도달률은 이 문제를 걸러내지 못한다.**

### ⑥ PCIe 대역폭 실측

`model_setup.hpp`에 프레임당 왕복 바이트 로그를 추가해뒀다. 준비 로그에서 값을 읽어 `FPS × 바이트`를 Gen3 x1 실효(~850 MB/s)와 비교.

- det: 204 MB/s = 링크의 24% → 병목 아님
- seg: 출력 10개(raw 텐서)라 **440 MB/s 추정 = 링크의 절반** → det/seg의 NNC% 차이(92% vs 85%)와 관련 있을 수 있음

### ⑦ (선택) 8L 3회 재측정

지금 비교 기준인 8L 51.33 FPS는 `results_B.csv`의 **1회 실행**이다. H10은 3회 평균. 8L도 3회로 맞추면 배율이 단단해진다.

---

## 5. 논문·보고서에 쓸 때 주의할 것

- **latency는 두 보드 모두 체류시간(sojourn)이다.** 큐 대기를 포함하며, 순수 연산이 아니다. 순수 연산은 `hw_latency`(H10 5.72 ms)로만 얻는다.
- **8L과 H10은 큐가 있는 위치가 다르다.** 8L은 호스트 vstream 큐, H10은 디바이스 내부 큐. 깊이는 맞췄지만 대기 장소는 원리적으로 못 맞춘다.
- **H10에서 영구히 얻을 수 없는 값**: 컨텍스트 전환 횟수·사유, 모델별 디바이스 점유율, `activation_*`. `.hrtt`가 0바이트이기 때문(`HRTT_ON_HAILO10H.md` 참고).
- **현재 3.23배는 하한값**이다. H10에 유리한 조건(전력 모드, x4 레인)이 적용되지 않은 상태에서 나온 수치다.
