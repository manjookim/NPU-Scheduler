# Hailo-8L → Hailo-10H 포팅 코드 리뷰

작성 2026-08-31 · 대상: `hailo_8L/{infer_scheduler.cpp, model_setup.hpp, model_runner.hpp, csv_writer.hpp}`
vs `hailo_10h/experiments/2026-08-24_det_seg_pose_workload_exp1/src/hailo10h_sched_bench.cpp`

> **결론 먼저** — 10배 차이는 하드웨어 차이도, 스케줄러 파라미터 누락도 아니다.
> **두 코드의 파이프라인 깊이(in-flight 요청 수)가 11.2 vs 0.77로 다르고, 그래서 `latency`의
> 정의 자체가 서로 다른 값을 재고 있다.** 8L은 포화된 큐의 체류시간(sojourn)을, H10은 요청
> 1개의 서비스시간(service)을 재고 있다. 부수적으로 H10 코드는 NPU를 24%밖에 못 쓰고 있어
> FPS는 오히려 8L보다 낮다.

---

## 0. 관측값 정리 (기존 CSV에서 그대로 계산)

| | 8L Gen3 B/D조건 (600장) | H10 기본값 (673장) |
|---|---|---|
| det 단독 FPS | **51.33** | **39.92** |
| det 단독 `det_latency_ms` | **218.87** | **19.17** |
| 3모델 동시, 모델당 FPS | 11.44 | 31.17 |
| 3모델 동시, 합계 FPS | 34.3 | **93.5** |
| 3모델 동시 `det_latency_ms` | 964.14 | 26.23 |
| NNC 가동률 | 92~99 % (2026-07-28 실험) | **23.26 %** (단독) / 72.58 % (3모델) |

**Little의 법칙(L = λ·W)으로 in-flight 요청 수를 역산하면:**

| 조건 | FPS(λ) | latency(W) | **평균 in-flight 프레임 수(L)** |
|---|---|---|---|
| 8L det 단독 | 51.33 | 0.2189 s | **11.23** |
| 8L 3모델 det | 11.44 | 0.9641 s | **11.03** |
| H10 det 단독 | 39.92 | 0.0192 s | **0.77** |
| H10 3모델 det | 31.17 | 0.0262 s | **0.82** |

8L은 조건이 5배 바뀌어도 항상 **11프레임**이 파이프라인 안에 떠 있고, H10은 항상 **1프레임 미만**이다.
`218.87 / 19.17 = 11.4` ≈ `11.23 / 0.77 = 14.6` — 관측된 "10배"는 in-flight 깊이 비율 그 자체다.

**보조 증거 3개:**

1. **H10은 겹침이 0이다.** `1000 / (19.1699 + 5.8073) = 40.04 FPS`, 실측 39.92 FPS.
   전처리와 추론이 소수점 둘째 자리까지 완전히 직렬이다(3모델도 31.25 vs 31.17로 동일).
2. **8L latency는 연산이 아니라 전송/큐가 지배한다.** PCIe Gen2→Gen3 전환만으로
   det 단독 latency가 419.57 → 218.87 ms(1.92배)로 줄었다(`comparison_summary.md`).
   NNC 연산량은 그대로인데 latency가 절반이 됐다 = 재고 있는 건 큐 체류시간이다.
3. **H10 NNC 실사용 시간은 프레임당 5.8 ms다.** `npu_percent 23.26 % × 25.05 ms 주기 = 5.83 ms`.
   `hailortcli run2 --measure-latency`의 `hw_latency 5.72 ms`, `benchmark` 166.66 FPS와 일치한다.
   즉 **호스트가 관측하는 19.17 ms 중 13.4 ms(70 %)는 RPC 왕복·DMA 대기이고, 파이프라이닝만
   하면 전부 숨길 수 있는 시간이다.**

---

## 1. 추론 파이프라인 및 로직 일관성 검증

### 1-1. 두 코드는 동시성 구조가 다르다 (핵심)

| 항목 | Hailo-8L | Hailo-10H |
|---|---|---|
| API | VStreams (`create_input_vstreams` / `write` / `read`) | InferModel Async (`run_async` / `wait`) |
| 모델당 스레드 | **2개** (writer + reader) | **1개** (worker) |
| 동시성 형태 | producer/consumer 파이프라인 | **완전 동기 lock-step** |
| 버퍼 | vstream 내부 큐(element별 기본 깊이 2) + HW 큐 | **입력 버퍼 1개 / Bindings 1개** |
| 전처리 위치 | writer 스레드 (reader의 추론·후처리와 겹침) | worker 스레드 **직렬** |
| in-flight | ~11 프레임 | **1 프레임** |

문제 지점은 `worker_loop()`의 이 4줄이다:

```cpp
auto job_exp = ctx->configured->run_async(*ctx->bindings);   // 비동기 제출
...
auto status = job_exp->wait(std::chrono::milliseconds(5000)); // 바로 기다림 → 동기
```

`run_async`로 제출하고 그 자리에서 `wait`하면 **동기 추론**이다. 다음 프레임의 `imread`조차
이 `wait`가 끝나야 시작된다. 8L의 writer 스레드는 큐가 받아주는 한 계속 밀어 넣고,
reader가 별도로 빼기 때문에 NPU가 놀지 않는다.

구조상 파이프라이닝이 불가능하게 되어 있는 점도 짚어야 한다: `ctx->bindings`와
`ctx->input_buffer`가 **모델당 1개**다. 두 번째 프레임을 제출하려면 첫 프레임이 아직 읽고 있는
입력 버퍼를 덮어써야 하므로, `wait`를 빼는 것만으로는 고칠 수 없고 **버퍼/Bindings 풀**이 필요하다.

### 1-2. 전/후처리 일관성 (비교 공정성에 영향)

| 항목 | 8L | H10 | 영향 |
|---|---|---|---|
| 리사이즈 | `letterbox()` (비율 유지 + 패딩) | `cv::resize()` (비율 무시) | 연산량 비슷, **정확도 비교 시엔 불일치** |
| 색변환 | letterbox 후 BGR2RGB | resize 전 BGR2RGB (**전체 원본 해상도에 대해 수행**) | H10이 약간 손해. 640×640으로 줄인 뒤 변환하는 게 맞다 |
| 출력 포맷 | `HAILO_FORMAT_TYPE_AUTO` (B/D조건) | InferModel 기본값 | **확인 필요** — InferModel이 FLOAT32로 자동 변환 중이면 역양자화 비용이 H10 latency에 포함된다 |
| 후처리 | `ENABLE_POSTPROCESS` 스위치 | 없음(항상 off) | CSV에서 `postprocess_ms_*`=NaN으로 명시됨, OK |
| power mode | `HAILO_POWER_MODE_ULTRA_PERFORMANCE` 명시 | **미설정**(기본 PERFORMANCE) | H10에서 이 설정이 유효한지 확인 필요 |
| 이미지 장수 | 600 (`NUM_IMAGES 600`) | 673 (전체) | FPS엔 영향 적지만 `run_time_s`/`total_time_*_s`는 직접 비교 불가 |

`ctx.infer_model->output(name)->set_format_type(...)`로 출력 타입을 명시해 8L의 AUTO와 조건을
맞추는 것을 권장한다(현재 무엇으로 협상되는지 로그로 한 번 찍어볼 것).

### 1-3. 그 외 발견

- `list_images()`의 확장자 검사가 "뒤 4글자"라서 `.jpg`/`.png`는 점 포함, `.jpeg`는 점 없이
  비교된다(동작은 하지만 규칙이 섞여 있다). `rfind('.')` 기준으로 통일할 것. 8L은
  `IMG_DIR` 전체를 읽고 `NUM_IMAGES`로 자르므로, **두 코드가 실제로 같은 파일 목록을
  같은 순서로 읽는지** 한 번 대조해 볼 것.
- 실패 프레임에서 `continue`할 때 `frame_count`가 늘지 않는데 `t_start`~`t_end` 구간에는
  포함된다. 실패가 많으면 FPS가 조용히 낮게 나온다. 실패 카운터를 따로 세워 로그로 남길 것.
- `job_exp->wait(5000ms)` 타임아웃 시 그냥 넘어간다. 8L은 `HAILO_TIMEOUT`이면 성공할 때까지
  **재시도**해서 프레임을 유실하지 않는다. 조건 불일치다.

---

## 2. 스케줄러 파라미터 누락의 영향 분석

### 2-1. 결론: 이번 성능 차이의 원인이 **아니다**

`SchedParams`의 기본값 주석은 정확하다. 그리고 8L 코드의 실제 설정값을 보면:

| 파라미터 | 8L `infer_scheduler.cpp` | HailoRT 기본값 (H10에서 setter 미호출) | 동일? |
|---|---|---|---|
| batch | 1 | `HAILO_DEFAULT_BATCH_SIZE`(0=auto) | 사실상 동일(in-flight 1이라 auto도 1) |
| threshold | 1 | 1 | **동일** |
| timeout | 0 ms | 0 ms | **동일** |
| priority | 15 (세 모델 전부) | 16 = NORMAL (세 모델 전부) | **값은 다르나 전부 같은 값 → 둘 다 Round-Robin** |

priority는 절대값이 아니라 **모델 간 상대 순서**만 의미가 있다. 세 모델이 모두 15든 모두 16이든
스케줄링 결과는 같다. 따라서 **setter를 주석 처리한 것과 8L 설정은 스케줄링상 등가**이며,
성능 차이를 설명하지 못한다. "파라미터를 안 줘서 느린 것 같다"는 가설은 여기서 기각된다.

### 2-2. 다만 지금 구조에서는 파라미터 스윕 자체가 성립하지 않는다

`HRTT_ON_HAILO10H.md` §9에 기록된 실측이 이걸 정확히 보여준다:

> 기본값 28.6 ms / NNC 67.5 % → `--threshold 2 --timeout_ms 50` 주면 **94.2 ms / NNC 13.0 %**

이건 "파라미터가 먹는다"는 증거이면서 동시에 **현재 코드의 결함이 드러난 증거**다.
in-flight가 1이므로 스케줄러 큐에 쌓인 요청은 **항상 1개**다. `threshold=2`는 영원히 충족될 수
없고, 매 프레임 `timeout_ms=50`을 다 기다린 뒤에야 실행된다. 그래서 94.2 ms ≈ 50 ms + 28.6 ms,
NNC 13 %로 폭락한 것이다.

> **규칙: `threshold ≤ 동시 in-flight 요청 수`**
> 8L에서 "threshold ≤ batch"로 알고 있던 제약의 일반형이다. 8L은 vstream이 알아서 ~11개를
> 채워주니 threshold를 올릴 여지가 있었지만, H10 코드는 1이라 threshold∈{1}만 유효하다.

`batch` 역시 마찬가지다. Async API에서 batch N은 **N개의 요청이 큐에 동시에 있어야** 묶인다.
in-flight 1에서 `--batch 8`은 배치를 만들지 못하고 오히려 대기만 늘린다.

즉 **파이프라이닝을 넣기 전까지는 threshold/timeout/batch 스윕 결과가 전부 "timeout 대기시간
측정"이 되어버린다.** 스케줄러 실험의 전제조건이 파이프라이닝이다.

### 2-3. H10 아키텍처 특성

스케줄러가 디바이스 안 `hailort_server`에 있으므로(§HRTT 문서 §1), 호스트의 `run_async`는
RPC 요청 1건이다. 요청을 1건씩만 보내면 **디바이스 스케줄러는 매번 "큐에 1개짜리 네트워크"를
보게 되고**, 다모델 실행 시 프레임마다 컨텍스트 스위치가 강제된다. 3모델 동시 실행에서
NNC 72.58 %가 나온 건 모델 3개 덕에 우연히 큐에 3건이 있었기 때문이고, 단독 실행에서
23.26 %로 떨어지는 게 그 반증이다. **동시성을 워크로드 개수에 의존하지 말고 모델별
in-flight 깊이로 확보해야** 워크로드 스케일링 실험이 의미를 갖는다.

---

## 3. 10배 성능 차이의 핵심 가설 (우선순위 순)

### 가설 1 — 측정 정의 불일치 (latency 10배의 ~전부를 설명, 신뢰도 상)

- 8L `det_latency_ms` = `enq_ts[i]`(write 호출 직전) → `deq_ts[i]`(모든 출력 read 완료).
  writer가 `INPUT_FPS=0`으로 최대 속도로 밀기 때문에 큐가 항상 포화 → **큐 대기시간 포함
  체류시간(sojourn latency)**. Little의 법칙으로 11.2프레임 × 19.5 ms/프레임 ≈ 219 ms.
- H10 `det_latency_ms` = `run_async` 제출 → `wait` 완료, in-flight 1 → **순수 서비스시간**.
- 검증: 8L의 in-flight가 조건과 무관하게 11.0~11.2로 고정. PCIe Gen2→Gen3에서 latency만
  1.92배 개선(연산 불변).
- **두 값은 다른 물리량이며, 지금 상태로는 나란히 놓으면 안 된다.**

### 가설 2 — H10의 lock-step 동기 루프가 RPC 왕복을 노출 (FPS 손실 원인, 신뢰도 상)

- `hw_latency` 5.72 ms vs 호스트 관측 19.17 ms → **프레임당 13.4 ms가 RPC/DMA 왕복**.
- H10은 추론 스택이 디바이스 안이라 호스트↔디바이스 RPC가 8L의 로컬 함수 호출보다 훨씬 비싸다.
  8L은 파이프라인이 이 비용을 가려줬고, H10 코드는 매 프레임 정면으로 맞는다.
- 결과: `benchmark` 상한 166.66 FPS 대비 **39.92 FPS = 23.9 %**.
- 예상 개선: in-flight 4~8이면 단독 모델 120~165 FPS 도달 가능.

### 가설 3 — 전처리가 추론과 전혀 겹치지 않음 (FPS 23 % 손실, 신뢰도 상)

- `1000/(19.1699 + 5.8073) = 40.04` vs 실측 39.92. 겹침 0 %가 산술적으로 확정된다.
- 8L은 writer 스레드가 전처리하는 동안 reader가 이전 프레임을 처리한다.
- 전처리 5.81 ms 중 `imread`(JPEG 디코딩)가 대부분이다. 파이프라이닝만 넣어도
  이 시간은 추론 뒤로 숨는다.

### 참고: 하드웨어 자체는 문제가 아니다

3모델 동시 실행 합계 FPS는 8L 34.3 vs H10 93.5로 **H10이 이미 2.7배 빠르다**.
단독 실행에서만 H10이 지는 것(39.9 vs 51.3)은 위 세 가설의 결과이지 성능 부족이 아니다.

---

## 4. CSV 로깅 파이프라인 정합성 및 HRTT 데이터 처리 가이드

### 4-1. 현재 스키마는 "완전히 동일"하지 않다

컬럼 수: **8L 49개 vs H10 46개**. 공통 컬럼의 상대 순서는 같지만 위치 인덱스가 어긋난다.

**H10에서 빠진 8개 (전부 HRTT 전용):**
`switches_per_s`(20), `idle_time_pct`(21), `avg_latency_det`(24), `activation_det`(26),
`avg_latency_seg`(28), `activation_seg`(30), `avg_latency_pose`(32), `activation_pose`(34)

**H10에만 있는 4개:** `label`(맨 앞), `det_frame_count`, `seg_frame_count`, `pose_frame_count`

`label`이 **맨 앞**에 들어간 게 특히 문제다. `pd.read_csv(...).iloc[:, 12]`처럼 위치로 접근하는
기존 스크립트(`fill_hrtt_columns*.py`, `make_avg_csv.py`)가 전부 어긋난다.

### 4-2. 권장: 8L 49컬럼을 **한 글자도 바꾸지 않고 앞에 두고**, 확장 컬럼은 뒤에 붙인다

```
[0..48]  8L csv_writer.hpp HEADER 원문 그대로 (49개)
[49..]   H10 확장: label, det_frame_count, seg_frame_count, pose_frame_count,
                   inflight_depth, service_latency_ms_det/seg/pose
```

이러면 `df10.iloc[:, :49]`와 `df8`이 바로 concat/비교 가능하고, 기존 파이썬 스크립트가
그대로 돈다. 아래 헤더를 `csv_schema_8l.hpp`로 만들어 두 프로젝트가 **같은 파일을 공유**하면
스키마 드리프트가 원천 차단된다.

```cpp
#pragma once
// csv_schema_8l.hpp — 8L/H10 공용 CSV 스키마 단일 정의(SSOT).
// [0..48]은 hailo_8L/csv_writer.hpp의 HEADER와 바이트 단위로 동일해야 한다. 절대 순서 변경 금지.
// [49..]는 H10 확장 — 뒤에만 추가할 것(앞/중간 삽입 시 8L 파서가 전부 깨진다).

#define CSV_SCHEMA_8L_CORE \
    "run_id,use_det,use_seg,use_pose,batch," \
    "threshold_det,threshold_seg,threshold_pose,timeout_ms," \
    "priority_det,priority_seg,priority_pose," \
    "det_latency_ms,seg_latency_ms,pose_latency_ms," \
    "cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches," \
    "npu_percent,switches_per_s,idle_time_pct,run_time_s," \
    "avg_fps_det,avg_latency_det,max_latency_det,activation_det," \
    "avg_fps_seg,avg_latency_seg,max_latency_seg,activation_seg," \
    "avg_fps_pose,avg_latency_pose,max_latency_pose,activation_pose," \
    "total_time_det_s,total_time_seg_s,total_time_pose_s," \
    "avg_preprocess_ms_det,avg_preprocess_ms_seg,avg_preprocess_ms_pose," \
    "postprocess_ms_det,postprocess_ms_seg,postprocess_ms_pose," \
    "total_time_ms_det,total_time_ms_seg,total_time_ms_pose," \
    "total_time_ms_nopp_det,total_time_ms_nopp_seg,total_time_ms_nopp_pose"

#define CSV_SCHEMA_H10_EXT \
    "label,det_frame_count,seg_frame_count,pose_frame_count," \
    "inflight_depth,service_latency_ms_det,service_latency_ms_seg,service_latency_ms_pose"

#define CSV_HEADER_H10  CSV_SCHEMA_8L_CORE "," CSV_SCHEMA_H10_EXT
```

### 4-3. HRTT 전용 지표의 표준 핸들링

원칙 세 가지:

1. **값이 아니라 "출처"를 타입으로 표현한다.** 미측정을 `-1`이나 `0`으로 흘려보내지 말고,
   플랫폼에서 원리적으로 얻을 수 없는 값은 컴파일 타임에 `NA`로 못 박는다.
2. **문자열은 `"NaN"`으로 통일한다.** 8L `dtos()`가 이미 그렇게 쓰고 있고,
   `pandas.read_csv`는 `NaN`을 기본으로 결측 처리한다. 빈 문자열도 결측이 되지만
   8L 파일과 바이트 비교가 안 되므로 `"NaN"`을 쓴다.
3. **어떤 상황에서도 컬럼 개수는 고정한다.** 실패 경로에서 조기 `return`하며 행을 안 쓰거나
   짧은 행을 쓰면 안 된다.

```cpp
// metric_cell.hpp — 미측정/미지원 값을 타입으로 구분해 CSV에 안전하게 쓴다.
#include <cmath>
#include <sstream>
#include <string>

enum class Origin {
    MEASURED,        // 이 실행에서 실측함
    NOT_APPLICABLE,  // 이 조건에서 의미 없음 (비활성 모델, 후처리 미수행 등)
    HRTT_ONLY,       // HRTT에서만 얻을 수 있음 → H10에서는 영구 결측
    DEFERRED,        // 나중에 후처리 스크립트가 채울 예정
};

struct Cell {
    Origin origin = Origin::NOT_APPLICABLE;
    double value = 0.0;

    static Cell measured(double v) {
        // NaN/inf가 실측값으로 들어오는 사고 방지 — 조용히 섞이면 평균이 통째로 오염된다.
        if (!std::isfinite(v)) return Cell{Origin::NOT_APPLICABLE, 0.0};
        return Cell{Origin::MEASURED, v};
    }
    static Cell na()       { return Cell{Origin::NOT_APPLICABLE, 0.0}; }
    static Cell hrtt_only(){ return Cell{Origin::HRTT_ONLY, 0.0}; }
    static Cell deferred() { return Cell{Origin::DEFERRED, 0.0}; }

    std::string str() const {
        if (origin != Origin::MEASURED) return "NaN";   // 8L dtos()와 동일 규약
        std::ostringstream os; os << value; return os.str();
    }
};

inline std::ostream& operator<<(std::ostream& os, const Cell& c) { return os << c.str(); }
```

사용 예 — HRTT 컬럼을 **위치를 유지한 채** 명시적으로 비운다:

```cpp
// H10에서 원리적으로 못 얻는 값(HRTT_ON_HAILO10H.md §7). 컬럼은 남기고 값만 NaN.
const Cell switches_per_s = Cell::hrtt_only();
const Cell idle_time_pct  = Cell::hrtt_only();
const Cell activation[3]  = {Cell::hrtt_only(), Cell::hrtt_only(), Cell::hrtt_only()};

// 8L에서는 HRTT가 채웠지만 H10에서는 호스트 실측으로 대체하는 값.
// avg_latency_*는 det_latency_ms와 정의가 겹치므로 중복 기재하지 말고 NaN으로 두는 편이
// 낫다 — 채워 넣으면 "HRTT 유래 값"과 "호스트 실측"이 같은 파일에서 구분 불가능해진다.
const Cell avg_latency[3] = {Cell::hrtt_only(), Cell::hrtt_only(), Cell::hrtt_only()};
const Cell avg_fps[3]  = { Cell::measured(fps(0)),    ... };  // 호스트 실측으로 대체 (문서에 명기)
const Cell max_lat[3]  = { Cell::measured(maxlat(0)), ... };  // 호스트 실측으로 대체
```

**분석 단계 가드**(플랫폼 간 비교 시 결측을 0으로 오해하는 사고 방지):

```python
HRTT_ONLY = ["switches_per_s", "idle_time_pct",
             "avg_latency_det", "avg_latency_seg", "avg_latency_pose",
             "activation_det", "activation_seg", "activation_pose"]

def load(path, platform):
    df = pd.read_csv(path)                       # "NaN" → np.nan 자동 처리
    df["platform"] = platform
    if platform == "H10":
        bad = [c for c in HRTT_ONLY if c in df and df[c].notna().any()]
        assert not bad, f"H10에서 나올 수 없는 HRTT 값이 채워져 있음: {bad}"
    return df

# 비교는 반드시 두 플랫폼 모두 값이 있는 컬럼으로만
common = [c for c in df8.columns if c in df10.columns
          and df8[c].notna().any() and df10[c].notna().any()]
```

### 4-4. 반드시 CSV/문서에 남겨야 할 메타

지금 CSV만 보고는 8L 219 ms와 H10 19 ms가 다른 물리량이라는 걸 알 수 없다. 최소한 이 셋을
확장 컬럼이나 동봉 `meta.json`에 남길 것: **`inflight_depth`**, **`n_images`**(600 vs 673),
**`latency_definition`**(`sojourn` / `service`). 이게 없으면 6개월 뒤 이 CSV로 잘못된 결론이 난다.

---

## 5. 코드 리팩토링 및 벤치마크 가이드

### 5-1. 최소 수정으로 파이프라이닝 넣기 (효과 가장 큼)

핵심은 **모델당 슬롯 D개**(버퍼 + Bindings)를 두고, 슬롯이 빌 때까지만 기다리는 것이다.

```cpp
// 슬롯 = 프레임 1개가 in-flight 상태로 점유하는 자원 묶음.
struct Slot {
    ConfiguredInferModel::Bindings bindings;
    std::shared_ptr<uint8_t> in_buf;
    std::vector<std::shared_ptr<uint8_t>> out_bufs;
};

// D개 슬롯을 순환 사용. D는 CLI 옵션(--inflight)으로 스윕할 것.
// prepare_worker()에서 아래를 D번 반복해 ctx.slots에 채운다(각 슬롯이 자기 입출력 버퍼를 가짐).

static void worker_loop(WorkerCtx *ctx, const std::vector<std::string> *images, int D) {
    CtxSwitches cs0 = read_thread_ctx_switches();

    std::mutex m;
    std::condition_variable cv;
    int outstanding = 0;                       // 현재 in-flight 개수
    std::vector<double> lat(images->size(), -1.0), svc(images->size(), -1.0);
    std::atomic<size_t> done{0}, failed{0};
    auto t_start = Clock::now();

    for (size_t idx = 0; idx < images->size(); ++idx) {
        Slot &slot = ctx->slots[idx % D];

        // (1) 전처리 — 이 시간 동안 앞선 D-1개 프레임이 NPU에서 돌고 있다.
        double prep_b = host_trace::now_us();
        cv::Mat img = cv::imread((*images)[idx]);
        if (img.empty()) { ++failed; continue; }
        cv::Mat resized, rgb;
        cv::resize(img, resized, cv::Size(ctx->spec.input_w, ctx->spec.input_h));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);   // 리사이즈 후 변환이 더 싸다
        std::memcpy(slot.in_buf.get(), rgb.data, ctx->input_frame_size);
        double prep_e = host_trace::now_us();
        ctx->trace->add_prep(idx, prep_b, prep_e);

        // (2) 8L의 enq_ts와 같은 지점 — 큐가 받아줄 때까지의 대기를 포함시켜야
        //     det_latency_ms가 8L과 같은 물리량(sojourn)이 된다.
        double t_enq = host_trace::now_us();
        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&]{ return outstanding < D; });
            ++outstanding;
        }
        // 슬롯이 D개뿐이므로 재사용 안전을 위해 HailoRT 큐 여유도 함께 확인.
        (void)ctx->configured->wait_for_async_ready(std::chrono::milliseconds(5000));

        double t_sub = host_trace::now_us();
        auto job = ctx->configured->run_async(slot.bindings,
            [&, idx, t_enq, t_sub](const AsyncInferCompletionInfo &info) {
                double t_end = host_trace::now_us();
                if (info.status == HAILO_SUCCESS) {
                    lat[idx] = (t_end - t_enq) / 1000.0;   // sojourn — 8L 호환
                    svc[idx] = (t_end - t_sub) / 1000.0;   // service — 순수 왕복
                    ++done;
                } else { ++failed; }
                ctx->trace->add_infer(idx, t_sub, t_end,
                                      (t_sub - t_enq) / 1000.0, info.status == HAILO_SUCCESS);
                { std::lock_guard<std::mutex> lk(m); --outstanding; }
                cv.notify_one();
            });
        if (!job) { std::lock_guard<std::mutex> lk(m); --outstanding; cv.notify_one(); ++failed; continue; }
        job->detach();
    }

    // (3) 남은 in-flight 배수
    { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [&]{ return outstanding == 0; }); }
    ...
}
```

주의점:

- **콜백은 HailoRT 내부 스레드에서 불린다.** 콜백 안에서 무거운 작업(후처리, 파일 I/O, 락 경합)을
  하면 그게 그대로 다음 프레임 지연이 된다. 위처럼 타임스탬프 기록과 카운터만 만질 것.
- 슬롯 D개를 넘어서는 제출이 없도록 `outstanding < D` 조건을 반드시 지킬 것. 어기면 아직 읽지
  않은 입력 버퍼를 덮어써서 **조용히 틀린 결과**가 나온다.
- `wait_for_async_ready` / `AsyncInferCompletionInfo`의 정확한 시그니처는 설치본
  `hailo/infer_model.hpp`에서 확인할 것 (HailoRT 5.3.0 기준).

### 5-2. 권장 스케줄러 파라미터

파이프라이닝 적용 **후에** 아래 순서로 스윕한다. 적용 전에는 §2-2 이유로 결과가 무의미하다.

| 실험 | 설정 | 기대 |
|---|---|---|
| A. 베이스라인 | `--inflight 1` (현재 코드와 동일) | 39.9 FPS / 19.2 ms 재현 확인 |
| B. 깊이 스윕 | `--inflight 1,2,4,8,16`, 파라미터 무설정 | FPS 포화점(무릎) 탐색. 단독 모델 120~165 FPS 예상 |
| C. threshold | B의 최적 D 고정, `--threshold 1..D` | **threshold ≤ D 범위에서만.** D 초과 시 timeout 대기로 폭락 |
| D. timeout | `--timeout_ms 0,1,5,10` | threshold 미달 시 강제 실행. 0이 아닌 값은 tail latency를 만든다 |
| E. batch | `--batch 1,2,4,8` (batch ≤ D) | H10 NNC 배치 효율 확인 |
| F. priority | 3모델 동시, `--priority 16,24,16` 등 | 상대값만 의미 있음. 전부 같으면 RR |

시작점 권장: **`--inflight 4 --threshold 1 --timeout_ms 0 --priority 16`**
(threshold=1/timeout=0은 지연 최소화 기본값이고, 깊이만 확보해도 대부분의 이득이 나온다.)

### 5-3. 순수 NPU 시간과 E2E 분리 측정

한 프레임의 시간을 이렇게 4단으로 쪼갠다:

```
t_prep_begin ──prep── t_enq ──큐대기── t_submit ──RPC+DMA+NNC── t_complete ──post── t_e2e_end
             └ preprocess_ms ┘└ queue_wait_ms ┘└─ service_ms ─┘└ postprocess_ms ┘
             └───────────────────── e2e_ms ─────────────────────────────────────┘
             └────────── sojourn_ms (8L det_latency_ms와 동일 정의) ──────────┘
```

| 측정 대상 | 방법 | 비고 |
|---|---|---|
| **순수 NNC 연산시간** | `hailortcli run2 -t 5 --measure-latency set-net <hef>` → `hw_latency` | **단독 모델만.** yolov8s = 5.72 ms |
| 순수 NNC(간접, 다모델 가능) | `npu_percent × run_time_s ÷ 총 프레임수` | 이미 CSV에 있는 값으로 계산 가능. det 단독 = 5.83 ms로 위와 일치 |
| RPC+DMA 오버헤드 | `service_ms − hw_latency` | det 단독 = 13.4 ms |
| 큐 대기 | `t_submit − t_enq` (위 코드의 `enqueue_ms`) | 이미 host_trace가 기록 중 |
| E2E | `prep + sojourn + post` | 현재 `total_time_ms_*`가 이 정의 |
| 실효 처리량 | `frame_count / run_time_s` | in-flight 깊이와 함께 기록해야 해석 가능 |

추가로 **in-flight 점유율 분포**는 이미 `host_trace`의 `_events.csv`로 계산 가능하다
(스모크 검증에서 3개 54.6 % / 2개 44.4 %로 뽑아본 그 값). 깊이 스윕 시 이 분포를 같이 뽑으면
"D를 늘렸는데 실제로 안 찼다"는 상황을 바로 잡아낼 수 있다.

### 5-4. 공정 비교 체크리스트

- [ ] 이미지 장수 통일 (673 또는 600 중 하나로)
- [ ] `letterbox` vs `resize` 통일 (또는 차이를 문서에 명기)
- [ ] 출력 포맷 통일 (8L B/D조건 = AUTO)
- [ ] 후처리 on/off 조건 통일
- [ ] **latency 정의 통일** — 8L의 sojourn에 맞추거나, 8L도 in-flight 1로 재측정
- [ ] `inflight_depth`, `n_images`, `latency_definition`을 CSV에 기록

---

## 부록 — 우선순위별 실행 순서

1. **(반나절)** `--inflight` 옵션 + 슬롯 풀 구현 → D=1로 현재 결과 재현되는지 확인
2. **(1시간)** D=1,2,4,8,16 스윕 → FPS/latency 곡선. 여기서 대부분의 답이 나온다
3. **(1시간)** CSV 스키마를 `csv_schema_8l.hpp` 공용 정의로 교체, `label`을 뒤로 이동
4. **(반나절)** 최적 D에서 threshold/timeout/batch/priority 스윕 재실행
5. 8L도 동일 조건(가능하면 sojourn/service 양쪽)으로 재측정해 표를 다시 그린다
