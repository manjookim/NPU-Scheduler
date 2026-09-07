# in-flight 조건 통일 — 근거와 실행 절차

작성 2026-09-07 · 대상: `hailo_10h/parity_port/`

> **한 줄 요약** — in-flight를 "전 모델 공통 D=12"에서 **"조합별·모델별로 8L 실측 L에 맞춘 D"**로
> 바꿨다. 어떤 값을 쓸지는 운영자가 명령줄에서 고르는 게 아니라 벤치가 기준표를 보고 정하고,
> 런이 끝나면 실제로 맞았는지 스스로 검증해 로그에 OK/NG로 찍는다.

---

## 1. 왜 바꿨나

in-flight는 "맞춰져 있던 조건을 우리가 건드린 것"이 아니라 **원래부터 어긋나 있던 통제되지 않은
변수**다. Little의 법칙(L = λ·W)으로 역산하면:

| | 8L | H10 (수정 전 코드) |
|---|---|---|
| det 단독 | **11.25** | **0.77** |
| 3모델 det | **11.03** | **0.82** |

구 H10 코드는 `run_async()` 직후 `wait()`해서 in-flight가 1이었다. 즉 같은 `latency_ms`라는
이름 아래 **8L은 체류시간(sojourn)을, H10은 서비스시간(service)을** 재고 있었다.

슬롯 풀 도입으로 깊이는 확보했지만, 그때 쓴 **전 모델 공통 D=12는 det에만 맞았다**:

| 조건 | 8L 실측 L | H10 (D=12) 실측 L | 어긋남 |
|---|---|---|---|
| det | 11.24 | 11.60 | 3% |
| **seg** | **10.37** | **12.17** | **17%** |
| **3모델** | **11.03** | **12.66** | **15%** |

**8L의 in-flight는 우리가 설정한 값이 아니라 vstream 파이프라인에서 모델마다 다르게 나타나는
결과값**이기 때문이다. 그래서 하나의 상수로는 맞출 수 없다.

---

## 2. 8L 기준표 — 어떻게 뽑았나

**출처**: `hailo_8L/experiments/2026-08-27_abcd_pp_format_sweep_gen3/csv/results_B.csv`
(B조건 = 후처리 OFF + 출력 AUTO. H10의 현재 실행 조건과 같은 조건이다.)

**산출식**: `L = FPS × latency`, 이때 `FPS = 600장 ÷ total_time_<model>_s`
(`avg_fps_*`는 HRTT 미수집으로 NaN이라 `total_time`에서 역산한다.)

### 조합별·모델별 8L 실측 L (B조건)

| 조합 | det | seg | pose |
|---|---|---|---|
| det 단독 | **11.25** | | |
| seg 단독 | | **10.38** | |
| pose 단독 | | | **10.21** |
| det+seg | **11.10** | **9.30** | |
| det+pose | **11.13** | | **9.23** |
| seg+pose | | **9.28** | **9.17** |
| det+seg+pose | **11.03** | **9.17** | **9.09** |

### 재현성 확인 (A/B/C 세 조건 교차)

| 모델 | A | B | C | 판정 |
|---|---|---|---|---|
| det 단독 | 11.25 | 11.25 | 11.25 | 편차 0% — 구조값 |
| pose 단독 | 10.16 | 10.21 | 10.21 | 편차 0.5% |
| 3모델 det | 11.02 | 11.03 | 11.03 | 편차 0.1% |
| seg 단독 | 17.83 | 10.38 | 10.52 | **A만 이상치** — A는 후처리 ON 조건이라 seg 후처리가 latency에 섞였다. 그래서 기준은 B조건을 쓴다 |

det가 조건이 5배 바뀌어도 11.0~11.3으로 고정되는 것이 이 값들이 노이즈가 아니라 파이프라인
구조에서 나오는 값이라는 근거다.

### 재현 커맨드

```bash
cd hailo_8L/experiments/2026-08-27_abcd_pp_format_sweep_gen3
python3 - <<'PY'
import csv
N=600
for r in csv.DictReader(open('csv/results_B.csv',newline='',encoding='utf-8')):
    act=[m for m in ('det','seg','pose') if r[f'use_{m}']=='1']
    for m in act:
        fps=N/float(r[f'total_time_{m}_s']); lat=float(r[f'{m}_latency_ms'])
        print(f"{'_'.join(act):14}{m:6} FPS={fps:7.2f} lat={lat:8.2f}ms  L={fps*lat/1000:6.2f}")
PY
```

---

## 3. 코드 변경

| 파일 | 변경 |
|---|---|
| `infer_scheduler_h10.cpp` | `#define INFLIGHT 12` → `INFLIGHT_FALLBACK` + **`L8L[8][3]` 기준표**(mask = det\|seg<<1\|pose<<2) |
| | `RunOpts.inflight` 를 `int` → `int[3]` (+ `inflight_explicit`) |
| | `--inflight` 가 `auto`(기본) / 단일값 / `det,seg,pose` 3값을 받음 |
| | `main`에 **깊이 결정 블록** — 활성 조합의 목표 L을 찾아 `D = round(L)`로 자동 설정 |
| | 시작 로그에 모델별 `D`와 목표 `L` 출력 |
| | 종료 시 **in-flight 검증표**(목표 L / 실측 L / 어긋남 % / OK·NG, 허용 ±5%) |
| | `prepare_infer_ctx(..., opt.inflight[i], ...)` — 모델별 슬롯 수 전달 |
| `csv_writer.hpp` | 확장 컬럼 `inflight_depth`(스칼라 1개) → **`inflight_det,inflight_seg,inflight_pose`** |
| `run_workload_sweep_h10.sh` | `INFLIGHT` 기본값 `12` → `auto`, 파일명 태그 `if12` → `ifmatched8L` |
| `model_setup.hpp`, `build.sh` | 주석·사용예 갱신 |

`model_runner.hpp`는 손대지 않았다 — `SLOTS = ctx.slots.size()`로 이미 컨텍스트별 값을 쓴다.

### 자동 결정 결과 (D = round(목표 L))

| 조합 | det | seg | pose |
|---|---|---|---|
| det 단독 | D=11 | | |
| seg 단독 | | D=10 | |
| pose 단독 | | | D=10 |
| det+seg | D=11 | D=9 | |
| det+pose | D=11 | | D=9 |
| seg+pose | | D=9 | D=9 |
| det+seg+pose | D=11 | D=9 | D=9 |

---

## 4. 실행

```bash
ssh -o ServerAliveInterval=30 npu-rpi5@155.230.16.157 -p 40020
cd ~/hailo10h_sched_exp1/parity_port
# (수정된 4개 파일을 먼저 이 경로로 복사할 것)
./build.sh

# 7조합 x 3회, in-flight는 조합별 자동 매칭
nohup ./run_workload_sweep_h10.sh > logs/sweep_matched.log 2>&1 &
tail -f logs/sweep_matched.log
```

대조군이 필요하면 같은 스크립트에 환경변수만 바꾼다 (CSV 파일명이 자동으로 분리된다):

```bash
INFLIGHT=12 ./run_workload_sweep_h10.sh    # 예전 공통값 — 매칭 전후 비교용
INFLIGHT=1  ./run_workload_sweep_h10.sh    # 최초 코드와 동일(파이프라인 없음)
```

---

## 5. 판정

런마다 로그 끝에 이 표가 찍힌다.

```
---------- in-flight 검증 (8L 목표 대비) ----------
  model    D   target_L     meas_L       dev  verdict
  det     11      11.03      11.20     +1.5%  OK
  seg      9       9.17       9.25     +0.9%  OK
  pose     9       9.09       9.15     +0.7%  OK
  => 이 런은 8L과 같은 파이프라인 깊이에서 측정되었다 (허용 오차 +-5%)
```

전 런 일괄 확인:

```bash
grep -A6 'in-flight 검증' logs/infer_matched8L_*_run*.log | grep -E '  (det|seg|pose) |=>'
```

**NG가 뜨면**: 그 모델의 `D`를 어긋난 방향으로 1 조정해 재측정한다.
`dev`가 +면 `D`를 1 낮추고, -면 1 올린다.

```bash
./infer_scheduler_h10 1 csv/probe.csv --models det,seg,pose --num_images 600 --inflight 11,8,8
```

`enq_ts`를 **슬롯 확보 이전**에 찍기 때문에(8L의 `write()` 직전과 같은 지점) 슬롯 대기 중인
프레임도 L에 잡힌다. 그래서 실측 L은 `D-1 ~ D+1` 범위가 정상이고, `round(L)`은 출발점일 뿐
반드시 정답은 아니다.

---

## 6. 이걸로도 해결되지 않는 것

깊이를 맞춰도 **여전히 남는 조건 차이**다. 보고서에 그대로 적을 것.

1. **큐의 위치가 다르다.** 8L은 호스트 vstream 큐, H10은 디바이스 내부 큐다. H10에는
   VStreams 경로 자체가 없다 — `create_configure_params()`가 `HAILO_NOT_IMPLEMENTED(7)`로
   실패하고 HailoRT가 *"use InferModel instead"* 라고 안내한다. **깊이는 맞출 수 있어도
   대기 장소는 원리적으로 못 맞춘다.**

2. **도착률이 여전히 다르다.** 양쪽 다 `INPUT_FPS 0`(포화)이라 각자의 최대 속도로 돌고 있다.
   부하가 다른 두 큐잉 시스템의 latency 비교는 in-flight를 맞추든 말든 성립하지 않는다.
   실제로 `latency 비 3.133 = 처리량 비 3.234 ÷ in-flight 비 1.032`로 항등식이다.
   → **고정 도착률 스윕**(8L에 이미 있는 `2026-08-04_singlemodel_fps_sweep_exp1`의
   INPUT_FPS = 5/10/15/20/25)을 H10에서 동일하게 돌려야 한다. 다음 순서 1순위.

3. **D를 8L에 맞춘 것이 공정한가**라는 반론은 `run_inflight_sweep.sh`(D = 1,2,4,8,12,16)의
   곡선으로만 방어된다. 아직 미실행.
