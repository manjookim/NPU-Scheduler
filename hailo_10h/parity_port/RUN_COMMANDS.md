# Hailo-10H 실행 명령어 (npu-rpi5)

**접속: `ssh npu-rpi5@155.230.16.157 -p 40020` / `scp -P 40020`**

전제: 이 폴더(`parity_port/`)를 npu-rpi5의 `~/hailo10h_sched_exp1/` 아래에 둔다.
HEF는 `~/hailo10h_sched_exp1/resources/`, 데이터셋은 `~/datasets/sampled_val2017`에 있다고 가정한다.
(소스의 `DET_HEF`/`SEG_HEF`/`POSE_HEF`/`IMG_DIR`이 `/home/npu-rpi5/...` 절대경로로 박혀 있다.)

---

## 0. 파일 전송

접속 정보: **`npu-rpi5@155.230.16.157 -p 40020`** (Hailo-10H 보드)
참고 — Hailo-8L 보드는 `npu-rpi1@155.230.16.157 -p 40021` 로 포트가 다르다. 헷갈리지 말 것.

### 방법 A — Windows에서 직접 복사 (기존 실험 방식과 동일)

```powershell
# Windows PowerShell. scp는 포트 옵션이 대문자 -P 다.
scp -P 40020 -r C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hailo_10h\parity_port `
      npu-rpi5@155.230.16.157:~/hailo10h_sched_exp1/
```

### 방법 B — git (RPi에 저장소가 클론되어 있을 때)

```bash
ssh npu-rpi5@155.230.16.157 -p 40020
cd ~/NPU-Scheduler && git pull
cp -r hailo_10h/parity_port ~/hailo10h_sched_exp1/
```

---

## 1. 사전 확인

```bash
ssh npu-rpi5@155.230.16.157 -p 40020

whoami                                  # npu-rpi5 여야 소스의 절대경로가 맞는다
hailortcli scan                         # 디바이스 인식 확인
hailortcli fw-control identify          # HAILO10H / FW 버전 확인
ls -la ~/hailo10h_sched_exp1/resources/*.hef
ls ~/datasets/sampled_val2017 | wc -l   # 673 이 나와야 정상
```

계정명이 `npu-rpi5`가 아니면 경로만 자기 홈으로 바꾼 뒤 빌드한다:

```bash
cd ~/hailo10h_sched_exp1/parity_port
sed -i "s|/home/npu-rpi5|$HOME|g" infer_scheduler_h10.cpp
```

---

## 2. 빌드

```bash
cd ~/hailo10h_sched_exp1/parity_port
chmod +x build.sh run_workload_sweep_h10.sh run_inflight_sweep.sh
./build.sh
```

수동으로 빌드하려면:

```bash
g++ -O2 -std=c++17 infer_scheduler_h10.cpp -o infer_scheduler_h10 -I. \
    $(pkg-config --cflags --libs opencv4) -lhailort -lpthread
```

---

## 3. 스모크 테스트 (CSV 저장 안 함)

CSV 경로를 안 주면 기록하지 않는다. 먼저 소량으로 동작만 확인한다.

```bash
# det 단독 60장 (기본 백엔드 = infer, 슬롯 12개)
./infer_scheduler_h10 1 --models det --num_images 60

# 3모델 동시 60장
./infer_scheduler_h10 1 --models det,seg,pose --num_images 60

# 깊이를 1로 (기존 문제 있던 H10 코드와 같은 조건 — 대조용)
./infer_scheduler_h10 1 --models det --num_images 60 --inflight 1
```

> **H10은 `--api infer` 하나뿐이다.** `--api vstreams`는 `create_configure_params`가
> `HAILO_NOT_IMPLEMENTED(7)`로 실패한다 — HailoRT가 "use InferModel instead"라고 직접 안내한다.
> vstreams 코드는 8L 대조용으로만 남겨두었다.

**확인할 것**

- `backend=infer (InferModel Async), inflight=12` 로 시작하는가
- `[스케줄러] ... setter 미호출 — HailoRT 기본값 사용` 이 모델마다 찍히는가
- `[조건] 후처리 없음(출력 포맷 AUTO, FLOAT32 변환 없음)` 줄이 있는가
- `ULTRA_PERFORMANCE configure 실패` 경고가 떴는가 (떴다면 8L과 전력모드 조건이 달라진 것)
- 맨 아래 `[in-flight 추정]` 값 — 8L 실측이 11.0~11.2였다. 0.8 근처면 파이프라인이 안 찬 것이다
- `NPU: xx%` 가 NaN이 아닌가 (`hailortcli monitor`가 자식 프로세스로 뜬다 — 별도 터미널 불필요)

`--api vstreams`를 주면 다음처럼 실패한다(정상 동작이다 — H10이 지원하지 않는 경로다):

```
[HailoRT] [error] Not supported. Did you try calling `create_configure_params` on H10?
                  If so, use InferModel instead
Detection configure params 실패, status=7
```

---

## 4. 본 실험 — 8L default_workload와 같은 7조건 × 3회

```bash
cd ~/hailo10h_sched_exp1/parity_port
mkdir -p csv logs

# 기본: infer 백엔드, 슬롯 12개 — 8L의 in-flight ≈ 11 에 맞춘 조건
./run_workload_sweep_h10.sh
#  -> csv/results_h10_default_workload_infer.csv

# 깊이를 바꿔 한 세트 더 (파일명이 겹치므로 CSV를 따로 지정)
INFLIGHT=1 CSV=csv/results_h10_workload_inflight1.csv ./run_workload_sweep_h10.sh
```

기본값: 이미지 600장(8L 실험과 동일), 조건 7개(det / seg / pose / det,seg / det,pose / seg,pose / det,seg,pose), 각 3회 = 21회.

환경변수로 바꿀 수 있다:

```bash
NUM_IMAGES=673 REPEAT=5 CSV=csv/my_run.csv IMG_DIR=$HOME/datasets/sampled_val2017 \
  ./run_workload_sweep_h10.sh
```

---

## 5. in-flight 깊이 스윕 (10배 차이 원인 확인용)

```bash
MODELS=det           ./run_inflight_sweep.sh     # D = 1,2,4,8,12,16 × 3회
MODELS=det,seg,pose  ./run_inflight_sweep.sh
DEPTHS="1 4 16 32"   MODELS=det ./run_inflight_sweep.sh
```

각 로그의 `[in-flight 추정]` 줄이 실제로 D만큼 찼는지 확인한다.
D를 CSV에서도 구분하려면 `CSV_EXTENSION_COLUMNS`를 1로 바꿔 다시 빌드한다:

```bash
sed -i 's/#define CSV_EXTENSION_COLUMNS  0/#define CSV_EXTENSION_COLUMNS  1/' infer_scheduler_h10.cpp
./build.sh
```

---

## 6. 단일 실행 (직접 CSV 지정)

```bash
# 위치 인자: <run_id> <csv_path>  (8L과 같은 규약)
./infer_scheduler_h10 1 csv/results.csv --models det,seg,pose
./infer_scheduler_h10 2 csv/results.csv --models det,seg,pose --api infer --inflight 12
```

사용 가능한 옵션:

| 옵션 | 의미 |
|---|---|
| `--api infer\|vstreams` | 백엔드 선택 (기본 infer. H10에서 vstreams는 미지원) |
| `--inflight N` | infer 백엔드의 슬롯 수 = in-flight 깊이 |
| `--models det,seg,pose` | 실행할 모델 조합 |
| `--batch N` 또는 `N,N,N` | batch size (실제로 적용되는 유일한 파라미터) |
| `--num_images N` | 사용할 이미지 수 (0 = 전체) |
| `--img_dir PATH` | 데이터셋 경로 |
| `--csv PATH` | CSV 경로 (위치 인자 대신 써도 됨) |
| `--run_id N` | run_id |
| `--no_npu_monitor` | `hailortcli monitor` 미실행 → `npu_percent` = NaN |

`--threshold` / `--timeout_ms` / `--priority` 는 **없다**. 스케줄러 setter가 주석 처리된 상태라
값을 줘도 적용되지 않기 때문이며, 넣으면 에러로 종료한다.
쓰려면 `model_setup.hpp`의 `apply_scheduler_params()` 주석을 먼저 푼다.

---

## 7. 결과 확인 / 회수

```bash
# RPi에서 헤더 + 몇 줄 확인
head -1 csv/results_h10_default_workload_infer.csv | tr ',' '\n' | nl   # 49컬럼인지
column -s, -t < csv/results_h10_default_workload_infer.csv | less -S
```

```powershell
# Windows에서 회수 (scp는 대문자 -P)
scp -P 40020 -r npu-rpi5@155.230.16.157:~/hailo10h_sched_exp1/parity_port/csv `
      C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hailo_10h\parity_port\

scp -P 40020 -r npu-rpi5@155.230.16.157:~/hailo10h_sched_exp1/parity_port/logs `
      C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hailo_10h\parity_port\
```

8L 결과와 합칠 때:

```python
import pandas as pd
df8  = pd.read_csv("hailo_8L/experiments/2026-08-27_abcd_pp_format_sweep_gen3/csv/results_B.csv")
df10 = pd.read_csv("hailo_10h/parity_port/csv/results_h10_default_workload_infer.csv")
assert list(df8.columns) == list(df10.columns)[:49]     # 스키마 동일 확인
df = pd.concat([df8.assign(board="8L"), df10.assign(board="10H")], ignore_index=True)
```

---

## 8. 문제가 생기면

| 증상 | 조치 |
|---|---|
| `configure params 실패, status=7` | H10은 vstreams 미지원. `--api infer`(기본값)로 실행 |
| `Expected<T>::operator=` 컴파일 에러 | `Expected`는 이동 대입이 delete다. 재시도마다 새 `Expected`를 받을 것 |
| `wait_for_async_ready` 컴파일 에러 | `model_runner.hpp`에서 인자에 `, 1` 추가 |
| `이미지 없음` | `--img_dir` 로 실제 경로 지정, `ls` 로 .jpg 존재 확인 |
| `npu_percent` 가 NaN | `hailortcli monitor` 가 단독으로 도는지 먼저 확인 |
| CSV `헤더 불일치로 중단` | 다른 컬럼 구성의 기존 파일이다. 새 파일명을 쓰거나 옮길 것 |
| `[in-flight 추정]` 이 1 미만 | 파이프라인이 안 찼다. infer 백엔드면 `--inflight` 를 올릴 것 |
