# Priority 실험 재실행 런북 (2026-06-19)

수정 반영: latency 추론구간만 측정 / letterbox 전처리 / context switch 스레드 합산 / HRTT 경로 버그 수정 / 3회 반복 + 실험폴더 자동생성

- RPi: `rpi1@155.230.16.157 -p 40021`
- RPi 작업 디렉토리: `~/hailo_cpp_test/`
- PC 프로젝트: `C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\`

---

## 1. PC → RPi 파일 전송 (PowerShell / bash, 프로젝트 폴더에서)

> 경로 끝에 `\` 또는 `/` 붙이지 말 것 (bash EOF 오류)

```bash
scp -P 40021 infer_scheduler.cpp auto_experiment_all.sh hailo_utilization.py parse_npu_log.py scheduler_mon_pb2.py rpi1@155.230.16.157:~/hailo_cpp_test/
```

## 2. RPi 접속 후 빌드 1회 확인 (먼저 통과 여부만 체크)

```bash
ssh rpi1@155.230.16.157 -p 40021
cd ~/hailo_cpp_test
g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread && echo "BUILD OK"
```

- `BUILD OK` 안 뜨면 컴파일 에러 메시지 확인 (특히 `/proc/thread-self` 관련 아님 — 헤더/링크 문제일 가능성)

## 3. 전체 실험 실행 (189회, nohup 백그라운드)

```bash
cd ~/hailo_cpp_test
chmod +x auto_experiment_all.sh
nohup ./auto_experiment_all.sh > experiment_log.txt 2>&1 &
echo "PID: $!"
```

- 로그 첫 줄의 **`실험 폴더: experiments/2026-06-19_priority_expN`** 경로(N값)를 기록해 둘 것 → 다운로드 시 사용

## 4. 진행 확인

```bash
tail -f ~/hailo_cpp_test/experiment_log.txt        # 종료는 Ctrl+C (실험은 계속 돌아감)
# 첫 조건만 확인: HRTT가 실제로 생성되는지
ls -lt ~/hailo_cpp_test/experiments/2026-06-19_priority_exp1/traces/ | head
```

- 중단하려면 `ps aux | grep auto_experiment` → `kill <PID>`

## 5. 결과 다운로드 (PC, 프로젝트 폴더에서) — N은 실제 폴더 번호로 교체

```bash
# CSV
scp -P 40021 "rpi1@155.230.16.157:~/hailo_cpp_test/experiments/2026-06-19_priority_exp1/results_all.csv" "C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler"

# HRTT 전체
scp -P 40021 "rpi1@155.230.16.157:~/hailo_cpp_test/experiments/2026-06-19_priority_exp1/traces/*.hrtt" "C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hrtt"
```

## 6. Excel 4시트 정리 (PC, 프로젝트 폴더에서)

```bash
python3 make_excel.py results_all.csv results_summary.xlsx
```

- Sheet1 Single / Sheet2 Multi2 / Sheet3 Multi3 / Sheet4 조건별 평균(mean·std)

---

## 주의사항
- 실험 폴더는 매 실행마다 자동 증가(`exp1`, `exp2`…)되므로 기존 데이터 수동 삭제 불필요
- starvation 케이스(priority 차이 15 이상): 해당 모델 latency가 ~10000ms로 왜곡 → 분석 시 별도 처리
- HRTT는 앞 30초만 기록(`HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30`) → HTML "미완료"는 트레이스 창 한계이지 실패 아님
- nohup이므로 SSH 끊겨도 실험 계속 진행됨
