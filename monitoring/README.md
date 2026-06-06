# 실험 진행 가이드

## 사전 작업 (매 실험 전)
`infer_scheduler.cpp` 상단 파라미터 수정:
```cpp
#define BATCH_SIZE     4  // ← 변경
#define THRESHOLD      0
#define TIMEOUT_MS     0
#define PRIORITY       0
```

---

## 모드 1 (노트북)

### ① SCP 전송 및 컴파일 (노트북 CMD)
```powershell
scp -P 40021 C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\infer_scheduler.cpp rpi1@155.230.16.157:/home/rpi1/hailo_cpp_test/
ssh rpi1@155.230.16.157 -p 40021 "cd ~/hailo_cpp_test && g++ infer_scheduler.cpp -o infer_scheduler -lhailort \$(pkg-config --cflags --libs opencv4)"
```

### ② NPU 모니터링 (Raspberry Pi 터미널 1)
```bash
cd ~/hailo_cpp_test
source ~/hailo_platform_venv/bin/activate
python3 hailo_utilization.py
```

### ③ 추론 실행 (Raspberry Pi 터미널 2)
```bash
cd ~/hailo_cpp_test
bash run_experiment.sh
```

### ④ 결과 파일 다운로드 (노트북 CMD)
```powershell
scp -P 40021 rpi1@155.230.16.157:/home/rpi1/hailo_cpp_test/results.csv C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\
```

### ⑤ HRTT 파일 다운로드 (노트북 CMD)
```powershell
scp -P 40021 "rpi1@155.230.16.157:/home/rpi1/hailo_cpp_test/traces/*.hrtt" C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hrtt_reports\hrtt\
```

### ⑥ HRTT 분석 (노트북 WSL2)
```bash
source ~/hailo_venv/bin/activate
hailo runtime-profiler "/mnt/c/Users/sset0/jungmin-claude/StudentExperiment/NPUscheduler/hrtt_reports/hrtt/(파일명).hrtt"
cp /mnt/c/Users/sset0/runtime_report.html "/mnt/c/Users/sset0/jungmin-claude/StudentExperiment/NPUscheduler/hrtt_reports/html/runtime_report_batchX.html"
```

---

## 모드 2 (데스크탑)

### ① SCP 전송 및 컴파일 (데스크탑 CMD)
```powershell
scp -P 40021 C:\Users\admin\jungmin-claude\StudentExperiment\NPUscheduler\infer_scheduler.cpp rpi1@155.230.16.157:/home/rpi1/hailo_cpp_test/
ssh rpi1@155.230.16.157 -p 40021 "cd ~/hailo_cpp_test && g++ infer_scheduler.cpp -o infer_scheduler -lhailort \$(pkg-config --cflags --libs opencv4)"
```

### ② NPU 모니터링 (Raspberry Pi 터미널 1)
```bash
cd ~/hailo_cpp_test
source ~/hailo_platform_venv/bin/activate
python3 hailo_utilization.py
```

### ③ 추론 실행 (Raspberry Pi 터미널 2)
```bash
cd ~/hailo_cpp_test
bash run_experiment.sh
```

### ④ 결과 파일 다운로드 (데스크탑 CMD)
```powershell
scp -P 40021 rpi1@155.230.16.157:/home/rpi1/hailo_cpp_test/results.csv C:\Users\admin\jungmin-claude\StudentExperiment\NPUscheduler\
```

### ⑤ HRTT 파일 다운로드 (데스크탑 CMD)
```powershell
scp -P 40021 "rpi1@155.230.16.157:/home/rpi1/hailo_cpp_test/traces/*.hrtt" C:\Users\admin\jungmin-claude\StudentExperiment\NPUscheduler\hrtt_reports\hrtt\
```

### ⑥ HRTT 분석 (데스크탑 WSL2)
```bash
source ~/hailo_venv/bin/activate
hailo runtime-profiler "/mnt/c/Users/admin/jungmin-claude/StudentExperiment/NPUscheduler/hrtt_reports/hrtt/(파일명).hrtt"
cp /home/gosungosun/runtime_report.html "/mnt/c/Users/admin/jungmin-claude/StudentExperiment/NPUscheduler/hrtt_reports/html/runtime_report_batchX.html"
```

---

## 주의사항
- 터미널 1 (NPU 모니터링) 먼저 실행 후 터미널 2 실행
- HRTT 파일명은 매번 타임스탬프가 달라지므로 확인 필요
- `hrtt_reports/html/` 파일명에 파라미터 값 명시 (예: `runtime_report_batch4.html`)


