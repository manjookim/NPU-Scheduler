# Threshold 실험 런북 (priority = threshold)

**실험 설계**
- 각 모델 scheduler threshold = 해당 모델 priority (0이면 0 그대로) — cpp `THRESHOLD_EQ_PRIORITY=1`
- priority 0/15/31, batch=1, timeout=0 (default)
- 2개 모델(27) + 3개 모델(27) = 54조건 × 3회 = **162회** (단일 모델 제외)
- 검증 이미지 모델당 100장, 메트릭 전체 수집(latency / cpu / mem / voluntary·nonvoluntary ctx / npu)

---

## 0. 데이터셋 준비 (RPi)
조교님 첨부 zip을 RPi에서 압축 해제 후, cpp의 `IMG_DIR` 경로와 일치시킬 것.
```bash
# 예: 압축 푼 경로가 ~/datasets/coco/val2017/ 이면 수정 불필요 (현재 IMG_DIR 기본값)
grep IMG_DIR ~/hailo_cpp_test/infer_scheduler.cpp
# 경로가 다르면 infer_scheduler.cpp의 #define IMG_DIR 수정
```

## 1. 파일 전송 (PC, 프로젝트 폴더에서)
```bash
scp -P 40021 infer_scheduler.cpp auto_experiment_threshold.sh rpi1@155.230.16.157:~/hailo_cpp_test/
```

## 2. 빌드 확인 + 실행 (RPi)
```bash
ssh rpi1@155.230.16.157 -p 40021
cd ~/hailo_cpp_test
g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread && echo "BUILD OK"

chmod +x auto_experiment_threshold.sh
nohup ./auto_experiment_threshold.sh > threshold_log.txt 2>&1 &
echo "PID: $!"
```

## 3. 진행 확인
```bash
head -20 threshold_log.txt      # "threshold = priority", "사용 이미지 수: 100장" 확인
tail -f threshold_log.txt
```
- 실험 폴더: `experiments/{날짜}_threshold_exp1/`

## 4. 결과 다운로드 (PC) — 폴더명/날짜 실제값으로 교체
```bash
scp -P 40021 "rpi1@155.230.16.157:~/hailo_cpp_test/experiments/2026-06-30_threshold_exp1/results_all.csv" "C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\results_threshold.csv"

mkdir hrtt\threshold
scp -P 40021 "rpi1@155.230.16.157:~/hailo_cpp_test/experiments/2026-06-30_threshold_exp1/traces/*.hrtt" "C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hrtt\threshold"
```

## 5. HRTT→HTML 변환 (WSL2)
```bash
cd /mnt/c/Users/sset0/jungmin-claude/StudentExperiment/NPUscheduler
source ~/hailo_venv/bin/activate
mkdir -p html/threshold
for f in hrtt/threshold/*.hrtt; do
    name=$(basename "$f" .hrtt)
    hailo runtime-profiler "$f"
    mv runtime_report.html "html/threshold/${name}.html"
done
```

## 6. Excel 정리 (PC)
```bash
python3 make_excel.py results_threshold.csv results_threshold.xlsx
```
- Sheet1(단일=빈값), Sheet2(2개), Sheet3(3개), Sheet4(조건별 평균)
- CSV의 `threshold` 컬럼은 `=priority`로 기록됨(실제 threshold는 각 priority 컬럼 값과 동일)

## 7. 업로드 (deadline 6/30)
- GitHub: 프로젝트 폴더에서 `git add . && git commit && git push`
- Notion / Drive: results_threshold.xlsx + html/threshold 업로드
