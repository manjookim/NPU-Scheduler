#!/bin/bash
# 학습 완료 후: 공식 테스트셋 평가 -> GTSRB 캘리브레이션으로 HEF 컴파일
set -u
source ~/hailo_venv/bin/activate
cd ~/mnv2_gtsrb || exit 1

echo "===== [1] GTSRB 공식 테스트셋 평가 (12,630장) ====="
python -u eval_gtsrb.py --model mnv2_gtsrb.keras --data ~/gtsrb 2>&1 | grep -vE "^\[info\]|WARNING" | tail -8

echo ""
echo "===== [2] HEF 컴파일 (GTSRB 실제 캘리브레이션 1024장) ====="
python -u compile_hailo.py \
  --model mnv2_gtsrb.tflite \
  --name mobilenet_v2_1_0_gtsrb \
  --out mobilenet_v2_1.0_gtsrb.hef \
  --calib-dir ~/gtsrb/GTSRB/Final_Training/Images \
  --calib-n 1024 2>&1 | tr '\r' '\n' | grep -aE "파싱|최적화|컴파일|캘리브|contexts|Successful|저장|Traceback|rror" | tail -20

echo ""
echo "===== 산출물 ====="
ls -la ~/mnv2_gtsrb/mobilenet_v2_1.0_gtsrb.hef 2>/dev/null || echo "  HEF 생성 실패"
