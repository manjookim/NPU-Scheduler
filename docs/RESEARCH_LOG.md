# 연구일지

## 2026-07-10 — 스케줄러 파라미터 적용 검증

### 목적
* Hailo-8L에서 Detection/Segmentation/Pose 3개 모델 동시 추론 시, 모델별로 다르게 설정한 priority/threshold/timeout/batch가 실제로 하드웨어 스케줄러에 적용되는지 검증

### 1. 코드 구현 (infer_scheduler.cpp)
* 입력: dummy buffer → COCO val2017 실제 이미지(600장, letterbox 640 + BGR2RGB 전처리)로 전환
* 모델별 개별 파라미터 구조체(ModelConfig: priority/threshold/timeout_ms/batch) 도입
* 모델별 writer/reader 스레드 비동기 추론(run_model_async)
* CSV 저장 로직 제거 — 콘솔 로그 + HRTT 트레이스만으로 검증 범위 축소

### 2. 하드웨어 제약 발견: threshold ≤ batch_size
* threshold > batch로 설정 시 set_scheduler_threshold가 HAILO_INVALID_ARGUMENT 반환
* 에러 메시지: "Threshold must be equal or lower than the maximum batch size!"
* 공식 문서에 명시되지 않은 제약 — 런타임 에러로 최초 발견
* 조치: threshold ≤ batch로 재조정
  * DET: batch=4, threshold=4
  * SEG: batch=8, threshold=6
  * POSE: batch=2, threshold=2
* 코드에 방어 로직 추가: threshold > batch면 실행 전 경고 로그 출력

### 3. 검증 방법론
* 1차: hailo runtime-profiler로 생성한 HTML 리포트의 "Network Parameters" 패널 확인
  * priority/batch는 정확히 표시되나 threshold/timeout은 항상 0으로 표시됨
  * Hailo 프로파일러 자체의 UI 렌더링 버그로 판단 (실제 미적용이 아님, PROJECT_HANDOFF.md §6 기록)
* 2차: HRTT(.hrtt) 내부 core_op_set_value 트레이스 직접 파싱 (verify_params.py 작성)
  * core_op_set_value = HailoRT가 스케줄러에 실제로 적용한 값을 기록하는 내부 트레이스 — setter 반환값보다 신뢰도 높음
  * 파일명이 아닌 트레이스 데이터만 근거로 판정하도록 로직 단순화 (파일명 파싱 로직 제거)

### 4. 교차검증 실험 (verify_params.py 신뢰성 확인)
* SEG threshold를 의도적으로 batch보다 크게(99) 설정 → set_scheduler_threshold 강제 실패 유도
* 결과: verify_params.py 출력에서 seg만 threshold=(미적용!)으로 표시, det/pose는 설정값 그대로 정확히 표시
* 결론: core_op_set_value 트레이스가 입력값의 단순 echo가 아니라 실제 적용 성공 여부를 정확히 반영함을 확인

### 5. 미해결 질문 (QUESTION_FOR_TA.md 작성 → GitHub 업로드)
* threshold ≤ batch_size 제약이 공식 문서 어디에 명시되어 있는지
* setter가 HAILO_SUCCESS를 반환하면서 내부적으로 다른 값(예: 기본값)을 조용히 적용하는 경우가 있는지 — 명시적 거부 케이스만 검증됨, 이 부분은 미검증
* HailoRT v4.23.0에서 스케줄러 파라미터 관련 알려진 이슈가 있는지
* core_op_set_value 트레이스 기반 검증(verify_params.py)이 올바른 접근인지

### 6. 저장소 정리
* 목적/용도별 재구성: scripts/, tools/{hrtt, monitoring, results}/, results/{실험명}/, docs/, experiments/2026-07-10_scheduler_param_verification/
* .gitignore 버그 발견 및 수정: hrtt/, html/ 패턴에 선행 `/`가 없어 tools/hrtt/ 하위까지 무시되어 verify_params.py 등 소스코드가 git에 전혀 추적되지 않고 있었음 → `/hrtt/`, `/html/`로 루트 고정하여 수정
* 오늘 작업물(infer_scheduler.cpp, verify_params.py, QUESTION_FOR_TA.md)은 experiments/2026-07-10_scheduler_param_verification/에, 기존 실험 결과는 results/{실험별}/에 분리 보관
* 로컬 커밋 완료 (429f887) — GitHub push는 직접 진행

### 다음 단계
* `git push origin HEAD:master` 실행 확인 (로컬 브랜치명 main ↔ 원격 master 불일치로 인한 push 이슈 대응 중)
* 조교님 답변 수신 후 미해결 질문 반영
