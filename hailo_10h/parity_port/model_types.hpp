#pragma once
// model_types.hpp — 모델 실행에 쓰이는 공용 데이터 구조체 모음 (Hailo-10H 이식본)
//
// [2026-08-31 변경] 후처리를 코드에서 완전히 제거했다.
//   - OutRole / OutMeta(출력 텐서 role 분류용 타입)를 삭제했다. 출력 버퍼를 해석하지 않으므로
//     쓸 곳이 없다. output_classify.hpp도 함께 제거했다.
//   - 출력 포맷 FLOAT32 강제(FORCE_OUTPUT_FLOAT32)도 삭제했다. 출력은 항상
//     HAILO_FORMAT_TYPE_AUTO로 받는다 → HailoRT 내부 역양자화/재배열 비용이 latency에
//     섞이지 않는다(8L의 B/D조건과 동일).
//   - LetterboxMeta는 남긴다. letterbox()의 시그니처가 쓰고, 8L 전처리와 동일한 코드를
//     유지하기 위해서다(추론 경로에서는 nullptr로 넘긴다).

#include <cstdint>

// 모델 종류 — CSV의 det/seg/pose 컬럼 순서를 고정하고 로그 라벨을 붙이는 데만 쓴다.
// (후처리 제거로 "어떤 디코딩 경로를 탈지" 의미는 사라졌다.)
enum class ModelKind { DET, SEG, POSE };

// Detection 후처리에서 좌표를 원본 좌표계로 되돌릴 때 쓰던 전처리 메타.
// 후처리를 제거했으므로 현재 추론 경로에서는 채우지 않는다(letterbox 시그니처 호환용).
struct LetterboxMeta {
    float scale = 1.0f;      // 원본 -> 리사이즈 배율 (min(target/h, target/w))
    int pad_top = 0, pad_left = 0;
    int orig_w = 0, orig_h = 0;
};

// 모델별 실행 구성 (main에서 채움). save_csv에서도 참조하므로 전역에 둔다.
struct ModelConfig {
    const char* hef_path;
    const char* name;
    int priority;         // [주의] 스케줄러 setter를 주석 처리한 상태라 "실제 적용값"(HailoRT 기본값)이 들어간다
    int threshold;        //        CSV에 기록되는 값이므로 실제와 다르게 두면 안 된다
    uint32_t timeout_ms;
    int batch;            // batch만 실제로 적용된다(configure 시점 인자)
    bool active;
    ModelKind kind;
    int img_size = 640;
};

// 모델 1개(writer/reader 스레드 쌍)의 측정 결과.
struct ModelResult {
    double avg_latency_ms = -1;     // enqueue -> 모든 출력 수신 완료 (= 8L det_latency_ms와 같은 정의: sojourn)
    int frame_count = 0;
    long vol_ctx = 0;
    long nonvol_ctx = 0;
    double total_time_s = -1;       // 이 모델이 모든 입력을 처리하는 데 걸린 전체 시간(초)
    double avg_preprocess_ms = -1;  // 프레임당 평균 전처리(imread+letterbox) 시간 — 모델별 독립 측정
    double avg_postprocess_ms = -1; // 후처리를 제거했으므로 항상 -1(-> CSV에 NaN). 컬럼은 유지된다.
    double avg_total_time_ms = -1;  // 장당 전체 시간 = 전처리 + latency (후처리항 = 0)

    // ── [H10 확장] 코어 49컬럼 밖(확장 컬럼)으로만 나가는 값 ──
    double max_latency_ms = -1;     // 프레임별 latency 최댓값 (8L에서는 HRTT가 주던 값)

    // [주의] avg_service_ms는 "순수 연산시간"이 아니다. 제출~완료 구간이며 **디바이스 내부
    //   큐 대기를 포함**한다. in-flight가 D면 대략 D x (1/FPS)로 커진다
    //   (실측: D=12, 166 FPS -> service 69.25 ms). 순수 NNC 연산시간은 호스트에서 잴 수 없고
    //   `hailortcli run2 --measure-latency`의 hw_latency(yolov8s 5.72 ms)로만 얻는다.
    double avg_service_ms = -1;     // 제출 -> 완료 (디바이스 큐 대기 포함). infer 백엔드 전용

    // enqueue -> 제출 = 호스트 writer가 "빈 슬롯"을 기다린 시간 = 호스트에 남은 여유.
    // 이 값이 0으로 수렴하면 디바이스가 아니라 호스트(전처리)가 병목이라는 신호다.
    double avg_queue_wait_ms = -1;
    double fps = -1;                // frame_count / total_time_s
};
