#pragma once
// model_types.hpp — 모델 실행에 쓰이는 공용 데이터 구조체 모음 (Hailo-8)
// ModelKind/ModelConfig/ModelResult/OutRole/OutMeta. 로직 없이 순수 데이터 타입만 둔다.

#include <cstdint>

// 모델 종류 — 어떤 후처리 경로를 태울지 결정 (Detection=on-chip NMS, Seg/Pose=raw tensor CPU 디코딩)
enum class ModelKind { DET, SEG, POSE };

// 모델별 실행 구성 (main에서 채움). save_csv에서도 참조하므로 전역에 둔다.
struct ModelConfig {
    const char* hef_path;
    const char* name;
    int priority;
    int threshold;
    uint32_t timeout_ms;  // [진단용] HailoRT 내부적으로 uint32_t ms로 다뤄짐(HAILO_INFINITE_TIMEOUT=UINT32_MAX 근거) —
                           // int로 두면 4294967295 같은 큰 값을 이진탐색으로 테스트할 때 넘쳐서 음수가 됨.
    int batch;
    bool active;
    ModelKind kind;
};

// 모델 1개(writer/reader 스레드 쌍)의 측정 결과.
struct ModelResult {
    double avg_latency_ms = -1;
    int frame_count = 0;
    long vol_ctx = 0;
    long nonvol_ctx = 0;
    double total_time_s = -1;       // 이 모델이 모든 입력(약 670장)을 처리하는 데 걸린 전체 시간(초)
    double avg_preprocess_ms = -1;  // [2026-07-28] 프레임당 평균 전처리(imread+letterbox) 시간 — 모델별 독립 측정(과거엔 공유)
    double avg_postprocess_ms = -1; // 프레임당 평균 후처리(디코딩+NMS) 시간
    double avg_total_time_ms = -1;  // 장당 전체 시간 = 전처리 + latency + 후처리 (모두 이 모델 자신의 측정값)
};

// ========================= 후처리용 output vstream 분류 =========================
// Pose/Seg의 raw output vstream들은 채널 수(c)만 보면 role을 알 수 있다
// (box=64, score=1, kpts=51, cls=80, coeff=32, proto=32이지만 h/w=160x160으로 구분).
// vstream 생성 순서(HEF 선언 순서)에 의존하지 않도록 매 vstream의 get_info().shape로
// 동적으로 분류한다. [전제] 해당 output vstream은 FLOAT32 + NHWC로 생성되어 있어야 함
// (postprocess_hailo8.hpp의 TensorView가 그 전제로 버퍼를 인덱싱함).
enum class OutRole { OTHER, POSE_BOX, POSE_SCORE, POSE_KPTS, SEG_BOX, SEG_CLS, SEG_COEFF, SEG_PROTO };

struct OutMeta {
    OutRole role = OutRole::OTHER;
    int h = 0, w = 0, c = 0;
    int stride = 0;  // img_size / h (정사각 640 입력 가정)
    // Detection(NMS-by-class 출력) 전용 — get_info().nms_shape에서 얻음. Pose/Seg는 0으로 둠.
    int nms_number_of_classes = 0;
    int nms_max_bboxes_per_class = 0;
};
