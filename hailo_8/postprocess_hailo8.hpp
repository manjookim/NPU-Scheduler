/**
 * postprocess_hailo8.hpp
 * ------------------------------------------------------------------------
 * TAPPAS 프레임워크 없이, HailoRT raw output vstream 버퍼만으로 동작하는
 * YOLOv8-Pose / YOLOv8-Seg 후처리(디코딩+NMS) 구현.
 *
 * 참고(알고리즘 출처): hailo-ai/tappas
 *   core/hailo/libs/postprocesses/pose_estimation/yolov8pose_postprocess.cpp
 *   (박스 DFL 디코딩, 키포인트 디코딩, NMS 로직을 xtensor/HailoROI 의존성 없이
 *    std::vector 기반으로 이식함)
 * Segmentation은 Ultralytics YOLOv8-seg 표준 아키텍처(box+cls+mask_coeff 3-scale,
 * 공유 proto mask 160x160x32) 기반으로 동일한 DFL 디코딩 방식을 적용해 구현.
 *
 * [2026-07-26 수정] 조교님이 참고용으로 주신 파이썬 후처리 코드(hailo_platform 기반)와
 * 대조해 두 가지를 반영함:
 *   1) cls/score 채널은 HEF에 이미 sigmoid가 포함돼 있어 raw 값을 그대로 확률로 써야
 *      함(다시 sigmoid를 걸면 이중 sigmoid가 됨). 실기 테스트에서 NMS 후보가 그리드
 *      셀 수에 근접해 O(n^2)로 폭주했던 원인이 바로 이 이중 sigmoid 버그였음.
 *   2) Segmentation NMS는 클래스 무관이 아니라 클래스별로 따로 수행(파이썬 코드와 동일).
 * DFL 박스 디코딩 공식과 keypoint 좌표 공식은 원래 구현이 파이썬 참고 코드와 수학적으로
 * 동일함을 대조 확인해 그대로 유지함.
 *
 * [중요 — infer_scheduler_hailo8.cpp의 vstream 생성부와 반드시 짝이 맞아야 함]
 * Pose/Segmentation output vstream은
 *   - format type  = HAILO_FORMAT_TYPE_FLOAT32  (HailoRT가 qp_scale/qp_zp로
 *     역양자화까지 대신 해줌 → 이 헤더는 역양자화를 하지 않고 float를 그대로 읽음)
 *   - format order = HAILO_FORMAT_ORDER_NHWC     (parse-hef가 일부 출력을 FCR로
 *     표시했으므로 AUTO에 맡기지 않고 명시적으로 NHWC 고정)
 * 로 생성되어 있어야 아래 TensorView가 버퍼를 올바르게 인덱싱한다.
 * (Detection은 on-chip NMS(HailoRT-pp) 출력이라 이 헤더를 거치지 않음.)
 *
 * 이 파일은 "정확한 mAP 재현"이 목적이 아니라, 스케줄러 벤치마크에 필요한
 * "장당 전체 추론시간(전처리-추론-후처리)"을 재는 데 필요한 실제 디코딩+NMS
 * 연산량을 수행하는 것이 목적이다. 좌표 계산에 미세한 오차가 있어도 타이밍
 * 측정 목적에는 지장 없다.
 * ------------------------------------------------------------------------
 */
#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <opencv2/core.hpp>  // Segmentation mask 복원(coeff·proto 행렬곱)에 cv::Mat/gemm 사용

// ========================= 공통 유틸 =========================

struct PPBox {
    float x1, y1, x2, y2;  // 0~1 정규화 좌표
    float score;
    int class_id;
};

struct PoseDet {
    PPBox box;
    float kpts[17][3];  // x, y, score (정규화)
};

struct SegDet {
    PPBox box;
    std::vector<float> mask_coeffs;  // 32
    std::vector<float> mask;          // 160x160 (proto와 곱해서 만든 최종 마스크, threshold 전 raw 값)
};

namespace pp {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// raw output 버퍼를 감싸는 뷰. vstream을 FLOAT32+NHWC로 만들었으므로 이미
// 역양자화된 float가 (h, w, c) NHWC 순서로 나열되어 있다고 가정한다.
struct TensorView {
    const float* data = nullptr;
    int h = 0, w = 0, c = 0;

    float at(int y, int x, int ch) const {
        size_t idx = (size_t)(y * w + x) * c + ch;
        return data[idx];
    }
};

// DFL(distribution focal loss) softmax: reg_len+1(=16)개 bin -> 확률분포로 정규화
inline void softmax_inplace(float* v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; i++) mx = std::max(mx, v[i]);
    float sum = 0.f;
    for (int i = 0; i < n; i++) { v[i] = std::exp(v[i] - mx); sum += v[i]; }
    for (int i = 0; i < n; i++) v[i] /= (sum > 1e-9f ? sum : 1e-9f);
}

inline float iou(const PPBox& a, const PPBox& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    float uni = area_a + area_b - inter;
    return uni > 1e-9f ? inter / uni : 0.f;
}

// 클래스 무관 NMS (한 프레임 내 중복 억제)
// [중요] 실기 테스트에서 확인된 문제: score threshold를 넘는 raw candidate 수가
// (버그든 정상적인 raw-output 특성이든) 그리드 셀 수(8400=80x80+40x40+20x20)에 가깝게
// 나오면 이 O(n^2) 억제 루프가 수백 ms~초 단위로 폭주한다(예: n=8400 -> 약 3500만 IoU
// 비교). Ultralytics 등 표준 구현도 NMS 직전에 후보 수를 상한으로 자르므로
// (max_nms 관행) 여기서도 정렬 후 상위 max_candidates개만 남기고 잘라낸다.
template <typename T, typename GetBoxFn>
inline std::vector<T> nms(std::vector<T> dets, float iou_thr, GetBoxFn getBox, size_t max_candidates = 300) {
    std::sort(dets.begin(), dets.end(), [&](const T& a, const T& b) {
        return getBox(a).score > getBox(b).score;
    });
    if (dets.size() > max_candidates) dets.resize(max_candidates);
    std::vector<char> suppressed(dets.size(), 0);
    std::vector<T> keep;
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (iou(getBox(dets[i]), getBox(dets[j])) >= iou_thr) suppressed[j] = 1;
        }
    }
    return keep;
}

// DFL 박스 디코딩 공통 로직: box_raw(4 x reg_len+1) -> 중심점 기준 4방향 거리 -> xyxy
// cx, cy: 그리드 중심 (stride 단위 적용됨, 640 기준 픽셀 좌표)
inline PPBox decode_dfl_box(const float* box_raw /* [4][reg_len+1] flattened row-major */,
                             int reg_len_plus1, float cx, float cy, int stride, int img_size)
{
    float dist[4];
    for (int side = 0; side < 4; side++) {
        float bins[16];
        for (int b = 0; b < reg_len_plus1; b++) bins[b] = box_raw[side * reg_len_plus1 + b];
        softmax_inplace(bins, reg_len_plus1);
        float d = 0.f;
        for (int b = 0; b < reg_len_plus1; b++) d += b * bins[b];
        dist[side] = d * stride;
    }
    // side 순서: left, top, right, bottom (ultralytics 표준)
    float x1 = cx - dist[0];
    float y1 = cy - dist[1];
    float x2 = cx + dist[2];
    float y2 = cy + dist[3];

    PPBox box;
    box.x1 = x1 / img_size; box.y1 = y1 / img_size;
    box.x2 = x2 / img_size; box.y2 = y2 / img_size;
    box.score = 0.f; box.class_id = -1;
    return box;
}

} // namespace pp

// ========================= YOLOv8-Pose 후처리 =========================
// 입력: 3개 스케일(stride 8/16/32) 각각 box(64=4x16 DFL), score(1), kpts(51=17x3) 텐서.
// 채널 수 기준으로 role 자동 판별하므로 vstream 순서에 의존하지 않음.
struct PoseScaleTensors {
    pp::TensorView box;    // c=64
    pp::TensorView score;  // c=1
    pp::TensorView kpts;   // c=51
    int stride = 0;
};

inline std::vector<PoseDet> decode_pose(const std::vector<PoseScaleTensors>& scales,
                                         int img_size = 640,
                                         float score_thr = 0.3f,   // 조교님 참고 코드 conf_thres와 동일
                                         float iou_thr = 0.45f,    // 조교님 참고 코드 iou_thres와 동일
                                         float kpt_thr = 0.5f,
                                         size_t* raw_candidates = nullptr)  // 디버그용: NMS 전 후보 수(score>=thr 통과 개수)
{
    std::vector<PoseDet> dets;
    const int reg_len1 = 16;

    for (const auto& sc : scales) {
        if (!sc.box.data || !sc.score.data || !sc.kpts.data) continue;  // 스케일 그룹이 불완전하면 스킵
        for (int y = 0; y < sc.box.h; y++) {
            for (int x = 0; x < sc.box.w; x++) {
                // [중요] 조교님 참고 코드 주석: "하일로 출력은 이미 확률값이므로 sigmoid 생략".
                // HEF가 cls/score 헤드에 sigmoid를 이미 포함해 export되어 있어서, 여기서
                // 또 sigmoid를 걸면 이중 sigmoid가 되어 값이 전부 0.5~0.73 근처로 몰리고
                // score_thr을 사실상 모든 셀이 통과해버린다(실기 테스트에서 NMS 후보가
                // 그리드 셀 수에 근접해 O(n^2) NMS가 폭주한 원인이 바로 이것이었음).
                float score = sc.score.at(y, x, 0);
                if (score < score_thr) continue;

                float box_raw[64];
                for (int ch = 0; ch < 64; ch++) box_raw[ch] = sc.box.at(y, x, ch);

                float cx = (x + 0.5f) * sc.stride;
                float cy = (y + 0.5f) * sc.stride;
                PPBox box = pp::decode_dfl_box(box_raw, reg_len1, cx, cy, sc.stride, img_size);
                box.score = score;
                box.class_id = 0;  // person only

                PoseDet det;
                det.box = box;
                for (int k = 0; k < 17; k++) {
                    float kx = sc.kpts.at(y, x, k * 3 + 0);
                    float ky = sc.kpts.at(y, x, k * 3 + 1);
                    float ks = pp::sigmoid(sc.kpts.at(y, x, k * 3 + 2));
                    // ultralytics 방식: raw kpt 좌표는 stride 배율 + grid offset 보정
                    float px = (kx * 2.0f + (x)) * sc.stride;
                    float py = (ky * 2.0f + (y)) * sc.stride;
                    det.kpts[k][0] = px / img_size;
                    det.kpts[k][1] = py / img_size;
                    det.kpts[k][2] = ks;
                }
                (void)kpt_thr;
                dets.push_back(det);
            }
        }
    }

    if (raw_candidates) *raw_candidates = dets.size();
    return pp::nms(dets, iou_thr, [](const PoseDet& d) -> const PPBox& { return d.box; });
}

// ========================= YOLOv8-Seg 후처리 =========================
// 입력: 3개 스케일 각각 box(64), cls(80), mask_coeff(32) 텐서 + 공유 proto(160x160x32)
struct SegScaleTensors {
    pp::TensorView box;    // c=64
    pp::TensorView cls;    // c=80
    pp::TensorView coeff;  // c=32
    int stride = 0;
};

inline std::vector<SegDet> decode_seg(const std::vector<SegScaleTensors>& scales,
                                       const pp::TensorView& proto,  // 160x160x32
                                       int img_size = 640,
                                       float score_thr = 0.01f,   // 조교님 참고 코드 CONF_THRES와 동일
                                       float iou_thr = 0.65f,     // 조교님 참고 코드 IOU_THRES와 동일
                                       int num_classes = 80,
                                       bool compute_mask = true,
                                       size_t* raw_candidates = nullptr,  // 디버그용: NMS 전 후보 수
                                       size_t max_det = 300)      // 조교님 참고 코드 MAX_DET과 동일
{
    std::vector<SegDet> dets;
    const int reg_len1 = 16;

    for (const auto& sc : scales) {
        if (!sc.box.data || !sc.cls.data || !sc.coeff.data) continue;  // 스케일 그룹이 불완전하면 스킵
        for (int y = 0; y < sc.box.h; y++) {
            for (int x = 0; x < sc.box.w; x++) {
                // 최댓값 클래스 스코어 찾기
                // [중요] 조교님 참고 코드: cls 값을 그대로 score로 씀(sigmoid 없음) —
                // HEF가 cls 헤드에 sigmoid를 이미 포함해서 export했기 때문. Pose와 동일한
                // 이유로 여기서 다시 sigmoid를 걸면 이중 sigmoid가 되어 NMS가 폭주한다.
                float best_score = -1e9f; int best_cls = -1;
                for (int c = 0; c < num_classes; c++) {
                    float s = sc.cls.at(y, x, c);
                    if (s > best_score) { best_score = s; best_cls = c; }
                }
                float score = best_score;
                if (score < score_thr) continue;

                float box_raw[64];
                for (int ch = 0; ch < 64; ch++) box_raw[ch] = sc.box.at(y, x, ch);

                float cx = (x + 0.5f) * sc.stride;
                float cy = (y + 0.5f) * sc.stride;
                PPBox box = pp::decode_dfl_box(box_raw, reg_len1, cx, cy, sc.stride, img_size);
                box.score = score;
                box.class_id = best_cls;

                SegDet det;
                det.box = box;
                det.mask_coeffs.resize(32);
                for (int c = 0; c < 32; c++) det.mask_coeffs[c] = sc.coeff.at(y, x, c);
                dets.push_back(det);
            }
        }
    }

    if (raw_candidates) *raw_candidates = dets.size();

    // 조교님 참고 코드처럼 클래스별로 따로 NMS(다른 클래스끼리는 서로 억제하지 않음).
    // 부수 효과로 클래스당 후보 수가 전체보다 훨씬 작아져 O(n^2) 비용도 자연히 줄어든다.
    std::vector<std::vector<SegDet>> per_class(num_classes);
    for (auto& d : dets) {
        if (d.box.class_id >= 0 && d.box.class_id < num_classes)
            per_class[d.box.class_id].push_back(d);
    }
    std::vector<SegDet> kept;
    for (int c = 0; c < num_classes; c++) {
        if (per_class[c].empty()) continue;
        auto kc = pp::nms(per_class[c], iou_thr, [](const SegDet& d) -> const PPBox& { return d.box; });
        kept.insert(kept.end(), kc.begin(), kc.end());
    }
    if (kept.size() > max_det) kept.resize(max_det);

    // mask = sigmoid(coeff · proto), proto: 160x160x32. C++엔 numpy 같은 내장 벡터화가
    // 없어서, naive 3중 for문(검출당 160*160*32 스칼라 곱) 대신 이미 프로젝트에 물려 있는
    // OpenCV의 행렬곱(cv::Mat operator*, 내부적으로 SIMD/BLAS 백엔드 활용)으로 대체함.
    // 실기 테스트에서 naive 버전이 Segmentation 후처리를 프레임당 250ms대로 끌어올려서
    // latency 측정 자체를 오염시켰던 게 바로 이 부분이었음.
    if (compute_mask && proto.data && !kept.empty()) {
        int Ph = proto.h, Pw = proto.w;
        // proto 버퍼를 복사 없이 (Ph*Pw, 32) 행렬로 감싼다 (proto.data는 (h,w,c) NHWC라
        // 픽셀 순서로 32채널이 연속 — 그대로 (Ph*Pw)행 x 32열 행렬과 메모리 배치가 같음).
        cv::Mat proto_mat(Ph * Pw, 32, CV_32F, const_cast<float*>(proto.data));

        cv::Mat coeffs_mat((int)kept.size(), 32, CV_32F);
        for (size_t i = 0; i < kept.size(); i++)
            std::memcpy(coeffs_mat.ptr<float>((int)i), kept[i].mask_coeffs.data(), 32 * sizeof(float));

        cv::Mat masks_raw = coeffs_mat * proto_mat.t();  // (N, Ph*Pw) — 조교님 코드의 (proto_2d @ coeffs.T).T와 동일 연산
        cv::Mat masks_sig;
        cv::exp(-masks_raw, masks_sig);
        masks_sig += 1.0;
        cv::divide(1.0, masks_sig, masks_sig);  // sigmoid = 1 / (1 + exp(-x))

        for (size_t i = 0; i < kept.size(); i++) {
            kept[i].mask.resize((size_t)Ph * Pw);
            std::memcpy(kept[i].mask.data(), masks_sig.ptr<float>((int)i), (size_t)Ph * Pw * sizeof(float));
        }
    }

    return kept;
}
