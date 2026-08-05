/**
 * postprocess_8l.hpp
 * ------------------------------------------------------------------------
 * [2026-07-28] Hailo-8(rpi4)용 postprocess_hailo8.hpp를 Hailo-8L(rpi1)에 그대로 포팅.
 * TAPPAS 프레임워크 없이, HailoRT raw output vstream 버퍼만으로 동작하는
 * YOLOv8-Pose / YOLOv8-Seg 후처리(디코딩+NMS) 구현.
 *
 * 이 파일의 디코딩/NMS 로직 자체는 보드에 의존하지 않는다(입력은 output vstream의
 * FLOAT32 raw 버퍼일 뿐이라 Hailo-8/Hailo-8L 어느 쪽 HEF든 동일한 아키텍처(YOLOv8
 * Detection/Pose/Seg)라면 그대로 재사용 가능). Hailo-8L의 yolov8s_h8l.hef /
 * yolov8s_pose_h8l.hef 도 동일 아키텍처의 h8l 타깃 컴파일본이라고 가정하고 그대로
 * 적용함 — 실기에서 첫 프레임 디버그 출력(클래스 수/검출 개수/좌표 범위)으로
 * 반드시 정상 여부를 확인할 것(Hailo-8 쪽은 실기 테스트로 확인 완료, Hailo-8L은
 * 아직 미검증).
 *
 * 참고(알고리즘 출처): hailo-ai/tappas
 *   core/hailo/libs/postprocesses/pose_estimation/yolov8pose_postprocess.cpp
 *   (박스 DFL 디코딩, 키포인트 디코딩, NMS 로직을 xtensor/HailoROI 의존성 없이
 *    std::vector 기반으로 이식함)
 * Segmentation은 Ultralytics YOLOv8-seg 표준 아키텍처(box+cls+mask_coeff 3-scale,
 * 공유 proto mask 160x160x32) 기반으로 동일한 DFL 디코딩 방식을 적용해 구현.
 *
 * [중요 — infer_scheduler.cpp의 vstream 생성부와 반드시 짝이 맞아야 함]
 * Detection/Pose/Segmentation 세 모델 모두 output vstream을
 *   - format type  = HAILO_FORMAT_TYPE_FLOAT32  (HailoRT가 qp_scale/qp_zp로
 *     역양자화까지 대신 해줌 → 이 헤더는 역양자화를 하지 않고 float를 그대로 읽음)
 *   - Pose/Seg만 format order = HAILO_FORMAT_ORDER_NHWC 로 추가 고정
 *     (Detection은 NMS-by-class 출력이라 order 지정 대상이 아님)
 * 로 생성되어 있어야 아래 TensorView/decode_det이 버퍼를 올바르게 인덱싱한다.
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
// [중요] score threshold를 넘는 raw candidate 수가 그리드 셀 수(8400=80x80+40x40+20x20)에
// 가깝게 나오면 이 O(n^2) 억제 루프가 수백 ms~초 단위로 폭주한다. Ultralytics 등 표준
// 구현도 NMS 직전에 후보 수를 상한으로 자르므로(max_nms 관행) 여기서도 정렬 후
// 상위 max_candidates개만 남기고 잘라낸다.
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

// ========================= YOLOv8-Detection 후처리 (on-chip NMS 결과 파싱) =========================
// Detection은 HEF에 NMS가 이미 내장돼 있어(HailoRT-pp) 디코딩/NMS 연산 자체는 필요 없다.
// 다만 결과가 HAILO_NMS_BY_CLASS 포맷의 raw byte 버퍼로 나오므로, 이걸 구조화된 검출
// 리스트(PPBox)로 "파싱"하는 작업은 CPU에서 해야 하고, 이 파싱 비용을 Detection의
// "후처리(postprocess)"로 측정한다.
//
// 버퍼 레이아웃(HailoRT hailort_common.hpp의 get_nms_by_class_host_shape_size 계산 방식 기준,
// output vstream을 FLOAT32로 받았을 때):
//   클래스마다 고정 크기 청크가 순서대로 이어짐:
//     [count(1개, float로 인코딩된 실제 검출 개수),
//      (y_min, x_min, y_max, x_max, score) x max_bboxes_per_class]
//   count는 실제 검출 개수(<= max_bboxes_per_class)이고 그 뒤 미사용 슬롯은 don't-care.
//   number_of_classes / max_bboxes_per_class는 output vstream의 get_info().nms_shape에서
//   얻는다(HailoRTCommon::is_nms()가 true인 NMS 포맷 vstream 전용 필드).

// letterbox 전처리 메타데이터(원본 이미지 -> 640 패딩 좌표로의 변환 정보).
// infer_scheduler.cpp의 letterbox()가 전처리 시점에 채워서 프레임별로 들고 있다가,
// Detection 후처리에서 좌표 unpad(패딩 제거 + 스케일 역보정)에 쓴다.
struct LetterboxMeta {
    float scale = 1.0f;      // 원본 -> 리사이즈 배율 (min(target/h, target/w))
    int pad_top = 0, pad_left = 0;
    int orig_w = 0, orig_h = 0;
};

// COCO category_id 매핑: YOLO 0~79 contiguous class idx -> COCO 공식 category_id(1~90,
// non-contiguous, 표준 COCO80->91 매핑표). Ultralytics coco80_to_coco91_class()와 동일.
inline int coco_category_id(int yolo_class_id) {
    static const int coco_ids[80] = {
        1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
        22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,
        46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,
        67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90
    };
    if (yolo_class_id < 0 || yolo_class_id >= 80) return yolo_class_id;
    return coco_ids[yolo_class_id];
}

// meta == nullptr 이면 640 기준 정규화 좌표 + YOLO class_id 그대로 반환.
// meta != nullptr 이면 원본 이미지 픽셀 좌표로 unpad하고 class_id를 COCO category_id로 매핑.
inline std::vector<PPBox> decode_det(const float* data, int number_of_classes, int max_bboxes_per_class,
                                      float score_thr = 0.0f,   // on-chip NMS가 이미 자체 threshold로 걸러서 나오므로 기본은 다 통과
                                      const LetterboxMeta* meta = nullptr,
                                      int model_input_size = 640)
{
    std::vector<PPBox> dets;
    if (!data || number_of_classes <= 0 || max_bboxes_per_class <= 0) return dets;

    const size_t stride = 1 + 5 * (size_t)max_bboxes_per_class;  // 클래스 1개 청크의 float 개수
    for (int cls = 0; cls < number_of_classes; cls++) {
        size_t base = (size_t)cls * stride;
        int count = (int)(data[base] + 0.5f);  // float로 인코딩된 개수 -> 반올림
        if (count < 0) count = 0;
        if (count > max_bboxes_per_class) count = max_bboxes_per_class;  // 방어적 클램프(버퍼 초과 접근 방지)
        for (int i = 0; i < count; i++) {
            size_t o = base + 1 + (size_t)i * 5;
            float ymin = data[o + 0], xmin = data[o + 1], ymax = data[o + 2], xmax = data[o + 3], score = data[o + 4];
            if (score < score_thr) continue;

            PPBox box;
            if (meta && meta->scale > 1e-9f) {
                // 정규화(0~1, 640 기준) -> 640 픽셀 -> 패딩 제거 -> 스케일 역보정 -> 원본 이미지 픽셀
                float x1_640 = xmin * model_input_size, y1_640 = ymin * model_input_size;
                float x2_640 = xmax * model_input_size, y2_640 = ymax * model_input_size;
                box.x1 = (x1_640 - meta->pad_left) / meta->scale;
                box.y1 = (y1_640 - meta->pad_top)  / meta->scale;
                box.x2 = (x2_640 - meta->pad_left) / meta->scale;
                box.y2 = (y2_640 - meta->pad_top)  / meta->scale;
                // 패딩 영역에 걸친 박스가 원본 경계를 벗어날 수 있어 클램프
                box.x1 = std::max(0.f, std::min(box.x1, (float)meta->orig_w));
                box.y1 = std::max(0.f, std::min(box.y1, (float)meta->orig_h));
                box.x2 = std::max(0.f, std::min(box.x2, (float)meta->orig_w));
                box.y2 = std::max(0.f, std::min(box.y2, (float)meta->orig_h));
                box.class_id = coco_category_id(cls);
            } else {
                box.x1 = xmin; box.y1 = ymin; box.x2 = xmax; box.y2 = ymax;
                box.class_id = cls;
            }
            box.score = score;
            dets.push_back(box);
        }
    }
    return dets;
}

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
                                         float score_thr = 0.3f,
                                         float iou_thr = 0.45f,
                                         float kpt_thr = 0.5f,
                                         size_t* raw_candidates = nullptr)  // 디버그용: NMS 전 후보 수(score>=thr 통과 개수)
{
    std::vector<PoseDet> dets;
    const int reg_len1 = 16;

    for (const auto& sc : scales) {
        if (!sc.box.data || !sc.score.data || !sc.kpts.data) continue;  // 스케일 그룹이 불완전하면 스킵
        for (int y = 0; y < sc.box.h; y++) {
            for (int x = 0; x < sc.box.w; x++) {
                // [중요] HEF가 cls/score 헤드에 sigmoid를 이미 포함해 export되어 있어서,
                // 여기서 또 sigmoid를 걸면 이중 sigmoid가 되어 값이 전부 0.5~0.73 근처로
                // 몰리고 score_thr을 사실상 모든 셀이 통과해버린다(NMS 폭주 원인).
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
                                       float score_thr = 0.01f,
                                       float iou_thr = 0.65f,
                                       int num_classes = 80,
                                       bool compute_mask = true,
                                       size_t* raw_candidates = nullptr,  // 디버그용: NMS 전 후보 수
                                       size_t max_det = 300)
{
    std::vector<SegDet> dets;
    const int reg_len1 = 16;

    for (const auto& sc : scales) {
        if (!sc.box.data || !sc.cls.data || !sc.coeff.data) continue;  // 스케일 그룹이 불완전하면 스킵
        for (int y = 0; y < sc.box.h; y++) {
            for (int x = 0; x < sc.box.w; x++) {
                // 최댓값 클래스 스코어 찾기
                // [중요] cls 값을 그대로 score로 씀(sigmoid 없음) — HEF가 cls 헤드에
                // sigmoid를 이미 포함해서 export했기 때문(Pose와 동일 이유).
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

    // 클래스별로 따로 NMS(다른 클래스끼리는 서로 억제하지 않음). 부수 효과로 클래스당
    // 후보 수가 전체보다 훨씬 작아져 O(n^2) 비용도 자연히 줄어든다.
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

    // mask = sigmoid(coeff · proto), proto: 160x160x32. naive 3중 for문 대신 OpenCV의
    // 행렬곱(cv::Mat operator*, 내부적으로 SIMD/BLAS 백엔드 활용)으로 대체.
    if (compute_mask && proto.data && !kept.empty()) {
        int Ph = proto.h, Pw = proto.w;
        // proto 버퍼를 복사 없이 (Ph*Pw, 32) 행렬로 감싼다 (proto.data는 (h,w,c) NHWC라
        // 픽셀 순서로 32채널이 연속 — 그대로 (Ph*Pw)행 x 32열 행렬과 메모리 배치가 같음).
        cv::Mat proto_mat(Ph * Pw, 32, CV_32F, const_cast<float*>(proto.data));

        cv::Mat coeffs_mat((int)kept.size(), 32, CV_32F);
        for (size_t i = 0; i < kept.size(); i++)
            std::memcpy(coeffs_mat.ptr<float>((int)i), kept[i].mask_coeffs.data(), 32 * sizeof(float));

        cv::Mat masks_raw = coeffs_mat * proto_mat.t();  // (N, Ph*Pw)
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
