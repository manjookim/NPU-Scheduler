#pragma once
// output_classify.hpp — output vstream을 채널 수 기준으로 role(box/score/kpts/cls/coeff/proto)로
// 분류한다 (Hailo-8L). [의존] model_types.hpp(OutMeta/OutRole/ModelKind)가 먼저 include 되어야 함.

#include "hailo/hailort.hpp"
#include <vector>

using namespace hailort;  // main.cpp에서 이미 선언되지만 헤더 단독 가독성을 위해 재선언(중복 무해)

inline std::vector<OutMeta> classify_outputs(ModelKind kind, std::vector<OutputVStream>& outputs, int img_size = 640) {
    std::vector<OutMeta> metas(outputs.size());
    if (kind == ModelKind::DET) {
        // Detection은 보통 NMS-by-class 출력 vstream 1개. 원시 텐서가 아니라 이미 검출
        // 결과이므로 role 분류는 필요 없고, decode_det()가 버퍼를 파싱하는 데 필요한
        // 클래스 수/클래스당 최대 검출 수만 얻는다.
        for (size_t j = 0; j < outputs.size(); j++) {
            const auto& info = outputs[j].get_info();
            metas[j].nms_number_of_classes = (int)info.nms_shape.number_of_classes;
            metas[j].nms_max_bboxes_per_class = (int)info.nms_shape.max_bboxes_per_class;
        }
        return metas;
    }

    for (size_t j = 0; j < outputs.size(); j++) {
        const auto& info = outputs[j].get_info();
        int h = (int)info.shape.height, w = (int)info.shape.width, c = (int)info.shape.features;
        metas[j].h = h; metas[j].w = w; metas[j].c = c;
        metas[j].stride = (h > 0) ? (img_size / h) : 0;

        if (kind == ModelKind::POSE) {
            if (c == 64) metas[j].role = OutRole::POSE_BOX;
            else if (c == 1) metas[j].role = OutRole::POSE_SCORE;
            else if (c == 51) metas[j].role = OutRole::POSE_KPTS;
        } else if (kind == ModelKind::SEG) {
            if (c == 64) metas[j].role = OutRole::SEG_BOX;
            else if (c == 80) metas[j].role = OutRole::SEG_CLS;
            else if (c == 32) metas[j].role = (h == 160 && w == 160) ? OutRole::SEG_PROTO : OutRole::SEG_COEFF;
        }
    }
    return metas;
}
