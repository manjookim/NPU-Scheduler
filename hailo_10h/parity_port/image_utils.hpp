#pragma once
// image_utils.hpp — 이미지 전처리(letterbox) 및 파일 목록 유틸 (Hailo-10H 이식본)
// [8L 대비 변경점]
//   - LetterboxMeta의 정의 위치만 model_types.hpp로 옮겼다(후처리 없이 단독 빌드 가능하도록).
//     letterbox()/get_image_files()/now_ms()의 동작은 hailo_8L/image_utils.hpp와 동일하다.
//   - get_image_files()는 8L 원본 그대로다(.jpg/.JPG 부분 문자열 매칭). 조교님 코드의
//     파일 선택 규칙을 바꾸지 않기 위해 확장자 판정을 손대지 않았다 — IMG_DIR은 반드시
//     '/'로 끝나야 한다.

#include "model_types.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <dirent.h>
#include <algorithm>

// Letterbox: 비율 유지 resize + gray(114) 패딩 -> target_size x target_size
// (YOLOv8 학습 전처리와 동일. 8L과 완전히 같은 전처리를 써야 전처리 시간 비교가 성립한다.)
// meta_out이 non-null이면 Detection 후처리의 좌표 unpad에 필요한 scale/pad/원본크기를 채워준다.
inline cv::Mat letterbox(const cv::Mat& img, int target_size = 640, LetterboxMeta* meta_out = nullptr) {
    int orig_h = img.rows, orig_w = img.cols;
    float scale = std::min((float)target_size / orig_h, (float)target_size / orig_w);
    int new_h = (int)(orig_h * scale);
    int new_w = (int)(orig_w * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    int pad_top    = (target_size - new_h) / 2;
    int pad_bottom = target_size - new_h - pad_top;
    int pad_left   = (target_size - new_w) / 2;
    int pad_right  = target_size - new_w - pad_left;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    if (meta_out) {
        meta_out->scale = scale;
        meta_out->pad_top = pad_top;
        meta_out->pad_left = pad_left;
        meta_out->orig_w = orig_w;
        meta_out->orig_h = orig_h;
    }
    return out;
}

inline std::vector<std::string> get_image_files(const char* dir_path) {
    // [8L 로직 그대로] hailo_8L/image_utils.hpp와 동일. .jpg / .JPG 부분 문자열 매칭이고
    // dir_path 뒤에 슬래시를 붙이지 않으므로 IMG_DIR이 '/'로 끝나야 한다.
    // (조교님 코드의 파일 선택 규칙을 바꾸지 않기 위해 확장자 판정을 원본대로 유지한다.)
    std::vector<std::string> files;
    DIR* dir = opendir(dir_path);
    if (!dir) return files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (name.size() > 4 &&
            (name.find(".jpg") != std::string::npos || name.find(".JPG") != std::string::npos))
            files.push_back(std::string(dir_path) + name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());  // 실험 재현성을 위해 정렬(항상 같은 부분집합 사용)
    return files;
}

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
