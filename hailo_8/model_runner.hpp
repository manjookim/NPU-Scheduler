#pragma once
// model_runner.hpp — 모델 1개(Detection/Segmentation/Pose 중 하나)를 담당하는
// writer/reader 스레드 쌍(producer/consumer)으로 비동기 추론을 수행한다 (Hailo-8).
// [의존] infer_scheduler_hailo8.cpp의 파라미터 #define 블록(INPUT_FPS, ENABLE_POSTPROCESS)과
// `std::mutex print_mutex;` 정의보다 뒤에 include 되어야 한다 — 매크로/전역변수를 그대로
// 사용하기 때문. postprocess_hailo8.hpp의 decode_det/pose/seg, image_utils.hpp의
// letterbox/now_ms, sys_monitor.hpp의 CtxSwitches도 필요하다.

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <map>
#include <utility>

// ========================= 모델별 비동기 추론 (producer/consumer) =========================
// writer 스레드가 입력 큐를 채우고(큐가 가득 차면 write가 블로킹), reader 스레드가
// 그 모델의 "모든" 출력 vstream을 프레임 단위로 읽는다. INPUT_FPS>0이면 writer가 그
// 속도로 write를 지연시켜 큐가 점진적으로 쌓이도록 한다 — 이래야 threshold(큐 누적 수)
// 와 timeout(대기 시간)이 실제로 트리거될 조건이 생긴다(동기식으로 한꺼번에 밀어넣으면
// 큐가 항상 포화되어 threshold가 무의미해짐, memory/findings.md 참고).
inline void run_model_async(const char* model_name,
                     ModelKind kind,
                     std::vector<InputVStream>& inputs,
                     std::vector<OutputVStream>& outputs,
                     const std::vector<OutMeta>& out_meta,
                     const std::vector<std::string>& images,  // [2026-07-28] 파일 경로만 공유. 전처리(imread+letterbox)는
                                                               // 더이상 미리 한꺼번에 하지 않고 writer 스레드 루프 안에서
                                                               // 프레임마다 수행한다(조교님 요청: "1장씩 전처리-추론-후처리").
                     ModelResult& result,
                     int img_size = 640)  // [2026-08-06 추가] YOLOv5 nms_core(512x512) 등 640이 아닌
                                           // 입력 크기를 쓰는 모델 지원용. 기본값 640이라 기존 호출부는
                                           // 인자를 안 줘도 그대로 640으로 동작(하위 호환).
{
    if (inputs.empty() || outputs.empty()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] 입력/출력 vstream 없음, 스킵" << std::endl;
        return;
    }

    size_t N = images.size();
    size_t expected_frame_size = inputs[0].get_frame_size();  // letterbox 결과(640x640x3)와 비교용, 프레임 0 처리 후 검증
    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0);
    std::vector<LetterboxMeta> pre_meta(N);  // writer가 프레임마다 채우고, reader가 Detection unpad에 사용
                                              // (reader는 항상 writer가 그 인덱스를 다 쓴 뒤에만 읽으므로 안전 — deq_ts와 동일한 전제)
    long w_vol = 0, w_nonvol = 0, r_vol = 0, r_nonvol = 0;
    hailo_status write_status = HAILO_SUCCESS, read_status = HAILO_SUCCESS;

    double prep_total_ms = 0.0;
    long prep_count = 0;

    std::thread writer([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        const double interval_ms = (INPUT_FPS > 0) ? (1000.0 / INPUT_FPS) : 0.0;
        double next_t = now_ms();
        for (size_t i = 0; i < N; i++) {
            if (INPUT_FPS > 0) {
                double t = now_ms();
                if (t < next_t)
                    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(next_t - t));
                next_t += interval_ms;
            }

            // ── 전처리 (이 프레임만): imread -> letterbox -> BGR2RGB. 프레임당 소요시간 측정 ──
            double prep_t0 = now_ms();
            cv::Mat img = cv::imread(images[i]);
            cv::Mat lb;
            if (img.empty()) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "[" << model_name << "] [경고] 이미지 로드 실패: " << images[i]
                           << " (검은 화면으로 대체, 프레임 수/인덱스 정렬 유지)" << std::endl;
                lb = cv::Mat::zeros(img_size, img_size, CV_8UC3);
            } else {
                LetterboxMeta lm;
                lb = letterbox(img, img_size, &lm);
                cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
                pre_meta[i] = lm;
            }
            double prep_t1 = now_ms();
            prep_total_ms += (prep_t1 - prep_t0);
            prep_count++;

            if (i == 0 && lb.total() * lb.elemSize() != expected_frame_size) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "[" << model_name << "] [경고] 프레임 크기 불일치: 모델 기대="
                           << expected_frame_size << "B, 전처리 결과=" << (lb.total() * lb.elemSize())
                           << "B (letterbox 크기/채널 수를 모델 입력 shape에 맞게 조정할 것)" << std::endl;
            }

            enq_ts[i] = now_ms();
            // 낮은 우선순위로 starvation되어 입력버퍼가 안 비면 write가 HAILO_TIMEOUT을 낸다.
            // 프레임 유실 방지를 위해 timeout이면 성공할 때까지 재시도한다(threshold>=1이라
            // 높은 우선순위 모델이 자기 큐를 비우면 결국 이 모델도 스케줄되어 write가 통과).
            hailo_status status;
            do {
                status = inputs[0].write(MemoryView(lb.data, lb.total() * lb.elemSize()));
            } while (status == HAILO_TIMEOUT);
            if (HAILO_SUCCESS != status) { write_status = status; }
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary; w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    double pp_total_ms = 0.0;
    long pp_count = 0;

    std::thread reader([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        std::vector<std::vector<uint8_t>> obuf(outputs.size());
        for (size_t j = 0; j < outputs.size(); j++)
            obuf[j].resize(outputs[j].get_frame_size());

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < outputs.size(); j++) {
                // starvation 중이면 출력도 늦게 나와 read가 HAILO_TIMEOUT을 낼 수 있으므로 재시도.
                hailo_status status;
                do {
                    status = outputs[j].read(MemoryView(obuf[j].data(), obuf[j].size()));
                } while (status == HAILO_TIMEOUT);
                if (HAILO_SUCCESS != status) { read_status = status; }
            }
            deq_ts[i] = now_ms();  // 이 프레임의 모든 출력을 다 받은 시각

            // ── 후처리, 프레임당 소요시간 별도 측정 ──
            // Detection은 HEF에 on-chip NMS(HailoRT-pp)가 내장되어 있어 디코딩/NMS 연산
            // 자체는 없지만, NMS-by-class raw 버퍼를 구조화된 검출 리스트로 파싱하는
            // decode_det() 비용은 여기서 postprocess로 잡는다.
            double pp_t0 = now_ms();
#if ENABLE_POSTPROCESS
            if (kind == ModelKind::DET) {
                if (!out_meta.empty() && out_meta[0].nms_number_of_classes > 0 && out_meta[0].nms_max_bboxes_per_class > 0) {
                    // i번째 프레임의 letterbox 메타(scale/pad/원본크기)로 좌표 unpad + COCO id 매핑까지 수행.
                    const LetterboxMeta* lm = (i < pre_meta.size()) ? &pre_meta[i] : nullptr;
                    std::vector<PPBox> det_dets = decode_det(reinterpret_cast<const float*>(obuf[0].data()),
                                                              out_meta[0].nms_number_of_classes,
                                                              out_meta[0].nms_max_bboxes_per_class,
                                                              0.0f, lm, img_size);
                    if (i == 0) {
                        std::lock_guard<std::mutex> lock(print_mutex);
                        std::printf("  [디버그][%s] 첫 프레임: 클래스 %d개 x 클래스당 최대 %d개 파싱, 검출=%zu개",
                                    model_name, out_meta[0].nms_number_of_classes, out_meta[0].nms_max_bboxes_per_class, det_dets.size());
                        if (lm && !det_dets.empty())
                            std::printf(" (unpad 적용, 원본 %dx%d, 첫 검출 box=[%.1f,%.1f,%.1f,%.1f] coco_id=%d)",
                                        lm->orig_w, lm->orig_h, det_dets[0].x1, det_dets[0].y1, det_dets[0].x2, det_dets[0].y2, det_dets[0].class_id);
                        std::printf("\n");
                    }
                } else if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::printf("  [경고][%s] nms_shape 정보를 못 얻어 후처리 파싱을 건너뜀\n", model_name);
                }
            } else if (kind == ModelKind::POSE) {
                std::map<std::pair<int,int>, PoseScaleTensors> groups;
                for (size_t j = 0; j < outputs.size(); j++) {
                    const auto& m = out_meta[j];
                    pp::TensorView tv{ reinterpret_cast<const float*>(obuf[j].data()), m.h, m.w, m.c };
                    auto& g = groups[{m.h, m.w}];
                    g.stride = m.stride;
                    if (m.role == OutRole::POSE_BOX) g.box = tv;
                    else if (m.role == OutRole::POSE_SCORE) g.score = tv;
                    else if (m.role == OutRole::POSE_KPTS) g.kpts = tv;
                }
                std::vector<PoseScaleTensors> scales;
                scales.reserve(groups.size());
                for (auto& kv : groups) scales.push_back(kv.second);
                size_t total_cells = 0;
                for (auto& sc : scales) total_cells += (size_t)sc.box.h * sc.box.w;
                size_t raw_cand = 0;
                // threshold는 조교님 참고 코드 값(conf=0.3, iou=0.45)과 동일하게 맞춤
                std::vector<PoseDet> pose_dets = decode_pose(scales, 640, 0.3f, 0.45f, 0.5f, &raw_cand);
                if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::printf("  [디버그][%s] 첫 프레임: 그리드 셀 총 %zu개 중 NMS 전 후보=%zu개, 최종 검출=%zu개\n",
                                model_name, total_cells, raw_cand, pose_dets.size());
                }
            } else if (kind == ModelKind::SEG) {
                std::map<std::pair<int,int>, SegScaleTensors> groups;
                pp::TensorView proto_tv;
                for (size_t j = 0; j < outputs.size(); j++) {
                    const auto& m = out_meta[j];
                    pp::TensorView tv{ reinterpret_cast<const float*>(obuf[j].data()), m.h, m.w, m.c };
                    if (m.role == OutRole::SEG_PROTO) { proto_tv = tv; continue; }
                    auto& g = groups[{m.h, m.w}];
                    g.stride = m.stride;
                    if (m.role == OutRole::SEG_BOX) g.box = tv;
                    else if (m.role == OutRole::SEG_CLS) g.cls = tv;
                    else if (m.role == OutRole::SEG_COEFF) g.coeff = tv;
                }
                std::vector<SegScaleTensors> scales;
                scales.reserve(groups.size());
                for (auto& kv : groups) scales.push_back(kv.second);
                size_t total_cells = 0;
                for (auto& sc : scales) total_cells += (size_t)sc.box.h * sc.box.w;
                size_t raw_cand = 0;
                // threshold는 조교님 참고 코드 값(conf=0.01, iou=0.65)과 동일하게 맞춤
                std::vector<SegDet> seg_dets = decode_seg(scales, proto_tv, 640, 0.01f, 0.65f, 80, true, &raw_cand);
                if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::printf("  [디버그][%s] 첫 프레임: 그리드 셀 총 %zu개 중 NMS 전 후보=%zu개, 최종 검출=%zu개\n",
                                model_name, total_cells, raw_cand, seg_dets.size());
                }
            }
#endif  // ENABLE_POSTPROCESS
            double pp_t1 = now_ms();
            pp_total_ms += (pp_t1 - pp_t0);
            pp_count++;
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

    result.avg_preprocess_ms = (prep_count > 0) ? (prep_total_ms / prep_count) : -1;
    result.avg_postprocess_ms = (pp_count > 0) ? (pp_total_ms / pp_count) : -1;

    if (HAILO_SUCCESS != write_status || HAILO_SUCCESS != read_status) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] [경고] 추론 중 오류 (write=" << write_status
                   << ", read=" << read_status << ") — 아래 latency는 왜곡됐을 수 있음" << std::endl;
    }

    double sum = 0; int c = 0;
    for (size_t i = 0; i < N; i++)
        if (deq_ts[i] > enq_ts[i]) { sum += (deq_ts[i] - enq_ts[i]); c++; }

    result.avg_latency_ms = c > 0 ? sum / c : -1;
    result.frame_count = c;
    result.vol_ctx = w_vol + r_vol;
    result.nonvol_ctx = w_nonvol + r_nonvol;

    // 모델별 전체 추론 시간 = 첫 입력 enqueue ~ 마지막 출력 dequeue 구간
    // (해당 모델이 모든 프레임을 다 처리하는 데 걸린 실제 wall-time, NPU 공유 경쟁 포함)
    double first_enq = 0, last_deq = 0;
    for (size_t i = 0; i < N; i++) {
        if (enq_ts[i] > 0 && (first_enq == 0 || enq_ts[i] < first_enq)) first_enq = enq_ts[i];
        if (deq_ts[i] > last_deq) last_deq = deq_ts[i];
    }
    result.total_time_s = (last_deq > first_enq) ? (last_deq - first_enq) / 1000.0 : -1;

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[%s] 완료: 전처리=%.2f ms, 평균 Latency=%.2f ms, 후처리=%.2f ms, %d장, 전체 추론시간=%.2f s (async, INPUT_FPS=%d)\n",
                model_name, result.avg_preprocess_ms, result.avg_latency_ms, result.avg_postprocess_ms, result.frame_count, result.total_time_s, INPUT_FPS);
}
