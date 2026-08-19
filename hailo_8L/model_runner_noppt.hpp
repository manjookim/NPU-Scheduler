#pragma once
// model_runner_noppt.hpp — model_runner.hpp의 "후처리 시간 자체를 측정하지 않는" 변형.
// [2026-08-06] 조교님 요청: ENABLE_POSTPROCESS 매크로로 후처리를 끄는 방식(시간이 0에
// 가깝게 찍히긴 하지만 pp_t0/pp_t1 타이머 자체는 여전히 코드에 존재)이 아니라, 아예
// 후처리 디코딩 호출과 그 타이머 코드를 통째로 들어낸 별도 파일을 요청함.
// model_runner.hpp와 diff는 정확히 이 부분뿐이다:
//   - pp_total_ms/pp_count 변수, pp_t0/pp_t1 타이밍, decode_det/pose/seg 호출,
//     #if ENABLE_POSTPROCESS 블록 자체를 전부 삭제
//   - result.avg_postprocess_ms 대입 코드 삭제 -> ModelResult 기본값(-1, "미측정")
//     그대로 유지되어 CSV에는 항상 NaN으로 기록됨(csv_writer.hpp 로직 그대로 재사용)
//   - 나머지(전처리 측정, latency 측정, INPUT_FPS, DEBUG_WRITE_TIMING 큐 진단, ctx switch
//     측정)는 model_runner.hpp와 100% 동일
// [의존] infer_scheduler_noppt.cpp의 파라미터 #define 블록(INPUT_FPS, DEBUG_WRITE_TIMING)과
// `std::mutex print_mutex;` 정의보다 뒤에 include 되어야 한다. decode_det/pose/seg를
// 호출하지 않으므로 postprocess_8l.hpp에 대한 의존은 없다(있어도 무해, 단순 미사용).

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <map>
#include <utility>

// ========================= 모델별 비동기 추론 (producer/consumer, 후처리 시간 미측정) =====
inline void run_model_async(const char* model_name,
                     ModelKind kind,
                     std::vector<InputVStream>& inputs,
                     std::vector<OutputVStream>& outputs,
                     const std::vector<OutMeta>& out_meta,
                     const std::vector<std::string>& images,
                     ModelResult& result)
{
    if (inputs.empty() || outputs.empty()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] 입력/출력 vstream 없음, 스킵" << std::endl;
        return;
    }

    size_t N = images.size();
    size_t expected_frame_size = inputs[0].get_frame_size();
    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0);
    std::vector<LetterboxMeta> pre_meta(N);  // 이 버전은 unpad 좌표복원(decode_det)을 호출하지 않아
                                              // 실제로는 안 쓰이지만, writer 쪽 코드를 model_runner.hpp와
                                              // 동일하게 유지하기 위해 그대로 둠(diff 최소화 목적).
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
                lb = cv::Mat::zeros(640, 640, CV_8UC3);
            } else {
                LetterboxMeta lm;
                lb = letterbox(img, 640, &lm);
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
            // 프레임 유실 방지를 위해 timeout이면 성공할 때까지 재시도한다.
#if DEBUG_WRITE_TIMING
            double w_t0 = now_ms();
#endif
            hailo_status status;
            do {
                status = inputs[0].write(MemoryView(lb.data, lb.total() * lb.elemSize()));
            } while (status == HAILO_TIMEOUT);
            if (HAILO_SUCCESS != status) { write_status = status; }
#if DEBUG_WRITE_TIMING
            if (i < 40) {
                double w_t1 = now_ms();
                std::lock_guard<std::mutex> lock(print_mutex);
                std::printf("  [큐진단][%s] frame=%zu write_blocking_ms=%.2f\n", model_name, i, w_t1 - w_t0);
            }
#endif
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary; w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

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
            // [2026-08-06] 후처리 디코딩 호출 + 타이머(pp_t0/pp_t1/pp_total_ms) 자체를
            // 통째로 제거함 — model_runner.hpp와 다른 부분은 이 지점뿐이다.
            // (obuf에 담긴 raw output은 여기서 그냥 버려짐, 다음 프레임에서 덮어씀)
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

#if DEBUG_WRITE_TIMING
    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::printf("  [큐진단-공식API][%s] input vstream 큐 accumulator:\n", model_name);
        for (auto& kv : inputs[0].get_queue_size_accumulators()) {
            for (auto& acc : kv.second) {
                if (!acc) continue;
                auto res = acc->get();
                std::printf("    element=%s  min=%.2f mean=%.2f max=%.2f (n=%zu)\n",
                    kv.first.c_str(),
                    res.min() ? res.min().value() : -1.0, res.mean() ? res.mean().value() : -1.0,
                    res.max() ? res.max().value() : -1.0, res.count() ? res.count().value() : (size_t)0);
            }
        }
        for (size_t j = 0; j < outputs.size(); j++) {
            std::printf("  [큐진단-공식API][%s] output[%zu] vstream 큐 accumulator:\n", model_name, j);
            for (auto& kv : outputs[j].get_queue_size_accumulators()) {
                for (auto& acc : kv.second) {
                    if (!acc) continue;
                    auto res = acc->get();
                    std::printf("    element=%s  min=%.2f mean=%.2f max=%.2f (n=%zu)\n",
                        kv.first.c_str(),
                        res.min() ? res.min().value() : -1.0, res.mean() ? res.mean().value() : -1.0,
                        res.max() ? res.max().value() : -1.0, res.count() ? res.count().value() : (size_t)0);
                }
            }
        }
    }
#endif

    result.avg_preprocess_ms = (prep_count > 0) ? (prep_total_ms / prep_count) : -1;
    // result.avg_postprocess_ms 는 대입하지 않음 -> ModelResult 기본값 -1 그대로 유지
    // (csv_writer.hpp가 -1을 NaN으로 기록하는 기존 관례를 그대로 따름 = "이 버전은 후처리
    // 시간을 원래 측정하지 않는다"는 의미가 CSV에도 명확히 남는다)

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

    double first_enq = 0, last_deq = 0;
    for (size_t i = 0; i < N; i++) {
        if (enq_ts[i] > 0 && (first_enq == 0 || enq_ts[i] < first_enq)) first_enq = enq_ts[i];
        if (deq_ts[i] > last_deq) last_deq = deq_ts[i];
    }
    result.total_time_s = (last_deq > first_enq) ? (last_deq - first_enq) / 1000.0 : -1;

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[%s] 완료: 전처리=%.2f ms, 평균 Latency=%.2f ms, 후처리=(미측정), %d장, 전체 추론시간=%.2f s (async, INPUT_FPS=%d)\n",
                model_name, result.avg_preprocess_ms, result.avg_latency_ms, result.frame_count, result.total_time_s, INPUT_FPS);
}
