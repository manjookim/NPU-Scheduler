#pragma once
// model_runner.hpp — 모델 1개(Detection/Segmentation/Pose 중 하나)를 담당하는
// writer/reader 스레드 쌍(producer/consumer)으로 비동기 추론을 수행한다 (Hailo-10H 이식본).
//
// [의존] infer_scheduler_h10.cpp의 파라미터 #define 블록(INPUT_FPS, DEBUG_WRITE_TIMING)과
//        `std::mutex print_mutex;` 정의보다 뒤에 include 되어야 한다.
//
// ============================ 8L 대비 핵심 차이 ============================
// 측정 정의는 8L과 완전히 동일하게 유지한다:
//   enq_ts[i] = 이 프레임을 파이프라인에 넘기기 직전(= write 호출 직전 / 슬롯 확보 직전)
//   deq_ts[i] = 이 프레임의 모든 출력을 다 받은 시각
//   avg_latency_ms = mean(deq_ts - enq_ts)   ← 큐 대기시간을 포함하는 체류시간(sojourn)
// 이 정의를 지켜야 8L의 det_latency_ms(219 ms 등)와 같은 물리량이 된다. run_async 직후
// wait()하는 구조는 in-flight가 1이 되어 "서비스시간"을 재게 되므로 값이 10배 작아진다 —
// 그게 기존 H10 코드에서 10배 차이가 났던 원인이므로, 여기서는 두 백엔드 모두
// 파이프라인을 채운 상태에서 측정한다.
//
//   백엔드 A run_model_async_vstreams(): 8L 코드 그대로. 큐 깊이는 HailoRT가 정한다.
//   백엔드 B run_model_async_infer()   : InferModel Async + 슬롯 N개. 큐 깊이 = INFLIGHT.
//
// [2026-08-31] 후처리(디코딩/NMS)를 코드에서 완전히 제거했다.
//   - reader 스레드는 "출력을 다 받는 것"까지만 하고 버퍼 내용을 해석하지 않는다.
//   - 출력 포맷 FLOAT32 강제도 없앴다(model_setup.hpp) → HailoRT 내부 역양자화/재배열
//     비용이 latency에 섞이지 않는다. 8L의 B/D조건(ENABLE_POSTPROCESS=0,
//     FORCE_OUTPUT_FLOAT32=0)과 동일한 조건이다.
//   - CSV의 postprocess_ms_* 컬럼은 그대로 두고 값만 NaN으로 나간다(스키마 49컬럼 유지).

#include "hailo/hailort.hpp"
#include "model_types.hpp"
#include "model_setup.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// 공용: 프레임 1장 전처리 (imread -> letterbox -> BGR2RGB). 8L writer 스레드 로직과 동일.
// 실패 시 검은 화면으로 대체해 프레임 수/인덱스 정렬을 유지한다(8L과 같은 규약).
// 후처리를 제거했으므로 LetterboxMeta는 채우지 않는다(unpad에 쓸 곳이 없다).
// ─────────────────────────────────────────────────────────────────────────────
inline cv::Mat preprocess_one(const char* model_name, const std::string& path, int img_size)
{
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] [경고] 이미지 로드 실패: " << path
                  << " (검은 화면으로 대체, 프레임 수/인덱스 정렬 유지)" << std::endl;
        return cv::Mat::zeros(img_size, img_size, CV_8UC3);
    }
    cv::Mat lb = letterbox(img, img_size, nullptr);
    cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
    return lb;
}

// ─────────────────────────────────────────────────────────────────────────────
// 공용: enq/deq 타임스탬프 배열로 ModelResult를 채운다. 두 백엔드가 같은 함수를 쓰므로
// latency/total_time의 정의가 백엔드에 따라 흔들리지 않는다.
// 후처리를 제거했으므로 avg_postprocess_ms는 항상 -1(-> CSV에 NaN)이다.
// ─────────────────────────────────────────────────────────────────────────────
inline void finalize_result(ModelResult& result,
                            const std::vector<double>& enq_ts,
                            const std::vector<double>& deq_ts,
                            double prep_total_ms, long prep_count,
                            double pp_total_ms, long pp_count,
                            double svc_total_ms, double qw_total_ms, long svc_count,
                            long vol, long nonvol)
{
    const size_t N = enq_ts.size();

    result.avg_preprocess_ms  = (prep_count > 0) ? (prep_total_ms / prep_count) : -1;
    // [8L 로직 유지] 8L은 ENABLE_POSTPROCESS=0일 때도 후처리 구간을 계측해서 ~1e-4 ms를
    //   기록한다(-1/NaN이 아니다). 실제 CSV 확인: results_B.csv postprocess_ms_det=0.00010376.
    //   이 이식본도 같은 규약을 따른다 — 디코딩 코드는 없지만 구간 계측은 남긴다.
    result.avg_postprocess_ms = (pp_count > 0) ? (pp_total_ms / pp_count) : -1;

    double sum = 0, mx = -1; int c = 0;
    for (size_t i = 0; i < N; i++) {
        if (deq_ts[i] > enq_ts[i]) {
            double lat = deq_ts[i] - enq_ts[i];
            sum += lat; c++;
            if (lat > mx) mx = lat;
        }
    }
    result.avg_latency_ms = c > 0 ? sum / c : -1;
    result.max_latency_ms = mx;
    result.frame_count = c;
    result.vol_ctx = vol;
    result.nonvol_ctx = nonvol;

    // 모델별 전체 추론 시간 = 첫 입력 enqueue ~ 마지막 출력 dequeue 구간 (8L과 동일 정의)
    double first_enq = 0, last_deq = 0;
    for (size_t i = 0; i < N; i++) {
        if (enq_ts[i] > 0 && (first_enq == 0 || enq_ts[i] < first_enq)) first_enq = enq_ts[i];
        if (deq_ts[i] > last_deq) last_deq = deq_ts[i];
    }
    result.total_time_s = (last_deq > first_enq) ? (last_deq - first_enq) / 1000.0 : -1;
    result.fps = (result.total_time_s > 0) ? (result.frame_count / result.total_time_s) : -1;

    result.avg_service_ms    = (svc_count > 0) ? (svc_total_ms / svc_count) : -1;
    result.avg_queue_wait_ms = (svc_count > 0) ? (qw_total_ms  / svc_count) : -1;
}

// ═════════════════════════════════════════════════════════════════════════════
//  백엔드 A — VStreams (Hailo-8L run_model_async()와 1:1)
// ═════════════════════════════════════════════════════════════════════════════
// writer 스레드가 입력 큐를 채우고(큐가 가득 차면 write가 블로킹), reader 스레드가
// 그 모델의 "모든" 출력 vstream을 프레임 단위로 읽는다. INPUT_FPS>0이면 writer가 그
// 속도로 write를 지연시켜 큐가 점진적으로 쌓이도록 한다.
inline void run_model_async_vstreams(const char* model_name,
                                     std::vector<InputVStream>& inputs,
                                     std::vector<OutputVStream>& outputs,
                                     const std::vector<std::string>& images,
                                     ModelResult& result,
                                     int img_size)
{
    if (inputs.empty() || outputs.empty()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] 입력/출력 vstream 없음, 스킵" << std::endl;
        return;
    }

    const size_t N = images.size();
    const size_t expected_frame_size = inputs[0].get_frame_size();
    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0);
    long w_vol = 0, w_nonvol = 0, r_vol = 0, r_nonvol = 0;
    hailo_status write_status = HAILO_SUCCESS, read_status = HAILO_SUCCESS;

    double prep_total_ms = 0.0; long prep_count = 0;
    double pp_total_ms = 0.0;   long pp_count = 0;   // 8L과 동일하게 후처리 구간을 계측한다

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

            // ── 전처리 (이 프레임만) ──
            double prep_t0 = now_ms();
            cv::Mat lb = preprocess_one(model_name, images[i], img_size);
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
            // 스케줄러 경합으로 입력버퍼가 안 비면 write가 HAILO_TIMEOUT을 낸다.
            // 프레임 유실 방지를 위해 timeout이면 성공할 때까지 재시도한다. (8L과 동일)
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
        // 출력 버퍼는 "받기만" 한다 — 후처리를 제거했으므로 내용은 해석하지 않는다.
        std::vector<std::vector<uint8_t>> obuf(outputs.size());
        for (size_t j = 0; j < outputs.size(); j++)
            obuf[j].resize(outputs[j].get_frame_size());

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < outputs.size(); j++) {
                hailo_status status;
                do {
                    status = outputs[j].read(MemoryView(obuf[j].data(), obuf[j].size()));
                } while (status == HAILO_TIMEOUT);
                if (HAILO_SUCCESS != status) { read_status = status; }
            }
            deq_ts[i] = now_ms();  // 이 프레임의 모든 출력을 다 받은 시각

            // ── [8L 로직 유지] 후처리 구간 계측. 디코딩 코드는 제거했으므로 본문은 비어 있고
            //    값은 ~1e-4 ms가 된다 — 8L의 ENABLE_POSTPROCESS=0 실행과 같은 결과다.
            double pp_t0 = now_ms();
            double pp_t1 = now_ms();
            pp_total_ms += (pp_t1 - pp_t0);
            pp_count++;
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

#if DEBUG_WRITE_TIMING
    // [진단용] HailoRT 공식 큐 크기 accumulator 값 출력 (element별 min/mean/max).
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

    if (HAILO_SUCCESS != write_status || HAILO_SUCCESS != read_status) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] [경고] 추론 중 오류 (write=" << write_status
                  << ", read=" << read_status << ") — 아래 latency는 왜곡됐을 수 있음" << std::endl;
    }

    finalize_result(result, enq_ts, deq_ts, prep_total_ms, prep_count, pp_total_ms, pp_count,
                    0.0, 0.0, 0,   // vstreams 백엔드는 service/queue_wait 분리 측정 불가 -> NaN
                    w_vol + r_vol, w_nonvol + r_nonvol);

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[%s] 완료(vstreams): 전처리=%.2f ms, 평균 Latency=%.2f ms, 최대=%.2f ms, "
                "%d장, 전체 추론시간=%.2f s, FPS=%.2f (후처리 없음, INPUT_FPS=%d)\n",
                model_name, result.avg_preprocess_ms, result.avg_latency_ms, result.max_latency_ms,
                result.frame_count, result.total_time_s, result.fps, INPUT_FPS);
}

// ═════════════════════════════════════════════════════════════════════════════
//  백엔드 B — InferModel Async + 슬롯 풀
// ═════════════════════════════════════════════════════════════════════════════
// 8L의 writer/reader 구조를 그대로 유지한다:
//   writer 스레드 : 전처리 -> 빈 슬롯 확보(= 8L의 write() 블로킹에 해당) -> run_async 제출
//   HailoRT 콜백  : 완료 시각(deq_ts) 기록 + 완료 큐에 밀어넣기 (가벼운 일만)
//   reader 스레드 : 완료 큐에서 꺼내 슬롯 반납 (후처리를 제거했으므로 그 외 일은 없다)
// 슬롯 수(inflight)가 곧 in-flight 깊이다. 8L vstream의 실측 in-flight가 ~11이었으므로
// INFLIGHT=8~16이 8L과 가장 비슷한 조건이 된다. INFLIGHT=1이면 기존(문제 있던) H10 코드와 동일.
inline void run_model_async_infer(const char* model_name,
                                  InferCtx& ctx,
                                  const std::vector<std::string>& images,
                                  ModelResult& result,
                                  int img_size)
{
    const size_t N = images.size();
    const int SLOTS = (int)ctx.slots.size();

    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0), sub_ts(N, 0.0);
    std::vector<int> slot_of(N, -1);
    std::vector<char> ok_of(N, 0);

    long w_vol = 0, w_nonvol = 0, r_vol = 0, r_nonvol = 0;
    double prep_total_ms = 0.0; long prep_count = 0;
    double pp_total_ms = 0.0;   long pp_count = 0;   // 8L과 동일하게 후처리 구간을 계측한다
    std::atomic<int> submit_failures{0};

    // ── 빈 슬롯 풀 ──
    std::mutex free_mtx;
    std::condition_variable free_cv;
    std::deque<int> free_slots;
    for (int s = 0; s < SLOTS; s++) free_slots.push_back(s);

    // ── 완료 큐 (콜백 -> reader) ──
    std::mutex done_mtx;
    std::condition_variable done_cv;
    std::deque<std::pair<size_t, bool>> done_q;   // (frame_idx, ok)

    // reader가 몇 개를 기다려야 하는지. 정상 종료면 N이지만, writer가 중간에 끊기면
    // 그 시점까지 실제로 큐에 들어갈 개수로 확정된다(reader 무한대기 방지).
    std::atomic<size_t> enqueued{0};
    std::atomic<bool> writer_done{false};

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

            // (1) 전처리 — 이 시간 동안 앞선 SLOTS-1개 프레임이 NPU에서 돌고 있다(8L writer와 동일).
            double prep_t0 = now_ms();
            cv::Mat lb = preprocess_one(model_name, images[i], img_size);
            double prep_t1 = now_ms();
            prep_total_ms += (prep_t1 - prep_t0);
            prep_count++;

            // (2) 8L의 enq_ts와 같은 지점 — 큐가 받아줄 때까지의 대기를 포함시켜야
            //     avg_latency_ms가 8L과 같은 물리량(sojourn)이 된다.
            enq_ts[i] = now_ms();

            int slot = -1;
            {
                std::unique_lock<std::mutex> lk(free_mtx);
                // 데드락 방지용 상한(5분). 정상 동작에서는 절대 걸리지 않는다 —
                // 걸린다면 콜백이 오지 않는다는 뜻이므로 원인을 반드시 확인할 것.
                if (!free_cv.wait_for(lk, std::chrono::seconds(300),
                                      [&] { return !free_slots.empty(); })) {
                    lk.unlock();
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cerr << "[" << model_name << "] [치명적] 5분간 빈 슬롯 없음 — "
                                 "완료 콜백이 오지 않고 있다. 프레임 " << i << "에서 중단." << std::endl;
                    break;
                }
                slot = free_slots.front();
                free_slots.pop_front();
            }
            slot_of[i] = slot;

            auto& S = ctx.slots[slot];
            const size_t src_bytes = lb.total() * lb.elemSize();
            if (src_bytes != S.in_size) {
                if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cerr << "[" << model_name << "] [경고] 프레임 크기 불일치: 모델 기대="
                              << S.in_size << "B, 전처리 결과=" << src_bytes << "B" << std::endl;
                }
                std::memcpy(S.in_buf.get(), lb.data, std::min(S.in_size, src_bytes));
            } else {
                std::memcpy(S.in_buf.get(), lb.data, S.in_size);
            }

            // HailoRT 내부 큐가 받아줄 준비가 될 때까지 대기(8L write()의 블로킹에 대응).
            auto ready_st = ctx.configured->wait_for_async_ready(std::chrono::milliseconds(300000));
            if (ready_st != HAILO_SUCCESS) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "[" << model_name << "] wait_for_async_ready 실패 status=" << ready_st << std::endl;
            }

            sub_ts[i] = now_ms();
            const size_t idx = i;
            auto job_exp = ctx.configured->run_async(*S.bindings,
                [&, idx](const AsyncInferCompletionInfo& info) {
                    // 콜백은 HailoRT 내부 스레드에서 불린다 — 타임스탬프와 큐 push만 한다.
                    deq_ts[idx] = now_ms();
                    bool ok = (info.status == HAILO_SUCCESS);
                    {
                        std::lock_guard<std::mutex> lk(done_mtx);
                        done_q.emplace_back(idx, ok);
                    }
                    done_cv.notify_one();
                });

            if (!job_exp) {
                // 제출 실패 — 슬롯을 즉시 반납하고 reader에게도 알려 개수 정렬을 유지한다.
                // [중요] slot_of[i]를 -1로 되돌린 뒤에 done_q에 넣어야 한다. 그러지 않으면
                //        reader가 같은 슬롯을 한 번 더 free_slots에 넣어(이중 반납) 풀 크기가
                //        INFLIGHT를 넘고, 두 프레임이 같은 입력 버퍼를 동시에 쓰게 된다.
                submit_failures++;
                slot_of[i] = -1;
                {
                    std::lock_guard<std::mutex> lk(free_mtx);
                    free_slots.push_back(slot);
                }
                free_cv.notify_one();
                enqueued++;
                {
                    std::lock_guard<std::mutex> lk(done_mtx);
                    done_q.emplace_back(i, false);
                }
                done_cv.notify_one();
                continue;
            }
            enqueued++;
            job_exp->detach();   // 소멸자에서 블로킹하지 않도록 분리 — 완료는 콜백으로 받는다.
        }
        writer_done.store(true);
        done_cv.notify_all();    // reader가 종료 조건을 다시 확인하도록 깨운다
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary; w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    std::thread reader([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();

        size_t processed = 0;
        int idle_ticks = 0;
        while (true) {
            size_t idx; bool ok;
            {
                std::unique_lock<std::mutex> lk(done_mtx);
                if (!done_cv.wait_for(lk, std::chrono::seconds(1),
                                      [&] { return !done_q.empty(); })) {
                    lk.unlock();
                    // 완료 큐가 비어 있다 — 정상 종료인지 멈춘 것인지 구분한다.
                    if (writer_done.load() && processed >= enqueued.load()) break;
                    if (++idle_ticks > 300) {   // 5분간 아무 완료도 없음
                        std::lock_guard<std::mutex> lock(print_mutex);
                        std::cerr << "[" << model_name << "] [치명적] 5분간 완료 콜백 없음 — "
                                  << processed << "/" << enqueued.load() << " 처리 후 중단." << std::endl;
                        break;
                    }
                    continue;
                }
                idle_ticks = 0;
                idx = done_q.front().first;
                ok  = done_q.front().second;
                done_q.pop_front();
            }
            ok_of[idx] = ok ? 1 : 0;

            // ── [8L 로직 유지] 후처리 구간 계측(디코딩 코드는 없으므로 본문은 비어 있다).
            //    8L reader 스레드가 프레임마다 하던 것과 같은 자리·같은 방식이다.
            double pp_t0 = now_ms();
            double pp_t1 = now_ms();
            pp_total_ms += (pp_t1 - pp_t0);
            pp_count++;

            // 출력 버퍼를 해석하지 않고 슬롯만 반납한다.
            int slot = slot_of[idx];
            if (slot >= 0) {
                {
                    std::lock_guard<std::mutex> lk(free_mtx);
                    free_slots.push_back(slot);
                }
                free_cv.notify_one();
            }

            processed++;
            if (writer_done.load() && processed >= enqueued.load()) break;
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

    // service / queue_wait 집계 (8L에서는 잴 수 없던 값 — 확장 컬럼으로만 나간다)
    // service = sub_ts -> deq_ts. 디바이스 내부 큐 대기를 포함하므로 in-flight에 비례해 커진다.
    // queue_wait = enq_ts -> sub_ts. 호스트가 빈 슬롯을 기다린 시간(= 호스트 여유분).
    double svc_total = 0.0, qw_total = 0.0; long svc_count = 0;
    for (size_t i = 0; i < N; i++) {
        if (!ok_of[i] || deq_ts[i] <= sub_ts[i] || sub_ts[i] <= 0) continue;
        svc_total += (deq_ts[i] - sub_ts[i]);
        qw_total  += (sub_ts[i] - enq_ts[i]);
        svc_count++;
    }

    if (submit_failures.load() > 0) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] [경고] run_async 제출 실패 " << submit_failures.load()
                  << "건 — latency/FPS가 왜곡됐을 수 있음" << std::endl;
    }

    finalize_result(result, enq_ts, deq_ts, prep_total_ms, prep_count, pp_total_ms, pp_count,
                    svc_total, qw_total, svc_count,
                    w_vol + r_vol, w_nonvol + r_nonvol);

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[%s] 완료(infer, slots=%d): 전처리=%.2f ms, 평균 Latency=%.2f ms, 최대=%.2f ms, "
                "서비스=%.2f ms, 큐대기=%.2f ms, %d장, 전체=%.2f s, FPS=%.2f (후처리 없음, INPUT_FPS=%d)\n",
                model_name, SLOTS, result.avg_preprocess_ms, result.avg_latency_ms, result.max_latency_ms,
                result.avg_service_ms, result.avg_queue_wait_ms,
                result.frame_count, result.total_time_s, result.fps, INPUT_FPS);
}
