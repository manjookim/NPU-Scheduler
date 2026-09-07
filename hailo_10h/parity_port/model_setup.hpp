#pragma once
// model_setup.hpp — VDevice에 각 모델(HEF) 로드 + configure, 그리고 실행 자원
// (vstream 또는 InferModel 슬롯) 생성까지 담당 (Hailo-10H 이식본).
//
// [의존] infer_scheduler_h10.cpp의 파라미터 #define 블록(DEBUG_WRITE_TIMING,
//        VSTREAM_QUEUE_SIZE, POWER_MODE_ULTRA)보다 뒤에 include 되어야 한다.
//
// ============================ 8L 대비 핵심 차이 ============================
// (1) VDevice 생성: 8L은 hailo_init_vdevice_params()로 params를 만들고 scheduling_algorithm을
//     ROUND_ROBIN으로 지정했다. H10에서도 같은 코드가 동작하지만, VDevice::create()가
//     로컬 VDeviceBase가 아니라 VDeviceHrpcClient(RPC 클라이언트)를 돌려준다. 즉 스케줄러
//     실체는 디바이스 안 hailort_server에 있다.
// (2) 스케줄러 파라미터: [2026-08-31] threshold/timeout/priority setter를 **주석 처리**했다.
//     아래 apply_scheduler_params() 참고. HailoRT 기본 동작(threshold=1, timeout=0ms,
//     priority=16=NORMAL) 그대로 돌린다.
// (3) power_mode: H10 펌웨어가 ULTRA_PERFORMANCE를 거부할 수 있어, 실패하면 PERFORMANCE로
//     한 번 폴백한 뒤 어느 쪽이 적용됐는지 출력한다(8L은 무조건 ULTRA였다).
// (4) 출력 포맷: 항상 HAILO_FORMAT_TYPE_AUTO. 후처리를 제거했으므로 FLOAT32 강제 변환
//     경로 자체를 코드에서 없앴다(8L의 B/D조건과 동일 — 역양자화 비용이 latency에 안 섞인다).
// (5) 백엔드 2종: vstreams(8L과 1:1) / infer(InferModel Async + 슬롯 풀).
//     [실측 2026-08-31] H10은 vstreams 경로를 **지원하지 않는다** — create_configure_params가
//     HAILO_NOT_IMPLEMENTED(7)로 실패하며 HailoRT가 "use InferModel instead"라고 안내한다.
//     따라서 H10의 기본 백엔드는 infer다. vstreams 코드는 8L 대조용으로 남겨둔다.
//     infer 백엔드는 in-flight 깊이를 명시적으로 조절할 수 있어, 8L의 "큐 포화 ~11프레임"
//     상태를 재현·통제할 수 있다 — 구조적 동등성은 이쪽으로 확보한다.

#include "hailo/hailort.hpp"
#include "model_types.hpp"

#include <sys/mman.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace hailort;

// ─────────────────────────────────────────────────────────────────────────────
// 공용: 페이지 정렬 버퍼 (InferModel Async 경로는 DMA 매핑 효율을 위해 정렬 버퍼를 권장)
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<uint8_t> page_aligned_alloc(size_t size) {
    void* addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr),
                                    [size](void* p) { munmap(p, size); });
}

// ─────────────────────────────────────────────────────────────────────────────
//  스케줄러 파라미터 — [2026-08-31] 요청에 따라 setter 3종을 주석 처리했다.
//
//  주석 처리 상태에서의 실제 동작(HailoRT 기본값):
//     threshold = 1        (infer_model.hpp "The default threshold is 1")
//     timeout   = 0 ms     (infer_model.hpp "The default timeout is 0ms")
//     priority  = 16       (HAILO_SCHEDULER_PRIORITY_NORMAL)
//  8L은 priority를 15로 줬지만 세 모델이 전부 같은 값이라 스케줄링 결과는 동일하다(Round-Robin).
//  즉 이 주석 처리로 8L 대비 스케줄링 동작이 달라지지는 않는다.
//
//  batch는 여기서 다루지 않는다 — configure() 시점 인자라 아래 configure_models() /
//  prepare_infer_ctx()에서 그대로 적용된다.
//
//  다시 켜려면: 아래 세 줄의 주석을 풀고, infer_scheduler_h10.cpp의 #define 값을 원하는
//  값으로 바꾼 뒤 재컴파일할 것. (CSV의 threshold/timeout/priority 컬럼은 "실제 적용값"을
//  적어야 하므로, 주석을 푸는 순간 #define 값이 곧 기록값이 된다.)
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
inline void apply_scheduler_params(T& target, const ModelConfig& m) {
    (void)target;  // setter를 호출하지 않으므로 미사용

    // ── 스케줄러 파라미터 API (주석 처리됨) ──────────────────────────────────
    // auto st_thr = target.set_scheduler_threshold((uint32_t)m.threshold);
    // auto st_to  = target.set_scheduler_timeout(std::chrono::milliseconds(m.timeout_ms));
    // auto st_pri = target.set_scheduler_priority((uint8_t)m.priority);
    // std::printf("  [적용확인] %-13s: threshold=%d [%s], timeout=%ums [%s], priority=%d [%s]\n",
    //             m.name,
    //             m.threshold,  (st_thr == HAILO_SUCCESS ? "OK" : "실패"),
    //             m.timeout_ms, (st_to  == HAILO_SUCCESS ? "OK" : "실패"),
    //             m.priority,   (st_pri == HAILO_SUCCESS ? "OK" : "실패"));
    // if (st_thr != HAILO_SUCCESS || st_to != HAILO_SUCCESS || st_pri != HAILO_SUCCESS)
    //     std::printf("  [경고] %s 일부 파라미터 적용 실패! (thr=%d to=%d pri=%d) — H10은 설정값 "
    //                 "read-back이 불가하므로 이 로그가 유일한 근거다.\n",
    //                 m.name, (int)st_thr, (int)st_to, (int)st_pri);
    // ────────────────────────────────────────────────────────────────────────

    std::printf("  [스케줄러] %-13s: setter 미호출 — HailoRT 기본값 사용 "
                "(threshold=%d, timeout=%ums, priority=%d). batch=%d만 실제 적용.\n",
                m.name, m.threshold, m.timeout_ms, m.priority, m.batch);
}

// ═════════════════════════════════════════════════════════════════════════════
//  백엔드 A: VStreams  (Hailo-8L 코드와 1:1 대응)
// ═════════════════════════════════════════════════════════════════════════════

// ── 모델별: HEF 로드 -> configure(batch, power_mode) ──
// 성공한 모델은 network_groups/active_model_idx에 순서대로 채워진다.
inline hailo_status configure_models(std::unique_ptr<VDevice>& vdevice,
                                     std::vector<ModelConfig>& models,
                                     std::vector<std::shared_ptr<ConfiguredNetworkGroup>>& network_groups,
                                     std::vector<int>& active_model_idx)
{
    for (size_t i = 0; i < models.size(); i++) {
        auto& m = models[i];
        if (!m.active) continue;

        auto hef_exp = Hef::create(m.hef_path);
        if (!hef_exp) {
            std::cerr << m.name << " HEF 로드 실패: " << m.hef_path
                      << " status=" << hef_exp.status() << std::endl;
            return hef_exp.status();
        }
        auto hef = hef_exp.release();

        auto cfg_exp = vdevice->create_configure_params(hef);
        if (!cfg_exp) {
            std::cerr << m.name << " configure params 실패, status=" << cfg_exp.status() << std::endl;
            // [실측 2026-08-31, npu-rpi5] H10에서는 여기서 HAILO_NOT_IMPLEMENTED(7)이 난다.
            // HailoRT가 직접 이렇게 말한다:
            //   "Not supported. Did you try calling `create_configure_params` on H10?
            //    If so, use InferModel instead"
            // 즉 VStreams/ConfiguredNetworkGroup 경로는 H10에서 아예 제공되지 않는다.
            std::cerr << "  [H10] VStreams/ConfiguredNetworkGroup 경로는 Hailo-10H에서 지원되지 않습니다.\n"
                         "        --api infer 로 실행하세요 (InferModel Async + 슬롯 풀).\n"
                         "        예: ./infer_scheduler_h10 1 csv/results.csv --api infer --inflight 12\n";
            return cfg_exp.status();
        }
        auto cfg = cfg_exp.value();
        for (auto& ng_param : cfg) {
            ng_param.second.batch_size = m.batch;
#if POWER_MODE_ULTRA
            ng_param.second.power_mode = HAILO_POWER_MODE_ULTRA_PERFORMANCE;
#else
            ng_param.second.power_mode = HAILO_POWER_MODE_PERFORMANCE;
#endif
        }

        // [주의] hailort::Expected<T>는 이동 대입(operator=)이 delete 되어 있다
        //        (/usr/include/hailo/expected.hpp:393). 그래서 "ngs_exp = ..."로 재시도 결과를
        //        덮어쓸 수 없고, 폴백 경로마다 새 Expected를 따로 받아야 한다.
        std::shared_ptr<ConfiguredNetworkGroup> network_group;
        {
            auto ngs_exp = vdevice->configure(hef, cfg);
            if (ngs_exp) {
                network_group = ngs_exp.value()[0];
            } else {
                const hailo_status first_st = ngs_exp.status();
#if POWER_MODE_ULTRA
                // [H10 추가] H10 펌웨어가 ULTRA_PERFORMANCE를 거부하는 경우가 있어 폴백을 둔다.
                // 8L은 항상 ULTRA로 돌았으므로, 폴백이 걸리면 실험 노트에 반드시 남길 것.
                std::printf("  [경고] %s: ULTRA_PERFORMANCE configure 실패(status=%d) — "
                            "PERFORMANCE로 폴백해 재시도한다(8L과 조건이 달라짐, 결과 해석 시 주의).\n",
                            m.name, (int)first_st);
                auto cfg2_exp = vdevice->create_configure_params(hef);
                if (!cfg2_exp) {
                    std::cerr << m.name << " configure params(폴백) 실패, status="
                              << cfg2_exp.status() << std::endl;
                    return cfg2_exp.status();
                }
                auto cfg2 = cfg2_exp.value();
                for (auto& ng_param : cfg2) {
                    ng_param.second.batch_size = m.batch;
                    ng_param.second.power_mode = HAILO_POWER_MODE_PERFORMANCE;
                }
                auto ngs2_exp = vdevice->configure(hef, cfg2);
                if (!ngs2_exp) {
                    std::cerr << m.name << " configure 실패(ULTRA=" << (int)first_st
                              << ", PERFORMANCE=" << ngs2_exp.status() << ")" << std::endl;
                    std::cerr << "  [힌트] H10에서 VStreams/ConfiguredNetworkGroup 경로가 막혀 있으면 "
                                 "--api infer 백엔드로 실행할 것(InferModel Async + 슬롯 풀).\n";
                    return ngs2_exp.status();
                }
                network_group = ngs2_exp.value()[0];
#else
                std::cerr << m.name << " configure 실패, status=" << first_st << std::endl;
                std::cerr << "  [힌트] H10에서 VStreams/ConfiguredNetworkGroup 경로가 막혀 있으면 "
                             "--api infer 백엔드로 실행할 것(InferModel Async + 슬롯 풀).\n";
                return first_st;
#endif
            }
        }

        // 스케줄러 파라미터: setter는 주석 처리된 상태다(위 apply_scheduler_params 참고).
        apply_scheduler_params(*network_group, m);

        network_groups.push_back(network_group);
        active_model_idx.push_back((int)i);
    }
    return HAILO_SUCCESS;
}

// ── vstream 생성 (입력/출력 timeout을 크게 잡음) ──
// [중요] vstream 기본 timeout은 10초다. 스케줄러 경합으로 굶으면 write가 HAILO_TIMEOUT으로
// 실패하고 내부 파이프라인 스레드가 죽어 프레임이 유실된다. timeout을 크게 주어 방지한다. (8L과 동일)
//
// [2026-08-31] 출력 포맷은 항상 HAILO_FORMAT_TYPE_AUTO다. 후처리를 제거해 출력 버퍼를
// 해석하지 않으므로 FLOAT32/NHWC 강제 변환이 필요 없고, 그 변환 비용이 latency에 섞이지도 않는다.
inline hailo_status create_all_vstreams(
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>>& network_groups,
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>>& vstreams_per_ng)
{
    const uint32_t VSTREAM_TIMEOUT_MS = 300000;  // 5분 — starvation 대기 여유 (8L과 동일)

    for (size_t k = 0; k < network_groups.size(); k++) {
        auto& ng = network_groups[k];

        auto in_params  = ng->make_input_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                              VSTREAM_TIMEOUT_MS, VSTREAM_QUEUE_SIZE);
        auto out_params = ng->make_output_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                              VSTREAM_TIMEOUT_MS, VSTREAM_QUEUE_SIZE);

        if (!in_params)  { std::cerr << "input vstream params 실패, status="  << in_params.status()  << std::endl; return in_params.status(); }
        if (!out_params) { std::cerr << "output vstream params 실패, status=" << out_params.status() << std::endl; return out_params.status(); }

#if DEBUG_WRITE_TIMING
        // [진단용] HailoRT 공식 큐 크기 accumulator로 실질 큐 깊이를 직접 측정한다.
        for (auto& kv : in_params.value())
            kv.second.pipeline_elements_stats_flags = HAILO_PIPELINE_ELEM_STATS_MEASURE_QUEUE_SIZE;
        for (auto& kv : out_params.value())
            kv.second.pipeline_elements_stats_flags = HAILO_PIPELINE_ELEM_STATS_MEASURE_QUEUE_SIZE;
#endif

        auto in_vs  = ng->create_input_vstreams(in_params.value());
        auto out_vs = ng->create_output_vstreams(out_params.value());
        if (!in_vs)  { std::cerr << "input vstream 생성 실패, status="  << in_vs.status()  << std::endl; return in_vs.status(); }
        if (!out_vs) { std::cerr << "output vstream 생성 실패, status=" << out_vs.status() << std::endl; return out_vs.status(); }

        vstreams_per_ng.emplace_back(std::make_pair(in_vs.release(), out_vs.release()));
    }
    return HAILO_SUCCESS;
}

// ═════════════════════════════════════════════════════════════════════════════
//  백엔드 B: InferModel Async + 슬롯 풀
//  (8L의 "writer/reader 스레드 + 큐" 구조를 유지하되, in-flight 깊이를 명시적으로 통제)
// ═════════════════════════════════════════════════════════════════════════════

// 슬롯 = 프레임 1개가 in-flight 상태로 점유하는 자원 묶음.
// 슬롯이 1개뿐이면 run_async 직후 wait하는 것과 같아져 파이프라인이 사라진다 —
// 슬롯 수는 모델별로 다르게 들어온다(infer_scheduler_h10.cpp의 L8L 기준표).
// 8L 실측 L: det 11.0~11.3 / seg 9.2~10.4 / pose 9.1~10.2 — 전 모델 공통값을 쓰면 안 된다.
struct InferSlot {
    std::shared_ptr<ConfiguredInferModel::Bindings> bindings;
    std::shared_ptr<uint8_t> in_buf;
    size_t in_size = 0;
    std::vector<std::shared_ptr<uint8_t>> out_bufs;   // 수신용 — 내용은 해석하지 않는다(후처리 제거)
};

struct InferCtx {
    std::shared_ptr<InferModel> model;
    std::shared_ptr<ConfiguredInferModel> configured;
    std::vector<InferSlot> slots;
    std::string input_name;
    std::vector<std::string> output_names;
    int model_idx = -1;   // models[] 인덱스 (0=Det, 1=Seg, 2=Pose)
};

// ── InferModel 준비: create_infer_model -> set_batch_size -> configure -> 슬롯 N개 ──
// [8L 대비] configure_models()+create_all_vstreams()가 하던 일을 한 함수에서 처리한다.
// create_infer_model/configure/create_bindings는 동시성이 검증된 경로가 아니므로
// 메인 스레드에서 직렬로만 호출한다(실제 동시 실행은 워커 스레드의 run_async에서만 일어남).
// 출력 포맷은 지정하지 않는다(= HailoRT 기본/AUTO). 후처리를 제거했으므로 FLOAT32 변환 불필요.
inline hailo_status prepare_infer_ctx(VDevice& vdevice, const ModelConfig& m, int model_idx,
                                      int inflight, InferCtx& ctx)
{
    auto model_exp = vdevice.create_infer_model(m.hef_path);
    if (!model_exp) {
        std::cerr << m.name << " create_infer_model 실패: " << m.hef_path
                  << " status=" << model_exp.status() << std::endl;
        return model_exp.status();
    }
    ctx.model = model_exp.release();
    ctx.model_idx = model_idx;

    // batch는 configure() 전에 정해져야 한다 (8L이 configure_params.batch_size로 주던 것과 같은 자리).
    ctx.model->set_batch_size((uint16_t)m.batch);

    auto configured_exp = ctx.model->configure();
    if (!configured_exp) {
        std::cerr << m.name << " configure 실패, status=" << configured_exp.status() << std::endl;
        return configured_exp.status();
    }
    ctx.configured = std::make_shared<ConfiguredInferModel>(configured_exp.release());

    // 스케줄러 파라미터: setter는 주석 처리된 상태다(위 apply_scheduler_params 참고).
    apply_scheduler_params(*ctx.configured, m);

    ctx.input_name = ctx.model->get_input_names()[0];
    ctx.output_names = ctx.model->get_output_names();

    const size_t in_size = ctx.model->input(ctx.input_name)->get_frame_size();

    // 프레임당 PCIe 왕복 바이트를 로그로 남긴다 — H2D(입력) + D2H(출력 전부).
    // 실측 FPS를 곱하면 링크 사용량이 나오고, PCIe Gen3 x1(실효 ~850 MB/s)과 비교하면
    // 전송이 병목인지 아닌지 바로 판단할 수 있다.
    size_t total_out_bytes = 0;

    // ── 슬롯 풀 생성 ──
    // 슬롯마다 자기 입력/출력 버퍼와 Bindings를 갖는다. 이걸 공유하면 아직 디바이스가 읽고 있는
    // 버퍼를 다음 프레임이 덮어써서 조용히 틀린 결과가 나온다.
    ctx.slots.resize(inflight);
    for (int s = 0; s < inflight; s++) {
        auto bindings_exp = ctx.configured->create_bindings();
        if (!bindings_exp) {
            std::cerr << m.name << " create_bindings(slot " << s << ") 실패, status="
                      << bindings_exp.status() << std::endl;
            return bindings_exp.status();
        }
        ctx.slots[s].bindings = std::make_shared<ConfiguredInferModel::Bindings>(bindings_exp.release());

        ctx.slots[s].in_size = in_size;
        ctx.slots[s].in_buf = page_aligned_alloc(in_size);
        auto st = ctx.slots[s].bindings->input(ctx.input_name)
                      ->set_buffer(MemoryView(ctx.slots[s].in_buf.get(), in_size));
        if (st != HAILO_SUCCESS) {
            std::cerr << m.name << " input set_buffer(slot " << s << ") 실패, status=" << st << std::endl;
            return st;
        }

        for (const auto& name : ctx.output_names) {
            size_t out_size = ctx.model->output(name)->get_frame_size();
            if (s == 0) total_out_bytes += out_size;   // 슬롯 0에서만 합산(PCIe 대역폭 계산용)
            auto buf = page_aligned_alloc(out_size);
            auto ost = ctx.slots[s].bindings->output(name)->set_buffer(MemoryView(buf.get(), out_size));
            if (ost != HAILO_SUCCESS) {
                std::cerr << m.name << " output(" << name << ") set_buffer(slot " << s
                          << ") 실패, status=" << ost << std::endl;
                return ost;
            }
            ctx.slots[s].out_bufs.push_back(buf);
        }
    }

    std::printf("  [준비완료] %-13s: input_frame_size=%zu, outputs=%zu(%zu B), "
                "프레임당 PCIe 왕복=%.2f MB, inflight_slots=%d\n",
                m.name, in_size, ctx.output_names.size(), total_out_bytes,
                (double)(in_size + total_out_bytes) / 1e6, inflight);
    return HAILO_SUCCESS;
}
