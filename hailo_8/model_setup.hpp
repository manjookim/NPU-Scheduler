#pragma once
// model_setup.hpp — VDevice에 각 모델(HEF) 로드+configure, 스케줄러 파라미터 설정,
// 그리고 vstream 생성까지 담당 (Hailo-8). main()에서 순서대로 호출되는 두 함수로 구성.
// classify_outputs()는 output_classify.hpp에서 온다.

#include "hailo/hailort.hpp"
#include <vector>
#include <memory>
#include <cstdio>
#include <iostream>

// ── 모델별: HEF 로드 -> configure(batch) -> 스케줄러 파라미터(threshold/timeout/priority) 설정 ──
// 성공한 모델은 network_groups/active_model_idx에 순서대로 채워진다.
// 반환값이 HAILO_SUCCESS가 아니면 main()은 그 값으로 그대로 종료해야 한다(원래 코드의 조기 return과 동일).
inline hailo_status configure_models(std::unique_ptr<VDevice>& vdevice,
                              std::vector<ModelConfig>& models,
                              std::vector<std::shared_ptr<ConfiguredNetworkGroup>>& network_groups,
                              std::vector<int>& active_model_idx)
{
    for (size_t i = 0; i < models.size(); i++) {
        auto& m = models[i];
        if (!m.active) continue;

        auto hef_exp = Hef::create(m.hef_path);
        if (!hef_exp) { std::cerr << m.name << " HEF 로드 실패" << std::endl; return hef_exp.status(); }
        auto hef = hef_exp.release();

        auto cfg_exp = vdevice->create_configure_params(hef);
        if (!cfg_exp) { std::cerr << m.name << " configure params 실패" << std::endl; return cfg_exp.status(); }
        auto cfg = cfg_exp.value();
        for (auto& ng_param : cfg) {
            ng_param.second.batch_size = m.batch;
            ng_param.second.power_mode = HAILO_POWER_MODE_ULTRA_PERFORMANCE;
        }

        auto ngs_exp = vdevice->configure(hef, cfg);
        if (!ngs_exp) { std::cerr << m.name << " configure 실패" << std::endl; return ngs_exp.status(); }
        auto network_group = ngs_exp.value()[0];

        // 실기 확인된 제약: threshold는 batch_size 이하여야 함 (모델 정의부 주석 참고).
        // 어길 경우 set_scheduler_threshold가 실패하며 threshold는 기본값(1)으로 남는다.
        if (m.threshold > m.batch)
            std::printf("  [경고] %s: threshold(%d) > batch(%d) — set_scheduler_threshold가 실패할 것으로 예상됨. "
                        "THRESHOLD_* <= BATCH_*로 맞출 것.\n", m.name, m.threshold, m.batch);

        // 스케줄러 파라미터 설정 (network_group.hpp 공식 시그니처와 동일)
        auto st_thr = network_group->set_scheduler_threshold((uint32_t)m.threshold);
        auto st_to  = network_group->set_scheduler_timeout(std::chrono::milliseconds(m.timeout_ms));
        auto st_pri = network_group->set_scheduler_priority((uint8_t)m.priority);
        std::printf("  [적용확인] %-13s: batch=%d, threshold=%d [%s], timeout=%ums [%s], priority=%d [%s]\n",
            m.name, m.batch,
            m.threshold,  (st_thr == HAILO_SUCCESS ? "OK" : "실패"),
            m.timeout_ms, (st_to  == HAILO_SUCCESS ? "OK" : "실패"),
            m.priority,   (st_pri == HAILO_SUCCESS ? "OK" : "실패"));
        if (st_thr != HAILO_SUCCESS || st_to != HAILO_SUCCESS || st_pri != HAILO_SUCCESS)
            std::printf("  [경고] %s 일부 파라미터 적용 실패! (thr=%d to=%d pri=%d)\n",
                        m.name, (int)st_thr, (int)st_to, (int)st_pri);

        network_groups.push_back(network_group);
        active_model_idx.push_back((int)i);
    }
    return HAILO_SUCCESS;
}

// ── vstream 생성 (입력/출력 timeout을 크게 잡음) ──
// [중요] vstream 기본 timeout은 10초다. 우선순위가 낮은 모델은 높은 모델이 끝날 때까지
// 10초 넘게 굶을 수 있는데, 그러면 write가 HAILO_TIMEOUT으로 실패하고 내부 파이프라인
// 스레드가 죽어 프레임이 유실된다(재시도로도 복구 불가). timeout을 크게 주어 방지한다.
// (공식 API: ConfiguredNetworkGroup::make_input/output_vstream_params + create_input/output_vstreams)
inline hailo_status create_all_vstreams(
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>>& network_groups,
    const std::vector<ModelConfig>& models,
    const std::vector<int>& active_model_idx,
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>>& vstreams_per_ng,
    std::vector<std::vector<OutMeta>>& out_meta_per_ng)
{
    const uint32_t VSTREAM_TIMEOUT_MS = 300000;  // 5분 — starvation 대기 여유
    for (size_t k = 0; k < network_groups.size(); k++) {
        auto& ng = network_groups[k];
        ModelKind kind = models[active_model_idx[k]].kind;

        auto in_params  = ng->make_input_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                              VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        // [2026-07-26 변경] 세 모델 모두 FLOAT32로 통일(조교님 참고 파이썬 코드와 동일하게).
        // Detection도 decode_det()가 NMS-by-class 버퍼를 float 배열로 가정하고 파싱하므로
        // AUTO에 맡기지 않고 명시적으로 FLOAT32를 요청해야 레이아웃이 보장된다.
        // Segmentation/Pose는 raw tensor라서 order도 NHWC로 추가 고정한다(parse-hef가
        // 일부 출력을 FCR로 표시했으므로 AUTO에 맡기지 않음).
        hailo_format_type_t out_fmt_type = HAILO_FORMAT_TYPE_FLOAT32;
        auto out_params = ng->make_output_vstream_params(false, out_fmt_type,
                              VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!in_params)  { std::cerr << "input vstream params 실패, status="  << in_params.status()  << std::endl; return in_params.status(); }
        if (!out_params) { std::cerr << "output vstream params 실패, status=" << out_params.status() << std::endl; return out_params.status(); }

        if (kind != ModelKind::DET) {
            for (auto& kv : out_params.value())
                kv.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        }

        auto in_vs  = ng->create_input_vstreams(in_params.value());
        auto out_vs = ng->create_output_vstreams(out_params.value());
        if (!in_vs)  { std::cerr << "input vstream 생성 실패, status="  << in_vs.status()  << std::endl; return in_vs.status(); }
        if (!out_vs) { std::cerr << "output vstream 생성 실패, status=" << out_vs.status() << std::endl; return out_vs.status(); }

        auto out_vs_val = out_vs.release();
        out_meta_per_ng.push_back(classify_outputs(kind, out_vs_val));
        vstreams_per_ng.emplace_back(std::make_pair(in_vs.release(), std::move(out_vs_val)));
    }
    return HAILO_SUCCESS;
}
