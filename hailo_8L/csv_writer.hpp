#pragma once
// csv_writer.hpp — 실험 결과를 CSV에 append 저장 (Hailo-8L). 추론 단계에서 직접 측정
// 가능한 값만 채우고, HRTT/모니터 전용 값(npu_percent 등)은 NaN으로 남겨 이후
// 파이프라인(tools/monitoring/parse_npu_log.py, HRTT 파서)에서 같은 행을 찾아 채운다.
//
// 추론 중 직접 채우는 값:
//   run_id, use_*, batch, threshold_*, timeout_ms, priority_*   (실행 구성)
//   det/seg/pose_latency_ms                                     (이 코드가 측정한 end-to-end latency)
//   cpu_percent, mem_percent, voluntary/nonvoluntary_ctx_switches
//   run_time_s                                                  (추론 구간 실측 wall-time)
//   avg_preprocess_ms_*, postprocess_ms_*, total_time_ms_*      [2026-07-28] Hailo-8에서 포팅.
//                                                                 batch는 기존처럼 단일 컬럼 유지.

#include <string>
#include <sstream>
#include <fstream>
#include <mutex>
#include <cstdio>
#include <iostream>
#include <vector>

inline std::string dtos(double v) {           // 음수(-1)=미측정/비활성 → NaN
    if (v < 0) return "NaN";
    std::ostringstream os; os << v; return os.str();
}

inline void save_csv(const std::string& csv_path, int run_id,
              const std::vector<ModelConfig>& models,   // [0]=Det, [1]=Seg, [2]=Pose 고정 순서
              const ModelResult results[3],
              double cpu_percent, double mem_percent,
              long vol_ctx, long nonvol_ctx, double run_time_s)
{
    static const char* HEADER =
        "run_id,use_det,use_seg,use_pose,batch,"
        "threshold_det,threshold_seg,threshold_pose,timeout_ms,"
        "priority_det,priority_seg,priority_pose,"
        "det_latency_ms,seg_latency_ms,pose_latency_ms,"
        "cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches,"
        "npu_percent,switches_per_s,idle_time_pct,run_time_s,"
        "avg_fps_det,avg_latency_det,max_latency_det,activation_det,"
        "avg_fps_seg,avg_latency_seg,max_latency_seg,activation_seg,"
        "avg_fps_pose,avg_latency_pose,max_latency_pose,activation_pose,"
        "total_time_det_s,total_time_seg_s,total_time_pose_s,"          // 모델별 전체 추론시간(추론 중 실측)
        // [2026-07-28] Hailo-8에서 포팅: 전처리(모델별 독립 측정)/후처리/장당전체시간.
        "avg_preprocess_ms_det,avg_preprocess_ms_seg,avg_preprocess_ms_pose,"
        "postprocess_ms_det,postprocess_ms_seg,postprocess_ms_pose,"     // 장당 평균 후처리(디코딩+NMS)시간
        "total_time_ms_det,total_time_ms_seg,total_time_ms_pose,"        // 장당 전체시간 = 전처리+latency+후처리(측정된 경우만, 안 쟀으면 후처리항=0)
        // [2026-08-06] 후처리를 뺀 전체시간 = 전처리+latency만(후처리 측정 여부와 무관하게
        // 항상 이 두 값만 더함). total_time_ms_*와 나란히 두고 "후처리 포함 vs 제외"를
        // 같은 행에서 바로 비교할 수 있게 하기 위한 컬럼.
        "total_time_ms_nopp_det,total_time_ms_nopp_seg,total_time_ms_nopp_pose";

    // 파일이 없거나 비어 있으면 헤더부터 쓴다.
    bool need_header = true;
    {
        std::ifstream chk(csv_path);
        if (chk.good() && chk.peek() != std::ifstream::traits_type::eof())
            need_header = false;
    }
    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[CSV] 열기 실패: " << csv_path << std::endl;
        return;
    }
    if (need_header) f << HEADER << "\n";

    // 비활성 모델의 latency는 NaN(-1) 처리
    double det_lat  = models[0].active ? results[0].avg_latency_ms : -1;
    double seg_lat  = models[1].active ? results[1].avg_latency_ms : -1;
    double pose_lat = models[2].active ? results[2].avg_latency_ms : -1;
    double det_tot  = models[0].active ? results[0].total_time_s : -1;
    double seg_tot  = models[1].active ? results[1].total_time_s : -1;
    double pose_tot = models[2].active ? results[2].total_time_s : -1;
    double det_pp   = models[0].active ? results[0].avg_postprocess_ms : -1;
    double seg_pp   = models[1].active ? results[1].avg_postprocess_ms : -1;
    double pose_pp  = models[2].active ? results[2].avg_postprocess_ms : -1;
    double det_prep  = models[0].active ? results[0].avg_preprocess_ms : -1;
    double seg_prep  = models[1].active ? results[1].avg_preprocess_ms : -1;
    double pose_prep = models[2].active ? results[2].avg_preprocess_ms : -1;
    double det_ttl  = models[0].active ? results[0].avg_total_time_ms : -1;
    double seg_ttl  = models[1].active ? results[1].avg_total_time_ms : -1;
    double pose_ttl = models[2].active ? results[2].avg_total_time_ms : -1;
    // 후처리 제외 전체시간 = 전처리 + latency만(후처리 측정 여부와 무관하게 항상 이렇게 계산).
    // 둘 중 하나라도 없으면(-1) 전체를 NaN 처리.
    auto nopp_total = [](double prep, double lat) {
        return (prep >= 0 && lat >= 0) ? (prep + lat) : -1;
    };
    double det_nopp  = models[0].active ? nopp_total(det_prep, det_lat)  : -1;
    double seg_nopp  = models[1].active ? nopp_total(seg_prep, seg_lat)  : -1;
    double pose_nopp = models[2].active ? nopp_total(pose_prep, pose_lat) : -1;

    std::ostringstream row;
    row << run_id << ','
        << (models[0].active ? 1 : 0) << ','
        << (models[1].active ? 1 : 0) << ','
        << (models[2].active ? 1 : 0) << ','
        << models[0].batch << ','                                  // batch (8L 실험 설계상 모델 통일값, 단일 컬럼 유지)
        << models[0].threshold << ',' << models[1].threshold << ',' << models[2].threshold << ','
        << models[0].timeout_ms << ','                             // timeout_ms (실험에서 모델 통일)
        << models[0].priority << ',' << models[1].priority << ',' << models[2].priority << ','
        << dtos(det_lat) << ',' << dtos(seg_lat) << ',' << dtos(pose_lat) << ','
        << dtos(cpu_percent) << ',' << dtos(mem_percent) << ','
        << vol_ctx << ',' << nonvol_ctx << ','
        // ↓ 여기서부터 HRTT/모니터 단계에서 채울 값 → NaN
        << "NaN" << ','                                            // npu_percent
        << "NaN" << ','                                            // switches_per_s
        << "NaN" << ','                                            // idle_time_pct
        << dtos(run_time_s) << ','                                 // run_time_s (실측)
        << "NaN,NaN,NaN,NaN,"                                      // det: avg_fps,avg_latency,max_latency,activation
        << "NaN,NaN,NaN,NaN,"                                      // seg
        << "NaN,NaN,NaN,NaN,"                                      // pose
        // ↓ 모델별 전체 추론시간 (추론 중 실측, 비활성=NaN)
        << dtos(det_tot) << ',' << dtos(seg_tot) << ',' << dtos(pose_tot) << ','
        << dtos(det_prep) << ',' << dtos(seg_prep) << ',' << dtos(pose_prep) << ','
        << dtos(det_pp) << ',' << dtos(seg_pp) << ',' << dtos(pose_pp) << ','
        << dtos(det_ttl) << ',' << dtos(seg_ttl) << ',' << dtos(pose_ttl) << ','
        << dtos(det_nopp) << ',' << dtos(seg_nopp) << ',' << dtos(pose_nopp);
    f << row.str() << "\n";
    f.close();

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[CSV] 저장: %s (run_id=%d, 추론 측정값 기록, HRTT값은 NaN)\n",
                csv_path.c_str(), run_id);
}
