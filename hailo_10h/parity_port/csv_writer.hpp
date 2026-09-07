#pragma once
// csv_writer.hpp — 실험 결과를 CSV에 append 저장 (Hailo-10H 이식본).
//
// ┌──────────────────────────────────────────────────────────────────────────┐
// │ [스키마 계약] 아래 HEADER 문자열은 hailo_8L/csv_writer.hpp의 HEADER와      │
// │ 바이트 단위로 동일하다(49컬럼). 컬럼을 추가·삭제·재배열하지 말 것.        │
// │ 8L 결과 CSV와 그대로 concat/비교되며, 기존 파이썬 스크립트                │
// │ (fill_hrtt_columns*.py, make_avg_csv.py, parse_npu_log.py)가 위치 기반으로 │
// │ 접근하므로 한 칸만 밀려도 전부 깨진다.                                    │
// │ H10 전용 값이 필요하면 CSV_EXTENSION_COLUMNS=1로 "맨 뒤에만" 붙일 것.      │
// └──────────────────────────────────────────────────────────────────────────┘
//
// ── 값의 출처 구분 (H10) ─────────────────────────────────────────────────────
//  [실측]      run_id, use_*, batch, det/seg/pose_latency_ms, cpu_percent, mem_percent,
//              ctx_switches, run_time_s, total_time_*_s, avg_preprocess_ms_*,
//              total_time_ms_*, total_time_ms_nopp_*
//  [기본값]    threshold_*, timeout_ms, priority_* — [2026-08-31] 스케줄러 setter를 주석
//              처리했으므로 여기에는 HailoRT 기본값(1 / 0ms / 16)이 들어간다. 실제 적용값과
//              같아야 하므로 infer_scheduler_h10.cpp의 #define도 그 값으로 맞춰져 있다.
//              batch만 실제로 적용된 값이다.
//  [영구 NaN]  postprocess_ms_det/seg/pose — [2026-08-31] 후처리를 코드에서 제거했다.
//              컬럼은 8L 스키마 유지를 위해 그대로 두고 값만 NaN으로 쓴다. 이 때문에
//              total_time_ms_*(전처리+latency+후처리)와 total_time_ms_nopp_*(전처리+latency)가
//              같은 값이 된다 — 8L의 계산 규약을 그대로 따른 결과다.
//  [디바이스]  npu_percent — 8L은 HAILO_MONITOR=1 + parse_npu_log.py로 사후 기입했고,
//              H10은 /tmp/hmon_files가 생기지 않으므로 hailortcli monitor를 파싱해 채운다.
//              평균 정의(NNC>0 샘플만)는 8L parse_npu_log.py와 동일하게 유지했다.
//  [호스트대체] avg_fps_*, max_latency_* — 8L에서는 HRTT가 채우던 값. H10에는 HRTT가 없어
//              호스트 실측으로 채운다(FILL_HOST_DERIVED=0으로 끄면 8L 런타임 출력과 동일하게 NaN).
//  [영구 NaN]  switches_per_s, idle_time_pct, activation_det/seg/pose, avg_latency_det/seg/pose
//              → H10에서 원리적으로 얻을 수 없다(HRTT_ON_HAILO10H.md §7). 컬럼은 유지하고
//                값만 "NaN"으로 쓴다. avg_latency_*는 det_latency_ms와 정의가 겹쳐서,
//                호스트 값으로 채우면 "HRTT 유래"와 "호스트 실측"이 한 파일에서 구분 불가능해지므로
//                의도적으로 비운다.

#include "model_types.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// 8L과 동일 규약: 음수(-1) = 미측정/비활성 → NaN.
// 문자열은 반드시 "NaN"으로 쓴다(빈 칸이 아니라). pandas.read_csv가 기본 결측으로 파싱하고,
// 8L 파일과 바이트 비교도 가능하다.
inline std::string dtos(double v) {
    if (v < 0) return "NaN";
    std::ostringstream os; os << v; return os.str();
}

// HRTT 전용 / H10에서 얻을 수 없는 값을 쓰는 자리임을 코드에서 드러내기 위한 상수.
// 값이 아니라 "출처"를 표시하는 용도 — 나중에 누가 0이나 -1로 바꿔 넣는 사고를 막는다.
static const char* const HRTT_ONLY = "NaN";   // H10에서 원리적으로 측정 불가
static const char* const NOT_MEASURED = "NaN";  // 이 실행 조건에서는 측정하지 않음

inline void save_csv(const std::string& csv_path, int run_id,
                     const std::vector<ModelConfig>& models,   // [0]=Det, [1]=Seg, [2]=Pose 고정 순서
                     const ModelResult results[3],
                     double cpu_percent, double mem_percent,
                     long vol_ctx, long nonvol_ctx, double run_time_s,
                     double npu_percent = -1.0,                 // [H10] hailortcli monitor 평균. 없으면 -1 -> NaN
                     const int* inflight_depth = nullptr,       // [H10 확장] 모델별 D 3개. nullptr -> NaN
                     const char* backend = "")                  // [H10 확장] "vstreams" / "infer"
{
    // ─── 8L csv_writer.hpp HEADER 원문 (49컬럼). 절대 수정 금지. ───
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
        "total_time_det_s,total_time_seg_s,total_time_pose_s,"
        "avg_preprocess_ms_det,avg_preprocess_ms_seg,avg_preprocess_ms_pose,"
        "postprocess_ms_det,postprocess_ms_seg,postprocess_ms_pose,"
        "total_time_ms_det,total_time_ms_seg,total_time_ms_pose,"
        "total_time_ms_nopp_det,total_time_ms_nopp_seg,total_time_ms_nopp_pose";

#if CSV_EXTENSION_COLUMNS
    // 확장 컬럼은 "맨 뒤"에만 붙인다. df.iloc[:, :49]로 자르면 8L과 그대로 비교된다.
    static const char* HEADER_EXT =
        ",backend,inflight_det,inflight_seg,inflight_pose,"
        "n_images_det,n_images_seg,n_images_pose,"
        "service_latency_ms_det,service_latency_ms_seg,service_latency_ms_pose,"
        "queue_wait_ms_det,queue_wait_ms_seg,queue_wait_ms_pose";
#endif

    // 파일이 없거나 비어 있으면 헤더부터 쓴다. (8L과 동일)
    bool need_header = true;
    {
        std::ifstream chk(csv_path);
        if (chk.good() && chk.peek() != std::ifstream::traits_type::eof())
            need_header = false;
    }

    // 기존 파일에 이어 붙일 때 헤더가 다르면 멈춘다 — 예전 스키마 파일에 새 행이 섞이는 사고 방지.
    if (!need_header) {
        std::ifstream chk(csv_path);
        std::string first;
        if (std::getline(chk, first)) {
            if (!first.empty() && first.back() == '\r') first.pop_back();
            std::string expect = HEADER;
#if CSV_EXTENSION_COLUMNS
            expect += HEADER_EXT;
#endif
            if (first != expect) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "[CSV] 헤더 불일치로 중단: " << csv_path << "\n"
                          << "      (다른 컬럼 구성의 파일입니다. 다른 파일명을 쓰거나 파일을 옮기세요)"
                          << std::endl;
                return;
            }
        }
    }

    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[CSV] 열기 실패: " << csv_path << std::endl;
        return;
    }
    if (need_header) {
        f << HEADER;
#if CSV_EXTENSION_COLUMNS
        f << HEADER_EXT;
#endif
        f << "\n";
    }

    // 비활성 모델의 값은 전부 -1(-> NaN) 처리. (8L과 동일)
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

    // 후처리 제외 전체시간 = 전처리 + latency만. 둘 중 하나라도 없으면 NaN. (8L과 동일)
    auto nopp_total = [](double prep, double lat) {
        return (prep >= 0 && lat >= 0) ? (prep + lat) : -1;
    };
    double det_nopp  = models[0].active ? nopp_total(det_prep,  det_lat)  : -1;
    double seg_nopp  = models[1].active ? nopp_total(seg_prep,  seg_lat)  : -1;
    double pose_nopp = models[2].active ? nopp_total(pose_prep, pose_lat) : -1;

    // [호스트 대체] 8L에서 HRTT가 채우던 avg_fps_* / max_latency_*.
#if FILL_HOST_DERIVED
    std::string fps_s[3], maxlat_s[3];
    for (int i = 0; i < 3; i++) {
        fps_s[i]    = models[i].active ? dtos(results[i].fps)            : std::string(NOT_MEASURED);
        maxlat_s[i] = models[i].active ? dtos(results[i].max_latency_ms) : std::string(NOT_MEASURED);
    }
#else
    // 8L 런타임 출력과 완전히 동일하게(전부 NaN) 남기고 싶을 때.
    std::string fps_s[3]    = {HRTT_ONLY, HRTT_ONLY, HRTT_ONLY};
    std::string maxlat_s[3] = {HRTT_ONLY, HRTT_ONLY, HRTT_ONLY};
#endif

    std::ostringstream row;
    row << run_id << ','
        << (models[0].active ? 1 : 0) << ','
        << (models[1].active ? 1 : 0) << ','
        << (models[2].active ? 1 : 0) << ','
        << models[0].batch << ','                                  // batch (실험 설계상 모델 통일값, 단일 컬럼 유지)
        << models[0].threshold << ',' << models[1].threshold << ',' << models[2].threshold << ','
        << models[0].timeout_ms << ','                             // timeout_ms (실험에서 모델 통일)
        << models[0].priority << ',' << models[1].priority << ',' << models[2].priority << ','
        << dtos(det_lat) << ',' << dtos(seg_lat) << ',' << dtos(pose_lat) << ','
        << dtos(cpu_percent) << ',' << dtos(mem_percent) << ','
        << vol_ctx << ',' << nonvol_ctx << ','
        << dtos(npu_percent) << ','                                // [디바이스] hailortcli monitor
        << HRTT_ONLY << ','                                        // switches_per_s   — H10 측정 불가
        << HRTT_ONLY << ','                                        // idle_time_pct    — H10 측정 불가
        << dtos(run_time_s) << ','
        // det: avg_fps, avg_latency, max_latency, activation
        << fps_s[0] << ',' << HRTT_ONLY << ',' << maxlat_s[0] << ',' << HRTT_ONLY << ','
        << fps_s[1] << ',' << HRTT_ONLY << ',' << maxlat_s[1] << ',' << HRTT_ONLY << ','
        << fps_s[2] << ',' << HRTT_ONLY << ',' << maxlat_s[2] << ',' << HRTT_ONLY << ','
        << dtos(det_tot)  << ',' << dtos(seg_tot)  << ',' << dtos(pose_tot)  << ','
        << dtos(det_prep) << ',' << dtos(seg_prep) << ',' << dtos(pose_prep) << ','
        << dtos(det_pp)   << ',' << dtos(seg_pp)   << ',' << dtos(pose_pp)   << ','
        << dtos(det_ttl)  << ',' << dtos(seg_ttl)  << ',' << dtos(pose_ttl)  << ','
        << dtos(det_nopp) << ',' << dtos(seg_nopp) << ',' << dtos(pose_nopp);

#if CSV_EXTENSION_COLUMNS
    row << ',' << backend;
    for (int i = 0; i < 3; i++)
        row << ',' << ((inflight_depth && models[i].active) ? std::to_string(inflight_depth[i])
                                                            : std::string(NOT_MEASURED));
    for (int i = 0; i < 3; i++)
        row << ',' << (models[i].active ? std::to_string(results[i].frame_count) : std::string(NOT_MEASURED));
    for (int i = 0; i < 3; i++)
        row << ',' << (models[i].active ? dtos(results[i].avg_service_ms) : std::string(NOT_MEASURED));
    for (int i = 0; i < 3; i++)
        row << ',' << (models[i].active ? dtos(results[i].avg_queue_wait_ms) : std::string(NOT_MEASURED));
#else
    (void)inflight_depth; (void)backend;
#endif

    f << row.str() << "\n";
    f.close();

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[CSV] 저장: %s (run_id=%d, 컬럼 49개 = 8L 스키마 동일, HRTT 전용값은 NaN)\n",
                csv_path.c_str(), run_id);
}
