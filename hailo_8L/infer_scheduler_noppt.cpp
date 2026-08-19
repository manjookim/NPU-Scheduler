/**
 * infer_scheduler_noppt.cpp  ("no post-process timing")
 * ------------------------------------------------------------------------
 * infer_scheduler.cpp의 "후처리 시간을 아예 측정하지 않는" 별도 버전.
 *
 * [2026-08-06] 조교님 요청: ENABLE_POSTPROCESS 매크로로 후처리 디코딩을 끄는 방식은
 * 후처리 시간이 0에 가깝게 찍히도록 만들 뿐, pp_t0/pp_t1 타이머 코드 자체는 여전히
 * 남아있는 구조였다. 조교님이 "아예 전체 후처리 시간 측정하는 것을 없앤 코드"를
 * 요청하여, 매크로 토글이 아니라 완전히 별도 파일로 분리했다.
 *
 * infer_scheduler.cpp와의 차이는 정확히 이 두 가지뿐이다:
 *   1) model_runner.hpp 대신 model_runner_noppt.hpp를 include — 후처리 디코딩 호출
 *      (decode_det/pose/seg)과 그 타이머(pp_t0/pp_t1/pp_total_ms/pp_count) 코드 자체가
 *      통째로 삭제되어 있음. result.avg_postprocess_ms는 아예 대입되지 않아 ModelResult
 *      기본값(-1)로 남고, csv_writer.hpp의 기존 관례에 따라 CSV에는 NaN으로 기록된다
 *      ("이 실행에서는 후처리 시간을 측정하지 않았다"는 의미가 데이터에도 명확히 남음).
 *   2) ENABLE_POSTPROCESS 매크로 자체를 제거 — 더 이상 쓰이지 않으므로.
 * 그 외 파라미터 #define 블록, HEF 경로, main() 오케스트레이션, CSV 스키마는
 * infer_scheduler.cpp와 100% 동일하다(analysis 스크립트/컬럼 호환성 유지 목적).
 *
 * 빌드 (RPi):
 *   g++ infer_scheduler_noppt.cpp -o infer_scheduler_noppt -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
 *
 * 실행:
 *   ./infer_scheduler_noppt [run_id] [csv_path]
 * ------------------------------------------------------------------------
 */

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <algorithm>
#include <unistd.h>
#include <map>
#include <utility>
#include <memory>

#include "postprocess_8l.hpp"  // decode_det/pose/seg는 이 버전에서 호출되지 않지만, 타입 선언은
                                // 참조하지 않으므로 include 자체는 무해함(빌드 호환성 유지 목적).

using namespace hailort;

// ========================= 파라미터 설정 (모델별로 다르게) =========================
// [주의] 이 블록의 #define 값들은 hailo_8L/scripts/*.sh 가 sed로 직접 편집한다 —
// 매크로 이름/줄 형식을 바꾸면 자동화 스크립트가 깨지니 그대로 유지할 것.
// (infer_scheduler.cpp와 동일한 블록. ENABLE_POSTPROCESS만 빠져 있음 — 이 파일은
//  후처리 자체를 호출하지 않으므로 그 매크로가 존재하지 않는다.)

#define BATCH_DET       4
#define BATCH_SEG       8
#define BATCH_POSE      2

#define THRESHOLD_DET   1
#define THRESHOLD_SEG   1
#define THRESHOLD_POSE  1

#define TIMEOUT_DET_MS   0
#define TIMEOUT_SEG_MS   0
#define TIMEOUT_POSE_MS  0

#define PRIORITY_DET    15
#define PRIORITY_SEG    15
#define PRIORITY_POSE   15

#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1

// [진단용, 2026-07-28] 1=batch_size 커지면 vstream 큐도 커지는지 확인하는 write() 블로킹시간
// 계측(처음 40프레임만 출력). 평소엔 0으로 둘 것.
#define DEBUG_WRITE_TIMING  0

// 입력 속도 제한 (모델당 초당 프레임 수). 0 = 제한 없음.
#define INPUT_FPS       0

// 사용할 검증 이미지 수 (0 = IMG_DIR의 전체 이미지 사용)
#define NUM_IMAGES      600
// =====================================================================================

// HEF 경로 (Raspberry Pi 5, hailo-rpi5-examples 리소스)
#define DET_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"
#define SEG_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef"

// 입력 데이터셋 경로: 조교 제공 sampled_val2017 (RPi에 이미 전송 완료).
#define IMG_DIR  "/home/rpi1/datasets/sampled_val2017/"

std::mutex print_mutex;

#include "model_types.hpp"
#include "sys_monitor.hpp"
#include "image_utils.hpp"
#include "output_classify.hpp"
#include "model_setup.hpp"
#include "model_runner_noppt.hpp"   // <- infer_scheduler.cpp와 유일하게 다른 include
#include "csv_writer.hpp"

// ========================= main =========================
// infer_scheduler.cpp의 main()과 완전히 동일한 오케스트레이션. avg_postprocess_ms가
// 항상 -1(미측정)로 남기 때문에 아래 avg_total_time_ms 계산은 자동으로
// "전처리 + latency"만 반영한다(원본 코드 그대로, 수정 불필요).

int main(int argc, char* argv[])
{
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;
    std::string csv_path = (argc > 2) ? argv[2] : "";

    pid_t my_pid = getpid();
    std::printf("PID: %d, Run ID: %d (후처리 시간 미측정 버전)\n", my_pid, run_id);

    hailo_vdevice_params_t vdevice_params;
    hailo_init_vdevice_params(&vdevice_params);
    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    auto vdevice_exp = VDevice::create(vdevice_params);
    if (!vdevice_exp) {
        std::cerr << "VDevice 생성 실패, status=" << vdevice_exp.status() << std::endl;
        return (int)vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();
    std::cout << "VDevice 생성 성공!" << std::endl;

    std::vector<ModelConfig> models = {
        {DET_HEF,  "Detection",    PRIORITY_DET,  THRESHOLD_DET,  TIMEOUT_DET_MS,  BATCH_DET,  (bool)USE_DET,  ModelKind::DET},
        {SEG_HEF,  "Segmentation", PRIORITY_SEG,  THRESHOLD_SEG,  TIMEOUT_SEG_MS,  BATCH_SEG,  (bool)USE_SEG,  ModelKind::SEG},
        {POSE_HEF, "Pose",         PRIORITY_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, BATCH_POSE, (bool)USE_POSE, ModelKind::POSE},
    };

    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> network_groups;
    std::vector<int> active_model_idx;
    hailo_status cfg_status = configure_models(vdevice, models, network_groups, active_model_idx);
    if (cfg_status != HAILO_SUCCESS) return (int)cfg_status;

    if (network_groups.empty()) {
        std::cerr << "활성화된 모델이 없습니다 (USE_DET/USE_SEG/USE_POSE 확인)" << std::endl;
        return 1;
    }

    std::vector<std::string> images = get_image_files(IMG_DIR);
    if (images.empty()) {
        std::cerr << "[경고] IMG_DIR(" << IMG_DIR << ")에서 이미지를 찾지 못함. "
                  << "경로를 확인하고 #define IMG_DIR을 수정할 것." << std::endl;
        return 1;
    }
    if (NUM_IMAGES > 0 && images.size() > (size_t)NUM_IMAGES)
        images.resize(NUM_IMAGES);
    std::printf("사용 이미지 수: %zu장 (경로: %s)\n\n", images.size(), IMG_DIR);

    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_ng;
    std::vector<std::vector<OutMeta>> out_meta_per_ng;
    hailo_status vs_status = create_all_vstreams(network_groups, models, active_model_idx,
                                                  vstreams_per_ng, out_meta_per_ng);
    if (vs_status != HAILO_SUCCESS) return (int)vs_status;

    CpuStats cpu_start = read_cpu_stats();
    double t_run_start = now_ms();

    ModelResult results[3];
    std::vector<std::thread> threads;
    for (size_t k = 0; k < vstreams_per_ng.size(); k++) {
        int mi = active_model_idx[k];
        threads.emplace_back(run_model_async, models[mi].name, models[mi].kind,
            std::ref(vstreams_per_ng[k].first), std::ref(vstreams_per_ng[k].second),
            std::cref(out_meta_per_ng[k]),
            std::cref(images), std::ref(results[mi]));
    }
    for (auto& t : threads) t.join();

    // 장당 전체시간(전처리-추론-후처리) = 전처리 + latency + 후처리(이 버전은 항상 0,
    // avg_postprocess_ms가 -1로 남아있기 때문).
    for (int i = 0; i < 3; i++) {
        if (models[i].active && results[i].avg_latency_ms >= 0) {
            double prep = (results[i].avg_preprocess_ms >= 0) ? results[i].avg_preprocess_ms : 0.0;
            double pp = (results[i].avg_postprocess_ms >= 0) ? results[i].avg_postprocess_ms : 0.0;
            results[i].avg_total_time_ms = prep + results[i].avg_latency_ms + pp;
        }
    }

    double run_time_s = (now_ms() - t_run_start) / 1000.0;
    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();

    long vol_ctx = 0, nonvol_ctx = 0;
    for (auto& r : results) { vol_ctx += r.vol_ctx; nonvol_ctx += r.nonvol_ctx; }

    std::printf("\n========== 실험 결과 (Run ID: %d, 후처리 시간 미측정) ==========\n", run_id);
    if (USE_DET)  std::printf("Detection    : 전처리=%.2fms, latency=%.2fms, 후처리=(미측정), 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[0].avg_preprocess_ms, results[0].avg_latency_ms, results[0].avg_total_time_ms,
                              results[0].frame_count, BATCH_DET, THRESHOLD_DET, (uint32_t)TIMEOUT_DET_MS, PRIORITY_DET);
    if (USE_SEG)  std::printf("Segmentation : 전처리=%.2fms, latency=%.2fms, 후처리=(미측정), 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[1].avg_preprocess_ms, results[1].avg_latency_ms, results[1].avg_total_time_ms,
                              results[1].frame_count, BATCH_SEG, THRESHOLD_SEG, (uint32_t)TIMEOUT_SEG_MS, PRIORITY_SEG);
    if (USE_POSE) std::printf("Pose         : 전처리=%.2fms, latency=%.2fms, 후처리=(미측정), 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[2].avg_preprocess_ms, results[2].avg_latency_ms, results[2].avg_total_time_ms,
                              results[2].frame_count, BATCH_POSE, THRESHOLD_POSE, (uint32_t)TIMEOUT_POSE_MS, PRIORITY_POSE);
    std::printf("CPU: %.2f%%, MEM: %.2f%%, Ctx Switch(vol/nonvol): %ld/%ld\n", final_cpu, final_mem, vol_ctx, nonvol_ctx);
    std::printf("================================================\n");
    std::printf("HRTT 트레이스를 PC/WSL에서 `hailo runtime-profiler <파일>.hrtt`로 변환한 뒤,\n"
                "core_op_set_value 이벤트에서 위 [적용확인] 값과 실제 적용값이 일치하는지 확인할 것.\n");

    if (!csv_path.empty())
        save_csv(csv_path, run_id, models, results,
                 final_cpu, final_mem, vol_ctx, nonvol_ctx, run_time_s);

    return HAILO_SUCCESS;
}
