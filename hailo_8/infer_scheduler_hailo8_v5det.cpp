/**
 * infer_scheduler_hailo8_v5det.cpp
 * ------------------------------------------------------------------------
 * [2026-08-06 신규] infer_scheduler_hailo8.cpp의 사본 - Detection만 YOLOv5
 * NPU측 후처리(nms_core) HEF로 교체하고, Segmentation/Pose는 기존과 동일하게
 * 호스트 CPU 후처리(YOLOv8-seg/pose, decode_seg/decode_pose)로 그대로 둔 버전.
 *
 * "Detection만 NPU에서 돌리고 나머지(Seg/Pose)는 CPU로" 라는 요청에 대응 - 근거는
 * PROJECT_SUMMARY.md 6번 항목 / 이 대화에서 재확인한 대로 Segmentation/Pose는 Hailo DFC의
 * nms_postprocess() 대상 자체가 아니라(nn_core/auto 옵션 없음) CPU 외 선택지가 없음.
 * Detection(YOLOv5)만 유일하게 NPU측 후처리가 가능함.
 *
 * [원본과의 차이]
 *   - DET_HEF: yolov8s.hef(engine=cpu) -> yolov5xs_wo_spp_nms_core.hef(engine=nn_core/auto)
 *   - DET_IMG_SIZE: 640 -> 512 (이 HEF의 입력 크기가 512x512x3이라 다름. SEG/POSE는 640 그대로)
 *   - BATCH_*, THRESHOLD_*, TIMEOUT_*_MS, PRIORITY_*: 전부 사용자 지정 기본값
 *     (batch=1, threshold=1, timeout=0ms, priority=0)으로 통일 - 원본은 BATCH_DET=4,
 *     BATCH_SEG=8, BATCH_POSE=2, PRIORITY_*=15였음. 스케줄러 파라미터 자체의 효과를
 *     보려는 실험이 아니라 "NPU 후처리 도입이 미치는 영향"이 목적이라 셋 다 동일한
 *     기준선(default_workload 실험 관례와 동일)으로 맞춤.
 *   - decode_det()/model_setup.hpp/model_runner.hpp 등 로직은 전혀 안 건드림 - DET도
 *     기존과 동일하게 NMS-by-class 버퍼를 "파싱"만 한다(NMS 연산 자체는 이제 호스트가
 *     아니라 칩에서 끝나고 나온 결과라는 점만 다름). SEG/POSE 로직도 완전히 동일.
 *
 * [주의] 원본 infer_scheduler_hailo8.cpp는 그대로 두었다 - hailo_8/scripts/*.sh 가 그
 * 파일을 sed로 직접 편집하므로, 이 파일을 건드리면 기존 자동화가 깨진다. 이 파일은
 * 완전히 별도 빌드 산출물(별도 실행 파일)이다.
 *
 * HEF 준비 (RPi, 아직 없으면):
 *   wget -P ~/hailo_cpp_test/resources/ \
 *     https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov5xs_wo_spp_nms_core.hef
 *   hailortcli parse-hef ~/hailo_cpp_test/resources/yolov5xs_wo_spp_nms_core.hef
 *
 * 빌드:
 *   g++ infer_scheduler_hailo8_v5det.cpp -o infer_scheduler_hailo8_v5det -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
 * 실행:
 *   ./infer_scheduler_hailo8_v5det [run_id] [csv_path]
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

#include "postprocess_hailo8.hpp"

using namespace hailort;

// ========================= 파라미터 (사용자 지정 기본값, 세 모델 동일) =========================
#define BATCH_DET       1
#define BATCH_SEG       1
#define BATCH_POSE      1

#define THRESHOLD_DET   1
#define THRESHOLD_SEG   1
#define THRESHOLD_POSE  1

#define TIMEOUT_DET_MS   0
#define TIMEOUT_SEG_MS   0
#define TIMEOUT_POSE_MS  0

#define PRIORITY_DET    0
#define PRIORITY_SEG    0
#define PRIORITY_POSE   0

#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1

#define ENABLE_POSTPROCESS  1
#define INPUT_FPS       0
#define NUM_IMAGES      0

#define DET_IMG_SIZE   512   // yolov5xs_wo_spp_nms_core 입력 크기 (SEG/POSE는 640 그대로 - model_types.hpp 기본값)
// =====================================================================================

// HEF 경로 - DET만 YOLOv5 NPU-postprocess 변종, SEG/POSE는 원본과 동일(CPU 후처리).
#define DET_HEF  "/home/rpi4/hailo_cpp_test/resources/yolov5xs_wo_spp_nms_core.hef"
#define SEG_HEF  "/home/rpi4/hailo_cpp_test/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/rpi4/hailo_cpp_test/resources/yolov8s_pose.hef"

#define IMG_DIR  "/home/rpi4/hailo_cpp_test/datasets/sampled_val2017/"

std::mutex print_mutex;

#include "model_types.hpp"
#include "sys_monitor.hpp"
#include "image_utils.hpp"
#include "output_classify.hpp"
#include "model_setup.hpp"
#include "model_runner.hpp"
#include "csv_writer.hpp"

int main(int argc, char* argv[])
{
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;
    std::string csv_path = (argc > 2) ? argv[2] : "";

    pid_t my_pid = getpid();
    std::printf("PID: %d, Run ID: %d\n", my_pid, run_id);
    std::printf("Detection = YOLOv5 nms_core(NPU 후처리, %dx%d) / Segmentation,Pose = YOLOv8(CPU 후처리, 640x640)\n", DET_IMG_SIZE, DET_IMG_SIZE);

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
        {DET_HEF,  "Detection-YOLOv5-NPU", PRIORITY_DET,  THRESHOLD_DET,  TIMEOUT_DET_MS,  BATCH_DET,  (bool)USE_DET,  ModelKind::DET,  DET_IMG_SIZE},
        {SEG_HEF,  "Segmentation",         PRIORITY_SEG,  THRESHOLD_SEG,  TIMEOUT_SEG_MS,  BATCH_SEG,  (bool)USE_SEG,  ModelKind::SEG},
        {POSE_HEF, "Pose",                 PRIORITY_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, BATCH_POSE, (bool)USE_POSE, ModelKind::POSE},
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

    ModelResult results[3];  // index: Detection=0, Segmentation=1, Pose=2
    std::vector<std::thread> threads;
    for (size_t k = 0; k < vstreams_per_ng.size(); k++) {
        int mi = active_model_idx[k];
        threads.emplace_back(run_model_async, models[mi].name, models[mi].kind,
            std::ref(vstreams_per_ng[k].first), std::ref(vstreams_per_ng[k].second),
            std::cref(out_meta_per_ng[k]),
            std::cref(images), std::ref(results[mi]), models[mi].img_size);
    }
    for (auto& t : threads) t.join();

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

    std::printf("\n========== 실험 결과 (Run ID: %d) ==========\n", run_id);
    if (USE_DET)  std::printf("Detection(YOLOv5-NPU): 전처리=%.2fms, latency=%.2fms, 후처리(파싱)=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d, img_size=%d\n",
                              results[0].avg_preprocess_ms, results[0].avg_latency_ms, results[0].avg_postprocess_ms, results[0].avg_total_time_ms,
                              results[0].frame_count, BATCH_DET, THRESHOLD_DET, (uint32_t)TIMEOUT_DET_MS, PRIORITY_DET, DET_IMG_SIZE);
    if (USE_SEG)  std::printf("Segmentation(CPU)    : 전처리=%.2fms, latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[1].avg_preprocess_ms, results[1].avg_latency_ms, results[1].avg_postprocess_ms, results[1].avg_total_time_ms,
                              results[1].frame_count, BATCH_SEG, THRESHOLD_SEG, (uint32_t)TIMEOUT_SEG_MS, PRIORITY_SEG);
    if (USE_POSE) std::printf("Pose(CPU)            : 전처리=%.2fms, latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[2].avg_preprocess_ms, results[2].avg_latency_ms, results[2].avg_postprocess_ms, results[2].avg_total_time_ms,
                              results[2].frame_count, BATCH_POSE, THRESHOLD_POSE, (uint32_t)TIMEOUT_POSE_MS, PRIORITY_POSE);
    std::printf("CPU: %.2f%%, MEM: %.2f%%, Ctx Switch(vol/nonvol): %ld/%ld\n", final_cpu, final_mem, vol_ctx, nonvol_ctx);
    std::printf("================================================\n");

    if (!csv_path.empty())
        save_csv(csv_path, run_id, models, results,
                 final_cpu, final_mem, vol_ctx, nonvol_ctx, run_time_s);

    return HAILO_SUCCESS;
}
