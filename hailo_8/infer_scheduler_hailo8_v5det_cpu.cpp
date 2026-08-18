/**
 * infer_scheduler_hailo8_v5det_cpu.cpp
 * ------------------------------------------------------------------------
 * [2026-08-07 신규] infer_scheduler_hailo8_v5det.cpp(Detection=YOLOv5 NPU측
 * 후처리 버전)의 짝이 되는 "CPU측 후처리" 기준선. Detection을 같은 YOLOv5
 * 아키텍처·같은 입력 크기(512x512)의 CPU-engine HEF로 바꾼 것 외에는
 * v5det.cpp와 완전히 동일하다 — 즉 이 파일과 infer_scheduler_hailo8_v5det.cpp
 * 두 개를 같은 조건(batch=1/threshold=1/timeout=0/priority=0)으로 각각 실행해
 * CSV를 나란히 비교하면 "Detection 후처리(NMS)를 NPU에서 vs CPU에서 돌렸을 때"
 * 차이만 보는 통제 실험이 된다.
 *
 * [이번 대화에서 확정한 실험 범위]
 * 사용자 요청은 원래 "3개 모델을 전부 YOLOv5로 바꾸자"였으나, 공식 hailo_model_zoo
 * (github.com/hailo-ai/hailo_model_zoo) 재조사 결과:
 *   - Detection: YOLOv5 존재, NPU(nms_core/auto)·CPU(engine=cpu) 둘 다 준비 가능,
 *     출력 포맷이 기존 decode_det()가 파싱하는 HAILO_NMS_BY_CLASS 그대로라 로직 변경 불필요.
 *   - Segmentation: YOLOv5-seg(yolov5s_seg.hef 등)는 존재하지만 박스 디코딩이
 *     앵커 기반이라(YOLOv8의 DFL 기반 decode_seg()와 수학이 다름) 새 디코딩 로직을
 *     작성해야 하고 실기 없이는 검증도 불가능 — 이번 실험 범위에서 제외하기로 사용자와 합의.
 *   - Pose: YOLOv5 계열 자체가 공식 Model Zoo에 없음(포즈 추정은 YOLOv8부터 추가된 기능이라
 *     애초에 "YOLOv5-Pose" 공식 모델이 존재하지 않음) — 어떤 방법으로도 대체 불가.
 * 따라서 이 실험은 "Detection만 YOLOv5로 바꾸고, NPU 후처리 1회 + CPU 후처리 1회를
 * 비교"로 범위를 좁혔다. Segmentation/Pose는 두 실행 모두 기존 YOLOv8(CPU 후처리) 그대로.
 *
 * [infer_scheduler_hailo8_v5det.cpp와의 차이점 — 이것뿐임]
 *   - DET_HEF: yolov5xs_wo_spp_nms_core.hef(engine=nn_core/auto) -> yolov5xs_wo_spp.hef(engine=cpu)
 *     두 HEF는 hailo_model_zoo 공식 문서(docs/public_models/HAILO8/HAILO8_object_detection.rst)에
 *     나란히 등재된 "같은 아키텍처(yolov5xs_wo_spp)·같은 입력 크기(512x512x3)" 쌍이라
 *     NPU/CPU 후처리 차이만 격리해서 보기에 가장 적합한 조합이다(모델 크기가 다르면
 *     그 차이가 섞여 들어감).
 *   - Detection 모델 name 라벨: "Detection-YOLOv5-NPU" -> "Detection-YOLOv5-CPU"
 *   - 나머지(SEG_HEF/POSE_HEF, 파라미터 4종(batch/threshold/timeout/priority), decode_det()
 *     등 헤더 재사용, DET_IMG_SIZE=512)는 전부 동일.
 *
 * HEF 준비 (RPi, 아직 없으면):
 *   wget -P ~/hailo_cpp_test/resources/ \
 *     https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov5xs_wo_spp.hef
 *   hailortcli parse-hef ~/hailo_cpp_test/resources/yolov5xs_wo_spp.hef   # Architecture: HAILO8 확인
 *
 * 빌드:
 *   g++ infer_scheduler_hailo8_v5det_cpu.cpp -o infer_scheduler_hailo8_v5det_cpu -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
 * 실행:
 *   ./infer_scheduler_hailo8_v5det_cpu [run_id] [csv_path]
 *
 * [주의] 원본 infer_scheduler_hailo8.cpp / infer_scheduler_hailo8_v5det.cpp는 그대로 두었다 —
 * hailo_8/scripts/*.sh 가 원본 파일을 sed로 직접 편집하므로, 이 파일을 건드리면 기존
 * 자동화가 깨진다. 이 파일은 완전히 별도 빌드 산출물(별도 실행 파일)이다.
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

#define DET_IMG_SIZE   512   // yolov5xs_wo_spp 입력 크기 (SEG/POSE는 640 그대로 - model_types.hpp 기본값)
// =====================================================================================

// HEF 경로 - DET만 YOLOv5 CPU-postprocess 변종(engine=cpu), SEG/POSE는 원본과 동일(CPU 후처리).
#define DET_HEF  "/home/rpi4/hailo_cpp_test/resources/yolov5xs_wo_spp.hef"
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
    std::printf("Detection = YOLOv5 wo_spp(CPU 후처리, %dx%d) / Segmentation,Pose = YOLOv8(CPU 후처리, 640x640)\n", DET_IMG_SIZE, DET_IMG_SIZE);

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
        {DET_HEF,  "Detection-YOLOv5-CPU", PRIORITY_DET,  THRESHOLD_DET,  TIMEOUT_DET_MS,  BATCH_DET,  (bool)USE_DET,  ModelKind::DET,  DET_IMG_SIZE},
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
    if (USE_DET)  std::printf("Detection(YOLOv5-CPU): 전처리=%.2fms, latency=%.2fms, 후처리(디코딩+NMS)=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d, img_size=%d\n",
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
    std::printf("참고: infer_scheduler_hailo8_v5det.cpp(Detection NPU 후처리) 결과와 같은 run_id 개수/조건으로\n"
                "      비교하려면 이 CSV와 그쪽 CSV를 나란히 놓고 det_latency_ms / postprocess_ms_det /\n"
                "      total_time_ms_det, cpu_percent 컬럼을 대조할 것.\n");

    if (!csv_path.empty())
        save_csv(csv_path, run_id, models, results,
                 final_cpu, final_mem, vol_ctx, nonvol_ctx, run_time_s);

    return HAILO_SUCCESS;
}
