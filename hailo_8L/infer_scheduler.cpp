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

#include "postprocess_8l.hpp"

using namespace hailort;

#define BATCH_DET       1
#define BATCH_SEG       1
#define BATCH_POSE      1

// threshold: 네트워크그룹이 스케줄될 "자격"을 얻기 위한 최소 누적 요청 수 (기본값 1)
// 반드시 threshold <= 같은 줄의 batch_size (위 제약 참고)
#define THRESHOLD_DET   1
#define THRESHOLD_SEG   1
#define THRESHOLD_POSE  1

// timeout(ms): threshold 미달이어도 최소 1프레임 + 이 시간이 지나면 강제로 실행 자격 부여 (기본값 0)
#define TIMEOUT_DET_MS   0
#define TIMEOUT_SEG_MS   0
#define TIMEOUT_POSE_MS  0

// priority: 0~31, 클수록 우선 (기본값 16=NORMAL). 스케줄 가능한 여러 모델 중 이 값이
// 가장 큰 모델부터 확인한다. 동일하면 Round-Robin.
#define PRIORITY_DET    15
#define PRIORITY_SEG    15
#define PRIORITY_POSE   15

// 어떤 모델을 이번 실행에 포함할지 (0/1)
#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1

// [진단용] 1=Pose/Seg 디코딩+NMS 수행(정상), 0=디코딩 스킵(후처리 시간 거의 0으로 고정).
#define ENABLE_POSTPROCESS  0

// [진단용, 2026-08-26] output vstream 포맷을 FLOAT32+NHWC로 강제할지 여부.
// ENABLE_POSTPROCESS와 별도로 독립 조절 가능하게 분리함 — "후처리는 끄되 포맷 변환 비용만
// 남겨서" 그 비용 자체의 영향을 따로 측정하는 실험용.
//   ENABLE_POSTPROCESS=0, FORCE_OUTPUT_FLOAT32=0 -> 조교님과 완전 동일 조건 (현재 이 상태)
//   ENABLE_POSTPROCESS=0, FORCE_OUTPUT_FLOAT32=1 -> 후처리 없음 + FLOAT32 변환 비용만 남김
//   ENABLE_POSTPROCESS=1, FORCE_OUTPUT_FLOAT32=1 -> 평소 정상 운영 조건(후처리 O, FLOAT32 필요)
#define FORCE_OUTPUT_FLOAT32  0

// [진단용, 2026-07-28] 1=batch_size 커지면 vstream 큐도 커지는지 확인하는 write() 블로킹시간
// 계측(처음 40프레임만 출력). 평소엔 0으로 둘 것 — 이 실험 때만 잠깐 1로 켬.
#define DEBUG_WRITE_TIMING  0

// 입력 속도 제한 (모델당 초당 프레임 수). 0 = 제한 없음(최대 속도로 큐를 채움 -> 큐가
// 항상 포화 상태가 되어 threshold/timeout 효과가 거의 관측되지 않음, findings.md 참고).
#define INPUT_FPS       0

// 사용할 검증 이미지 수 (0 = IMG_DIR의 전체 이미지 사용)
#define NUM_IMAGES      600
// =====================================================================================

// HEF 경로 (Raspberry Pi 5, hailo-rpi5-examples 리소스)
// [2026-08-26] 계정명 rpi1 -> npu-rpi1 변경에 따라 홈 디렉토리 경로도 변경됨(/home/npu-rpi1).
#define DET_HEF  "/home/npu-rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"
#define SEG_HEF  "/home/npu-rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/npu-rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef"

// 입력 데이터셋 경로: 조교 제공 sampled_val2017 (RPi에 이미 전송 완료).
// RPi의 실제 저장 위치가 다르면 이 값만 수정 후 재컴파일할 것.
#define IMG_DIR  "/home/npu-rpi1/datasets/sampled_val2017/"
std::mutex print_mutex;

#include "model_types.hpp"
#include "sys_monitor.hpp"
#include "image_utils.hpp"
#include "output_classify.hpp"
#include "model_setup.hpp"
#include "model_runner.hpp"
#include "csv_writer.hpp"

// ========================= main =========================
// 오케스트레이션만 담당: VDevice 생성 -> 모델 설정(model_setup.hpp) -> 이미지 목록 로드
// -> vstream 생성(model_setup.hpp) -> 모델별 스레드 실행(model_runner.hpp) -> 결과 요약
// 출력 -> CSV 저장(csv_writer.hpp). 실제 로직은 각 헤더로 분리되어 있다.

int main(int argc, char* argv[])
{
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;
    std::string csv_path = (argc > 2) ? argv[2] : "";   // argv[2] 있으면 CSV 저장

    pid_t my_pid = getpid();
    std::printf("PID: %d, Run ID: %d\n", my_pid, run_id);

    // ── VDevice 생성 (스케줄러 Round-Robin) ──
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

    // ── 모델별: HEF 로드 -> configure(batch) -> 스케줄러 파라미터 설정 (model_setup.hpp) ──
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> network_groups;
    std::vector<int> active_model_idx;  // network_groups[k] <-> models[active_model_idx[k]]
    hailo_status cfg_status = configure_models(vdevice, models, network_groups, active_model_idx);
    if (cfg_status != HAILO_SUCCESS) return (int)cfg_status;

    if (network_groups.empty()) {
        std::cerr << "활성화된 모델이 없습니다 (USE_DET/USE_SEG/USE_POSE 확인)" << std::endl;
        return 1;
    }

    // ── val2017 이미지 파일 목록만 로드 ──
    // [2026-07-28 변경, Hailo-8에서 포팅] 전처리를 여기서 미리 한꺼번에 수행하지 않는다.
    // 예전엔 모든 모델이 공유하는 letterbox 결과를 여기서 한 번에 만들어뒀는데,
    // 이제는 파일 경로만 넘기고 각 모델의 writer 스레드가 프레임마다
    // "전처리(imread+letterbox) -> 추론 -> [reader 스레드에서] 후처리"를 1장씩 수행한다.
    std::vector<std::string> images = get_image_files(IMG_DIR);
    if (images.empty()) {
        std::cerr << "[경고] IMG_DIR(" << IMG_DIR << ")에서 이미지를 찾지 못함. "
                  << "경로를 확인하고 #define IMG_DIR을 수정할 것." << std::endl;
        return 1;
    }
    if (NUM_IMAGES > 0 && images.size() > (size_t)NUM_IMAGES)
        images.resize(NUM_IMAGES);
    std::printf("사용 이미지 수: %zu장 (경로: %s)\n\n", images.size(), IMG_DIR);

    // ── vstream 생성 (model_setup.hpp) ──
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_ng;
    std::vector<std::vector<OutMeta>> out_meta_per_ng;  // vstreams_per_ng와 동일 인덱스로 정렬
    hailo_status vs_status = create_all_vstreams(network_groups, models, active_model_idx,
                                                  vstreams_per_ng, out_meta_per_ng);
    if (vs_status != HAILO_SUCCESS) return (int)vs_status;

    // ── 측정 시작 ──
    CpuStats cpu_start = read_cpu_stats();
    double t_run_start = now_ms();

    ModelResult results[3];  // index: Detection=0, Segmentation=1, Pose=2 (models 배열과 동일 순서)
    std::vector<std::thread> threads;
    for (size_t k = 0; k < vstreams_per_ng.size(); k++) {
        int mi = active_model_idx[k];
        // [수정] std::thread는 함수의 기본 인자를 모르므로 img_size를 명시적으로 넘겨야 함
        // (run_model_async의 img_size=640 기본값은 일반 함수 호출에서만 적용되고, std::thread
        // 생성자를 통한 간접 호출에서는 무시되어 컴파일 에러가 남).
        threads.emplace_back(run_model_async, models[mi].name, models[mi].kind,
            std::ref(vstreams_per_ng[k].first), std::ref(vstreams_per_ng[k].second),
            std::cref(out_meta_per_ng[k]),
            std::cref(images), std::ref(results[mi]), models[mi].img_size);
    }
    for (auto& t : threads) t.join();

    // [2026-08-26] join() 직후 바로 측정 — 뒤의 avg_total_time_ms 계산 루프가 끼어들기 전에
    // 타임스탬프를 찍어서 run_time_s(추론 구간 실측 wall-time)를 더 엄밀하게 잰다.
    double run_time_s = (now_ms() - t_run_start) / 1000.0;

    // 장당 전체시간(전처리-추론-후처리) = 이 모델 자신의 평균 전처리 + 평균 latency + 평균 후처리시간
    for (int i = 0; i < 3; i++) {
        if (models[i].active && results[i].avg_latency_ms >= 0) {
            double prep = (results[i].avg_preprocess_ms >= 0) ? results[i].avg_preprocess_ms : 0.0;
            double pp = (results[i].avg_postprocess_ms >= 0) ? results[i].avg_postprocess_ms : 0.0;
            results[i].avg_total_time_ms = prep + results[i].avg_latency_ms + pp;
        }
    }

    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();

    long vol_ctx = 0, nonvol_ctx = 0;
    for (auto& r : results) { vol_ctx += r.vol_ctx; nonvol_ctx += r.nonvol_ctx; }

    std::printf("\n========== 실험 결과 (Run ID: %d) ==========\n", run_id);
    if (USE_DET)  std::printf("Detection    : 전처리=%.2fms, latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[0].avg_preprocess_ms, results[0].avg_latency_ms, results[0].avg_postprocess_ms, results[0].avg_total_time_ms,
                              results[0].frame_count, BATCH_DET, THRESHOLD_DET, (uint32_t)TIMEOUT_DET_MS, PRIORITY_DET);
    if (USE_SEG)  std::printf("Segmentation : 전처리=%.2fms, latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[1].avg_preprocess_ms, results[1].avg_latency_ms, results[1].avg_postprocess_ms, results[1].avg_total_time_ms,
                              results[1].frame_count, BATCH_SEG, THRESHOLD_SEG, (uint32_t)TIMEOUT_SEG_MS, PRIORITY_SEG);
    if (USE_POSE) std::printf("Pose         : 전처리=%.2fms, latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[2].avg_preprocess_ms, results[2].avg_latency_ms, results[2].avg_postprocess_ms, results[2].avg_total_time_ms,
                              results[2].frame_count, BATCH_POSE, THRESHOLD_POSE, (uint32_t)TIMEOUT_POSE_MS, PRIORITY_POSE);
    std::printf("CPU: %.2f%%, MEM: %.2f%%, Ctx Switch(vol/nonvol): %ld/%ld\n", final_cpu, final_mem, vol_ctx, nonvol_ctx);
    std::printf("================================================\n");
    std::printf("HRTT 트레이스를 PC/WSL에서 `hailo runtime-profiler <파일>.hrtt`로 변환한 뒤,\n"
                "core_op_set_value 이벤트에서 위 [적용확인] 값과 실제 적용값이 일치하는지 확인할 것.\n");

    // ── CSV 저장 (argv[2]로 경로가 주어졌을 때만) ──
    // 추론 중 측정 가능한 값만 채우고, HRTT/모니터 전용 값은 NaN으로 남긴다.
    if (!csv_path.empty())
        save_csv(csv_path, run_id, models, results,
                 final_cpu, final_mem, vol_ctx, nonvol_ctx, run_time_s);

    return HAILO_SUCCESS;
}
