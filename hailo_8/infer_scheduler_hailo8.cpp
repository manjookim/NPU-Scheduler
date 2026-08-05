/**
 * infer_scheduler_hailo8.cpp
 * ------------------------------------------------------------------------
 * [Hailo-8 (진짜 8, RPi5 보드명 rpi4) 포트 — infer_scheduler.cpp(Hailo-8L/rpi1용)의
 *  HEF 경로/IMG_DIR만 Hailo-8 환경에 맞게 바꾼 사본. 스케줄러 로직은 동일함.]
 *
 * Hailo-8 (Raspberry Pi 5) 위에서 Detection / Segmentation / Pose 세 모델을
 * HailoRT Model Scheduler(ROUND_ROBIN)로 "동시에" 추론하면서, 모델별로
 *   priority / threshold / timeout / batch_size
 * 를 독립적으로 설정해 벤치마킹하기 위한 C++ 프로그램.
 *
 * [2026-08-02] 코드 정리(스파게티 코드 개선): 예전엔 전처리/후처리/CSV저장/스케줄러
 * 설정 등 거의 모든 로직이 main() 안에 몰려 있었는데, 기능별로 헤더 파일로 분리했다
 * (Hailo-8L 쪽과 동일한 구조로 맞춤).
 *   - model_types.hpp    : ModelKind/ModelConfig/ModelResult/OutRole/OutMeta (데이터 타입)
 *   - sys_monitor.hpp    : CPU/메모리/컨텍스트 스위치 측정
 *   - image_utils.hpp    : letterbox 전처리, 이미지 파일 목록, now_ms()
 *   - output_classify.hpp: output vstream 채널수 기반 role 분류
 *   - model_setup.hpp    : HEF 로드→configure→스케줄러 파라미터 설정, vstream 생성
 *   - model_runner.hpp   : 모델별 writer/reader 스레드(1장씩 전처리→추론→후처리)
 *   - csv_writer.hpp     : 결과 CSV 저장
 * 이 파일엔 실험 자동화 스크립트가 sed로 직접 편집하는 파라미터 #define 블록과 main()만
 * 남겼다 — 파일명과 #define 위치를 그대로 유지했기 때문에 hailo_8/scripts/*.sh 의
 * sed 편집 로직은 수정 없이 그대로 동작한다. 위 헤더들은 전부 이 파일에 #include 되어
 * 결국 하나의 번역단위(.cpp 1개)로 컴파일되므로, 빌드 명령어도 바뀌지 않는다.
 *
 * 입력 데이터: COCO val2017 샘플 이미지(IMG_DIR, 기본 최대 600장) — 더미 버퍼가 아닌
 *            실제 이미지를 읽어 letterbox 전처리 후 추론에 사용한다.
 *
 * 스케줄러 파라미터 적용 근거 (공식 문서/예제, github.com/hailo-ai/hailort, hailo8 브랜치):
 *   - hailort/libhailort/include/hailo/network_group.hpp
 *       set_scheduler_timeout(const std::chrono::milliseconds&, network_name="")
 *         -> 기본값 0ms. "적어도 한 번의 요청이 들어온 뒤" 이 시간이 지나면
 *            threshold 미달이어도 강제로 실행 자격을 얻는다.
 *       set_scheduler_threshold(uint32_t, network_name="")
 *         -> 기본값 1. 큐에 threshold개 요청이 쌓여야 실행 자격을 얻는다(단, timeout이
 *            먼저 지나면 그 전에도 자격을 얻음 — 위 timeout 설명 참고).
 *       set_scheduler_priority(uint8_t, network_name="")
 *         -> 기본값 HAILO_SCHEDULER_PRIORITY_NORMAL. 값이 클수록 우선.
 *       (주의: 세 함수 모두 "network_name 지정 시 특정 네트워크만 설정"은 아직 미지원 —
 *        네트워크그룹 전체 단위로만 적용됨. 본 코드는 모델당 네트워크그룹을 하나씩
 *        따로 configure하므로 문제 없음.)
 *   - hailort/libhailort/examples/cpp/switch_network_groups_example/switch_network_groups_example.cpp
 *       VDevice::create -> create_configure_params(hef) -> batch_size 설정 -> configure(hef)
 *       -> set_scheduler_timeout/threshold/priority -> VStreamsBuilder::create_vstreams(*ng, {}, FORMAT_TYPE)
 *     본 파일의 구조/시그니처는 위 공식 예제를 그대로 따른다.
 *
 * 참고(프로젝트 문서): PROJECT_HANDOFF.md, README.md, docs/setup.md, memory/findings.md
 *   - threshold 효과를 실제로 관측하려면 timeout > 0 이어야 하고(§7),
 *     입력을 한꺼번에 다 밀어넣지 말고 NPU 처리량 근처 속도로 흘려보내야
 *     큐가 상시 포화되지 않아 threshold/timeout이 의미를 가진다.
 *     -> 아래 INPUT_FPS 로 입력 속도를 제한할 수 있다 (0 = 제한 없음/최대속도).
 *
 * 빌드 (RPi):
 *   g++ infer_scheduler_hailo8.cpp -o infer_scheduler_hailo8 -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
 *
 * 실행:
 *   ./infer_scheduler_hailo8 [run_id] [csv_path]
 *
 * HRTT 트레이스를 PC/WSL에서 `hailo runtime-profiler <파일>.hrtt`로 변환한 HTML에서
 * `core_op_set_value` 이벤트로 실제 적용된 threshold/timeout/priority 값을 확인할 수 있다
 * (PROJECT_HANDOFF.md §6 참고. setter가 HAILO_SUCCESS를 반환해도 실제 반영은
 * HRTT로 재확인하는 것이 정확함).
 *
 * HRTT 트레이스:
 *   export HAILO_TRACE=scheduler
 *   export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
 *   export HAILO_TRACE_PATH=/path/to/traces
 *   export HAILO_MONITOR=1
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

// ========================= 파라미터 설정 (모델별로 다르게) =========================
// [실기 확인된 제약, 공식 문서에 명시 안 됨] threshold는 반드시 그 모델의 batch_size
// 이하여야 한다. 초과 시 set_scheduler_threshold가 HAILO_INVALID_ARGUMENT로 실패하고
// (HailoRT 로그: "Threshold must be equal or lower than the maximum batch size!"),
// 해당 모델은 threshold가 기본값(1)으로 남는다 — [적용확인] 로그의 [실패] 표시로 알 수 있음.
// 즉 아래 THRESHOLD_* <= BATCH_* 를 항상 지킬 것.
// [주의] 이 블록의 #define 값들은 hailo_8/scripts/*.sh 가 sed로 직접 편집한다 —
// 매크로 이름/줄 형식을 바꾸면 자동화 스크립트가 깨지니 그대로 유지할 것.

// 모델별 batch_size (네트워크그룹 입출력 큐 크기 — 공식 문서: 클수록 pre/post-process가
// 하드웨어 추론과 병렬화되어 다른 스케줄러 파라미터가 더 잘 작동함)
#define BATCH_DET       4
#define BATCH_SEG       8
#define BATCH_POSE      2

// threshold: 네트워크그룹이 스케줄될 "자격"을 얻기 위한 최소 누적 요청 수 (기본값 1)
// 반드시 threshold <= 같은 줄의 batch_size (위 제약 참고)
// [검증 완료] threshold(99) > batch(8)로 깨서 테스트한 결과, core_op_set_value 트레이스에
// 거부된 값은 실제로 안 남는 것을 확인함(verify_params.py로 (미적용!) 확인). 정상값으로 복구.
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
// 후처리 추가가 backpressure를 통해 latency 자체를 얼마나 밀어올리는지 같은 batch 조건에서
// A/B 비교하기 위한 스위치. 평소엔 1로 둘 것.
#define ENABLE_POSTPROCESS  1

// 입력 속도 제한 (모델당 초당 프레임 수). 0 = 제한 없음(최대 속도로 큐를 채움 -> 큐가
// 항상 포화 상태가 되어 threshold/timeout 효과가 거의 관측되지 않음, findings.md 참고).
// threshold/timeout 효과를 보고 싶다면 0보다 큰 값(예: NPU 처리량 근처)으로 설정할 것.
#define INPUT_FPS       0

// 사용할 검증 이미지 수 (0 = IMG_DIR의 전체 이미지 사용)
// [Hailo-8] 8L(NUM_IMAGES=600)과 달리 IMG_DIR(sampled_val2017, 673장) 전체를 사용하도록 변경.
#define NUM_IMAGES      0
// =====================================================================================

// HEF 경로 (Hailo-8, rpi4 보드: ~/hailo_cpp_test/resources/, Model Zoo v2.14.0/hailo8)
#define DET_HEF  "/home/rpi4/hailo_cpp_test/resources/yolov8s.hef"
#define SEG_HEF  "/home/rpi4/hailo_cpp_test/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/rpi4/hailo_cpp_test/resources/yolov8s_pose.hef"

// 입력 데이터셋 경로: 조교 제공 sampled_val2017 (RPi에 이미 전송 완료).
// RPi의 실제 저장 위치가 다르면 이 값만 수정 후 재컴파일할 것.
#define IMG_DIR  "/home/rpi4/hailo_cpp_test/datasets/sampled_val2017/"

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
    // [2026-07-28 변경] 조교님 요청: 전처리를 여기서 미리 한꺼번에 수행하지 않는다.
    // 예전엔 모든 모델이 공유하는 letterbox 결과를 여기서 한 번에 만들어뒀는데,
    // 이제는 파일 경로만 넘기고 각 모델의 writer 스레드가 프레임마다
    // "전처리(imread+letterbox) -> 추론 -> [reader 스레드에서] 후처리"를 1장씩 수행한다
    // (model_runner.hpp::run_model_async 참고). 모델마다 독립적으로 읽으므로 전처리도
    // 모델별로 따로 측정된다.
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
        threads.emplace_back(run_model_async, models[mi].name, models[mi].kind,
            std::ref(vstreams_per_ng[k].first), std::ref(vstreams_per_ng[k].second),
            std::cref(out_meta_per_ng[k]),
            std::cref(images), std::ref(results[mi]));
    }
    for (auto& t : threads) t.join();

    // 장당 전체시간(전처리-추론-후처리) = 이 모델 자신의 평균 전처리 + 평균 latency + 평균 후처리시간
    // ([2026-07-28] 전처리가 이제 모델별 독립 측정이라 공유값을 더하지 않음)
    for (int i = 0; i < 3; i++) {
        if (models[i].active && results[i].avg_latency_ms >= 0) {
            double prep = (results[i].avg_preprocess_ms >= 0) ? results[i].avg_preprocess_ms : 0.0;
            double pp = (results[i].avg_postprocess_ms >= 0) ? results[i].avg_postprocess_ms : 0.0;
            results[i].avg_total_time_ms = prep + results[i].avg_latency_ms + pp;
        }
    }

    double run_time_s = (now_ms() - t_run_start) / 1000.0;   // 추론 구간 실측 wall-time(초)
    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();

    long vol_ctx = 0, nonvol_ctx = 0;
    for (auto& r : results) { vol_ctx += r.vol_ctx; nonvol_ctx += r.nonvol_ctx; }

    std::printf("\n========== 실험 결과 (Run ID: %d) ==========\n", run_id);
    // [2026-07-28] 전처리가 모델별 독립 측정이라 공유 한 줄 대신 모델별 요약 줄에 포함시킴.
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
