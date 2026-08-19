/**
 * infer_yolov5_hailo8.cpp
 * ------------------------------------------------------------------------
 * [2026-08-06 신규] YOLOv5 "NPU(neural core)측 후처리" 단독 벤치마크 (Hailo-8, rpi4).
 *
 * 배경 / 목적
 * -----------
 * PROJECT_SUMMARY.md §6에서 조사한 대로, Hailo Dataflow Compiler의 nms_postprocess()는
 * 모델 아키텍처별로 NMS를 어디서 돌릴지(엔진) 선택할 수 있다:
 *   - engine=cpu      : NMS 전체를 호스트 CPU에서 수행 (우리 기존 yolov8s.hef가 이 모드 —
 *                        hailo_model_zoo의 yolov8*.alls에 engine=cpu가 박혀 있음. 앱 입장에선
 *                        "on-chip NMS 결과"처럼 보이지만 실제 IoU 연산은 HailoRT가 호스트 CPU
 *                        스레드에서 수행한다.)
 *   - engine=nn_core  : NMS 전체를 칩 위 neural core(NPU 안의 전용 후처리 클러스터)에서 수행.
 *                        공식 문서(Hailo DFC User Guide, Model Script 레퍼런스) 기준
 *                        YOLOv5 / SSD / Centernet만 지원.
 *   - engine=auto     : YOLOv5 전용. bbox 디코딩 + score threshold는 neural core에서,
 *                        IoU 필터링만 CPU에서.
 * 즉 "YOLOv5는 NPU에서 후처리 가능"이 성립하려면 hailo_model_zoo가 기본 배포하는
 * engine=cpu HEF가 아니라, engine=nn_core/auto로 컴파일된 HEF가 필요하다 — 이건 우리 코드가
 * 아니라 컴파일 시점(Dataflow Compiler)에 결정되는 값이다.
 *
 * 사용 HEF: yolov5xs_wo_spp_nms_core.hef
 *   - 출처: 공식 hailo_model_zoo(github.com/hailo-ai/hailo_model_zoo) 공개 배포 HEF.
 *     docs/public_models/HAILO8/HAILO8_object_detection.rst에 등록되어 있고,
 *     실 다운로드 링크(Hailo 공식 S3)로 존재를 확인함:
 *     https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov5xs_wo_spp_nms_core.hef
 *   - "_nms_core" 접미사 모델(예: yolov6n_0.2.1_nms_core.alls 실물 확인)은 hailo_model_zoo에서
 *     nms_postprocess(..., engine=nn_core)로 컴파일된 것들에 붙는 명명 규칙. CHANGELOG.rst는
 *     이 계열을 "contains bbox decoding and confidence thresholding on Hailo-8"로 설명함 —
 *     즉 최소 bbox 디코딩+score threshold는 확실히 neural core에서 돈다(auto 모드와 동일 특징).
 *     [주의] hailo_model_zoo가 nn_core와 auto 둘 다에 "_nms_core" 접미사를 재사용해온 이력이 있어
 *     이 특정 HEF가 둘 중 정확히 어느 쪽인지는 100% 단정하지 못했다 — 실기에서
 *     `hailortcli parse-hef yolov5xs_wo_spp_nms_core.hef`로 postprocess 메타데이터를 확인하거나,
 *     조교님께 재확인 요청할 것. 어느 쪽이든 기존 yolov8s(engine=cpu)보다 후처리 연산의 상당
 *     부분이 호스트 CPU에서 NPU 칩으로 옮겨간다는 사실 자체는 동일하게 성립.
 *   - 입력 크기가 512x512x3이다(다른 세 모델은 640x640x3) — hailo_model_zoo가 공개 배포하는
 *     YOLOv5 "_nms_core" HEF가 이 xs(extra-small) 변종 하나뿐이라 다른 입력 크기 선택지가 없음.
 *     model_types.hpp::ModelConfig::img_size로 처리(모델 코드/model_runner.hpp가 이제 img_size를
 *     파라미터로 받으므로 letterbox/decode_det 둘 다 자동으로 512 기준으로 동작함).
 *
 * 파라미터: 사용자 지정 기본값 그대로 사용 — batch=1, threshold=1, timeout=0ms, priority=0.
 *          (스케줄러 파라미터 자체의 효과를 보려는 실험이 아니라 "NPU 후처리 자체의 타이밍"이
 *           목적이므로 모델 1개만 단독 실행 — 큐 경쟁/스케줄링 영향 배제.)
 *
 * 측정: 기존 3모델 프레임워크와 동일하게 model_runner.hpp::run_model_async()가
 *   전처리(imread+letterbox) / latency(enqueue~dequeue) / 후처리(decode_det 파싱) /
 *   장당 전체시간을 프레임별로 측정한다. Detection의 "후처리" 측정값은 여기서는 순수
 *   버퍼 파싱 비용만 잡는다는 점에 유의(NMS 연산 자체는 애초에 이 시점 이전에
 *   호스트 CPU가 아니라 칩에서 이미 끝난 상태로 나옴 — engine=cpu였던 yolov8s와 비교할 때
 *   "decode_det 파싱 시간"이 아니라 "이 모델을 돌리는 동안 호스트 CPU 총 사용률
 *   (cpu_percent 컬럼)"이 더 의미 있는 비교 지표가 될 가능성이 높음 — CSV에 같이 기록해둠).
 *
 * 재사용: model_types/sys_monitor/image_utils/output_classify/model_setup/model_runner/
 *   postprocess_hailo8 헤더를 기존 3모델 파일과 동일하게 그대로 재사용한다(로직 변경 없음,
 *   model_runner.hpp에 img_size 파라미터만 추가됨 — 기본값 640이라 기존 3모델 파일 동작은
 *   그대로임). csv_writer.hpp는 Det/Seg/Pose 3슬롯 고정 스키마라 재사용하지 않고, 이 파일
 *   전용의 단순 1행 CSV를 아래에 직접 작성한다.
 *
 * 빌드 (RPi, 기존 infer_scheduler_hailo8.cpp와 동일 방식):
 *   g++ infer_yolov5_hailo8.cpp -o infer_yolov5_hailo8 -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
 *
 * 실행:
 *   ./infer_yolov5_hailo8 [run_id] [csv_path]
 *
 * [2026-08-19 추가] "Det 단일모델, v8s(CPU 후처리) vs v5-nms_core(NPU 후처리), FPS=60,
 * 3회 반복 후 평균" 실험은 이 파일을 그대로 사용하고 scripts/run_det_v8s_vs_v5npu_fps60.sh가
 * USE_CPU_BASELINE_INSTEAD(0=v5/NPU, 1=v8s/CPU)와 INPUT_FPS를 sed로 자동 토글해 양쪽을
 * 순서대로 빌드/실행한다 — 로직 변경 없음, INPUT_FPS 기본값만 0→60으로 바꿈. 평균/xlsx는
 * scripts/make_avg_csv_det_v8s_vs_v5npu.py, scripts/build_xlsx_det_v8s_vs_v5npu.py 참고.
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

// ========================= 파라미터 (사용자 지정 기본값) =========================
// [주의] 자동화 스크립트(scripts/run_yolov5_nms_core.sh)가 필요시 sed로 이 블록을 편집할 수
// 있으니 매크로 이름/줄 형식을 유지할 것 (기존 infer_scheduler_hailo8.cpp 관례와 동일).
#define BATCH_YOLOV5      1
#define THRESHOLD_YOLOV5  1
#define TIMEOUT_YOLOV5_MS 0
#define PRIORITY_YOLOV5   0

#define ENABLE_POSTPROCESS  1   // decode_det() 파싱 비용까지 측정(=1). 0이면 파싱 스킵.
#define INPUT_FPS           60  // [2026-08-19 변경, 기존 0] 사용자 요청: NPU vs CPU 후처리
                                 // 비교 실험은 FPS=60 기준으로 통일. 모델 1개 단독 실행이라
                                 // 스케줄러 starvation 문제 자체는 없지만(경쟁 상대가 없음),
                                 // "후처리 담당 프로세서 차이만 빼고 나머지는 전부 동일 환경"
                                 // 이라는 실험 조건을 맞추기 위해 입력 속도를 고정한다 — 두
                                 // 조건(v8s/v5) 모두 이 스크립트가 같은 값을 적용함
                                 // (scripts/run_det_v8s_vs_v5npu_fps60.sh 참고).
#define NUM_IMAGES          0   // 0 = IMG_DIR 전체 사용

#define YOLOV5_IMG_SIZE   512   // yolov5xs_wo_spp_nms_core 모델 입력 크기(다른 세 모델은 640)
// =====================================================================================

// HEF 경로 — 기존 세 모델과 같은 디렉터리 관례. RPi에 아직 없으면 먼저 받아야 함:
//   wget -P ~/hailo_cpp_test/resources/ \
//     https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov5xs_wo_spp_nms_core.hef
//   hailortcli parse-hef ~/hailo_cpp_test/resources/yolov5xs_wo_spp_nms_core.hef   # Architecture: HAILO8 확인
#define YOLOV5_HEF "/home/rpi4/hailo_cpp_test/resources/yolov5xs_wo_spp_nms_core.hef"

// 비교 기준(선택): 기존 engine=cpu 경로인 yolov8s.hef와 같은 실행 조건(단독, 동일 파라미터)으로
// 돌리고 싶을 때 이 매크로를 1로 바꾸면 YOLOV5_HEF 대신 DET_HEF_CPU_BASELINE을 로드한다.
#define USE_CPU_BASELINE_INSTEAD 0
#define DET_HEF_CPU_BASELINE "/home/rpi4/hailo_cpp_test/resources/yolov8s.hef"

#define IMG_DIR  "/home/rpi4/hailo_cpp_test/datasets/sampled_val2017/"

std::mutex print_mutex;

#include "model_types.hpp"
#include "sys_monitor.hpp"
#include "image_utils.hpp"
#include "output_classify.hpp"
#include "model_setup.hpp"
#include "model_runner.hpp"

// 이 파일 전용 1행 CSV (Det/Seg/Pose 3슬롯을 가정하는 csv_writer.hpp와 스키마가 달라 별도 작성).
static void save_csv_single(const std::string& csv_path, int run_id, const ModelConfig& m,
                             const ModelResult& r, double cpu_percent, double mem_percent,
                             long vol_ctx, long nonvol_ctx, double run_time_s)
{
    static const char* HEADER =
        "run_id,hef_name,img_size,batch,threshold,timeout_ms,priority,"
        "frame_count,avg_preprocess_ms,avg_latency_ms,avg_postprocess_ms,avg_total_time_ms,"
        "total_time_s,run_time_s,cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches";

    bool need_header = true;
    { std::ifstream chk(csv_path); if (chk.good() && chk.peek() != std::ifstream::traits_type::eof()) need_header = false; }
    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) { std::cerr << "[CSV] 열기 실패: " << csv_path << std::endl; return; }
    if (need_header) f << HEADER << "\n";

    auto dtos = [](double v) { if (v < 0) return std::string("NaN"); std::ostringstream os; os << v; return os.str(); };

    f << run_id << ',' << m.name << ',' << m.img_size << ',' << m.batch << ',' << m.threshold << ','
      << m.timeout_ms << ',' << m.priority << ',' << r.frame_count << ','
      << dtos(r.avg_preprocess_ms) << ',' << dtos(r.avg_latency_ms) << ',' << dtos(r.avg_postprocess_ms) << ','
      << dtos(r.avg_total_time_ms) << ',' << dtos(r.total_time_s) << ',' << dtos(run_time_s) << ','
      << dtos(cpu_percent) << ',' << dtos(mem_percent) << ',' << vol_ctx << ',' << nonvol_ctx << "\n";
    f.close();
    std::printf("[CSV] 저장: %s (run_id=%d)\n", csv_path.c_str(), run_id);
}

int main(int argc, char* argv[])
{
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;
    std::string csv_path = (argc > 2) ? argv[2] : "";

    pid_t my_pid = getpid();
    std::printf("PID: %d, Run ID: %d\n", my_pid, run_id);
    std::printf("모델: %s\n", USE_CPU_BASELINE_INSTEAD ? "yolov8s (engine=cpu 기준선)" : "yolov5xs_wo_spp_nms_core (engine=nn_core/auto)");

    hailo_vdevice_params_t vdevice_params;
    hailo_init_vdevice_params(&vdevice_params);
    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;  // 모델 1개뿐이라 알고리즘 자체는 무관, 기존 파일과 설정만 통일
    auto vdevice_exp = VDevice::create(vdevice_params);
    if (!vdevice_exp) {
        std::cerr << "VDevice 생성 실패, status=" << vdevice_exp.status() << std::endl;
        return (int)vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();
    std::cout << "VDevice 생성 성공!" << std::endl;

    const char* hef_path = USE_CPU_BASELINE_INSTEAD ? DET_HEF_CPU_BASELINE : YOLOV5_HEF;
    int img_size = USE_CPU_BASELINE_INSTEAD ? 640 : YOLOV5_IMG_SIZE;
    const char* model_name = USE_CPU_BASELINE_INSTEAD ? "Detection-CPU-baseline" : "YOLOv5-NPU-postprocess";

    std::vector<ModelConfig> models = {
        {hef_path, model_name, PRIORITY_YOLOV5, THRESHOLD_YOLOV5, TIMEOUT_YOLOV5_MS, BATCH_YOLOV5, true, ModelKind::DET, img_size},
    };

    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> network_groups;
    std::vector<int> active_model_idx;
    hailo_status cfg_status = configure_models(vdevice, models, network_groups, active_model_idx);
    if (cfg_status != HAILO_SUCCESS) return (int)cfg_status;
    if (network_groups.empty()) { std::cerr << "모델 로드 실패" << std::endl; return 1; }

    std::vector<std::string> images = get_image_files(IMG_DIR);
    if (images.empty()) {
        std::cerr << "[경고] IMG_DIR(" << IMG_DIR << ")에서 이미지를 찾지 못함." << std::endl;
        return 1;
    }
    if (NUM_IMAGES > 0 && images.size() > (size_t)NUM_IMAGES) images.resize(NUM_IMAGES);
    std::printf("사용 이미지 수: %zu장 (경로: %s, img_size=%d)\n\n", images.size(), IMG_DIR, img_size);

    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_ng;
    std::vector<std::vector<OutMeta>> out_meta_per_ng;
    hailo_status vs_status = create_all_vstreams(network_groups, models, active_model_idx, vstreams_per_ng, out_meta_per_ng);
    if (vs_status != HAILO_SUCCESS) return (int)vs_status;

    CpuStats cpu_start = read_cpu_stats();
    double t_run_start = now_ms();

    ModelResult result;
    // 모델이 1개뿐이므로 run_model_async를 바로 호출(내부에서 writer/reader 스레드를 만들고 join까지 함).
    run_model_async(models[0].name, models[0].kind,
        vstreams_per_ng[0].first, vstreams_per_ng[0].second,
        out_meta_per_ng[0], images, result, models[0].img_size);

    if (result.avg_latency_ms >= 0) {
        double prep = (result.avg_preprocess_ms >= 0) ? result.avg_preprocess_ms : 0.0;
        double pp = (result.avg_postprocess_ms >= 0) ? result.avg_postprocess_ms : 0.0;
        result.avg_total_time_ms = prep + result.avg_latency_ms + pp;
    }

    double run_time_s = (now_ms() - t_run_start) / 1000.0;
    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();

    std::printf("\n========== 실험 결과 (Run ID: %d) ==========\n", run_id);
    std::printf("%s: 전처리=%.2fms, latency=%.2fms, 후처리(파싱)=%.2fms, 전체=%.2fms, %d장, "
                "batch=%d, threshold=%d, timeout=%ums, priority=%d, img_size=%d\n",
                model_name, result.avg_preprocess_ms, result.avg_latency_ms, result.avg_postprocess_ms,
                result.avg_total_time_ms, result.frame_count, BATCH_YOLOV5, THRESHOLD_YOLOV5,
                (uint32_t)TIMEOUT_YOLOV5_MS, PRIORITY_YOLOV5, img_size);
    std::printf("CPU: %.2f%%, MEM: %.2f%%, Ctx Switch(vol/nonvol): %ld/%ld, run_time=%.2fs\n",
                final_cpu, final_mem, result.vol_ctx, result.nonvol_ctx, run_time_s);
    std::printf("================================================\n");
    std::printf("참고: 여기서 '후처리'는 decode_det()의 NMS-by-class 버퍼 파싱 비용만 잡음 —\n"
                "      NMS(IoU 억제) 연산 자체가 호스트 CPU에서 도는지 칩에서 도는지는 이 숫자에\n"
                "      안 보인다. yolov8s(engine=cpu) 기준선과 비교하려면 위 CPU%% 컬럼과\n"
                "      HRTT 트레이스(core_op 이벤트)를 같이 볼 것.\n");

    if (!csv_path.empty())
        save_csv_single(csv_path, run_id, models[0], result, final_cpu, final_mem,
                         result.vol_ctx, result.nonvol_ctx, run_time_s);

    return HAILO_SUCCESS;
}
