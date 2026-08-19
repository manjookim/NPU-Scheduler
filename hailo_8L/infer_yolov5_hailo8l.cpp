/**
 * infer_yolov5_hailo8l.cpp
 * ------------------------------------------------------------------------
 * [2026-08-19 신규] YOLOv5 "NPU(neural core)측 후처리" 단독 벤치마크 (Hailo-8L, rpi1).
 * hailo_8/infer_yolov5_hailo8.cpp(Hailo-8/rpi4용, 2026-08-06 작성)를 Hailo-8L로 그대로
 * 포팅한 파일 — 사용자가 "앞으로는 Hailo-8L에서 실험한다(Hailo-8은 안 함)"고 정정해서
 * 만들어짐. 로직은 100% 동일하고, HEF 경로/데이터셋 경로/include 헤더만 8L 관례로 교체.
 *
 * 배경 / 목적
 * -----------
 * PROJECT_SUMMARY.md §6에서 조사한 대로, Hailo Dataflow Compiler의 nms_postprocess()는
 * 모델 아키텍처별로 NMS를 어디서 돌릴지(엔진) 선택할 수 있다:
 *   - engine=cpu      : NMS 전체를 호스트 CPU에서 수행 (우리 기존 yolov8s_h8l.hef가 이 모드 —
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
 * 사용 HEF: yolov5xs_wo_spp_nms_core.hef (Hailo-8L 타겟 컴파일본)
 *   - 출처: 공식 hailo_model_zoo(github.com/hailo-ai/hailo_model_zoo) 공개 배포 HEF.
 *     docs/public_models/HAILO8L/HAILO8L_object_detection.rst에 yolov5xs_wo_spp_nms_core
 *     항목으로 등록되어 있음(512x512x3 입력, Hailo-8용과 동일 이름·해상도, hailo8l 타겟으로
 *     따로 컴파일된 버전) — 실 다운로드 링크(Hailo 공식 S3):
 *     https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/yolov5xs_wo_spp_nms_core.hef
 *   - [주의] Hailo-8용 HEF(.../hailo8/yolov5xs_wo_spp_nms_core.hef)와 이름이 같지만 타겟
 *     아키텍처가 다른 별개 파일이다 — 절대 8용 HEF를 8L 보드에서 그대로 쓰면 안 됨
 *     (`hailortcli parse-hef`의 Architecture 필드가 HAILO8L인지 실행 전 반드시 확인).
 *   - 공식 문서 기준 참고 성능(Hailo-8L, batch=1): yolov5xs_wo_spp(CPU 후처리)=206 FPS,
 *     yolov5xs_wo_spp_nms_core(NPU 후처리)=57.2 FPS — Hailo-8에서 이미 관측된 "NPU 후처리
 *     조건이 오히려 더 느림" 경향과 방향이 일치함(엔진 자체의 처리량 차이로 보임, 우리
 *     실측값과 다를 수 있으니 참고용으로만 사용할 것).
 *   - "_nms_core" 접미사 모델은 hailo_model_zoo에서 nms_postprocess(..., engine=nn_core)로
 *     컴파일된 것들에 붙는 명명 규칙. CHANGELOG.rst는 이 계열을 "contains bbox decoding and
 *     confidence thresholding"으로 설명함 — 즉 최소 bbox 디코딩+score threshold는 확실히
 *     neural core에서 돈다(auto 모드와 동일 특징). nn_core/auto 중 정확히 어느 쪽인지는
 *     실기에서 `hailortcli parse-hef`로 postprocess 메타데이터를 확인할 것.
 *   - 입력 크기가 512x512x3이다(다른 세 모델은 640x640x3) — model_types.hpp::ModelConfig::
 *     img_size로 처리(model_runner.hpp가 img_size를 파라미터로 받으므로 letterbox/decode_det
 *     둘 다 자동으로 512 기준으로 동작함 — 2026-08-19 hailo_8에서 포팅한 변경사항).
 *
 * 파라미터: 사용자 지정 기본값 그대로 사용 — batch=1, threshold=1, timeout=0ms, priority=0,
 *          INPUT_FPS=60(사용자 요청, 2026-08-19). 모델 1개만 단독 실행 — 큐 경쟁/스케줄링
 *          영향 배제하고 "후처리 담당 프로세서 차이"만 격리해서 보는 것이 목적.
 *
 * 측정: 기존 3모델 프레임워크와 동일하게 model_runner.hpp::run_model_async()가
 *   전처리(imread+letterbox) / latency(enqueue~dequeue) / 후처리(decode_det 파싱) /
 *   장당 전체시간을 프레임별로 측정한다. Detection의 "후처리" 측정값은 여기서는 순수
 *   버퍼 파싱 비용만 잡는다는 점에 유의(NMS 연산 자체는 애초에 이 시점 이전에
 *   호스트 CPU가 아니라 칩에서 이미 끝난 상태로 나옴 — engine=cpu였던 yolov8s_h8l과 비교할 때
 *   "decode_det 파싱 시간"이 아니라 "이 모델을 돌리는 동안 호스트 CPU 총 사용률
 *   (cpu_percent 컬럼)"이 더 의미 있는 비교 지표가 될 가능성이 높음 — CSV에 같이 기록해둠).
 *
 * 재사용: model_types/sys_monitor/image_utils/output_classify/model_setup/model_runner/
 *   postprocess_8l 헤더를 기존 3모델 파일(infer_scheduler.cpp)과 동일하게 그대로 재사용한다
 *   (로직 변경 없음, model_runner.hpp/model_types.hpp에 img_size만 추가됨 — 기본값 640이라
 *   infer_scheduler.cpp/infer_scheduler_noppt.cpp 동작은 그대로임). csv_writer.hpp는
 *   Det/Seg/Pose 3슬롯 고정 스키마라 재사용하지 않고, 이 파일 전용의 단순 1행 CSV를
 *   아래에 직접 작성한다(hailo_8 버전과 완전히 동일한 스키마 — make_avg_csv_det_v8s_vs_v5npu.py
 *   /build_xlsx_det_v8s_vs_v5npu.py 그대로 재사용 가능).
 *
 * 빌드 (RPi, 기존 infer_scheduler.cpp와 동일 방식 — 8L은 rpi4 같은 -I 경로 이슈 없음):
 *   g++ infer_yolov5_hailo8l.cpp -o infer_yolov5_hailo8l -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
 *
 * 실행:
 *   ./infer_yolov5_hailo8l [run_id] [csv_path]
 *
 * "Det 단일모델, v8s_h8l(CPU 후처리) vs v5-nms_core(NPU 후처리), FPS=60, 3회 반복 후 평균"
 * 실험은 이 파일을 그대로 사용하고 scripts/run_det_v8s_vs_v5npu_fps60.sh가
 * USE_CPU_BASELINE_INSTEAD(0=v5/NPU, 1=v8s/CPU)를 sed로 자동 토글해 양쪽을 순서대로
 * 빌드/실행한다. 평균/xlsx는 scripts/make_avg_csv_det_v8s_vs_v5npu.py,
 * scripts/build_xlsx_det_v8s_vs_v5npu.py 참고.
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

#include "postprocess_8l.hpp"

using namespace hailort;

// ========================= 파라미터 (사용자 지정 기본값) =========================
// [주의] 자동화 스크립트(scripts/run_det_v8s_vs_v5npu_fps60.sh)가 필요시 sed로 이 블록을
// 편집할 수 있으니 매크로 이름/줄 형식을 유지할 것 (기존 infer_scheduler.cpp 관례와 동일).
#define BATCH_YOLOV5      1
#define THRESHOLD_YOLOV5  1
#define TIMEOUT_YOLOV5_MS 0
#define PRIORITY_YOLOV5   0

#define ENABLE_POSTPROCESS  1   // decode_det() 파싱 비용까지 측정(=1). 0이면 파싱 스킵.
#define INPUT_FPS           60  // [2026-08-19] 사용자 요청: NPU vs CPU 후처리 비교 실험은
                                 // FPS=60 기준으로 통일. 모델 1개 단독 실행이라 스케줄러
                                 // starvation 문제 자체는 없지만(경쟁 상대가 없음), "후처리
                                 // 담당 프로세서 차이만 빼고 나머지는 전부 동일 환경"이라는
                                 // 실험 조건을 맞추기 위해 입력 속도를 고정한다.
#define NUM_IMAGES          0   // 0 = IMG_DIR 전체 사용

#define YOLOV5_IMG_SIZE   512   // yolov5xs_wo_spp_nms_core 모델 입력 크기(다른 세 모델은 640)
// =====================================================================================

// HEF 경로 — 기존 세 모델(hailo-rpi5-examples/resources)과 같은 디렉터리 관례.
// yolov5xs_wo_spp_nms_core.hef가 아직 없으면 먼저 받아야 함(스크립트가 자동으로 받음):
//   wget -P /home/rpi1/hailo-rpi5-examples/resources/ \
//     https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/yolov5xs_wo_spp_nms_core.hef
//   hailortcli parse-hef /home/rpi1/hailo-rpi5-examples/resources/yolov5xs_wo_spp_nms_core.hef   # Architecture: HAILO8L 확인
#define YOLOV5_HEF "/home/rpi1/hailo-rpi5-examples/resources/yolov5xs_wo_spp_nms_core.hef"

// 비교 기준(선택): 기존 engine=cpu 경로인 yolov8s_h8l.hef와 같은 실행 조건(단독, 동일
// 파라미터)으로 돌리고 싶을 때 이 매크로를 1로 바꾸면 YOLOV5_HEF 대신
// DET_HEF_CPU_BASELINE을 로드한다. (README.md/infer_scheduler.cpp의 DET_HEF와 동일 경로.)
#define USE_CPU_BASELINE_INSTEAD 0
#define DET_HEF_CPU_BASELINE "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"

#define IMG_DIR  "/home/rpi1/datasets/sampled_val2017/"

std::mutex print_mutex;

#include "model_types.hpp"
#include "sys_monitor.hpp"
#include "image_utils.hpp"
#include "output_classify.hpp"
#include "model_setup.hpp"
#include "model_runner.hpp"

// 이 파일 전용 1행 CSV (Det/Seg/Pose 3슬롯을 가정하는 csv_writer.hpp와 스키마가 달라 별도 작성).
// [2026-08-19 확장] FPS 스윕(run_det_v8s_vs_v5npu_fpssweep.sh, INPUT_FPS={0,30})을 한 CSV에
// 같이 담기 위해 input_fps 컬럼 추가. 또한 기존 3모델 csv_writer.hpp가 갖고 있던 HRTT 유래
// 지표(npu_percent 등)를 이 실험에도 붙이기 위해 NaN 플레이스홀더 컬럼을 추가한다 —
// npu_percent는 실행 스크립트가 hailo_utilization.py 로그를 읽어 마지막 행에 바로 채우고,
// 나머지(switches_per_s/idle_time_pct/avg_fps_hrtt/avg_latency_hrtt/max_latency_hrtt/
// activation_hrtt)는 다운로드 후 fill_hrtt_columns_det_v8s_vs_v5npu.py가 .hrtt를 직접
// 파싱해서 채운다(csv_writer.hpp 3모델 스키마와 동일한 2단계 채움 방식).
static void save_csv_single(const std::string& csv_path, int run_id, const ModelConfig& m,
                             const ModelResult& r, double cpu_percent, double mem_percent,
                             long vol_ctx, long nonvol_ctx, double run_time_s)
{
    static const char* HEADER =
        "run_id,hef_name,img_size,batch,threshold,timeout_ms,priority,input_fps,"
        "frame_count,avg_preprocess_ms,avg_latency_ms,avg_postprocess_ms,avg_total_time_ms,"
        "total_time_s,run_time_s,cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches,"
        "npu_percent,switches_per_s,idle_time_pct,avg_fps_hrtt,avg_latency_hrtt,max_latency_hrtt,activation_hrtt";

    bool need_header = true;
    { std::ifstream chk(csv_path); if (chk.good() && chk.peek() != std::ifstream::traits_type::eof()) need_header = false; }
    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) { std::cerr << "[CSV] 열기 실패: " << csv_path << std::endl; return; }
    if (need_header) f << HEADER << "\n";

    auto dtos = [](double v) { if (v < 0) return std::string("NaN"); std::ostringstream os; os << v; return os.str(); };

    f << run_id << ',' << m.name << ',' << m.img_size << ',' << m.batch << ',' << m.threshold << ','
      << m.timeout_ms << ',' << m.priority << ',' << INPUT_FPS << ',' << r.frame_count << ','
      << dtos(r.avg_preprocess_ms) << ',' << dtos(r.avg_latency_ms) << ',' << dtos(r.avg_postprocess_ms) << ','
      << dtos(r.avg_total_time_ms) << ',' << dtos(r.total_time_s) << ',' << dtos(run_time_s) << ','
      << dtos(cpu_percent) << ',' << dtos(mem_percent) << ',' << vol_ctx << ',' << nonvol_ctx << ','
      << "NaN,NaN,NaN,NaN,NaN,NaN,NaN" << "\n";   // npu_percent~activation_hrtt: 실행/후처리 스크립트가 채움
    f.close();
    std::printf("[CSV] 저장: %s (run_id=%d)\n", csv_path.c_str(), run_id);
}

int main(int argc, char* argv[])
{
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;
    std::string csv_path = (argc > 2) ? argv[2] : "";

    pid_t my_pid = getpid();
    std::printf("PID: %d, Run ID: %d\n", my_pid, run_id);
    std::printf("모델: %s\n", USE_CPU_BASELINE_INSTEAD ? "yolov8s_h8l (engine=cpu 기준선)" : "yolov5xs_wo_spp_nms_core (engine=nn_core/auto)");

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
                "      안 보인다. yolov8s_h8l(engine=cpu) 기준선과 비교하려면 위 CPU%% 컬럼을\n"
                "      같이 볼 것.\n");

    if (!csv_path.empty())
        save_csv_single(csv_path, run_id, models[0], result, final_cpu, final_mem,
                         result.vol_ctx, result.nonvol_ctx, run_time_s);

    return HAILO_SUCCESS;
}
