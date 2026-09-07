// infer_scheduler_h10.cpp — Hailo-8L infer_scheduler.cpp의 Hailo-10H 이식본.
//
// 목적: 8L 코드의 "구조와 데이터 흐름"을 그대로 유지한 채 HailoRT 5.x / Hailo-10H에서
//       돌린다. 측정 정의(latency = enqueue→모든 출력 수신, total_time = 첫 enqueue→
//       마지막 dequeue)와 CSV 49컬럼 스키마를 8L과 100% 동일하게 유지해, 두 보드 결과를
//       같은 표에 놓고 비교할 수 있게 하는 것이 이 파일의 존재 이유다.
//
// ── [2026-08-31 변경 2가지] ──────────────────────────────────────────────────
//  (1) 후처리를 코드에서 완전히 제거했다.
//      - 디코딩/NMS 코드, output_classify.hpp, OutRole/OutMeta 전부 삭제.
//      - 출력 포맷 FLOAT32(+NHWC) 강제도 삭제 → 항상 HAILO_FORMAT_TYPE_AUTO.
//        HailoRT 내부 역양자화/재배열 비용이 latency에 섞이지 않는다.
//      - 8L의 B/D조건(ENABLE_POSTPROCESS=0, FORCE_OUTPUT_FLOAT32=0)과 동일한 조건이다.
//      - CSV의 postprocess_ms_* 컬럼은 그대로 두고 값만 NaN으로 기록한다(49컬럼 유지).
//  (2) 스케줄러 파라미터 API(threshold/timeout/priority setter)를 주석 처리했다.
//      - model_setup.hpp의 apply_scheduler_params() 안에 주석으로 남아 있다.
//      - HailoRT 기본 동작(threshold=1, timeout=0ms, priority=16=NORMAL)으로 돈다.
//      - batch만 실제로 적용된다(configure 시점 인자).
// ────────────────────────────────────────────────────────────────────────────
//
// 오케스트레이션만 담당: VDevice 생성 -> 모델 설정(model_setup.hpp) -> 이미지 목록 로드
// -> 실행 자원 생성 -> 모델별 스레드 실행(model_runner.hpp) -> 결과 요약 -> CSV 저장.
//
// ── 빌드 (npu-rpi5) ──
//   g++ -O2 -std=c++17 infer_scheduler_h10.cpp -o infer_scheduler_h10 -I. \
//       $(pkg-config --cflags --libs opencv4) -lhailort -lpthread
//
// ── 실행 ──
//   ./infer_scheduler_h10 <run_id> [csv_path] [옵션...]
//   예: ./infer_scheduler_h10 1 csv/results.csv --api infer --inflight 12
//
//   [중요] H10의 기본/유일 백엔드는 --api infer 다. VStreams 경로(--api vstreams)는
//          H10에서 HAILO_NOT_IMPLEMENTED(7)로 실패한다(실측 2026-08-31). 8L 대조용으로만 남겨둔다.
//   인자를 하나도 안 주면 아래 #define 값 그대로 동작한다 — 8L과 같은 sed 패치 방식 지원.

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace hailort;

// =====================================================================================
//  실험 파라미터 (8L infer_scheduler.cpp의 #define 블록과 같은 구성·같은 이름 유지)
//  스크립트에서 sed로 패치하는 방식이 8L과 동일하게 동작한다.
// =====================================================================================
#define BATCH_DET       1
#define BATCH_SEG       1
#define BATCH_POSE      1

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ [중요] 아래 THRESHOLD_* / TIMEOUT_*_MS / PRIORITY_* 는 현재 **적용되지 않는다**. │
// │ model_setup.hpp의 set_scheduler_threshold/timeout/priority 호출을 주석         │
// │ 처리했기 때문이다. 따라서 이 값들은 "HailoRT가 기본으로 쓰는 값"과 같게 맞춰    │
// │ 두었다 — CSV의 threshold_*/timeout_ms/priority_* 컬럼에 그대로 기록되는데,     │
// │ 실제 적용값과 다르면 CSV가 거짓말을 하게 되기 때문이다.                        │
// │   threshold = 1   (infer_model.hpp "The default threshold is 1")              │
// │   timeout   = 0ms (infer_model.hpp "The default timeout is 0ms")              │
// │   priority  = 16  (HAILO_SCHEDULER_PRIORITY_NORMAL)                           │
// │ 8L은 priority를 15로 줬지만 세 모델이 전부 같은 값이라 스케줄링 결과는 동일     │
// │ 하다(Round-Robin). 즉 주석 처리로 8L 대비 스케줄링 동작이 달라지지 않는다.      │
// │ 값을 실제로 바꾸려면 model_setup.hpp의 setter 주석을 먼저 풀 것.               │
// └──────────────────────────────────────────────────────────────────────────────┘
#define THRESHOLD_DET   1
#define THRESHOLD_SEG   1
#define THRESHOLD_POSE  1

#define TIMEOUT_DET_MS   0
#define TIMEOUT_SEG_MS   0
#define TIMEOUT_POSE_MS  0

#define PRIORITY_DET    16
#define PRIORITY_SEG    16
#define PRIORITY_POSE   16

// 어떤 모델을 이번 실행에 포함할지 (0/1)
#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1

// [진단용] 1=write() 블로킹시간 계측(처음 40프레임) + HailoRT 큐 accumulator 출력. 평소엔 0.
#define DEBUG_WRITE_TIMING  0

// 입력 속도 제한 (모델당 초당 프레임 수). 0 = 제한 없음(최대 속도로 큐를 채움 -> 큐가
// 항상 포화 상태). 8L 기본 실험과 같은 조건을 쓰려면 0으로 둘 것.
#define INPUT_FPS       0

// 사용할 검증 이미지 수 (0 = IMG_DIR의 전체 이미지 사용)
// [주의] 8L 기존 실험은 600장이다. 두 보드를 비교하려면 반드시 같은 값을 쓸 것.
#define NUM_IMAGES      600

// [H10 신규] 기본 백엔드와 in-flight 깊이.
//   API_VSTREAMS=0 -> InferModel Async + 슬롯 풀 (큐 깊이 = 모델별 in-flight)  ← H10 기본값
//   API_VSTREAMS=1 -> VStreams 경로 (8L과 1:1). **H10에서는 동작하지 않는다.**
//
// [실측 2026-08-31, npu-rpi5 / HailoRT 5.3.0]
//   vdevice->create_configure_params(hef) 가 HAILO_NOT_IMPLEMENTED(7)로 실패하고
//   HailoRT가 직접 이렇게 안내한다:
//     "Not supported. Did you try calling `create_configure_params` on H10?
//      If so, use InferModel instead"
//   즉 H10에는 VStreams/ConfiguredNetworkGroup 경로 자체가 없다. 그래서 기본값을 infer로 둔다.
//   vstreams 코드는 8L 대조용으로 남겨두었다(8L에서 이 파일을 그대로 빌드해 비교할 때 쓴다).
//
// [2026-09-07 변경] in-flight 깊이를 "전 모델 공통 1개"에서 "모델별 3개"로 분리했다.
//
//   왜: 8L의 in-flight는 우리가 설정한 값이 아니라 vstream 파이프라인에서 **모델마다
//   다르게 나타나는 결과값**이다(det 11.0~11.3 / seg 9.2~10.4 / pose 9.1~10.2). 전 모델
//   공통 D=12는 det에만 맞고(3% 오차) seg·3모델은 15~17% 어긋나 있었다. 즉 "in-flight를
//   맞췄다"는 진술이 det에서만 참이었다.
//
//   기본 동작: 아래 L8L 표에서 **활성 모델 조합에 해당하는 8L 실측 L**을 찾아
//   D = round(L)로 자동 결정한다. --inflight 로 덮어쓸 수 있다(단일값 또는 det,seg,pose).
#define API_VSTREAMS    0
#define INFLIGHT_FALLBACK  12   // L8L 표에 기준값이 없는 조합에서만 쓰인다

// [H10 신규] vstream 큐 크기. HAILO_DEFAULT_VSTREAM_QUEUE_SIZE = 8L과 동일 조건.
#define VSTREAM_QUEUE_SIZE  HAILO_DEFAULT_VSTREAM_QUEUE_SIZE

// [H10 신규] configure 시 ULTRA_PERFORMANCE 전력모드를 시도할지(8L은 항상 1). 실패 시 자동 폴백.
#define POWER_MODE_ULTRA    1

// [H10 신규] CSV 처리 옵션
//   CSV_EXTENSION_COLUMNS 0 -> 8L과 100% 동일한 49컬럼만 기록 (기본값, 권장)
//   CSV_EXTENSION_COLUMNS 1 -> 뒤에 backend/inflight/service_latency 등 확장 컬럼 추가
//   FILL_HOST_DERIVED     1 -> 8L에서 HRTT가 채우던 avg_fps_*/max_latency_*를 호스트 실측으로 채움
//   FILL_HOST_DERIVED     0 -> 그 컬럼들도 NaN (8L 런타임 출력과 완전히 동일)
#define CSV_EXTENSION_COLUMNS  0
#define FILL_HOST_DERIVED      1
// =====================================================================================

// HEF 경로 (Raspberry Pi 5 + Hailo-10H M.2). 8L용 _h8l HEF가 아니라 H10용으로 컴파일된
// HEF여야 한다 — HEF는 아키텍처별로 다르므로 8L 파일을 그대로 쓰면 로드에 실패한다.
#define DET_HEF  "/home/npu-rpi5/hailo10h_sched_exp1/resources/yolov8s.hef"
#define SEG_HEF  "/home/npu-rpi5/hailo10h_sched_exp1/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/npu-rpi5/hailo10h_sched_exp1/resources/yolov8s_pose_h10.hef"

// 입력 데이터셋 경로 (8L 실험과 같은 sampled_val2017).
#define IMG_DIR  "/home/npu-rpi5/datasets/sampled_val2017/"

std::mutex print_mutex;

#include "model_types.hpp"
#include "sys_monitor.hpp"
#include "image_utils.hpp"
#include "model_setup.hpp"
#include "model_runner.hpp"
#include "csv_writer.hpp"
#include "npu_monitor.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════
// 8L 실측 in-flight 기준표 — H10이 맞춰야 할 목표 깊이
// ═════════════════════════════════════════════════════════════════════════════
// 출처: hailo_8L/experiments/2026-08-27_abcd_pp_format_sweep_gen3/csv/results_B.csv
//       (B조건 = 후처리 OFF + 출력 AUTO. H10 현재 조건과 같은 조건이다)
// 산출: L = FPS x latency  (Little의 법칙),  FPS = 600장 / total_time_<model>_s
//       results_B.csv에는 avg_fps_*가 HRTT 미수집으로 NaN이라 total_time에서 역산한다.
// 재현성: A/B/C 세 조건에서 det 11.02~11.25로 편차 2% 이내 — 안정적인 구조값이다.
//
// [해석 주의] 여기서 D는 슬롯 수이고 L은 관측된 평균 점유 프레임 수다. enq_ts를 슬롯
//   확보 **이전**에 찍으므로(8L의 write() 직전과 같은 지점) 슬롯 대기 중인 프레임도
//   L에 포함된다. 따라서 실측 L은 대략 [D-1, D+1] 범위에 들어온다. round(L)을 D의
//   출발점으로 쓰고, 실행 후 아래 [in-flight 검증] 출력을 보고 ±1 조정하면 된다.
static const double L8L[8][3] = {
    /* mask 0 : (없음)      */ { -1.00, -1.00, -1.00 },
    /* mask 1 : det         */ { 11.25, -1.00, -1.00 },
    /* mask 2 : seg         */ { -1.00, 10.38, -1.00 },
    /* mask 3 : det+seg     */ { 11.10,  9.30, -1.00 },
    /* mask 4 : pose        */ { -1.00, -1.00, 10.21 },
    /* mask 5 : det+pose    */ { 11.13, -1.00,  9.23 },
    /* mask 6 : seg+pose    */ { -1.00,  9.28,  9.17 },
    /* mask 7 : det+seg+pose*/ { 11.03,  9.17,  9.09 },
};
// 조건이 맞았다고 볼 어긋남 한계(%). 이보다 크면 로그에 NG로 찍힌다.
static const double INFLIGHT_MATCH_TOL_PCT = 5.0;

static const char* const MODEL_KEY[3] = {"det", "seg", "pose"};

// CLI 오버라이드 — 8L은 #define + sed 패치만 썼다. 그 방식을 그대로 유지하면서
// (인자를 안 주면 #define 값 그대로), 스윕 편의를 위해 선택적 플래그를 얹었다.
// [주의] --threshold / --timeout_ms / --priority 플래그는 없다. 스케줄러 setter가 주석
//        처리된 상태라 값을 받아도 적용되지 않고 CSV만 거짓으로 만들기 때문이다.
//        스윕이 필요하면 model_setup.hpp의 setter 주석을 먼저 풀 것.
// ─────────────────────────────────────────────────────────────────────────────
struct RunOpts {
    int run_id = 1;
    std::string csv_path;
    std::string img_dir = IMG_DIR;
    int num_images = NUM_IMAGES;
    bool use_vstreams = (API_VSTREAMS != 0);
    int inflight[3] = {-1, -1, -1};   // -1 = 자동(L8L 표에서 결정). 인덱스 0=det,1=seg,2=pose
    bool inflight_explicit = false;   // true면 사용자가 --inflight로 직접 지정한 값
    bool npu_monitor = true;

    int batch[3] = {BATCH_DET, BATCH_SEG, BATCH_POSE};
    bool use[3]  = {(bool)USE_DET, (bool)USE_SEG, (bool)USE_POSE};
};

static std::vector<std::string> split_csv_arg(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}

// "1" -> 세 모델 모두 1 / "1,2,1" -> det,seg,pose 각각.
static bool parse_triple(const std::string& arg, int out[3], const char* flag) {
    auto parts = split_csv_arg(arg);
    if (parts.size() == 1) {
        out[0] = out[1] = out[2] = std::atoi(parts[0].c_str());
        return true;
    }
    if (parts.size() == 3) {
        for (int i = 0; i < 3; i++) out[i] = std::atoi(parts[i].c_str());
        return true;
    }
    std::cerr << flag << "는 값 1개 또는 3개(det,seg,pose)만 가능: " << arg << std::endl;
    return false;
}

static bool parse_args(int argc, char** argv, RunOpts& o) {
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::cerr << flag << " 값 없음" << std::endl; exit(1); }
            return argv[++i];
        };
        if (a.rfind("--", 0) != 0) {
            // 8L 호환 위치 인자: argv[1]=run_id, argv[2]=csv_path
            if (positional == 0)      o.run_id = std::atoi(a.c_str());
            else if (positional == 1) o.csv_path = a;
            else { std::cerr << "알 수 없는 위치 인자: " << a << std::endl; return false; }
            positional++;
            continue;
        }
        if (a == "--api") {
            std::string v = next("--api");
            if (v == "vstreams") o.use_vstreams = true;
            else if (v == "infer") o.use_vstreams = false;
            else { std::cerr << "--api는 vstreams 또는 infer" << std::endl; return false; }
        }
        else if (a == "--inflight") {
            // "auto"(기본) = L8L 표에 맞춰 자동 / "12" = 전 모델 12 / "11,9,9" = 모델별
            std::string v = next("--inflight");
            if (v == "auto") {
                o.inflight[0] = o.inflight[1] = o.inflight[2] = -1;
                o.inflight_explicit = false;
            } else {
                if (!parse_triple(v, o.inflight, "--inflight")) return false;
                o.inflight_explicit = true;
            }
        }
        else if (a == "--csv")        o.csv_path = next("--csv");
        else if (a == "--img_dir")    o.img_dir = next("--img_dir");
        else if (a == "--num_images") o.num_images = std::atoi(next("--num_images").c_str());
        else if (a == "--run_id")     o.run_id = std::atoi(next("--run_id").c_str());
        else if (a == "--no_npu_monitor") o.npu_monitor = false;
        else if (a == "--batch")      { if (!parse_triple(next("--batch"), o.batch, "--batch")) return false; }
        else if (a == "--models") {
            // "det,seg,pose" 형태. 지정한 모델만 활성화한다.
            o.use[0] = o.use[1] = o.use[2] = false;
            for (const auto& k : split_csv_arg(next("--models"))) {
                if (k == "det") o.use[0] = true;
                else if (k == "seg") o.use[1] = true;
                else if (k == "pose") o.use[2] = true;
                else { std::cerr << "알 수 없는 모델 키: " << k << " (det/seg/pose)" << std::endl; return false; }
            }
        }
        else if (a == "--threshold" || a == "--timeout_ms" || a == "--priority") {
            std::cerr << a << " 는 지원하지 않습니다 — 스케줄러 파라미터 setter가 주석 처리된\n"
                      << "상태라 값을 줘도 적용되지 않습니다. model_setup.hpp의\n"
                      << "apply_scheduler_params()에서 주석을 먼저 푸세요." << std::endl;
            return false;
        }
        else { std::cerr << "알 수 없는 옵션: " << a << std::endl; return false; }
    }
    return true;
}

// ========================= main =========================
int main(int argc, char* argv[])
{
    RunOpts opt;
    if (!parse_args(argc, argv, opt)) return 1;

    // ── in-flight 깊이 결정 ──────────────────────────────────────────────────
    // 활성 모델 조합(mask)에 해당하는 8L 실측 L을 표에서 찾아, 사용자가 --inflight로
    // 직접 주지 않은 모델만 round(L)로 채운다. "8L과 같은 조건"을 운영자의 명령줄이
    // 아니라 코드가 보장하게 만드는 것이 이 블록의 목적이다.
    const int mask = (opt.use[0] ? 1 : 0) | (opt.use[1] ? 2 : 0) | (opt.use[2] ? 4 : 0);
    double target_L[3] = { L8L[mask][0], L8L[mask][1], L8L[mask][2] };
    for (int i = 0; i < 3; i++) {
        if (!opt.use[i]) continue;
        if (opt.inflight[i] > 0) continue;                       // 사용자 지정값 우선
        opt.inflight[i] = (target_L[i] > 0) ? (int)(target_L[i] + 0.5) : INFLIGHT_FALLBACK;
        if (target_L[i] <= 0)
            std::printf("[경고] %s: 이 조합에 대한 8L 기준값이 표에 없어 D=%d(폴백)을 쓴다.\n",
                        MODEL_KEY[i], opt.inflight[i]);
    }

    pid_t my_pid = getpid();
    if (opt.use_vstreams) {
        std::printf("PID: %d, Run ID: %d, backend=vstreams (8L과 1:1)\n", my_pid, opt.run_id);
        std::printf("[경고] Hailo-10H는 VStreams/ConfiguredNetworkGroup 경로를 지원하지 않습니다.\n"
                    "       create_configure_params가 HAILO_NOT_IMPLEMENTED(7)로 실패합니다.\n"
                    "       H10에서는 --api infer 를 쓰세요(기본값). 이 경로는 8L 대조용입니다.\n");
    } else {
        std::printf("PID: %d, Run ID: %d, backend=infer (InferModel Async)\n", my_pid, opt.run_id);
        std::printf("[in-flight] 결정 방식 = %s\n",
                    opt.inflight_explicit ? "사용자 지정(--inflight)"
                                          : "8L 실측 L 자동 매칭(results_B.csv 기준표)");
        for (int i = 0; i < 3; i++) {
            if (!opt.use[i]) continue;
            if (target_L[i] > 0)
                std::printf("            %-4s D=%-2d   (8L 목표 L=%.2f)\n",
                            MODEL_KEY[i], opt.inflight[i], target_L[i]);
            else
                std::printf("            %-4s D=%-2d   (8L 기준값 없음)\n",
                            MODEL_KEY[i], opt.inflight[i]);
        }
    }
    std::printf("[조건] 후처리 없음(출력 포맷 AUTO, FLOAT32 변환 없음) / "
                "스케줄러 파라미터 setter 주석 처리(HailoRT 기본값)\n");

    if (!opt.use_vstreams) {
        for (int i = 0; i < 3; i++) {
            if (!opt.use[i]) continue;
            if (opt.inflight[i] < 1) {
                std::cerr << "--inflight(" << MODEL_KEY[i] << ")는 1 이상이어야 합니다." << std::endl;
                return 1;
            }
            if (opt.inflight[i] == 1)
                std::printf("[경고] %s: inflight=1은 run_async 직후 wait()하는 것과 같아 파이프라인이\n"
                            "       사라진다. 8L과 비교할 latency를 재려면 표 기준값(9~11)을 쓸 것.\n",
                            MODEL_KEY[i]);
        }
    }

    // ── VDevice 생성 (스케줄러 Round-Robin) ──
    // [8L 대비] 코드는 동일하다. 다만 H10에서는 VDevice::create()가 VDeviceHrpcClient를
    // 돌려주며, 실제 스케줄러는 디바이스 안 hailort_server에서 돈다.
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

    // priority/threshold/timeout에는 "실제 적용값"(= HailoRT 기본값)이 들어간다.
    // setter를 주석 처리했으므로 이 값들은 CSV 기록용이며 동작에 영향을 주지 않는다.
    std::vector<ModelConfig> models = {
        {DET_HEF,  "Detection",    PRIORITY_DET,  THRESHOLD_DET,  TIMEOUT_DET_MS,  opt.batch[0], opt.use[0], ModelKind::DET,  640},
        {SEG_HEF,  "Segmentation", PRIORITY_SEG,  THRESHOLD_SEG,  TIMEOUT_SEG_MS,  opt.batch[1], opt.use[1], ModelKind::SEG,  640},
        {POSE_HEF, "Pose",         PRIORITY_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, opt.batch[2], opt.use[2], ModelKind::POSE, 640},
    };

    // ── 이미지 목록 로드 (전처리는 미리 하지 않는다 — writer 스레드가 프레임마다 수행) ──
    // [중요] get_image_files()는 8L 원본 그대로라 dir_path 뒤에 슬래시를 붙이지 않는다.
    //        --img_dir로 슬래시 없는 경로가 들어오면 ".../sampled_val2017000001.jpg" 같은
    //        경로가 만들어져 전부 imread 실패 → 검은 화면으로 대체되며 조용히 잘못된 결과가
    //        나온다. 8L 함수를 손대지 않기 위해 호출부에서 슬래시를 보정한다.
    if (!opt.img_dir.empty() && opt.img_dir.back() != '/') opt.img_dir.push_back('/');
    std::vector<std::string> images = get_image_files(opt.img_dir.c_str());
    if (images.empty()) {
        std::cerr << "[경고] IMG_DIR(" << opt.img_dir << ")에서 이미지를 찾지 못함. 경로를 확인할 것."
                  << std::endl;
        return 1;
    }
    if (opt.num_images > 0 && images.size() > (size_t)opt.num_images)
        images.resize(opt.num_images);
    std::printf("사용 이미지 수: %zu장 (경로: %s)\n\n", images.size(), opt.img_dir.c_str());

    ModelResult results[3];  // index: Detection=0, Segmentation=1, Pose=2 (models 배열과 동일 순서)

    // ── 디바이스 NNC 가동률 수집 시작 (8L에서는 외부 파이썬 모니터가 하던 일) ──
    npu_monitor::NpuMonitor npu(opt.npu_monitor);
    if (opt.npu_monitor && !npu.start())
        std::cerr << "[npu_monitor] 시작 실패 — npu_percent는 NaN으로 기록됩니다" << std::endl;

    double run_time_s = -1;
    CpuStats cpu_start, cpu_end;

    if (opt.use_vstreams) {
        // ══════════ 백엔드 A: VStreams (8L과 1:1) ══════════
        std::vector<std::shared_ptr<ConfiguredNetworkGroup>> network_groups;
        std::vector<int> active_model_idx;
        hailo_status cfg_status = configure_models(vdevice, models, network_groups, active_model_idx);
        if (cfg_status != HAILO_SUCCESS) return (int)cfg_status;

        if (network_groups.empty()) {
            std::cerr << "활성화된 모델이 없습니다 (USE_DET/USE_SEG/USE_POSE 또는 --models 확인)" << std::endl;
            return 1;
        }

        std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_ng;
        hailo_status vs_status = create_all_vstreams(network_groups, vstreams_per_ng);
        if (vs_status != HAILO_SUCCESS) return (int)vs_status;

        cpu_start = read_cpu_stats();
        double t_run_start = now_ms();

        std::vector<std::thread> threads;
        for (size_t k = 0; k < vstreams_per_ng.size(); k++) {
            int mi = active_model_idx[k];
            // std::thread는 함수의 기본 인자를 모르므로 img_size를 명시적으로 넘긴다.
            threads.emplace_back(run_model_async_vstreams, models[mi].name,
                std::ref(vstreams_per_ng[k].first), std::ref(vstreams_per_ng[k].second),
                std::cref(images), std::ref(results[mi]), models[mi].img_size);
        }
        for (auto& t : threads) t.join();
        run_time_s = (now_ms() - t_run_start) / 1000.0;

    } else {
        // ══════════ 백엔드 B: InferModel Async + 슬롯 풀 ══════════
        std::vector<std::unique_ptr<InferCtx>> ctxs;
        for (int i = 0; i < 3; i++) {
            if (!models[i].active) continue;
            auto ctx = std::unique_ptr<InferCtx>(new InferCtx());
            hailo_status st = prepare_infer_ctx(*vdevice, models[i], i, opt.inflight[i], *ctx);
            if (st != HAILO_SUCCESS) {
                std::cerr << "준비 실패: " << models[i].name << std::endl;
                return (int)st;
            }
            ctxs.push_back(std::move(ctx));
        }
        if (ctxs.empty()) {
            std::cerr << "활성화된 모델이 없습니다 (USE_DET/USE_SEG/USE_POSE 또는 --models 확인)" << std::endl;
            return 1;
        }

        cpu_start = read_cpu_stats();
        double t_run_start = now_ms();

        std::vector<std::thread> threads;
        for (auto& ctx : ctxs) {
            int mi = ctx->model_idx;
            threads.emplace_back(run_model_async_infer, models[mi].name,
                std::ref(*ctx), std::cref(images), std::ref(results[mi]), models[mi].img_size);
        }
        for (auto& t : threads) t.join();
        run_time_s = (now_ms() - t_run_start) / 1000.0;
    }

    npu.stop();
    const double npu_avg = npu.avg_npu_percent();

    // 장당 전체시간 = 이 모델 자신의 평균 전처리 + 평균 latency + 평균 후처리 (8L과 동일한 식).
    // 후처리 디코딩은 없지만 구간 계측은 8L처럼 남겨두므로 pp는 ~1e-4 ms가 들어간다.
    for (int i = 0; i < 3; i++) {
        if (models[i].active && results[i].avg_latency_ms >= 0) {
            double prep = (results[i].avg_preprocess_ms >= 0) ? results[i].avg_preprocess_ms : 0.0;
            double pp   = (results[i].avg_postprocess_ms >= 0) ? results[i].avg_postprocess_ms : 0.0;
            results[i].avg_total_time_ms = prep + results[i].avg_latency_ms + pp;
        }
    }

    // [8L 로직 유지] 8L main은 위 루프가 끝난 뒤에 cpu_end를 읽는다. 같은 자리로 맞춘다.
    cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();

    long vol_ctx = 0, nonvol_ctx = 0;
    for (auto& r : results) { vol_ctx += r.vol_ctx; nonvol_ctx += r.nonvol_ctx; }

    std::printf("\n========== 실험 결과 (Run ID: %d) ==========\n", opt.run_id);
    static const char* LABEL[3] = {"Detection   ", "Segmentation", "Pose        "};
    for (int i = 0; i < 3; i++) {
        if (!models[i].active) continue;
        std::printf("%s : 전처리=%.2fms, latency=%.2fms, 전체=%.2fms, %d장, FPS=%.2f, batch=%d "
                    "(스케줄러 기본값: threshold=%d, timeout=%ums, priority=%d)\n",
                    LABEL[i], results[i].avg_preprocess_ms, results[i].avg_latency_ms,
                    results[i].avg_total_time_ms, results[i].frame_count, results[i].fps,
                    models[i].batch, models[i].threshold, models[i].timeout_ms, models[i].priority);
    }
    std::printf("CPU: %.2f%%, MEM: %.2f%%, Ctx Switch(vol/nonvol): %ld/%ld, NPU: %s%% (샘플 %zu개), run_time=%.2fs\n",
                final_cpu, final_mem, vol_ctx, nonvol_ctx, dtos(npu_avg).c_str(),
                npu.sample_count(), run_time_s);

    // ── in-flight 검증 ───────────────────────────────────────────────────────
    // Little의 법칙(L = FPS x latency)으로 실측 깊이를 역산해 8L 목표값과 대조한다.
    // 이 표가 OK로 차야 "두 보드를 같은 파이프라인 깊이에서 비교했다"고 말할 수 있다.
    // NG가 뜨면 그 모델의 D를 어긋난 방향으로 1 조정해 다시 돌릴 것.
    std::printf("\n---------- in-flight 검증 (8L 목표 대비) ----------\n");
    std::printf("  (target_L = 8L 실측 in-flight, meas_L = 이번 런의 FPS x latency, dev = 어긋남)\n");
    std::printf("  %-5s %4s %10s %10s %9s  %s\n", "model", "D", "target_L", "meas_L", "dev", "verdict");
    bool all_matched = true;
    for (int i = 0; i < 3; i++) {
        if (!models[i].active) continue;
        if (results[i].fps <= 0 || results[i].avg_latency_ms <= 0) {
            std::printf("  %-5s %4d %10s %10s %9s  %s\n",
                        MODEL_KEY[i], opt.inflight[i], "-", "MEAS_FAIL", "-", "NG");
            all_matched = false;
            continue;
        }
        const double L = results[i].fps * results[i].avg_latency_ms / 1000.0;
        if (target_L[i] <= 0) {
            std::printf("  %-5s %4d %10s %10.2f %9s  %s\n",
                        MODEL_KEY[i], opt.inflight[i], "n/a", L, "-", "NO_REF");
            continue;
        }
        const double dev = (L - target_L[i]) / target_L[i] * 100.0;
        const bool ok = (dev < 0 ? -dev : dev) <= INFLIGHT_MATCH_TOL_PCT;
        if (!ok) all_matched = false;
        std::printf("  %-5s %4d %10.2f %10.2f %+8.1f%%  %s\n",
                    MODEL_KEY[i], opt.inflight[i], target_L[i], L, dev, ok ? "OK" : "NG");
    }
    std::printf("  => %s (허용 오차 +-%.0f%%)\n",
                all_matched ? "이 런은 8L과 같은 파이프라인 깊이에서 측정되었다"
                            : "깊이가 맞지 않는 모델이 있다 — D를 조정해 재측정할 것",
                INFLIGHT_MATCH_TOL_PCT);
    std::printf("  (참고: enq_ts를 슬롯 확보 이전에 찍으므로 실측 L은 D-1 ~ D+1 범위가 정상이다)\n");
    std::printf("================================================\n");
    std::printf("[주의] H10은 .hrtt가 항상 0바이트다(스케줄러가 디바이스 안 hailort_server에 있음).\n"
                "       switches_per_s / idle_time_pct / activation_* / avg_latency_*는 CSV에 컬럼만\n"
                "       남기고 NaN으로 기록된다. postprocess_ms_*도 후처리를 제거했으므로 NaN이다.\n"
                "       자세한 근거는 HRTT_ON_HAILO10H.md 참고.\n");

    // ── CSV 저장 (argv[2] 또는 --csv로 경로가 주어졌을 때만) ──
    if (!opt.csv_path.empty())
        save_csv(opt.csv_path, opt.run_id, models, results,
                 final_cpu, final_mem, vol_ctx, nonvol_ctx, run_time_s,
                 npu_avg, opt.use_vstreams ? nullptr : opt.inflight,
                 opt.use_vstreams ? "vstreams" : "infer");

    return HAILO_SUCCESS;
}
