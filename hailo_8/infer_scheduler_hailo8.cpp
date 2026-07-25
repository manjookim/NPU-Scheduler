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
 *   ./infer_scheduler_hailo8 [run_id]
 *
 * 현재는 CSV 저장 없이, 콘솔의 [적용확인] 로그 + HRTT 트레이스만으로 파라미터
 * 적용 여부를 확인하는 것이 목적이다. 아래 환경변수를 설정하고 실행하면 HRTT가
 * 생성되고, PC/WSL에서 `hailo runtime-profiler <파일>.hrtt`로 변환한 HTML에서
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

#include "postprocess_hailo8.hpp"

using namespace hailort;

// ========================= 파라미터 설정 (모델별로 다르게) =========================
// [실기 확인된 제약, 공식 문서에 명시 안 됨] threshold는 반드시 그 모델의 batch_size
// 이하여야 한다. 초과 시 set_scheduler_threshold가 HAILO_INVALID_ARGUMENT로 실패하고
// (HailoRT 로그: "Threshold must be equal or lower than the maximum batch size!"),
// 해당 모델은 threshold가 기본값(1)으로 남는다 — [적용확인] 로그의 [실패] 표시로 알 수 있음.
// 즉 아래 THRESHOLD_* <= BATCH_* 를 항상 지킬 것.

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

// ========================= 시스템 모니터링 (CPU/MEM/Context Switch) =========================

struct CpuStats {
    long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0;
};

CpuStats read_cpu_stats() {
    CpuStats s;
    std::ifstream f("/proc/stat");
    std::string line;
    std::getline(f, line);
    sscanf(line.c_str(), "cpu %ld %ld %ld %ld %ld %ld %ld",
           &s.user, &s.nice, &s.system, &s.idle, &s.iowait, &s.irq, &s.softirq);
    return s;
}

double calc_cpu_usage(const CpuStats& s1, const CpuStats& s2) {
    long idle1 = s1.idle + s1.iowait;
    long idle2 = s2.idle + s2.iowait;
    long total1 = s1.user + s1.nice + s1.system + s1.idle + s1.iowait + s1.irq + s1.softirq;
    long total2 = s2.user + s2.nice + s2.system + s2.idle + s2.iowait + s2.irq + s2.softirq;
    long dt = total2 - total1;
    if (dt <= 0) return 0.0;
    return 100.0 * (1.0 - (double)(idle2 - idle1) / (double)dt);
}

double read_mem_usage() {
    std::ifstream f("/proc/meminfo");
    long total = 0, available = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("MemTotal:") != std::string::npos)
            sscanf(line.c_str(), "MemTotal: %ld kB", &total);
        if (line.find("MemAvailable:") != std::string::npos)
            sscanf(line.c_str(), "MemAvailable: %ld kB", &available);
    }
    if (total <= 0) return 0.0;
    return 100.0 * (1.0 - (double)available / (double)total);
}

struct CtxSwitches { long voluntary = 0, nonvoluntary = 0; };

// 호출한 스레드 자신의 context switch 읽기 (/proc/thread-self/status, Linux 3.17+).
// 워커 스레드는 join 시점에 이미 사라지므로, 각 스레드가 자기 값을 측정해 합산해야 정확하다.
CtxSwitches read_thread_ctx_switches() {
    CtxSwitches cs;
    std::ifstream f("/proc/thread-self/status");
    std::string line;
    long v;
    while (std::getline(f, line)) {
        if (sscanf(line.c_str(), "nonvoluntary_ctxt_switches: %ld", &v) == 1)
            cs.nonvoluntary = v;
        else if (sscanf(line.c_str(), "voluntary_ctxt_switches: %ld", &v) == 1)
            cs.voluntary = v;
    }
    return cs;
}

// ========================= 이미지 유틸 =========================

// Letterbox: 비율 유지 resize + gray(114) 패딩 -> target_size x target_size
// (YOLOv8 학습 전처리와 동일; 3개 모델 모두 640x640x3 입력, docs/setup.md 참고)
cv::Mat letterbox(const cv::Mat& img, int target_size = 640) {
    int orig_h = img.rows, orig_w = img.cols;
    float scale = std::min((float)target_size / orig_h, (float)target_size / orig_w);
    int new_h = (int)(orig_h * scale);
    int new_w = (int)(orig_w * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    int pad_top    = (target_size - new_h) / 2;
    int pad_bottom = target_size - new_h - pad_top;
    int pad_left   = (target_size - new_w) / 2;
    int pad_right  = target_size - new_w - pad_left;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right,
                        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

std::vector<std::string> get_image_files(const char* dir_path) {
    std::vector<std::string> files;
    DIR* dir = opendir(dir_path);
    if (!dir) return files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (name.size() > 4 &&
            (name.find(".jpg") != std::string::npos || name.find(".JPG") != std::string::npos))
            files.push_back(std::string(dir_path) + name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());  // 실험 재현성을 위해 정렬(항상 같은 부분집합 사용)
    return files;
}

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ========================= 모델별 비동기 추론 (producer/consumer) =========================
// writer 스레드가 입력 큐를 채우고(큐가 가득 차면 write가 블로킹), reader 스레드가
// 그 모델의 "모든" 출력 vstream을 프레임 단위로 읽는다. INPUT_FPS>0이면 writer가 그
// 속도로 write를 지연시켜 큐가 점진적으로 쌓이도록 한다 — 이래야 threshold(큐 누적 수)
// 와 timeout(대기 시간)이 실제로 트리거될 조건이 생긴다(동기식으로 한꺼번에 밀어넣으면
// 큐가 항상 포화되어 threshold가 무의미해짐, memory/findings.md 참고).
struct ModelResult {
    double avg_latency_ms = -1;
    int frame_count = 0;
    long vol_ctx = 0;
    long nonvol_ctx = 0;
    double total_time_s = -1;      // 이 모델이 모든 입력(약 670장)을 처리하는 데 걸린 전체 시간(초)
    double avg_postprocess_ms = -1; // 프레임당 평균 후처리(디코딩+NMS) 시간
    double avg_total_time_ms = -1;  // 장당 전체 시간 = 전처리(공유, 평균) + latency + 후처리
};

// 모델 종류 — 어떤 후처리 경로를 태울지 결정 (Detection=on-chip NMS, Seg/Pose=raw tensor CPU 디코딩)
enum class ModelKind { DET, SEG, POSE };

// 모델별 실행 구성 (main에서 채움). save_csv에서도 참조하므로 전역에 둔다.
struct ModelConfig {
    const char* hef_path;
    const char* name;
    int priority;
    int threshold;
    int timeout_ms;
    int batch;
    bool active;
    ModelKind kind;
};

// ========================= 후처리용 output vstream 분류 =========================
// Pose/Seg의 raw output vstream들은 채널 수(c)만 보면 role을 알 수 있다
// (box=64, score=1, kpts=51, cls=80, coeff=32, proto=32이지만 h/w=160x160으로 구분).
// vstream 생성 순서(HEF 선언 순서)에 의존하지 않도록 매 vstream의 get_info().shape로
// 동적으로 분류한다. [전제] 해당 output vstream은 FLOAT32 + NHWC로 생성되어 있어야 함
// (postprocess_hailo8.hpp의 TensorView가 그 전제로 버퍼를 인덱싱함).
enum class OutRole { OTHER, POSE_BOX, POSE_SCORE, POSE_KPTS, SEG_BOX, SEG_CLS, SEG_COEFF, SEG_PROTO };

struct OutMeta {
    OutRole role = OutRole::OTHER;
    int h = 0, w = 0, c = 0;
    int stride = 0;  // img_size / h (정사각 640 입력 가정)
};

std::vector<OutMeta> classify_outputs(ModelKind kind, std::vector<OutputVStream>& outputs, int img_size = 640) {
    std::vector<OutMeta> metas(outputs.size());
    if (kind == ModelKind::DET) return metas;  // Detection은 on-chip NMS 출력이라 분류 불필요

    for (size_t j = 0; j < outputs.size(); j++) {
        const auto& info = outputs[j].get_info();
        int h = (int)info.shape.height, w = (int)info.shape.width, c = (int)info.shape.features;
        metas[j].h = h; metas[j].w = w; metas[j].c = c;
        metas[j].stride = (h > 0) ? (img_size / h) : 0;

        if (kind == ModelKind::POSE) {
            if (c == 64) metas[j].role = OutRole::POSE_BOX;
            else if (c == 1) metas[j].role = OutRole::POSE_SCORE;
            else if (c == 51) metas[j].role = OutRole::POSE_KPTS;
        } else if (kind == ModelKind::SEG) {
            if (c == 64) metas[j].role = OutRole::SEG_BOX;
            else if (c == 80) metas[j].role = OutRole::SEG_CLS;
            else if (c == 32) metas[j].role = (h == 160 && w == 160) ? OutRole::SEG_PROTO : OutRole::SEG_COEFF;
        }
    }
    return metas;
}

void run_model_async(const char* model_name,
                     ModelKind kind,
                     std::vector<InputVStream>& inputs,
                     std::vector<OutputVStream>& outputs,
                     const std::vector<OutMeta>& out_meta,
                     const std::vector<cv::Mat>& pre,
                     ModelResult& result)
{
    if (inputs.empty() || outputs.empty()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] 입력/출력 vstream 없음, 스킵" << std::endl;
        return;
    }

    // 프레임 크기 검증 (letterbox 결과가 모델 입력과 안 맞으면 write가 실패/깨질 수 있음)
    size_t expected = inputs[0].get_frame_size();
    size_t actual = pre.empty() ? 0 : pre[0].total() * pre[0].elemSize();
    if (!pre.empty() && expected != actual) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] [경고] 프레임 크기 불일치: 모델 기대="
                   << expected << "B, 전처리 결과=" << actual
                   << "B (letterbox 크기/채널 수를 모델 입력 shape에 맞게 조정할 것)" << std::endl;
    }

    size_t N = pre.size();
    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0);
    long w_vol = 0, w_nonvol = 0, r_vol = 0, r_nonvol = 0;
    hailo_status write_status = HAILO_SUCCESS, read_status = HAILO_SUCCESS;

    std::thread writer([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        const double interval_ms = (INPUT_FPS > 0) ? (1000.0 / INPUT_FPS) : 0.0;
        double next_t = now_ms();
        for (size_t i = 0; i < N; i++) {
            if (INPUT_FPS > 0) {
                double t = now_ms();
                if (t < next_t)
                    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(next_t - t));
                next_t += interval_ms;
            }
            enq_ts[i] = now_ms();
            // 낮은 우선순위로 starvation되어 입력버퍼가 안 비면 write가 HAILO_TIMEOUT을 낸다.
            // 프레임 유실 방지를 위해 timeout이면 성공할 때까지 재시도한다(threshold>=1이라
            // 높은 우선순위 모델이 자기 큐를 비우면 결국 이 모델도 스케줄되어 write가 통과).
            hailo_status status;
            do {
                status = inputs[0].write(MemoryView(pre[i].data, pre[i].total() * pre[i].elemSize()));
            } while (status == HAILO_TIMEOUT);
            if (HAILO_SUCCESS != status) { write_status = status; }
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary; w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    double pp_total_ms = 0.0;
    long pp_count = 0;

    std::thread reader([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        std::vector<std::vector<uint8_t>> obuf(outputs.size());
        for (size_t j = 0; j < outputs.size(); j++)
            obuf[j].resize(outputs[j].get_frame_size());

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < outputs.size(); j++) {
                // starvation 중이면 출력도 늦게 나와 read가 HAILO_TIMEOUT을 낼 수 있으므로 재시도.
                hailo_status status;
                do {
                    status = outputs[j].read(MemoryView(obuf[j].data(), obuf[j].size()));
                } while (status == HAILO_TIMEOUT);
                if (HAILO_SUCCESS != status) { read_status = status; }
            }
            deq_ts[i] = now_ms();  // 이 프레임의 모든 출력을 다 받은 시각

            // ── 후처리 (디코딩 + NMS), 프레임당 소요시간 별도 측정 ──
            // Detection은 HEF에 on-chip NMS(HailoRT-pp)가 내장되어 있어 이 시점의 obuf가
            // 이미 최종 검출 결과이므로 추가 디코딩이 필요 없다(후처리시간 ≈ 0으로 기록).
            double pp_t0 = now_ms();
#if ENABLE_POSTPROCESS
            if (kind == ModelKind::POSE) {
                std::map<std::pair<int,int>, PoseScaleTensors> groups;
                for (size_t j = 0; j < outputs.size(); j++) {
                    const auto& m = out_meta[j];
                    pp::TensorView tv{ reinterpret_cast<const float*>(obuf[j].data()), m.h, m.w, m.c };
                    auto& g = groups[{m.h, m.w}];
                    g.stride = m.stride;
                    if (m.role == OutRole::POSE_BOX) g.box = tv;
                    else if (m.role == OutRole::POSE_SCORE) g.score = tv;
                    else if (m.role == OutRole::POSE_KPTS) g.kpts = tv;
                }
                std::vector<PoseScaleTensors> scales;
                scales.reserve(groups.size());
                for (auto& kv : groups) scales.push_back(kv.second);
                size_t total_cells = 0;
                for (auto& sc : scales) total_cells += (size_t)sc.box.h * sc.box.w;
                size_t raw_cand = 0;
                // threshold는 조교님 참고 코드 값(conf=0.3, iou=0.45)과 동일하게 맞춤
                std::vector<PoseDet> pose_dets = decode_pose(scales, 640, 0.3f, 0.45f, 0.5f, &raw_cand);
                if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::printf("  [디버그][%s] 첫 프레임: 그리드 셀 총 %zu개 중 NMS 전 후보=%zu개, 최종 검출=%zu개\n",
                                model_name, total_cells, raw_cand, pose_dets.size());
                }
            } else if (kind == ModelKind::SEG) {
                std::map<std::pair<int,int>, SegScaleTensors> groups;
                pp::TensorView proto_tv;
                for (size_t j = 0; j < outputs.size(); j++) {
                    const auto& m = out_meta[j];
                    pp::TensorView tv{ reinterpret_cast<const float*>(obuf[j].data()), m.h, m.w, m.c };
                    if (m.role == OutRole::SEG_PROTO) { proto_tv = tv; continue; }
                    auto& g = groups[{m.h, m.w}];
                    g.stride = m.stride;
                    if (m.role == OutRole::SEG_BOX) g.box = tv;
                    else if (m.role == OutRole::SEG_CLS) g.cls = tv;
                    else if (m.role == OutRole::SEG_COEFF) g.coeff = tv;
                }
                std::vector<SegScaleTensors> scales;
                scales.reserve(groups.size());
                for (auto& kv : groups) scales.push_back(kv.second);
                size_t total_cells = 0;
                for (auto& sc : scales) total_cells += (size_t)sc.box.h * sc.box.w;
                size_t raw_cand = 0;
                // threshold는 조교님 참고 코드 값(conf=0.01, iou=0.65)과 동일하게 맞춤
                std::vector<SegDet> seg_dets = decode_seg(scales, proto_tv, 640, 0.01f, 0.65f, 80, true, &raw_cand);
                if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::printf("  [디버그][%s] 첫 프레임: 그리드 셀 총 %zu개 중 NMS 전 후보=%zu개, 최종 검출=%zu개\n",
                                model_name, total_cells, raw_cand, seg_dets.size());
                }
            }
#endif  // ENABLE_POSTPROCESS
            double pp_t1 = now_ms();
            pp_total_ms += (pp_t1 - pp_t0);
            pp_count++;
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

    result.avg_postprocess_ms = (pp_count > 0) ? (pp_total_ms / pp_count) : -1;

    if (HAILO_SUCCESS != write_status || HAILO_SUCCESS != read_status) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] [경고] 추론 중 오류 (write=" << write_status
                   << ", read=" << read_status << ") — 아래 latency는 왜곡됐을 수 있음" << std::endl;
    }

    double sum = 0; int c = 0;
    for (size_t i = 0; i < N; i++)
        if (deq_ts[i] > enq_ts[i]) { sum += (deq_ts[i] - enq_ts[i]); c++; }

    result.avg_latency_ms = c > 0 ? sum / c : -1;
    result.frame_count = c;
    result.vol_ctx = w_vol + r_vol;
    result.nonvol_ctx = w_nonvol + r_nonvol;

    // 모델별 전체 추론 시간 = 첫 입력 enqueue ~ 마지막 출력 dequeue 구간
    // (해당 모델이 모든 프레임을 다 처리하는 데 걸린 실제 wall-time, NPU 공유 경쟁 포함)
    double first_enq = 0, last_deq = 0;
    for (size_t i = 0; i < N; i++) {
        if (enq_ts[i] > 0 && (first_enq == 0 || enq_ts[i] < first_enq)) first_enq = enq_ts[i];
        if (deq_ts[i] > last_deq) last_deq = deq_ts[i];
    }
    result.total_time_s = (last_deq > first_enq) ? (last_deq - first_enq) / 1000.0 : -1;

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[%s] 완료: 평균 Latency=%.2f ms, 후처리=%.2f ms, %d장, 전체 추론시간=%.2f s (async, INPUT_FPS=%d)\n",
                model_name, result.avg_latency_ms, result.avg_postprocess_ms, result.frame_count, result.total_time_s, INPUT_FPS);
}

// ========================= CSV 저장 =========================
// [핵심] 이 프로그램(추론 단계)에서 "직접 측정 가능한 값"만 채워서 한 행을 append 한다.
// HRTT/HTML 프로파일러로만 얻을 수 있는 값은 "NaN"으로 남겨두고, 이후
//   - npu_percent            : tools/monitoring/parse_npu_log.py
//   - idle_time_pct / switches_per_s / 모델별 avg_fps·avg_latency·max_latency·activation
//                            : HRTT(.hrtt→.html) 프로파일러 파싱
// 단계에서 같은 행을 찾아 채운다.
// 컬럼 순서 = results/nancheck_rerun/results_nancheck_full.csv 스키마와 완전히 동일.
//
// 추론 중 직접 채우는 값:
//   run_id, use_*, batch_det/seg/pose, threshold_*, timeout_ms, priority_*   (실행 구성)
//   det/seg/pose_latency_ms                                     (이 코드가 측정한 end-to-end latency)
//   cpu_percent, mem_percent, voluntary/nonvoluntary_ctx_switches
//   run_time_s                                                  (추론 구간 실측 wall-time)
//   avg_preprocess_ms, postprocess_ms_*, total_time_ms_*        (8L에는 없는 신규 열 — 배치 스윕 실험용)
static std::string dtos(double v) {           // 음수(-1)=미측정/비활성 → NaN
    if (v < 0) return "NaN";
    std::ostringstream os; os << v; return os.str();
}

void save_csv(const std::string& csv_path, int run_id,
              const std::vector<ModelConfig>& models,   // [0]=Det, [1]=Seg, [2]=Pose 고정 순서
              const ModelResult results[3],
              double cpu_percent, double mem_percent,
              long vol_ctx, long nonvol_ctx, double run_time_s,
              double avg_preprocess_ms)
{
    static const char* HEADER =
        "run_id,use_det,use_seg,use_pose,batch_det,batch_seg,batch_pose,"
        "threshold_det,threshold_seg,threshold_pose,timeout_ms,"
        "priority_det,priority_seg,priority_pose,"
        "det_latency_ms,seg_latency_ms,pose_latency_ms,"
        "cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches,"
        "npu_percent,switches_per_s,idle_time_pct,run_time_s,"
        "avg_fps_det,avg_latency_det,max_latency_det,activation_det,"
        "avg_fps_seg,avg_latency_seg,max_latency_seg,activation_seg,"
        "avg_fps_pose,avg_latency_pose,max_latency_pose,activation_pose,"
        "total_time_det_s,total_time_seg_s,total_time_pose_s,"          // 모델별 전체 추론시간(추론 중 실측)
        "avg_preprocess_ms,"                                             // 장당 평균 전처리시간(공유, 모델 무관 동일값)
        "postprocess_ms_det,postprocess_ms_seg,postprocess_ms_pose,"     // 장당 평균 후처리(디코딩+NMS)시간
        "total_time_ms_det,total_time_ms_seg,total_time_ms_pose";        // 장당 전체시간 = 전처리+latency+후처리

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
    double det_ttl  = models[0].active ? results[0].avg_total_time_ms : -1;
    double seg_ttl  = models[1].active ? results[1].avg_total_time_ms : -1;
    double pose_ttl = models[2].active ? results[2].avg_total_time_ms : -1;

    std::ostringstream row;
    row << run_id << ','
        << (models[0].active ? 1 : 0) << ','
        << (models[1].active ? 1 : 0) << ','
        << (models[2].active ? 1 : 0) << ','
        // batch는 8L과 달리 모델별로 다른 값을 쓸 수 있어(배치사이즈 스윕 실험) 3열로 분리 기록.
        << models[0].batch << ',' << models[1].batch << ',' << models[2].batch << ','
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
        << dtos(avg_preprocess_ms) << ','
        << dtos(det_pp) << ',' << dtos(seg_pp) << ',' << dtos(pose_pp) << ','
        << dtos(det_ttl) << ',' << dtos(seg_ttl) << ',' << dtos(pose_ttl);
    f << row.str() << "\n";
    f.close();

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[CSV] 저장: %s (run_id=%d, 추론 측정값 기록, HRTT값은 NaN)\n",
                csv_path.c_str(), run_id);
}

// ========================= main =========================

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

    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> network_groups;
    std::vector<int> active_model_idx;  // network_groups[k] <-> models[active_model_idx[k]]

    // ── 모델별: HEF 로드 -> configure(batch) -> 스케줄러 파라미터(threshold/timeout/priority) 설정 ──
    for (size_t i = 0; i < models.size(); i++) {
        auto& m = models[i];
        if (!m.active) continue;

        auto hef_exp = Hef::create(m.hef_path);
        if (!hef_exp) { std::cerr << m.name << " HEF 로드 실패" << std::endl; return (int)hef_exp.status(); }
        auto hef = hef_exp.release();

        auto cfg_exp = vdevice->create_configure_params(hef);
        if (!cfg_exp) { std::cerr << m.name << " configure params 실패" << std::endl; return (int)cfg_exp.status(); }
        auto cfg = cfg_exp.value();
        for (auto& ng_param : cfg) {
            ng_param.second.batch_size = m.batch;
            ng_param.second.power_mode = HAILO_POWER_MODE_ULTRA_PERFORMANCE;
        }

        auto ngs_exp = vdevice->configure(hef, cfg);
        if (!ngs_exp) { std::cerr << m.name << " configure 실패" << std::endl; return (int)ngs_exp.status(); }
        auto network_group = ngs_exp.value()[0];

        // 실기 확인된 제약: threshold는 batch_size 이하여야 함 (위 정의부 주석 참고).
        // 어길 경우 set_scheduler_threshold가 실패하며 threshold는 기본값(1)으로 남는다.
        if (m.threshold > m.batch)
            std::printf("  [경고] %s: threshold(%d) > batch(%d) — set_scheduler_threshold가 실패할 것으로 예상됨. "
                        "THRESHOLD_* <= BATCH_*로 맞출 것.\n", m.name, m.threshold, m.batch);

        // 스케줄러 파라미터 설정 (network_group.hpp 공식 시그니처와 동일)
        auto st_thr = network_group->set_scheduler_threshold((uint32_t)m.threshold);
        auto st_to  = network_group->set_scheduler_timeout(std::chrono::milliseconds(m.timeout_ms));
        auto st_pri = network_group->set_scheduler_priority((uint8_t)m.priority);
        std::printf("  [적용확인] %-13s: batch=%d, threshold=%d [%s], timeout=%dms [%s], priority=%d [%s]\n",
            m.name, m.batch,
            m.threshold,  (st_thr == HAILO_SUCCESS ? "OK" : "실패"),
            m.timeout_ms, (st_to  == HAILO_SUCCESS ? "OK" : "실패"),
            m.priority,   (st_pri == HAILO_SUCCESS ? "OK" : "실패"));
        if (st_thr != HAILO_SUCCESS || st_to != HAILO_SUCCESS || st_pri != HAILO_SUCCESS)
            std::printf("  [경고] %s 일부 파라미터 적용 실패! (thr=%d to=%d pri=%d)\n",
                        m.name, (int)st_thr, (int)st_to, (int)st_pri);

        network_groups.push_back(network_group);
        active_model_idx.push_back((int)i);
    }

    if (network_groups.empty()) {
        std::cerr << "활성화된 모델이 없습니다 (USE_DET/USE_SEG/USE_POSE 확인)" << std::endl;
        return 1;
    }

    // ── val2017 이미지 로드 + letterbox 전처리 (한 번만 수행, 모든 모델이 공유) ──
    std::vector<std::string> images = get_image_files(IMG_DIR);
    if (images.empty()) {
        std::cerr << "[경고] IMG_DIR(" << IMG_DIR << ")에서 이미지를 찾지 못함. "
                  << "경로를 확인하고 #define IMG_DIR을 수정할 것." << std::endl;
        return 1;
    }
    if (NUM_IMAGES > 0 && images.size() > (size_t)NUM_IMAGES)
        images.resize(NUM_IMAGES);
    std::printf("사용 이미지 수: %zu장 (경로: %s)\n", images.size(), IMG_DIR);

    double t_prep_start = now_ms();
    std::vector<cv::Mat> pre;
    pre.reserve(images.size());
    for (auto& path : images) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;
        cv::Mat lb = letterbox(img, 640);
        cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
        pre.push_back(lb);
    }
    double t_prep_end = now_ms();
    // 장당 평균 전처리시간 — 모든 모델이 동일한 letterbox 전처리를 공유하므로 모델별로
    // 따로 재지 않고, 전체 전처리 구간을 이미지 수로 나눈 평균값을 공통으로 사용한다.
    double avg_preprocess_ms = pre.empty() ? 0.0 : (t_prep_end - t_prep_start) / pre.size();
    std::printf("전처리 완료: %zu장 (약 %.0f MB), 장당 평균 전처리시간=%.3f ms\n\n",
                pre.size(), pre.size() * 640.0 * 640.0 * 3 / 1e6, avg_preprocess_ms);

    // ── vstream 생성 (입력/출력 timeout을 크게 잡음) ──
    // [중요] vstream 기본 timeout은 10초다. 우선순위가 낮은 모델은 높은 모델이 끝날 때까지
    // 10초 넘게 굶을 수 있는데, 그러면 write가 HAILO_TIMEOUT으로 실패하고 내부 파이프라인
    // 스레드가 죽어 프레임이 유실된다(재시도로도 복구 불가). timeout을 크게 주어 방지한다.
    // (공식 API: ConfiguredNetworkGroup::make_input/output_vstream_params + create_input/output_vstreams)
    const uint32_t VSTREAM_TIMEOUT_MS = 300000;  // 5분 — starvation 대기 여유
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_ng;
    std::vector<std::vector<OutMeta>> out_meta_per_ng;  // vstreams_per_ng와 동일 인덱스로 정렬
    for (size_t k = 0; k < network_groups.size(); k++) {
        auto& ng = network_groups[k];
        ModelKind kind = models[active_model_idx[k]].kind;

        auto in_params  = ng->make_input_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                              VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        // Detection: on-chip NMS 출력이라 AUTO 유지(기존 검증된 그대로).
        // Segmentation/Pose: raw tensor라서 후처리(DFL 디코딩) 대상 — FLOAT32로 명시 요청해
        // HailoRT가 qp_scale/qp_zp 역양자화까지 대신 하게 하고, order도 NHWC로 고정한다
        // (parse-hef가 일부 출력을 FCR로 표시했으므로 AUTO에 맡기지 않음).
        hailo_format_type_t out_fmt_type = (kind == ModelKind::DET) ? HAILO_FORMAT_TYPE_AUTO : HAILO_FORMAT_TYPE_FLOAT32;
        auto out_params = ng->make_output_vstream_params(false, out_fmt_type,
                              VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!in_params)  { std::cerr << "input vstream params 실패, status="  << in_params.status()  << std::endl; return (int)in_params.status(); }
        if (!out_params) { std::cerr << "output vstream params 실패, status=" << out_params.status() << std::endl; return (int)out_params.status(); }

        if (kind != ModelKind::DET) {
            for (auto& kv : out_params.value())
                kv.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        }

        auto in_vs  = ng->create_input_vstreams(in_params.value());
        auto out_vs = ng->create_output_vstreams(out_params.value());
        if (!in_vs)  { std::cerr << "input vstream 생성 실패, status="  << in_vs.status()  << std::endl; return (int)in_vs.status(); }
        if (!out_vs) { std::cerr << "output vstream 생성 실패, status=" << out_vs.status() << std::endl; return (int)out_vs.status(); }

        auto out_vs_val = out_vs.release();
        out_meta_per_ng.push_back(classify_outputs(kind, out_vs_val));
        vstreams_per_ng.emplace_back(std::make_pair(in_vs.release(), std::move(out_vs_val)));
    }

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
            std::cref(pre), std::ref(results[mi]));
    }
    for (auto& t : threads) t.join();

    // 장당 전체시간(전처리-추론-후처리) = 공유 평균 전처리시간 + 모델별 평균 latency + 모델별 평균 후처리시간
    for (int i = 0; i < 3; i++) {
        if (models[i].active && results[i].avg_latency_ms >= 0) {
            double pp = (results[i].avg_postprocess_ms >= 0) ? results[i].avg_postprocess_ms : 0.0;
            results[i].avg_total_time_ms = avg_preprocess_ms + results[i].avg_latency_ms + pp;
        }
    }

    double run_time_s = (now_ms() - t_run_start) / 1000.0;   // 추론 구간 실측 wall-time(초)
    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();

    long vol_ctx = 0, nonvol_ctx = 0;
    for (auto& r : results) { vol_ctx += r.vol_ctx; nonvol_ctx += r.nonvol_ctx; }

    std::printf("\n========== 실험 결과 (Run ID: %d) ==========\n", run_id);
    std::printf("장당 평균 전처리시간(공유): %.3f ms\n", avg_preprocess_ms);
    if (USE_DET)  std::printf("Detection    : latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[0].avg_latency_ms, results[0].avg_postprocess_ms, results[0].avg_total_time_ms,
                              results[0].frame_count, BATCH_DET, THRESHOLD_DET, TIMEOUT_DET_MS, PRIORITY_DET);
    if (USE_SEG)  std::printf("Segmentation : latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[1].avg_latency_ms, results[1].avg_postprocess_ms, results[1].avg_total_time_ms,
                              results[1].frame_count, BATCH_SEG, THRESHOLD_SEG, TIMEOUT_SEG_MS, PRIORITY_SEG);
    if (USE_POSE) std::printf("Pose         : latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[2].avg_latency_ms, results[2].avg_postprocess_ms, results[2].avg_total_time_ms,
                              results[2].frame_count, BATCH_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, PRIORITY_POSE);
    std::printf("CPU: %.2f%%, MEM: %.2f%%, Ctx Switch(vol/nonvol): %ld/%ld\n", final_cpu, final_mem, vol_ctx, nonvol_ctx);
    std::printf("================================================\n");
    std::printf("HRTT 트레이스를 PC/WSL에서 `hailo runtime-profiler <파일>.hrtt`로 변환한 뒤,\n"
                "core_op_set_value 이벤트에서 위 [적용확인] 값과 실제 적용값이 일치하는지 확인할 것.\n");

    // ── CSV 저장 (argv[2]로 경로가 주어졌을 때만) ──
    // 추론 중 측정 가능한 값만 채우고, HRTT/모니터 전용 값은 NaN으로 남긴다.
    if (!csv_path.empty())
        save_csv(csv_path, run_id, models, results,
                 final_cpu, final_mem, vol_ctx, nonvol_ctx, run_time_s, avg_preprocess_ms);

    return HAILO_SUCCESS;
}
