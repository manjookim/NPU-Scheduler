/**
 * infer_scheduler.cpp
 * ------------------------------------------------------------------------
 * Hailo-8L (Raspberry Pi 5) 위에서 Detection / Segmentation / Pose 세 모델을
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
 *   g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
 *
 * 실행:
 *   ./infer_scheduler [run_id]
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
#define THRESHOLD_DET   4
#define THRESHOLD_SEG   6
#define THRESHOLD_POSE  2

// timeout(ms): threshold 미달이어도 최소 1프레임 + 이 시간이 지나면 강제로 실행 자격 부여 (기본값 0)
#define TIMEOUT_DET_MS   200
#define TIMEOUT_SEG_MS   500
#define TIMEOUT_POSE_MS  100

// priority: 0~31, 클수록 우선 (기본값 16=NORMAL). 스케줄 가능한 여러 모델 중 이 값이
// 가장 큰 모델부터 확인한다. 동일하면 Round-Robin.
#define PRIORITY_DET    15
#define PRIORITY_SEG    15
#define PRIORITY_POSE   15

// 어떤 모델을 이번 실행에 포함할지 (0/1)
#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1

// 입력 속도 제한 (모델당 초당 프레임 수). 0 = 제한 없음(최대 속도로 큐를 채움 -> 큐가
// 항상 포화 상태가 되어 threshold/timeout 효과가 거의 관측되지 않음, findings.md 참고).
// threshold/timeout 효과를 보고 싶다면 0보다 큰 값(예: NPU 처리량 근처)으로 설정할 것.
#define INPUT_FPS       0

// 사용할 검증 이미지 수 (0 = IMG_DIR의 전체 이미지 사용)
#define NUM_IMAGES      600
// =====================================================================================

// HEF 경로 (Raspberry Pi 5, hailo-rpi5-examples 리소스)
#define DET_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"
#define SEG_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef"

// val2017 데이터셋 경로 (docs/setup.md 기준 기본 경로).
// 라즈베리파이에 실제로 저장된 경로가 다르면 이 값만 수정 후 재컴파일할 것
// (예: sampled_val2017을 쓰는 경우 "/home/rpi1/datasets/sampled_val2017/").
#define IMG_DIR  "/home/rpi1/datasets/coco/val2017/"

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
};

void run_model_async(const char* model_name,
                     std::vector<InputVStream>& inputs,
                     std::vector<OutputVStream>& outputs,
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
            auto status = inputs[0].write(MemoryView(pre[i].data, pre[i].total() * pre[i].elemSize()));
            if (HAILO_SUCCESS != status) { write_status = status; }
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary; w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    std::thread reader([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        std::vector<std::vector<uint8_t>> obuf(outputs.size());
        for (size_t j = 0; j < outputs.size(); j++)
            obuf[j].resize(outputs[j].get_frame_size());

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < outputs.size(); j++) {
                auto status = outputs[j].read(MemoryView(obuf[j].data(), obuf[j].size()));
                if (HAILO_SUCCESS != status) { read_status = status; }
            }
            deq_ts[i] = now_ms();  // 이 프레임의 모든 출력을 다 받은 시각
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

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

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[%s] 완료: 평균 Latency=%.2f ms, %d장 (async, INPUT_FPS=%d)\n",
                model_name, result.avg_latency_ms, result.frame_count, INPUT_FPS);
}

// ========================= main =========================

int main(int argc, char* argv[])
{
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;

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

    struct ModelConfig {
        const char* hef_path;
        const char* name;
        int priority;
        int threshold;
        int timeout_ms;
        int batch;
        bool active;
    };
    std::vector<ModelConfig> models = {
        {DET_HEF,  "Detection",    PRIORITY_DET,  THRESHOLD_DET,  TIMEOUT_DET_MS,  BATCH_DET,  (bool)USE_DET},
        {SEG_HEF,  "Segmentation", PRIORITY_SEG,  THRESHOLD_SEG,  TIMEOUT_SEG_MS,  BATCH_SEG,  (bool)USE_SEG},
        {POSE_HEF, "Pose",         PRIORITY_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, BATCH_POSE, (bool)USE_POSE},
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

    std::vector<cv::Mat> pre;
    pre.reserve(images.size());
    for (auto& path : images) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;
        cv::Mat lb = letterbox(img, 640);
        cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
        pre.push_back(lb);
    }
    std::printf("전처리 완료: %zu장 (약 %.0f MB)\n\n",
                pre.size(), pre.size() * 640.0 * 640.0 * 3 / 1e6);

    // ── vstream 생성 (모델별로 한 번에: create_vstreams) ──
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_ng;
    for (auto& ng : network_groups) {
        auto vs_exp = VStreamsBuilder::create_vstreams(*ng, {}, HAILO_FORMAT_TYPE_AUTO);
        if (!vs_exp) { std::cerr << "vstream 생성 실패, status=" << vs_exp.status() << std::endl; return (int)vs_exp.status(); }
        vstreams_per_ng.emplace_back(vs_exp.release());
    }

    // ── 측정 시작 ──
    CpuStats cpu_start = read_cpu_stats();

    ModelResult results[3];  // index: Detection=0, Segmentation=1, Pose=2 (models 배열과 동일 순서)
    std::vector<std::thread> threads;
    for (size_t k = 0; k < vstreams_per_ng.size(); k++) {
        int mi = active_model_idx[k];
        threads.emplace_back(run_model_async, models[mi].name,
            std::ref(vstreams_per_ng[k].first), std::ref(vstreams_per_ng[k].second),
            std::cref(pre), std::ref(results[mi]));
    }
    for (auto& t : threads) t.join();

    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();

    long vol_ctx = 0, nonvol_ctx = 0;
    for (auto& r : results) { vol_ctx += r.vol_ctx; nonvol_ctx += r.nonvol_ctx; }

    std::printf("\n========== 실험 결과 (Run ID: %d) ==========\n", run_id);
    if (USE_DET)  std::printf("Detection    : latency=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[0].avg_latency_ms, results[0].frame_count, BATCH_DET, THRESHOLD_DET, TIMEOUT_DET_MS, PRIORITY_DET);
    if (USE_SEG)  std::printf("Segmentation : latency=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[1].avg_latency_ms, results[1].frame_count, BATCH_SEG, THRESHOLD_SEG, TIMEOUT_SEG_MS, PRIORITY_SEG);
    if (USE_POSE) std::printf("Pose         : latency=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[2].avg_latency_ms, results[2].frame_count, BATCH_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, PRIORITY_POSE);
    std::printf("CPU: %.2f%%, MEM: %.2f%%, Ctx Switch(vol/nonvol): %ld/%ld\n", final_cpu, final_mem, vol_ctx, nonvol_ctx);
    std::printf("================================================\n");
    std::printf("HRTT 트레이스를 PC/WSL에서 `hailo runtime-profiler <파일>.hrtt`로 변환한 뒤,\n"
                "core_op_set_value 이벤트에서 위 [적용확인] 값과 실제 적용값이 일치하는지 확인할 것.\n");

    return HAILO_SUCCESS;
}
