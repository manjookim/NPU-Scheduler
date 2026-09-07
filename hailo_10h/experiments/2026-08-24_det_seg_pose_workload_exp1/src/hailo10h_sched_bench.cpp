// hailo10h_sched_bench.cpp
// Hailo10H: det(yolov8s) / seg(yolov8s_seg) / pose(yolov8s_pose) 워크로드 조합 실험.
// 스케줄러 파라미터(priority/threshold/timeout/batch)는 옵션으로 주지 않으면 절대 건드리지
// 않는다 — VDevice::create() 기본값 그대로, 여러 InferModel을 한 VDevice에 올려서 HailoRT
// 내장 스케줄러가 시분할하게 둔다. 옵션을 준 경우에만 setter를 호출한다(H10에서도
// ConfiguredInferModelHrpcClient가 RPC로 디바이스 스케줄러에 전달해준다).
// 목적이 정확도가 아니라 워크로드 개수에 따른 latency/FPS 스케일링 측정이라, bbox/mask/keypoint
// 디코딩은 하지 않고 raw output buffer를 그대로 받아 타이밍만 잰다.
//
// [CSV 컬럼] hailo_8L/csv_writer.hpp의 컬럼 구성을 그대로 따르되, HRTT에서만 얻을 수 있는
// 것(switches_per_s / idle_time_pct / activation_*)은 뺐다 — H10은 추론 스택이 디바이스 안
// hailort_server에 있어서 호스트 .hrtt가 항상 0바이트다(HRTT_ON_HAILO10H.md 참고).
// 8L에서 HRTT가 채워주던 avg_fps_* / max_latency_*는 호스트 실측으로 대체해 채운다.
//
// 사용법:
//   ./hailo10h_sched_bench --models det,seg,pose --label det_seg_pose --run_id 1 \
//       --csv csv/results.csv --img_dir /home/npu-rpi5/datasets/sampled_val2017 [--num_images 0] \
//       [--trace_dir traces] [--no_trace] [--no_npu_monitor] \
//       [--batch N] [--threshold N|N,N,N] [--timeout_ms N] [--priority N|N,N,N]
//   (--threshold/--priority는 det,seg,pose 순서. 값 하나만 주면 세 모델에 같은 값 적용)
//
// 타임라인 트레이스: HailoRT의 .hrtt는 H10에서 항상 비므로(이유는 host_trace.hpp 주석 참고)
// 호스트에서 관측 가능한 구간을 host_trace가 직접 traces/에 남긴다. --no_trace로 끌 수 있다.

#include "hailo/hailort.hpp"
#include "host_trace.hpp"
#include "npu_monitor.hpp"
#include "sys_monitor.hpp"

#include <opencv2/opencv.hpp>

#include <sys/mman.h>
#include <dirent.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

using namespace hailort;
using Clock = std::chrono::steady_clock;

// 모델 컬럼 순서 — 8L CSV와 동일하게 det, seg, pose 고정.
static const char *MODEL_ORDER[3] = {"det", "seg", "pose"};

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size) {
    void *addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *p){ munmap(p, size); });
}

struct ModelSpec {
    std::string key;        // det / seg / pose
    std::string hef_path;
    int input_w = 640, input_h = 640;
};

static const std::map<std::string, ModelSpec> MODEL_REGISTRY = {
    {"det",  {"det",  "resources/yolov8s.hef",          640, 640}},
    {"seg",  {"seg",  "resources/yolov8s_seg.hef",       640, 640}},
    {"pose", {"pose", "resources/yolov8s_pose_h10.hef",  640, 640}},
};

// 스케줄러/배치 파라미터. set_* 플래그가 false면 setter를 아예 호출하지 않는다(= HailoRT 기본
// 동작 그대로). CSV에는 어느 쪽이든 "실제로 적용된 값"을 적어야 하므로, 초기값을 HailoRT가
// 헤더 주석에 명시한 기본값으로 잡아둔다.
//   batch     : HAILO_DEFAULT_BATCH_SIZE(0) = HailoRT가 자동 결정
//   threshold : 1        (infer_model.hpp "The default threshold is 1")
//   timeout   : 0 ms     (infer_model.hpp "The default timeout is 0ms")
//   priority  : 16       (HAILO_SCHEDULER_PRIORITY_NORMAL)
struct SchedParams {
    bool set_batch = false, set_threshold = false, set_timeout = false, set_priority = false;
    uint16_t batch = HAILO_DEFAULT_BATCH_SIZE;
    uint32_t threshold[3] = {1, 1, 1};
    uint32_t timeout_ms = 0;
    uint32_t priority[3] = {HAILO_SCHEDULER_PRIORITY_NORMAL,
                            HAILO_SCHEDULER_PRIORITY_NORMAL,
                            HAILO_SCHEDULER_PRIORITY_NORMAL};
};

static std::vector<std::string> split_csv_arg(const std::string &s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}

// "16" -> 세 모델 모두 16 / "16,20,16" -> det,seg,pose 각각. 그 외 개수는 에러.
static bool parse_triple(const std::string &arg, uint32_t out[3], const char *flag) {
    auto parts = split_csv_arg(arg);
    if (parts.size() == 1) {
        out[0] = out[1] = out[2] = (uint32_t)std::stoul(parts[0]);
        return true;
    }
    if (parts.size() == 3) {
        for (int i = 0; i < 3; ++i) out[i] = (uint32_t)std::stoul(parts[i]);
        return true;
    }
    std::cerr << flag << "는 값 1개 또는 3개(det,seg,pose)만 가능: " << arg << "\n";
    return false;
}

static int model_index(const std::string &key) {
    for (int i = 0; i < 3; ++i) if (key == MODEL_ORDER[i]) return i;
    return -1;
}

static std::vector<std::string> list_images(const std::string &dir) {
    std::vector<std::string> files;
    DIR *d = opendir(dir.c_str());
    if (!d) return files;
    struct dirent *ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() > 4) {
            std::string ext = name.substr(name.size() - 4);
            for (auto &c : ext) c = tolower(c);
            if (ext == ".jpg" || ext == "jpeg" || ext == ".png") {
                files.push_back(dir + "/" + name);
            }
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

struct ResultRow {
    size_t frame_count = 0;
    double avg_latency_ms = 0.0;   // run_async 제출 ~ wait 완료 (전처리 제외)
    double max_latency_ms = 0.0;   // 8L에서 HRTT가 주던 max_latency_* 대체 — 호스트 실측
    double fps = 0.0;
    double total_time_s = 0.0;
    double avg_preprocess_ms = -1.0;  // imread+cvtColor+resize+memcpy 장당 평균
    long vol_ctx = 0, nonvol_ctx = 0; // 이 모델 워커 스레드의 context switch 증가분
    bool ok = false;
};

static std::mutex g_result_mutex;
static std::map<std::string, ResultRow> g_results;  // key = det/seg/pose

struct WorkerCtx {
    ModelSpec spec;
    std::shared_ptr<InferModel> infer_model;
    std::shared_ptr<ConfiguredInferModel> configured;
    std::shared_ptr<ConfiguredInferModel::Bindings> bindings;
    std::string input_name;
    size_t input_frame_size = 0;
    std::shared_ptr<uint8_t> input_buffer;
    std::vector<std::shared_ptr<uint8_t>> output_buffers;  // guard only
    host_trace::ModelTrace *trace = nullptr;               // 이 워커의 타임라인 버퍼
};

// 메인 스레드에서 순차적으로 준비 (create_infer_model/configure/bindings는 동시성 검증 안 된
// 경로라 안전하게 직렬로 처리하고, 실제 동시 실행은 워커 스레드의 run_async 루프에서만 일어난다).
static bool prepare_worker(VDevice &vdevice, WorkerCtx &ctx, const SchedParams &sp) {
    auto infer_model_exp = vdevice.create_infer_model(ctx.spec.hef_path);
    if (!infer_model_exp) {
        std::cerr << "[" << ctx.spec.key << "] create_infer_model 실패: status="
                  << infer_model_exp.status() << std::endl;
        return false;
    }
    ctx.infer_model = infer_model_exp.release();

    // batch는 configure() 전에 정해져야 한다.
    if (sp.set_batch) ctx.infer_model->set_batch_size(sp.batch);

    auto configured_exp = ctx.infer_model->configure();
    if (!configured_exp) {
        std::cerr << "[" << ctx.spec.key << "] configure 실패: status="
                  << configured_exp.status() << std::endl;
        return false;
    }
    ctx.configured = std::make_shared<ConfiguredInferModel>(configured_exp.release());

    // 스케줄러 파라미터는 요청받은 것만 건드린다. H10에서도 RPC로 디바이스 스케줄러에 전달된다.
    const int mi = model_index(ctx.spec.key);
    if (sp.set_threshold) {
        auto st = ctx.configured->set_scheduler_threshold(sp.threshold[mi]);
        if (st != HAILO_SUCCESS) {
            std::cerr << "[" << ctx.spec.key << "] set_scheduler_threshold 실패: status=" << st << std::endl;
            return false;
        }
    }
    if (sp.set_timeout) {
        auto st = ctx.configured->set_scheduler_timeout(std::chrono::milliseconds(sp.timeout_ms));
        if (st != HAILO_SUCCESS) {
            std::cerr << "[" << ctx.spec.key << "] set_scheduler_timeout 실패: status=" << st << std::endl;
            return false;
        }
    }
    if (sp.set_priority) {
        auto st = ctx.configured->set_scheduler_priority((uint8_t)sp.priority[mi]);
        if (st != HAILO_SUCCESS) {
            std::cerr << "[" << ctx.spec.key << "] set_scheduler_priority 실패: status=" << st << std::endl;
            return false;
        }
    }

    auto bindings_exp = ctx.configured->create_bindings();
    if (!bindings_exp) {
        std::cerr << "[" << ctx.spec.key << "] create_bindings 실패: status="
                  << bindings_exp.status() << std::endl;
        return false;
    }
    ctx.bindings = std::make_shared<ConfiguredInferModel::Bindings>(bindings_exp.release());

    ctx.input_name = ctx.infer_model->get_input_names()[0];
    ctx.input_frame_size = ctx.infer_model->input(ctx.input_name)->get_frame_size();
    ctx.input_buffer = page_aligned_alloc(ctx.input_frame_size);
    auto st = ctx.bindings->input(ctx.input_name)->set_buffer(
        MemoryView(ctx.input_buffer.get(), ctx.input_frame_size));
    if (st != HAILO_SUCCESS) {
        std::cerr << "[" << ctx.spec.key << "] input set_buffer 실패: status=" << st << std::endl;
        return false;
    }

    for (const auto &out_name : ctx.infer_model->get_output_names()) {
        size_t out_size = ctx.infer_model->output(out_name)->get_frame_size();
        auto out_buf = page_aligned_alloc(out_size);
        auto ost = ctx.bindings->output(out_name)->set_buffer(MemoryView(out_buf.get(), out_size));
        if (ost != HAILO_SUCCESS) {
            std::cerr << "[" << ctx.spec.key << "] output(" << out_name << ") set_buffer 실패: status="
                      << ost << std::endl;
            return false;
        }
        ctx.output_buffers.push_back(out_buf);
    }

    std::cout << "[" << ctx.spec.key << "] 준비 완료 (input_frame_size=" << ctx.input_frame_size
              << ", outputs=" << ctx.output_buffers.size() << ")" << std::endl;
    return true;
}

static void worker_loop(WorkerCtx *ctx, const std::vector<std::string> *images) {
    // 컨텍스트 스위치는 스레드가 살아 있는 동안 자기 값을 읽어야 한다(8L model_runner.hpp와 동일).
    CtxSwitches cs0 = read_thread_ctx_switches();

    double sum_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double prep_total_ms = 0.0;
    long prep_count = 0;
    size_t frame_count = 0;
    size_t frame_idx = 0;
    auto t_start = Clock::now();

    for (const auto &img_path : *images) {
        const size_t idx = frame_idx++;
        double prep_begin_us = host_trace::now_us();
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;
        cv::Mat rgb, resized;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(ctx->spec.input_w, ctx->spec.input_h));
        if (resized.total() * resized.elemSize() != ctx->input_frame_size) {
            std::cerr << "[" << ctx->spec.key << "] 프레임 크기 불일치, 건너뜀" << std::endl;
            continue;
        }
        std::memcpy(ctx->input_buffer.get(), resized.data, ctx->input_frame_size);
        double prep_end_us = host_trace::now_us();
        ctx->trace->add_prep(idx, prep_begin_us, prep_end_us);
        prep_total_ms += (prep_end_us - prep_begin_us) / 1000.0;
        prep_count++;

        auto t0 = Clock::now();
        double infer_begin_us = host_trace::now_us();
        auto job_exp = ctx->configured->run_async(*ctx->bindings);
        // run_async가 반환하기까지 걸린 시간 = 디바이스측 큐가 꽉 차서 밀린 정도(backpressure).
        // H10에서는 스케줄러가 디바이스 안에 있어서, 호스트가 경합을 관측할 수 있는 거의 유일한 창구다.
        double enqueue_ms = (host_trace::now_us() - infer_begin_us) / 1000.0;
        if (!job_exp) {
            ctx->trace->add_infer(idx, infer_begin_us, host_trace::now_us(), enqueue_ms, false);
            continue;
        }
        auto status = job_exp->wait(std::chrono::milliseconds(5000));
        auto t1 = Clock::now();
        ctx->trace->add_infer(idx, infer_begin_us, host_trace::now_us(), enqueue_ms,
                              status == HAILO_SUCCESS);
        if (status != HAILO_SUCCESS) continue;

        double lat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum_latency_ms += lat_ms;
        if (lat_ms > max_latency_ms) max_latency_ms = lat_ms;
        frame_count++;
    }

    auto t_end = Clock::now();
    double total_time_s = std::chrono::duration<double>(t_end - t_start).count();
    CtxSwitches cs1 = read_thread_ctx_switches();

    ResultRow row;
    row.frame_count = frame_count;
    row.avg_latency_ms = frame_count ? sum_latency_ms / frame_count : 0.0;
    row.max_latency_ms = frame_count ? max_latency_ms : -1.0;
    row.fps = total_time_s > 0 ? (double)frame_count / total_time_s : 0.0;
    row.total_time_s = total_time_s;
    row.avg_preprocess_ms = prep_count ? prep_total_ms / prep_count : -1.0;
    row.vol_ctx = cs1.voluntary - cs0.voluntary;
    row.nonvol_ctx = cs1.nonvoluntary - cs0.nonvoluntary;
    row.ok = frame_count > 0;

    std::lock_guard<std::mutex> lk(g_result_mutex);
    g_results[ctx->spec.key] = row;
}

// 8L csv_writer.hpp와 같은 규약: 음수(-1) = 미측정/비활성 → NaN
static std::string dtos(double v) {
    if (v < 0) return "NaN";
    std::ostringstream os; os << v; return os.str();
}

// hailo_8L/csv_writer.hpp의 헤더에서 HRTT 전용 컬럼(switches_per_s, idle_time_pct,
// activation_det/seg/pose)만 뺀 것. label과 *_frame_count는 H10 스윕에서 조건 식별·검증에
// 필요해서 추가로 둔다.
static const char *CSV_HEADER =
    "label,run_id,use_det,use_seg,use_pose,batch,"
    "threshold_det,threshold_seg,threshold_pose,timeout_ms,"
    "priority_det,priority_seg,priority_pose,"
    "det_latency_ms,seg_latency_ms,pose_latency_ms,"
    "cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches,"
    "npu_percent,run_time_s,"
    "det_frame_count,avg_fps_det,max_latency_det,"
    "seg_frame_count,avg_fps_seg,max_latency_seg,"
    "pose_frame_count,avg_fps_pose,max_latency_pose,"
    "total_time_det_s,total_time_seg_s,total_time_pose_s,"
    "avg_preprocess_ms_det,avg_preprocess_ms_seg,avg_preprocess_ms_pose,"
    "postprocess_ms_det,postprocess_ms_seg,postprocess_ms_pose,"
    "total_time_ms_det,total_time_ms_seg,total_time_ms_pose,"
    "total_time_ms_nopp_det,total_time_ms_nopp_seg,total_time_ms_nopp_pose";

int main(int argc, char **argv) {
    std::string models_arg, label = "run", csv_path = "csv/results.csv";
    std::string img_dir = "/home/npu-rpi5/datasets/sampled_val2017";
    std::string trace_dir = "traces";
    int run_id = 1;
    int num_images = 0;  // 0 = 전체
    bool trace_enabled = true;
    bool npu_monitor_enabled = true;
    SchedParams sp;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *flag) -> std::string {
            if (i + 1 >= argc) { std::cerr << flag << " 값 없음\n"; exit(1); }
            return argv[++i];
        };
        if (a == "--models") models_arg = next("--models");
        else if (a == "--label") label = next("--label");
        else if (a == "--run_id") run_id = std::stoi(next("--run_id"));
        else if (a == "--csv") csv_path = next("--csv");
        else if (a == "--img_dir") img_dir = next("--img_dir");
        else if (a == "--num_images") num_images = std::stoi(next("--num_images"));
        else if (a == "--trace_dir") trace_dir = next("--trace_dir");
        else if (a == "--no_trace") trace_enabled = false;
        else if (a == "--no_npu_monitor") npu_monitor_enabled = false;
        else if (a == "--batch") { sp.batch = (uint16_t)std::stoul(next("--batch")); sp.set_batch = true; }
        else if (a == "--timeout_ms") { sp.timeout_ms = (uint32_t)std::stoul(next("--timeout_ms")); sp.set_timeout = true; }
        else if (a == "--threshold") {
            if (!parse_triple(next("--threshold"), sp.threshold, "--threshold")) return 1;
            sp.set_threshold = true;
        }
        else if (a == "--priority") {
            if (!parse_triple(next("--priority"), sp.priority, "--priority")) return 1;
            sp.set_priority = true;
        }
        else { std::cerr << "알 수 없는 옵션: " << a << "\n"; return 1; }
    }

    if (models_arg.empty()) {
        std::cerr << "--models det,seg,pose 형식으로 지정할 것\n";
        return 1;
    }
    auto model_keys = split_csv_arg(models_arg);

    // 기존 CSV에 이어 붙일 때 헤더가 다르면(= 예전 컬럼 구성) 멈춘다.
    // 예전 스윕 결과 파일에 새 스키마 행이 섞여 들어가는 사고를 막기 위함.
    {
        std::ifstream chk(csv_path);
        std::string first;
        if (chk.good() && std::getline(chk, first)) {
            if (!first.empty() && first.back() == '\r') first.pop_back();
            if (first != CSV_HEADER) {
                std::cerr << "CSV 헤더가 현재 컬럼 구성과 다릅니다: " << csv_path << "\n"
                          << "  (예전 스키마 파일로 보입니다. 다른 파일명을 쓰거나 파일을 옮기세요)\n";
                return 1;
            }
        }
    }

    std::vector<std::string> images = list_images(img_dir);
    if (images.empty()) {
        std::cerr << "이미지 없음: " << img_dir << "\n";
        return 1;
    }
    if (num_images > 0 && (int)images.size() > num_images) {
        images.resize(num_images);
    }
    std::cout << "이미지 " << images.size() << "장, 조건=" << label
              << ", run_id=" << run_id << ", 모델=" << models_arg << std::endl;

    // 트레이서를 VDevice보다 먼저 만들어 시간 원점을 확정한다.
    host_trace::HostTracer tracer(trace_dir, label, run_id, trace_enabled);

    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "VDevice::create 실패: status=" << vdevice_exp.status() << std::endl;
        return 1;
    }
    auto vdevice = vdevice_exp.release();

    std::vector<std::unique_ptr<WorkerCtx>> ctxs;
    for (const auto &key : model_keys) {
        auto it = MODEL_REGISTRY.find(key);
        if (it == MODEL_REGISTRY.end()) {
            std::cerr << "알 수 없는 모델 키: " << key << " (det/seg/pose만 가능)\n";
            return 1;
        }
        auto ctx = std::make_unique<WorkerCtx>();
        ctx->spec = it->second;
        ctx->trace = tracer.add_model(key, images.size());
        if (!prepare_worker(*vdevice, *ctx, sp)) {
            std::cerr << "준비 실패: " << key << std::endl;
            return 1;
        }
        ctxs.push_back(std::move(ctx));
    }

    // CPU/메모리 백그라운드 모니터
    std::atomic<bool> monitor_run{true};
    std::vector<double> cpu_samples, mem_samples;
    std::mutex sample_mutex;
    std::thread monitor_thread([&]() {
        CpuStats prev = read_cpu_stats();
        while (monitor_run.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            CpuStats cur = read_cpu_stats();
            double cpu = calc_cpu_usage(prev, cur);
            double mem = read_mem_usage();
            prev = cur;
            std::lock_guard<std::mutex> lk(sample_mutex);
            cpu_samples.push_back(cpu);
            mem_samples.push_back(mem);
        }
    });

    // 디바이스 NNC 가동률(= 8L의 npu_percent). 자식 프로세스로 hailortcli monitor를 띄운다.
    npu_monitor::NpuMonitor npu(npu_monitor_enabled);
    if (npu_monitor_enabled && !npu.start()) {
        std::cerr << "[npu_monitor] 시작 실패 — npu_percent는 NaN으로 기록됩니다" << std::endl;
    }

    auto run_t0 = Clock::now();
    std::vector<std::thread> workers;
    for (auto &ctx : ctxs) {
        workers.emplace_back(worker_loop, ctx.get(), &images);
    }
    for (auto &t : workers) t.join();
    double run_time_s = std::chrono::duration<double>(Clock::now() - run_t0).count();

    npu.stop();
    monitor_run.store(false);
    monitor_thread.join();

    double cpu_avg = 0.0, mem_avg = 0.0;
    {
        std::lock_guard<std::mutex> lk(sample_mutex);
        if (!cpu_samples.empty()) {
            for (double v : cpu_samples) cpu_avg += v;
            cpu_avg /= cpu_samples.size();
        }
        if (!mem_samples.empty()) {
            for (double v : mem_samples) mem_avg += v;
            mem_avg /= mem_samples.size();
        }
    }
    const double npu_avg = npu.avg_npu_percent();

    // 모델별 값 접근 헬퍼. 비활성 모델은 -1 → dtos()가 NaN으로 찍는다(8L과 같은 규약).
    auto has    = [&](int mi) { return g_results.count(MODEL_ORDER[mi]) > 0; };
    auto row_of = [&](int mi) -> const ResultRow & { return g_results[MODEL_ORDER[mi]]; };
    auto lat    = [&](int mi) { return has(mi) ? row_of(mi).avg_latency_ms : -1.0; };
    auto maxlat = [&](int mi) { return has(mi) ? row_of(mi).max_latency_ms : -1.0; };
    auto fps    = [&](int mi) { return has(mi) ? row_of(mi).fps : -1.0; };
    auto tot_s  = [&](int mi) { return has(mi) ? row_of(mi).total_time_s : -1.0; };
    auto prep   = [&](int mi) { return has(mi) ? row_of(mi).avg_preprocess_ms : -1.0; };
    auto frames = [&](int mi) -> std::string {
        if (!has(mi)) return "NaN";
        std::ostringstream o; o << row_of(mi).frame_count; return o.str();
    };
    // 8L 규약: 장당 전체시간 = 전처리 + latency + 후처리(측정한 경우만). 이 벤치는 디코딩을
    // 하지 않으므로 후처리항은 0 → total_time_ms_*와 total_time_ms_nopp_*가 같은 값이 된다.
    auto total_ms = [&](int mi) {
        double p = prep(mi), l = lat(mi);
        return (p >= 0 && l >= 0) ? (p + l) : -1.0;
    };

    long vol_sum = 0, nonvol_sum = 0;
    for (int mi = 0; mi < 3; ++mi) {
        if (!has(mi)) continue;
        vol_sum += row_of(mi).vol_ctx;
        nonvol_sum += row_of(mi).nonvol_ctx;
    }

    bool csv_exists = std::ifstream(csv_path).good();
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv.is_open()) {
        std::cerr << "CSV 열기 실패: " << csv_path << std::endl;
        return 1;
    }
    if (!csv_exists) csv << CSV_HEADER << "\n";

    csv << label << "," << run_id << ","
        << (has(0) ? 1 : 0) << "," << (has(1) ? 1 : 0) << "," << (has(2) ? 1 : 0) << ","
        << sp.batch << ","
        << sp.threshold[0] << "," << sp.threshold[1] << "," << sp.threshold[2] << ","
        << sp.timeout_ms << ","
        << sp.priority[0] << "," << sp.priority[1] << "," << sp.priority[2] << ","
        << dtos(lat(0)) << "," << dtos(lat(1)) << "," << dtos(lat(2)) << ","
        << dtos(cpu_avg) << "," << dtos(mem_avg) << ","
        << vol_sum << "," << nonvol_sum << ","
        << dtos(npu_avg) << "," << dtos(run_time_s) << ","
        << frames(0) << "," << dtos(fps(0)) << "," << dtos(maxlat(0)) << ","
        << frames(1) << "," << dtos(fps(1)) << "," << dtos(maxlat(1)) << ","
        << frames(2) << "," << dtos(fps(2)) << "," << dtos(maxlat(2)) << ","
        << dtos(tot_s(0)) << "," << dtos(tot_s(1)) << "," << dtos(tot_s(2)) << ","
        << dtos(prep(0)) << "," << dtos(prep(1)) << "," << dtos(prep(2)) << ","
        << "NaN,NaN,NaN,"   // postprocess_ms_* — 이 벤치는 디코딩/NMS를 하지 않는다
        << dtos(total_ms(0)) << "," << dtos(total_ms(1)) << "," << dtos(total_ms(2)) << ","
        << dtos(total_ms(0)) << "," << dtos(total_ms(1)) << "," << dtos(total_ms(2)) << "\n";
    csv.close();

    std::cout << "===== 결과 (" << label << " run_id=" << run_id << ") =====" << std::endl;
    for (const auto &kv : g_results) {
        std::cout << "  " << kv.first << ": frame_count=" << kv.second.frame_count
                  << " avg_latency_ms=" << kv.second.avg_latency_ms
                  << " max_latency_ms=" << kv.second.max_latency_ms
                  << " fps=" << kv.second.fps
                  << " prep_ms=" << kv.second.avg_preprocess_ms
                  << " ctx(v/nv)=" << kv.second.vol_ctx << "/" << kv.second.nonvol_ctx << std::endl;
    }
    std::cout << "  cpu_percent=" << cpu_avg << " mem_percent=" << mem_avg
              << " npu_percent=" << dtos(npu_avg)
              << " (샘플 " << npu.sample_count() << "개)"
              << " run_time_s=" << run_time_s << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    if (tracer.enabled()) {
        if (tracer.flush()) {
            std::cout << "타임라인: " << tracer.csv_path() << " / " << tracer.json_path()
                      << " (json은 ui.perfetto.dev 또는 chrome://tracing에서 열림)" << std::endl;
        } else {
            std::cerr << "타임라인 기록 실패 — " << trace_dir << " 디렉토리 확인" << std::endl;
        }
    }

    return 0;
}
