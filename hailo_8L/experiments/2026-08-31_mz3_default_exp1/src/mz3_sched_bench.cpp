// mz3_sched_bench.cpp — Hailo Model Zoo 3모델(ssd_mobilenet_v1 / deeplab_v3_mobilenet_v2_wo_dilation /
// mobilenet_v2_1.0) 스케줄러 벤치마크 (Hailo-8L).
//
// [이 프로그램의 핵심 조건] "파라미터 미설정(default)" 실험 전용.
//   - set_scheduler_threshold / set_scheduler_timeout / set_scheduler_priority 를 **호출하지 않는다**.
//   - configure_params 의 batch_size / power_mode 도 **건드리지 않는다** (HailoRT 기본값 그대로).
//   - 즉 HailoRT 스케줄러가 자기 기본값(priority=NORMAL(16), threshold=1, timeout=0, batch=default)
//     으로 동작할 때의 latency/FPS 를 측정한다.
//   - 예외: output/input vstream 의 host 측 timeout 만 크게 잡는다(기본 10초). 이건 스케줄링 정책이
//     아니라 "굶은 모델의 write/read 가 10초 만에 실패해 프레임이 유실되는 것"을 막는 host 측 안전장치라,
//     기존 실험 코드(infer_scheduler.cpp)와 동일하게 유지한다.
//
// 측정 방식은 기존 hailo_cpp_test/model_runner.hpp 와 동일하게 맞춰 이전 실험과 비교 가능하게 했다:
//   - writer 스레드: (프레임별 전처리) -> enq_ts[i]=now -> input.write()
//   - reader 스레드: output.read() -> deq_ts[i]=now
//   - latency = mean(deq_ts[i] - enq_ts[i])
//
// [전처리 주의] 이 3모델은 YOLO 계열이 아니라 letterbox 가 아닌 단순 resize 를 쓴다
//   (Model Zoo 의 해당 모델 전처리 기준). 따라서 cv::resize(정사각 입력) + BGR2RGB 만 수행한다.
//
// 빌드:
//   g++ -O2 -std=c++17 mz3_sched_bench.cpp -o mz3_sched_bench \
//       $(pkg-config --cflags --libs opencv4) -lhailort -lpthread

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "sys_monitor.hpp"

using namespace hailort;

static std::mutex print_mutex;

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ───────────────────────────── 모델 정의 ─────────────────────────────
struct Model {
    std::string key;        // ssd / deeplab / mnv2
    std::string name;       // CSV/로그 표시용
    std::string hef_path;
    int img_size = 0;       // 정사각 입력 한 변
    bool active = false;

    // 결과
    double avg_latency_ms = -1;
    double avg_preprocess_ms = -1;
    double total_time_s = -1;
    double fps = -1;
    int frame_count = 0;
    long vol_ctx = 0, nonvol_ctx = 0;
};

// ───────────────────────────── 이미지 목록 ─────────────────────────────
static std::vector<std::string> list_images(const std::string& dir, size_t want) {
    std::vector<std::string> v;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        std::cerr << "[에러] 이미지 디렉터리 열기 실패: " << dir << std::endl;
        return v;
    }
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string n = e->d_name;
        if (n.size() < 5) continue;
        std::string ext = n.substr(n.find_last_of('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == "jpg" || ext == "jpeg" || ext == "png")
            v.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(v.begin(), v.end());
    if (v.empty()) return v;
    // 원하는 프레임 수만큼 순환 반복해서 채운다 (이미지 수 < 프레임 수인 경우 대비)
    std::vector<std::string> out;
    out.reserve(want);
    for (size_t i = 0; i < want; i++) out.push_back(v[i % v.size()]);
    return out;
}

// ───────────────────────────── 모델 1개 실행 ─────────────────────────────
// writer/reader 스레드 쌍. 기존 model_runner.hpp 와 동일한 계측 지점.
static void run_model_async(Model& m,
                            std::vector<InputVStream>& inputs,
                            std::vector<OutputVStream>& outputs,
                            const std::vector<std::string>& images,
                            int input_fps) {
    if (inputs.empty() || outputs.empty()) {
        std::lock_guard<std::mutex> lk(print_mutex);
        std::cerr << "[" << m.name << "] 입력/출력 vstream 없음, 스킵" << std::endl;
        return;
    }

    const size_t N = images.size();
    const size_t expected_frame_size = inputs[0].get_frame_size();
    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0);
    long w_vol = 0, w_nonvol = 0, r_vol = 0, r_nonvol = 0;
    double prep_total_ms = 0.0;
    long prep_count = 0;
    hailo_status write_status = HAILO_SUCCESS, read_status = HAILO_SUCCESS;

    const double t_start = now_ms();

    std::thread writer([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        const double interval_ms = (input_fps > 0) ? (1000.0 / input_fps) : 0.0;
        double next_t = now_ms();

        for (size_t i = 0; i < N; i++) {
            if (input_fps > 0) {
                double t = now_ms();
                if (t < next_t)
                    std::this_thread::sleep_for(
                        std::chrono::duration<double, std::milli>(next_t - t));
                next_t += interval_ms;
            }

            // ── 전처리: imread -> resize(정사각) -> BGR2RGB ──
            double p0 = now_ms();
            cv::Mat img = cv::imread(images[i]);
            cv::Mat in;
            if (img.empty()) {
                std::lock_guard<std::mutex> lk(print_mutex);
                std::cerr << "[" << m.name << "] [경고] 이미지 로드 실패: " << images[i]
                          << " (검은 화면 대체)" << std::endl;
                in = cv::Mat::zeros(m.img_size, m.img_size, CV_8UC3);
            } else {
                cv::resize(img, in, cv::Size(m.img_size, m.img_size));
                cv::cvtColor(in, in, cv::COLOR_BGR2RGB);
            }
            if (!in.isContinuous()) in = in.clone();
            double p1 = now_ms();
            prep_total_ms += (p1 - p0);
            prep_count++;

            if (i == 0 && in.total() * in.elemSize() != expected_frame_size) {
                std::lock_guard<std::mutex> lk(print_mutex);
                std::cerr << "[" << m.name << "] [경고] 프레임 크기 불일치: 모델 기대="
                          << expected_frame_size
                          << "B, 전처리 결과=" << (in.total() * in.elemSize()) << "B" << std::endl;
            }

            enq_ts[i] = now_ms();
            hailo_status st;
            do {
                st = inputs[0].write(MemoryView(in.data, in.total() * in.elemSize()));
            } while (st == HAILO_TIMEOUT);
            if (st != HAILO_SUCCESS) write_status = st;
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary;
        w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    std::thread reader([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        std::vector<std::vector<uint8_t>> obuf(outputs.size());
        for (size_t j = 0; j < outputs.size(); j++)
            obuf[j].resize(outputs[j].get_frame_size());

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < outputs.size(); j++) {
                hailo_status st;
                do {
                    st = outputs[j].read(MemoryView(obuf[j].data(), obuf[j].size()));
                } while (st == HAILO_TIMEOUT);
                if (st != HAILO_SUCCESS) read_status = st;
            }
            deq_ts[i] = now_ms();  // 이 프레임의 모든 출력을 다 받은 시각
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary;
        r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

    const double t_end = now_ms();

    if (write_status != HAILO_SUCCESS || read_status != HAILO_SUCCESS) {
        std::lock_guard<std::mutex> lk(print_mutex);
        std::cerr << "[" << m.name << "] [경고] write_status=" << write_status
                  << " read_status=" << read_status << std::endl;
    }

    double sum = 0.0;
    int cnt = 0;
    for (size_t i = 0; i < N; i++) {
        if (deq_ts[i] > 0 && enq_ts[i] > 0) {
            sum += (deq_ts[i] - enq_ts[i]);
            cnt++;
        }
    }
    m.avg_latency_ms = (cnt > 0) ? (sum / cnt) : -1;
    m.frame_count = cnt;
    m.total_time_s = (t_end - t_start) / 1000.0;
    m.fps = (m.total_time_s > 0) ? (cnt / m.total_time_s) : -1;
    m.avg_preprocess_ms = (prep_count > 0) ? (prep_total_ms / prep_count) : -1;
    m.vol_ctx = w_vol + r_vol;
    m.nonvol_ctx = w_nonvol + r_nonvol;
}

// ───────────────────────────── CSV ─────────────────────────────
static void write_csv(const std::string& path, const std::string& tag, int run_id, int input_fps,
                      int frames, const std::vector<Model>& models, double cpu, double mem,
                      double wall_s) {
    bool need_header = true;
    {
        std::ifstream f(path);
        if (f.good() && f.peek() != std::ifstream::traits_type::eof()) need_header = false;
    }
    std::ofstream f(path, std::ios::app);
    if (!f) {
        std::cerr << "[에러] CSV 열기 실패: " << path << std::endl;
        return;
    }
    if (need_header) {
        f << "tag,run_id,input_fps,frames,"
             "use_ssd,use_deeplab,use_mnv2,"
             "ssd_latency_ms,deeplab_latency_ms,mnv2_latency_ms,"
             "ssd_fps,deeplab_fps,mnv2_fps,"
             "ssd_preprocess_ms,deeplab_preprocess_ms,mnv2_preprocess_ms,"
             "ssd_total_time_s,deeplab_total_time_s,mnv2_total_time_s,"
             "cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches,wall_time_s\n";
    }

    auto get = [&](const std::string& k) -> const Model* {
        for (auto& m : models)
            if (m.key == k) return &m;
        return nullptr;
    };
    auto fld = [&](const std::string& k, int which) {
        const Model* m = get(k);
        if (!m || !m->active) {
            f << "NaN";
            return;
        }
        switch (which) {
            case 0: f << m->avg_latency_ms; break;
            case 1: f << m->fps; break;
            case 2: f << m->avg_preprocess_ms; break;
            case 3: f << m->total_time_s; break;
        }
    };

    long vol = 0, nonvol = 0;
    for (auto& m : models)
        if (m.active) { vol += m.vol_ctx; nonvol += m.nonvol_ctx; }

    f << tag << "," << run_id << "," << input_fps << "," << frames << ",";
    for (auto k : {"ssd", "deeplab", "mnv2"}) {
        const Model* m = get(k);
        f << ((m && m->active) ? 1 : 0) << ",";
    }
    for (int which : {0, 1, 2, 3}) {
        for (auto k : {"ssd", "deeplab", "mnv2"}) { fld(k, which); f << ","; }
    }
    f << cpu << "," << mem << "," << vol << "," << nonvol << "," << wall_s << "\n";
}

// ───────────────────────────── main ─────────────────────────────
int main(int argc, char** argv) {
    std::string images_dir = "/home/npu-rpi1/datasets/sampled_val2017";
    std::string res_dir = "/home/npu-rpi1/mz3_exp/resources";
    std::string csv_path = "/home/npu-rpi1/mz3_exp/results_mz3_default.csv";
    std::string tag = "default";
    int input_fps = 0;   // 0 = 최대속도(무제한)
    int frames = 300;    // 모델당 프레임 수
    int run_id = 1;
    bool use_ssd = false, use_deeplab = false, use_mnv2 = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--ssd") use_ssd = std::stoi(next());
        else if (a == "--deeplab") use_deeplab = std::stoi(next());
        else if (a == "--mnv2") use_mnv2 = std::stoi(next());
        else if (a == "--fps") input_fps = std::stoi(next());
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--run-id") run_id = std::stoi(next());
        else if (a == "--images") images_dir = next();
        else if (a == "--res") res_dir = next();
        else if (a == "--csv") csv_path = next();
        else if (a == "--tag") tag = next();
        else {
            std::cerr << "알 수 없는 인자: " << a << std::endl;
            return 1;
        }
    }

    std::vector<Model> models = {
        {"ssd",     "ssd_mobilenet_v1", res_dir + "/ssd_mobilenet_v1.hef",                     300, use_ssd},
        {"deeplab", "deeplab_v3_mnv2",  res_dir + "/deeplab_v3_mobilenet_v2_wo_dilation.hef",  513, use_deeplab},
        {"mnv2",    "mobilenet_v2_1.0", res_dir + "/mobilenet_v2_1.0.hef",                     224, use_mnv2},
    };

    int n_active = 0;
    for (auto& m : models) if (m.active) n_active++;
    if (n_active == 0) {
        std::cerr << "실행할 모델이 없습니다. --ssd/--deeplab/--mnv2 중 최소 하나를 1로 주세요." << std::endl;
        return 1;
    }

    std::printf("=== MZ 3모델 스케줄러 벤치마크 (파라미터 미설정 / HailoRT 기본값) ===\n");
    std::printf("  tag=%s run_id=%d input_fps=%d frames=%d\n", tag.c_str(), run_id, input_fps, frames);
    std::printf("  활성 모델: ");
    for (auto& m : models) if (m.active) std::printf("%s ", m.name.c_str());
    std::printf("(%d개)\n", n_active);
    std::printf("  [주의] set_scheduler_threshold/timeout/priority 미호출, batch_size/power_mode 미지정\n");

    auto images = list_images(images_dir, (size_t)frames);
    if (images.empty()) {
        std::cerr << "[에러] 이미지가 없습니다: " << images_dir << std::endl;
        return 1;
    }
    std::printf("  이미지: %s (%zu 프레임)\n", images_dir.c_str(), images.size());

    // ── VDevice (스케줄러 사용). 스케줄링 알고리즘만 ROUND_ROBIN 으로 켜고, 그 외 파라미터는 손대지 않는다.
    hailo_vdevice_params_t vdevice_params;
    hailo_init_vdevice_params(&vdevice_params);
    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;

    auto vdevice_exp = VDevice::create(vdevice_params);
    if (!vdevice_exp) { std::cerr << "VDevice 생성 실패" << std::endl; return vdevice_exp.status(); }
    auto vdevice = vdevice_exp.release();

    // ── configure: 파라미터를 일절 수정하지 않는다 (기본값 그대로) ──
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> ngs;
    std::vector<int> active_idx;
    for (size_t i = 0; i < models.size(); i++) {
        auto& m = models[i];
        if (!m.active) continue;

        auto hef_exp = Hef::create(m.hef_path);
        if (!hef_exp) { std::cerr << m.name << " HEF 로드 실패: " << m.hef_path << std::endl; return hef_exp.status(); }
        auto hef = hef_exp.release();

        auto cfg_exp = vdevice->create_configure_params(hef);
        if (!cfg_exp) { std::cerr << m.name << " configure params 실패" << std::endl; return cfg_exp.status(); }
        auto cfg = cfg_exp.value();
        // ↑ 여기서 batch_size / power_mode 를 **수정하지 않는다** — 그것이 이 실험의 조건이다.

        auto ngs_exp = vdevice->configure(hef, cfg);
        if (!ngs_exp) { std::cerr << m.name << " configure 실패" << std::endl; return ngs_exp.status(); }
        ngs.push_back(ngs_exp.value()[0]);
        active_idx.push_back((int)i);
        std::printf("  [로드] %-18s <- %s\n", m.name.c_str(), m.hef_path.c_str());
    }

    // ── vstream 생성. format 은 AUTO(HEF 네이티브) — 강제 FLOAT32 변환 비용을 넣지 않는다.
    //    timeout 만 크게(5분) 잡아 starvation 시 프레임 유실 방지.
    const uint32_t VSTREAM_TIMEOUT_MS = 300000;
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vs;
    for (size_t k = 0; k < ngs.size(); k++) {
        auto in_p = ngs[k]->make_input_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                                                      VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        auto out_p = ngs[k]->make_output_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                                                        VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!in_p) { std::cerr << "input vstream params 실패" << std::endl; return in_p.status(); }
        if (!out_p) { std::cerr << "output vstream params 실패" << std::endl; return out_p.status(); }

        auto iv = VStreamsBuilder::create_input_vstreams(*ngs[k], in_p.value());
        auto ov = VStreamsBuilder::create_output_vstreams(*ngs[k], out_p.value());
        if (!iv) { std::cerr << "input vstream 생성 실패" << std::endl; return iv.status(); }
        if (!ov) { std::cerr << "output vstream 생성 실패" << std::endl; return ov.status(); }
        vs.emplace_back(iv.release(), ov.release());
    }

    // ── 실행 ──
    CpuStats cpu0 = read_cpu_stats();
    double mem0 = read_mem_usage();
    double wall0 = now_ms();

    std::vector<std::thread> threads;
    for (size_t k = 0; k < ngs.size(); k++) {
        Model& m = models[active_idx[k]];
        threads.emplace_back([&, k]() {
            run_model_async(m, vs[k].first, vs[k].second, images, input_fps);
        });
    }
    for (auto& t : threads) t.join();

    double wall_s = (now_ms() - wall0) / 1000.0;
    CpuStats cpu1 = read_cpu_stats();
    double cpu = calc_cpu_usage(cpu0, cpu1);
    double mem = (mem0 + read_mem_usage()) / 2.0;

    std::printf("\n=== 결과 ===\n");
    for (auto& m : models) {
        if (!m.active) continue;
        std::printf("  %-18s latency=%8.3f ms | fps=%8.2f | frames=%d | prep=%.3f ms | total=%.2f s\n",
                    m.name.c_str(), m.avg_latency_ms, m.fps, m.frame_count,
                    m.avg_preprocess_ms, m.total_time_s);
    }
    std::printf("  cpu=%.2f%% mem=%.2f%% wall=%.2f s\n", cpu, mem, wall_s);

    write_csv(csv_path, tag, run_id, input_fps, frames, models, cpu, mem, wall_s);
    std::printf("  -> CSV 기록: %s\n", csv_path.c_str());
    return 0;
}
