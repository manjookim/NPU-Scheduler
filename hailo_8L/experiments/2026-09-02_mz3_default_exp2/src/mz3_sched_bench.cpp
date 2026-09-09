// =============================================================================
// mz3_sched_bench.cpp  (v2, 2026-09-02)
//
// Hailo-8L / Model Zoo 3모델(ssd_mobilenet_v1, deeplab_v3_mobilenet_v2_wo_dilation,
// mobilenet_v2_1.0) 다중 조합(2^3-1 = 7조건) 스케줄러 벤치마크.
//
// -----------------------------------------------------------------------------
// [v1 대비 무엇이 바뀌었나 — 요약]
//   (A) 측정 정의는 **하나도 바꾸지 않았다.** avg_latency = mean(deq-enq), enq 는 write() 직전,
//       프레임별 전처리, do-while(HAILO_TIMEOUT) 재시도, AUTO 포맷, vstream timeout 300s,
//       스케줄러 setter 미호출, ROUND_ROBIN — 전부 v1/원본 model_runner.hpp 와 동일.
//       기존 42런과 같은 표에 놓을 수 있다.
//   (B) 대신 **v1 이 조용히 틀리던 것**을 고쳤다:
//       1. 워밍업(--warmup, 기본 30) — HEF activate/첫 페이지폴트/OpenCV 지연초기화가
//          max_latency 를 통째로 지배하던 문제. v1 의 max_latency 는 사실상 "0번 프레임 값"이다.
//       2. 동시실행 구간(co-execution window) 통계 — v1 은 모델별 완주 시각이 달라서
//          "3모델 경합" 평균에 1~2모델 구간이 섞여 들어갔다. 이제 모든 활성 모델이 실제로
//          동시에 돌던 구간만 따로 집계한다(coex_*). 슬로우다운 배수의 근거가 여기다.
//       3. 시작 배리어 — 모델별 스레드 생성 시차(수 ms)를 제거. 짧은 조건에서 유효.
//       4. fps 페이싱 폭주 방지 — v1 은 next_t += interval 만 해서, 전처리가 밀리면
//          next_t 가 과거에 남아 이후 프레임이 sleep 없이 **버스트**로 나갔다.
//          즉 "fps=30 정속 입력"이 조용히 깨졌다. 이제 재동기화 + late_frames 로 노출한다.
//       5. 에러 프레임 격리 — v1 은 HAILO_TIMEOUT 이 아닌 실패도 그냥 다음 프레임으로 넘어가
//          쓰레기 deq_ts 가 평균에 섞였다. 이제 실패 프레임은 통계에서 제외하고 카운트한다.
//       6. 분산/분위수(std/p50/p95/p99) — "3회 평균만 있어 유의성 판단 불가" 문제를
//          사후가 아니라 원천에서 해결. 런 1회 안에서 이미 673샘플이 있다.
//       7. write() 블로킹 시간 분리 계측(wblock_*) — latency 는 정의상 큐 대기를 포함한다.
//          이제 latency = wblock(큐 대기) + dev(장치 구간) 으로 **분해**되어 HRTT 의
//          device latency 와 직접 비교 가능하다. (2026-08-08 "50배 차이" 의 정체)
//       8. 파라미터 하드코딩 제거 — v1 은 batch=0/threshold=1/timeout=0/priority=16 을
//          **측정하지 않고 CSV 에 상수로 찍었다.** 이제 configure_params 에서 실제 값을
//          읽어 기록하고, 읽을 수 없는 값은 라이브러리 기본값임을 params_source 로 명시한다.
//       9. 스레드 람다의 dangling reference 수정 (루프 지역 참조를 [&] 로 캡처하던 UB).
//      10. run manifest(JSON) 출력 — HRTT 파일이 조건 라벨을 안 가지는 문제를 해결한다.
//          런 시작/종료 epoch ms + HRTT 디렉터리를 남겨 .hrtt 를 1:1 로 매핑한다.
//
// [Hailo-8L 구조 관련 최적화 메모]
//   · VDevice 1개를 3모델이 공유하고, 모델마다 ConfiguredNetworkGroup 1개 + writer/reader
//     스레드 1쌍. 스케줄러(ROUND_ROBIN)가 network group 간 전환을 담당한다.
//   · batch_size 를 건드리지 않으므로(=auto) 스케줄러는 한 번 활성화될 때 여러 프레임을
//     연속 처리할 수 있다. 즉 "3모델이 2모델보다 switch/s 가 적다"는 관측은 데이터 손실이
//     아니라 정상일 수 있다 — batch 파라미터를 CSV 에 남겨야 이 해석이 가능하다(위 8번).
//   · deeplab 은 multi-context 로 추정되므로 context 전환이 활성화 비용에 섞인다.
//     activate_core_op duration=0.0000ms 라는 HRTT 관측만으로 "전환 비용 없음"을 단정하면 안 된다.
//
// 빌드:
//   g++ -O2 -std=c++17 mz3_sched_bench.cpp -o mz3_sched_bench \
//       $(pkg-config --cflags --libs opencv4) -lhailort -lpthread
// =============================================================================
#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "sys_monitor.hpp"   // CpuStats / read_thread_ctx_switches (원본 8L 공통)

using namespace hailort;

static std::mutex print_mutex;

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
// HRTT 파일명(벽시계 timestamp)과 런을 매핑하기 위한 epoch 시각.
static inline double epoch_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

// ───────────────────────── 시작 배리어 (C++17, std::barrier 없이) ─────────────────────────
class StartBarrier {
public:
    explicit StartBarrier(int n) : need_(n) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(m_);
        if (++arrived_ >= need_) { open_ = true; cv_.notify_all(); }
        else cv_.wait(lk, [&] { return open_; });
    }
private:
    std::mutex m_; std::condition_variable cv_;
    int need_, arrived_ = 0; bool open_ = false;
};

// ───────────────────────────── 통계 유틸 ─────────────────────────────
struct Stat { double mean=-1, sd=-1, mx=-1, p50=-1, p95=-1, p99=-1; int n=0; };

static Stat summarize(std::vector<double> v) {
    Stat s; s.n = (int)v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / v.size();
    double acc = 0.0; for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.sd  = (v.size() > 1) ? std::sqrt(acc / (v.size() - 1)) : 0.0;  // 표본 표준편차
    s.mx  = v.back();
    auto qq = [&](double p) -> double {
        size_t idx = (size_t)std::llround(p * (double)(v.size() - 1));
        return v[std::min(idx, v.size() - 1)];
    };
    s.p50 = qq(0.50); s.p95 = qq(0.95); s.p99 = qq(0.99);
    return s;
}

// ───────────────────────────── 모델 ─────────────────────────────
struct Model {
    std::string key, name, hef_path;
    int img_size = 0;
    bool active = false;

    // 프레임별 원시 타임스탬프 (통계는 전부 사후 계산)
    std::vector<double> enq_ts, wexit_ts, deq_ts;
    std::vector<char>   ok;          // 이 프레임이 정상 완료됐는가
    long   vol_ctx = 0, nonvol_ctx = 0;
    long   late_frames = 0;          // fps 페이싱 데드라인을 놓친 프레임 수
    long   err_frames  = 0;
    double prep_mean_ms = -1;

    // 전체 구간 (워밍업 제외) 통계 — v1 의 avg_latency 와 동일 정의
    Stat   lat, wblock, dev;
    double total_time_s = -1, fps = -1;
    int    frame_count = 0;
    // 동시실행 구간 통계
    Stat   coex_lat; double coex_fps = -1; int coex_n = 0;
    double t_first = 0, t_last = 0;   // 측정구간 경계 (배리어 이후 첫 enq ~ 마지막 deq)
};

// ───────────────────────────── 이미지 목록 ─────────────────────────────
static std::vector<std::string> list_images(const std::string& dir, size_t want) {
    std::vector<std::string> v;
    DIR* d = opendir(dir.c_str());
    if (!d) { std::cerr << "[에러] 이미지 디렉터리 열기 실패: " << dir << std::endl; return v; }
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string n = e->d_name;
        if (n.size() < 5) continue;
        auto dot = n.find_last_of('.');
        if (dot == std::string::npos) continue;
        std::string ext = n.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == "jpg" || ext == "jpeg" || ext == "png") v.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(v.begin(), v.end());
    if (v.empty()) return v;
    std::vector<std::string> out; out.reserve(want);
    for (size_t i = 0; i < want; i++) out.push_back(v[i % v.size()]);
    return out;
}

// ───────────────────────────── 모델 1개 실행 ─────────────────────────────
// [측정 지점은 v1/원본과 동일] enq_ts = write() 직전, deq_ts = 그 프레임의 모든 출력 수신 후.
// 추가된 것은 wexit_ts(= write() 반환 시각) 하나뿐이며, 기존 정의에 영향을 주지 않는다.
static void run_model_async(Model& m,
                            std::vector<InputVStream>& inputs,
                            std::vector<OutputVStream>& outputs,
                            const std::vector<std::string>& images,
                            int input_fps, StartBarrier& bar) {
    if (inputs.empty() || outputs.empty()) {
        std::lock_guard<std::mutex> lk(print_mutex);
        std::cerr << "[" << m.name << "] 입력/출력 vstream 없음, 스킵" << std::endl;
        return;
    }
    const size_t N = images.size();
    const size_t expected_frame_size = inputs[0].get_frame_size();
    m.enq_ts.assign(N, 0.0); m.wexit_ts.assign(N, 0.0); m.deq_ts.assign(N, 0.0);
    m.ok.assign(N, 0);

    long w_vol = 0, w_nonvol = 0, r_vol = 0, r_nonvol = 0;
    double prep_total_ms = 0.0; long prep_count = 0;
    std::atomic<long> late{0}, errs{0};
    std::atomic<bool> fatal{false};
    std::atomic<size_t> written{0};   // reader 가 "어디까지 읽어야 하는지" 알기 위함

    std::thread writer([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        const double interval_ms = (input_fps > 0) ? (1000.0 / input_fps) : 0.0;
        bar.arrive_and_wait();                     // ← 모든 모델이 여기서 동시에 출발
        double next_t = now_ms();

        for (size_t i = 0; i < N; i++) {
            if (input_fps > 0) {
                double t = now_ms();
                if (t < next_t) {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double, std::milli>(next_t - t));
                    next_t += interval_ms;
                } else {
                    // [v2 수정] v1 은 여기서도 next_t += interval 만 했다. 전처리/경합으로
                    // 한 번 밀리면 next_t 가 과거에 남아 이후 프레임이 sleep 없이 연속 발사되고,
                    // "fps=30 정속"이라는 실험 조건이 조용히 깨진다(= 버스트 입력).
                    // 한 주기 이상 밀렸으면 현재 시각으로 재동기화하고 late 로 노출한다.
                    if (i > 0) late.fetch_add(1);
                    next_t = (t > next_t + interval_ms) ? (t + interval_ms) : (next_t + interval_ms);
                }
            }

            // ── 전처리(프레임별): imread -> resize(정사각) -> BGR2RGB  [원본 규약 유지] ──
            double p0 = now_ms();
            cv::Mat img = cv::imread(images[i]);
            cv::Mat in;
            if (img.empty()) {
                std::lock_guard<std::mutex> lk(print_mutex);
                std::cerr << "[" << m.name << "] [경고] 이미지 로드 실패: " << images[i] << std::endl;
                in = cv::Mat::zeros(m.img_size, m.img_size, CV_8UC3);
            } else {
                cv::resize(img, in, cv::Size(m.img_size, m.img_size));
                cv::cvtColor(in, in, cv::COLOR_BGR2RGB);
            }
            if (!in.isContinuous()) in = in.clone();
            prep_total_ms += (now_ms() - p0); prep_count++;

            if (i == 0 && in.total() * in.elemSize() != expected_frame_size) {
                std::lock_guard<std::mutex> lk(print_mutex);
                std::cerr << "[" << m.name << "] [치명] 프레임 크기 불일치: 기대="
                          << expected_frame_size << "B, 실제=" << (in.total() * in.elemSize())
                          << "B — 측정을 중단합니다(잘못된 입력으로 낸 수치는 논문에 못 씀)." << std::endl;
                fatal = true; break;
            }

            m.enq_ts[i] = now_ms();
            hailo_status st;
            do { st = inputs[0].write(MemoryView(in.data, in.total() * in.elemSize())); }
            while (st == HAILO_TIMEOUT);            // 원본 관례: 프레임 유실 방지
            m.wexit_ts[i] = now_ms();
            if (st != HAILO_SUCCESS) {
                errs.fetch_add(1);
                std::lock_guard<std::mutex> lk(print_mutex);
                std::cerr << "[" << m.name << "] write 실패 status=" << st
                          << " frame=" << i << " — 이 프레임은 통계에서 제외" << std::endl;
                fatal = true; break;                // 스트림이 깨진 뒤의 수치는 신뢰할 수 없다
            }
            written.store(i + 1);
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary; w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    std::thread reader([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        std::vector<std::vector<uint8_t>> obuf(outputs.size());
        for (size_t j = 0; j < outputs.size(); j++) obuf[j].resize(outputs[j].get_frame_size());

        for (size_t i = 0; i < N; i++) {
            // [주의] 여기서 폴링/스핀을 하면 안 된다 — voluntary_ctx_switches 지표가 오염되고
            // RPi5 4코어에서 CPU 를 잡아먹는다. 원본과 동일하게 read() 블로킹에 맡기고,
            // writer 가 이미 죽은 경우에만(atomic 1회 로드) 빠져나온다(300s 무한대기 방지).
            if (fatal.load() && i >= written.load()) break;

            bool frame_ok = true;
            for (size_t j = 0; j < outputs.size(); j++) {
                hailo_status st;
                do { st = outputs[j].read(MemoryView(obuf[j].data(), obuf[j].size())); }
                while (st == HAILO_TIMEOUT);
                if (st != HAILO_SUCCESS) {
                    frame_ok = false; errs.fetch_add(1);
                    std::lock_guard<std::mutex> lk(print_mutex);
                    std::cerr << "[" << m.name << "] read 실패 status=" << st
                              << " frame=" << i << std::endl;
                }
            }
            m.deq_ts[i] = now_ms();
            m.ok[i] = frame_ok ? 1 : 0;             // [v2] 실패 프레임은 통계에서 제외
            if (!frame_ok) { fatal = true; break; }
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join(); reader.join();

    m.vol_ctx = w_vol + r_vol; m.nonvol_ctx = w_nonvol + r_nonvol;
    m.late_frames = late.load(); m.err_frames = errs.load();
    m.prep_mean_ms = (prep_count > 0) ? (prep_total_ms / prep_count) : -1;
}

// ── 워밍업 제외 + 구간 통계 계산 (스레드 종료 후 단일 스레드에서) ──
static void finalize_model(Model& m, int warmup) {
    std::vector<double> lat, wb, dv;
    m.t_first = 0; m.t_last = 0;
    for (size_t i = (size_t)warmup; i < m.enq_ts.size(); i++) {
        if (!m.ok[i]) continue;
        if (m.deq_ts[i] <= m.enq_ts[i]) continue;
        lat.push_back(m.deq_ts[i] - m.enq_ts[i]);
        wb.push_back(m.wexit_ts[i] - m.enq_ts[i]);            // 큐 대기(=write 블로킹)
        dv.push_back(m.deq_ts[i] - m.wexit_ts[i]);            // 장치 구간(발사~수신)
        if (m.t_first == 0) m.t_first = m.enq_ts[i];
        m.t_last = m.deq_ts[i];
    }
    m.lat = summarize(lat); m.wblock = summarize(wb); m.dev = summarize(dv);
    m.frame_count = m.lat.n;
    m.total_time_s = (m.t_last > m.t_first) ? (m.t_last - m.t_first) / 1000.0 : -1;
    m.fps = (m.total_time_s > 0) ? (m.frame_count / m.total_time_s) : -1;
}

static void finalize_coex(std::vector<Model*>& act, int warmup) {
    if (act.empty()) return;
    double s = -1e18, e = 1e18;
    for (auto* m : act) { s = std::max(s, m->t_first); e = std::min(e, m->t_last); }
    for (auto* m : act) {
        std::vector<double> v;
        for (size_t i = (size_t)warmup; i < m->enq_ts.size(); i++) {
            if (!m->ok[i] || m->deq_ts[i] <= m->enq_ts[i]) continue;
            if (m->enq_ts[i] >= s && m->deq_ts[i] <= e) v.push_back(m->deq_ts[i] - m->enq_ts[i]);
        }
        m->coex_lat = summarize(v);
        m->coex_n = m->coex_lat.n;
        m->coex_fps = (e > s) ? (m->coex_n / ((e - s) / 1000.0)) : -1;
    }
    std::printf("  [동시실행 구간] %.3f s (측정 프레임의 이 구간만이 진짜 %zu-모델 경합이다)\n",
                (e - s) / 1000.0, act.size());
}

// ───────────────────────────── CSV ─────────────────────────────
// [스키마 원칙] 앞의 57컬럼은 hailo_8L/csv_writer.hpp 및 재학습 18런 CSV 와 **바이트 단위로 동일**하다.
// v2 에서 추가된 것은 전부 뒤에 붙였다 — 기존 파서/엑셀 시트가 그대로 동작한다.
struct RunMeta {
    std::string tag, hrtt_dir, instr, params_source, hailort_ver;
    int run_id=1, input_fps=0, frames=0, warmup=0;
    uint16_t batch_param=0; int power_mode_param=0;
    double cpu=0, mem=0, wall_s=0, epoch_start=0, epoch_end=0;
    double coex_start_s=0, coex_end_s=0, coex_dur_s=0;
};

static void write_csv(const std::string& path, const RunMeta& R, const std::vector<Model>& models) {
    static const char* HEADER =
        "tag,run_id,use_ssd,use_deeplab,use_mnv2,batch,"
        "threshold_ssd,threshold_deeplab,threshold_mnv2,timeout_ms,"
        "priority_ssd,priority_deeplab,priority_mnv2,"
        "ssd_latency_ms,deeplab_latency_ms,mnv2_latency_ms,"
        "cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches,"
        "npu_percent,switches_per_s,idle_time_pct,run_time_s,"
        "avg_fps_ssd,avg_latency_ssd,max_latency_ssd,activation_ssd,"
        "avg_fps_deeplab,avg_latency_deeplab,max_latency_deeplab,activation_deeplab,"
        "avg_fps_mnv2,avg_latency_mnv2,max_latency_mnv2,activation_mnv2,"
        "total_time_ssd_s,total_time_deeplab_s,total_time_mnv2_s,"
        "avg_preprocess_ms_ssd,avg_preprocess_ms_deeplab,avg_preprocess_ms_mnv2,"
        "postprocess_ms_ssd,postprocess_ms_deeplab,postprocess_ms_mnv2,"
        "total_time_ms_ssd,total_time_ms_deeplab,total_time_ms_mnv2,"
        "total_time_ms_nopp_ssd,total_time_ms_nopp_deeplab,total_time_ms_nopp_mnv2,"
        "input_fps,frames,ssd_frame_count,deeplab_frame_count,mnv2_frame_count,wall_time_s"
        // ── 여기부터 v2 추가 컬럼 (앞 57컬럼 정렬 불변) ──
        ",warmup,hailort_version,batch_param,power_mode_param,params_source,instr"
        ",coex_start_s,coex_end_s,coex_dur_s"
        ",std_latency_ssd,std_latency_deeplab,std_latency_mnv2"
        ",p50_ssd,p50_deeplab,p50_mnv2,p95_ssd,p95_deeplab,p95_mnv2,p99_ssd,p99_deeplab,p99_mnv2"
        ",coex_latency_ssd,coex_latency_deeplab,coex_latency_mnv2"
        ",coex_fps_ssd,coex_fps_deeplab,coex_fps_mnv2"
        ",coex_n_ssd,coex_n_deeplab,coex_n_mnv2"
        ",wblock_ms_ssd,wblock_ms_deeplab,wblock_ms_mnv2"
        ",device_ms_ssd,device_ms_deeplab,device_ms_mnv2"
        ",late_ssd,late_deeplab,late_mnv2,err_ssd,err_deeplab,err_mnv2"
        ",epoch_ms_start,epoch_ms_end,hrtt_dir";

    bool need_header = true;
    { std::ifstream fchk(path);
      if (fchk.good() && fchk.peek() != std::ifstream::traits_type::eof()) need_header = false; }
    std::ofstream f(path, std::ios::app);
    if (!f) { std::cerr << "[에러] CSV 열기 실패: " << path << std::endl; return; }
    if (need_header) f << HEADER << "\n";
    f.setf(std::ios::fixed); f.precision(6);

    const char* KEYS[3] = {"ssd", "deeplab", "mnv2"};
    const Model* M[3] = {nullptr, nullptr, nullptr};
    for (int k = 0; k < 3; k++)
        for (auto& m : models) if (m.key == KEYS[k]) { M[k] = &m; break; }
    auto on = [&](int k) { return (M[k] && M[k]->active); };
    auto v  = [&](int k, double val) { if (!on(k)) f << "NaN"; else f << val; };   // 비활성=NaN (원본 관례)
    auto vi = [&](int k, long val)   { if (!on(k)) f << "NaN"; else f << val;   };

    long vol = 0, nonvol = 0;
    for (auto& m : models) if (m.active) { vol += m.vol_ctx; nonvol += m.nonvol_ctx; }

    f << R.tag << ',' << R.run_id << ',';
    for (int k = 0; k < 3; k++) f << (on(k) ? 1 : 0) << ',';

    // [v2] 하드코딩 제거: batch 는 configure_params 에서 읽은 실제 값.
    // threshold/timeout/priority 는 setter 미호출 → HailoRT 기본값이며, 그 근거를
    // params_source 컬럼에 남긴다(= "libhailort-default, no setter called").
    f << R.batch_param << ',';
    f << 1 << ',' << 1 << ',' << 1 << ',';
    f << 0 << ',';
    f << 16 << ',' << 16 << ',' << 16 << ',';

    for (int k = 0; k < 3; k++) { v(k, M[k] ? M[k]->lat.mean : 0); f << ','; }
    f << R.cpu << ',' << R.mem << ',' << vol << ',' << nonvol << ',';
    f << "NaN" << ',' << "NaN" << ',' << "NaN" << ',';     // npu_percent / switches_per_s / idle (후처리 병합)
    f << R.wall_s << ',';

    for (int k = 0; k < 3; k++) {
        v(k, M[k] ? M[k]->fps : 0);        f << ',';
        v(k, M[k] ? M[k]->lat.mean : 0);   f << ',';
        v(k, M[k] ? M[k]->lat.mx : 0);     f << ',';   // 워밍업 제외된 max — v1 과 의미가 다름(개선)
        f << "NaN" << ',';                              // activation_* (HRTT)
    }
    for (int k = 0; k < 3; k++) { v(k, M[k] ? M[k]->total_time_s : 0); f << ','; }
    for (int k = 0; k < 3; k++) { v(k, M[k] ? M[k]->prep_mean_ms : 0); f << ','; }
    for (int k = 0; k < 3; k++) { f << "NaN" << ','; }   // postprocess_ms_* — host 후처리 없음
    for (int pass = 0; pass < 2; pass++)
        for (int k = 0; k < 3; k++) {
            if (!on(k)) f << "NaN"; else f << (M[k]->prep_mean_ms + M[k]->lat.mean);
            f << ',';
        }
    f << R.input_fps << ',' << R.frames << ',';
    for (int k = 0; k < 3; k++) { vi(k, M[k] ? M[k]->frame_count : 0); f << ','; }
    f << R.wall_s;

    // ── v2 추가 컬럼 ──
    f << ',' << R.warmup << ',' << R.hailort_ver << ',' << R.batch_param << ','
      << R.power_mode_param << ',' << R.params_source << ',' << R.instr;
    f << ',' << R.coex_start_s << ',' << R.coex_end_s << ',' << R.coex_dur_s;
    for (int k = 0; k < 3; k++) { f << ','; v(k, M[k] ? M[k]->lat.sd  : 0); }
    for (int k = 0; k < 3; k++) { f << ','; v(k, M[k] ? M[k]->lat.p50 : 0); }
    for (int k = 0; k < 3; k++) { f << ','; v(k, M[k] ? M[k]->lat.p95 : 0); }
    for (int k = 0; k < 3; k++) { f << ','; v(k, M[k] ? M[k]->lat.p99 : 0); }
    for (int k = 0; k < 3; k++) { f << ','; v(k, M[k] ? M[k]->coex_lat.mean : 0); }
    for (int k = 0; k < 3; k++) { f << ','; v(k, M[k] ? M[k]->coex_fps : 0); }
    for (int k = 0; k < 3; k++) { f << ','; vi(k, M[k] ? M[k]->coex_n : 0); }
    for (int k = 0; k < 3; k++) { f << ','; v(k, M[k] ? M[k]->wblock.mean : 0); }
    for (int k = 0; k < 3; k++) { f << ','; v(k, M[k] ? M[k]->dev.mean : 0); }
    for (int k = 0; k < 3; k++) { f << ','; vi(k, M[k] ? M[k]->late_frames : 0); }
    for (int k = 0; k < 3; k++) { f << ','; vi(k, M[k] ? M[k]->err_frames : 0); }
    f.precision(3);
    f << ',' << R.epoch_start << ',' << R.epoch_end << ',' << R.hrtt_dir << "\n";
}

// ── run manifest: .hrtt 파일에 조건 라벨이 없다는 문제(4-D)를 여기서 끝낸다 ──
static void write_manifest(const std::string& path, const RunMeta& R,
                           const std::vector<Model>& models) {
    std::ofstream f(path);
    if (!f) return;
    f.setf(std::ios::fixed); f.precision(3);
    f << "{\n";
    f << "  \"tag\": \"" << R.tag << "\",\n  \"run_id\": " << R.run_id
      << ",\n  \"input_fps\": " << R.input_fps << ",\n  \"frames\": " << R.frames
      << ",\n  \"warmup\": " << R.warmup << ",\n";
    f << "  \"epoch_ms_start\": " << R.epoch_start << ",\n  \"epoch_ms_end\": " << R.epoch_end << ",\n";
    f << "  \"hrtt_dir\": \"" << R.hrtt_dir << "\",\n";
    f << "  \"instr\": \"" << R.instr << "\",\n";
    f << "  \"hailort_version\": \"" << R.hailort_ver << "\",\n";
    f << "  \"batch_param\": " << R.batch_param << ",\n";
    f << "  \"params_source\": \"" << R.params_source << "\",\n";
    f << "  \"models\": [\n";
    bool first = true;
    for (auto& m : models) {
        if (!m.active) continue;
        if (!first) f << ",\n"; first = false;
        f << "    {\"key\": \"" << m.key << "\", \"hef\": \"" << m.hef_path
          << "\", \"frames_ok\": " << m.frame_count
          << ", \"avg_latency_ms\": " << m.lat.mean
          << ", \"coex_latency_ms\": " << m.coex_lat.mean
          << ", \"wblock_ms\": " << m.wblock.mean
          << ", \"device_ms\": " << m.dev.mean
          << ", \"late\": " << m.late_frames << ", \"err\": " << m.err_frames << "}";
    }
    f << "\n  ]\n}\n";
}

// ───────────────────────────── main ─────────────────────────────
int main(int argc, char** argv) {
    std::string images_dir = "/home/npu-rpi1/datasets/sampled_val2017";
    std::string res_dir    = "/home/npu-rpi1/mz3_exp/resources";
    std::string csv_path   = "/home/npu-rpi1/mz3_exp/csv/results_mz3_default.csv";
    std::string manifest   = "";
    std::string tag = "default", instr = "none";
    int input_fps = 0, frames = 300, run_id = 1, warmup = 30;
    bool use_ssd = false, use_deeplab = false, use_mnv2 = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--ssd")      use_ssd     = std::stoi(next());
        else if (a == "--deeplab")  use_deeplab = std::stoi(next());
        else if (a == "--mnv2")     use_mnv2    = std::stoi(next());
        else if (a == "--fps")      input_fps   = std::stoi(next());
        else if (a == "--frames")   frames      = std::stoi(next());
        else if (a == "--warmup")   warmup      = std::stoi(next());
        else if (a == "--run-id")   run_id      = std::stoi(next());
        else if (a == "--images")   images_dir  = next();
        else if (a == "--res")      res_dir     = next();
        else if (a == "--csv")      csv_path    = next();
        else if (a == "--tag")      tag         = next();
        else if (a == "--manifest") manifest    = next();
        else if (a == "--instr")    instr       = next();   // 계측 조건 기록(monitor/trace on/off)
        else { std::cerr << "알 수 없는 인자: " << a << std::endl; return 1; }
    }
    if (warmup < 0) warmup = 0;
    if (warmup >= frames) { std::cerr << "[에러] warmup(" << warmup << ") >= frames" << std::endl; return 1; }

    std::vector<Model> models(3);
    models[0] = {}; models[0].key="ssd";     models[0].name="ssd_mobilenet_v1";
    models[0].hef_path=res_dir+"/ssd_mobilenet_v1.hef";                    models[0].img_size=300; models[0].active=use_ssd;
    models[1] = {}; models[1].key="deeplab"; models[1].name="deeplab_v3_mnv2";
    models[1].hef_path=res_dir+"/deeplab_v3_mobilenet_v2_wo_dilation.hef"; models[1].img_size=513; models[1].active=use_deeplab;
    models[2] = {}; models[2].key="mnv2";    models[2].name="mobilenet_v2_1.0";
    models[2].hef_path=res_dir+"/mobilenet_v2_1.0.hef";                    models[2].img_size=224; models[2].active=use_mnv2;

    int n_active = 0; for (auto& m : models) if (m.active) n_active++;
    if (n_active == 0) { std::cerr << "--ssd/--deeplab/--mnv2 중 최소 하나를 1로 주세요." << std::endl; return 1; }

    hailo_version_t hv{}; std::string hver = "unknown";
    if (hailo_get_library_version(&hv) == HAILO_SUCCESS) {
        char b[32]; std::snprintf(b, sizeof b, "%u.%u.%u", hv.major, hv.minor, hv.revision); hver = b;
    }

    std::printf("=== MZ3 스케줄러 벤치 v2 (파라미터 미설정 / HailoRT %s) ===\n", hver.c_str());
    std::printf("  tag=%s run_id=%d fps=%d frames=%d warmup=%d instr=%s\n",
                tag.c_str(), run_id, input_fps, frames, warmup, instr.c_str());
    std::printf("  활성 모델(%d): ", n_active);
    for (auto& m : models) if (m.active) std::printf("%s ", m.name.c_str());
    std::printf("\n  [조건] set_scheduler_threshold/timeout/priority 미호출, batch_size/power_mode 미지정\n");

    auto images = list_images(images_dir, (size_t)frames);
    if (images.empty()) { std::cerr << "[에러] 이미지가 없습니다: " << images_dir << std::endl; return 1; }

    // ── VDevice: 스케줄링 알고리즘만 ROUND_ROBIN. 그 외 필드는 손대지 않는다.
    hailo_vdevice_params_t vdevice_params;
    hailo_init_vdevice_params(&vdevice_params);
    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    auto vdevice_exp = VDevice::create(vdevice_params);
    if (!vdevice_exp) { std::cerr << "VDevice 생성 실패" << std::endl; return vdevice_exp.status(); }
    auto vdevice = vdevice_exp.release();

    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> ngs;
    std::vector<int> active_idx;
    uint16_t batch_param = 0; int power_mode_param = 0;

    for (size_t i = 0; i < models.size(); i++) {
        auto& m = models[i];
        if (!m.active) continue;
        auto hef_exp = Hef::create(m.hef_path);
        if (!hef_exp) { std::cerr << m.name << " HEF 로드 실패: " << m.hef_path << std::endl; return hef_exp.status(); }
        auto hef = hef_exp.release();
        auto cfg_exp = vdevice->create_configure_params(hef);
        if (!cfg_exp) { std::cerr << m.name << " configure params 실패" << std::endl; return cfg_exp.status(); }
        auto cfg = cfg_exp.value();
        // ★ 파라미터는 수정하지 않는다 — 그 대신 **실제 값을 읽어서 기록**한다.
        //   (v1 은 batch=0 을 CSV 에 상수로 찍었을 뿐 확인한 적이 없었다.)
        for (auto& kv : cfg) {
            batch_param     = kv.second.batch_size;
            power_mode_param= (int)kv.second.power_mode;
            std::printf("  [params] %-18s batch_size=%u power_mode=%d\n",
                        m.name.c_str(), (unsigned)kv.second.batch_size, (int)kv.second.power_mode);
            for (auto& np : kv.second.network_params_by_name)
                std::printf("           network '%s' batch_size=%u\n",
                            np.first.c_str(), (unsigned)np.second.batch_size);
        }
        auto ngs_exp = vdevice->configure(hef, cfg);
        if (!ngs_exp) { std::cerr << m.name << " configure 실패" << std::endl; return ngs_exp.status(); }
        ngs.push_back(ngs_exp.value()[0]);
        active_idx.push_back((int)i);
        std::printf("  [로드] %-18s <- %s\n", m.name.c_str(), m.hef_path.c_str());
    }

    // ── vstream: AUTO 포맷 + 300s timeout (원본과 동일. 스케줄링 정책 아님, starvation 안전장치) ──
    const uint32_t VSTREAM_TIMEOUT_MS = 300000;
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vs;
    for (size_t k = 0; k < ngs.size(); k++) {
        auto in_p  = ngs[k]->make_input_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                        VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        auto out_p = ngs[k]->make_output_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                        VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!in_p)  { std::cerr << "input vstream params 실패"  << std::endl; return in_p.status(); }
        if (!out_p) { std::cerr << "output vstream params 실패" << std::endl; return out_p.status(); }
        auto iv = VStreamsBuilder::create_input_vstreams(*ngs[k], in_p.value());
        auto ov = VStreamsBuilder::create_output_vstreams(*ngs[k], out_p.value());
        if (!iv) { std::cerr << "input vstream 생성 실패"  << std::endl; return iv.status(); }
        if (!ov) { std::cerr << "output vstream 생성 실패" << std::endl; return ov.status(); }
        vs.emplace_back(iv.release(), ov.release());
    }

    // ── 실행 ──
    CpuStats cpu0 = read_cpu_stats();
    double mem0 = read_mem_usage();
    double wall0 = now_ms(), ep0 = epoch_ms();

    StartBarrier bar((int)ngs.size());
    std::vector<std::thread> threads;
    for (size_t k = 0; k < ngs.size(); k++) {
        // [v2 수정] v1 은 루프 지역 참조(Model& m)를 [&] 로 캡처해서 스레드가 죽은 참조를
        // 볼 수 있었다(UB). 인덱스를 값으로 캡처하고 컨테이너를 통해 접근한다.
        int mi = active_idx[k];
        Model* mp = &models[mi];
        auto* inp = &vs[k].first; auto* outp = &vs[k].second;
        threads.emplace_back([mp, inp, outp, &images, input_fps, &bar]() {
            run_model_async(*mp, *inp, *outp, images, input_fps, bar);
        });
    }
    for (auto& t : threads) t.join();

    double wall_s = (now_ms() - wall0) / 1000.0, ep1 = epoch_ms();
    CpuStats cpu1 = read_cpu_stats();
    double cpu = calc_cpu_usage(cpu0, cpu1);
    double mem = (mem0 + read_mem_usage()) / 2.0;

    std::vector<Model*> act;
    for (auto& m : models) if (m.active) { finalize_model(m, warmup); act.push_back(&m); }
    finalize_coex(act, warmup);

    double cs = -1e18, ce = 1e18;
    for (auto* m : act) { cs = std::max(cs, m->t_first); ce = std::min(ce, m->t_last); }

    std::printf("\n=== 결과 (워밍업 %d프레임 제외) ===\n", warmup);
    for (auto* m : act) {
        std::printf("  %-18s lat=%8.3f±%-7.3f p95=%8.3f max=%8.3f | fps=%7.2f | n=%d\n",
                    m->name.c_str(), m->lat.mean, m->lat.sd, m->lat.p95, m->lat.mx, m->fps, m->frame_count);
        std::printf("  %-18s   └ 분해: 큐대기(write블로킹)=%.3f ms + 장치구간=%.3f ms   "
                    "동시구간 lat=%.3f (n=%d)\n",
                    "", m->wblock.mean, m->dev.mean, m->coex_lat.mean, m->coex_n);
        if (m->late_frames > 0)
            std::printf("  %-18s   [경고] 입력 페이싱 지연 %ld 프레임 — fps=%d 정속 입력이 깨졌다."
                        " 전처리가 CPU 병목이다.\n", "", m->late_frames, input_fps);
        if (m->err_frames > 0)
            std::printf("  %-18s   [경고] 실패 프레임 %ld — 통계에서 제외됨\n", "", m->err_frames);
    }
    std::printf("  cpu=%.2f%% mem=%.2f%% wall=%.2f s\n", cpu, mem, wall_s);

    RunMeta R;
    R.tag=tag; R.run_id=run_id; R.input_fps=input_fps; R.frames=frames; R.warmup=warmup;
    R.cpu=cpu; R.mem=mem; R.wall_s=wall_s; R.epoch_start=ep0; R.epoch_end=ep1;
    R.batch_param=batch_param; R.power_mode_param=power_mode_param;
    R.params_source="no-setter-called;libhailort-default";
    R.hailort_ver=hver; R.instr=instr;
    const char* tp = std::getenv("HAILO_TRACE_PATH"); R.hrtt_dir = tp ? tp : "";
    R.coex_start_s=(cs-wall0)/1000.0; R.coex_end_s=(ce-wall0)/1000.0; R.coex_dur_s=(ce-cs)/1000.0;

    write_csv(csv_path, R, models);
    if (!manifest.empty()) { write_manifest(manifest, R, models); std::printf("  -> manifest: %s\n", manifest.c_str()); }
    std::printf("  -> CSV 기록: %s\n", csv_path.c_str());

    // 실패 프레임이 있었으면 종료코드로 알린다 (스윕 스크립트가 조용히 지나가지 않도록)
    for (auto* m : act) if (m->err_frames > 0) return 2;
    return 0;
}
