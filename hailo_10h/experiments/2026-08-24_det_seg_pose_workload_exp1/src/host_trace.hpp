#pragma once
// host_trace.hpp — Hailo-10H용 호스트측 추론 타임라인 트레이서
//
// 왜 이게 필요한가 (HailoRT .hrtt가 H10에서 0바이트인 이유):
//   Hailo-10H는 HailoRT가 DISCRETE로 분류하는 디바이스라, VDevice::create()가
//   VDeviceHandle(로컬 스택)이 아니라 VDeviceHrpcClient(RPC 클라이언트)를 돌려준다
//   (libhailort/src/vdevice/vdevice.cpp). 즉 VDevice·스케줄러·코어옵 activate/switch·
//   VDMA 스트림이 전부 디바이스 안에서 도는 hailort_server 프로세스 쪽에 있고,
//   호스트 프로세스에는 RPC 껍데기만 남는다.
//   .hrtt를 쓰는 SchedulerProfilerHandler는 "자기 프로세스 안에서 발생한" TRACE 이벤트만
//   모으는데, 호스트에서 발화할 수 있는 지점이 사실상 없다(파일 경로로 Hef를 직접 파싱할 때
//   나오는 HefLoadedTrace 하나뿐). 그래서 파일은 열리지만 빈 protobuf가 직렬화돼 0바이트가 된다.
//   → 디바이스 내부 스케줄링은 호스트에서 못 보므로, 호스트에서 실제로 관측 가능한
//     "요청 제출 → 완료" 구간을 우리가 직접 기록한다.
//
// 출력 (label/run_id 기준 파일명):
//   <dir>/<label>_run<N>_events.csv  원자료 — 모델별·프레임별 구간
//   <dir>/<label>_run<N>_trace.json  Chrome Trace Event 포맷
//                                    (ui.perfetto.dev 또는 chrome://tracing 에서 그대로 열림)
//
// 측정 중에는 워커 스레드마다 자기 vector에만 push_back 하고(락 없음), 전부 끝난 뒤
// 메인 스레드에서 한 번에 flush 한다 — 트레이싱이 측정값을 흔들지 않게.

#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace host_trace {

using TraceClock = std::chrono::steady_clock;

// 모든 스레드가 공유하는 시간 원점. 첫 호출 시점으로 고정되므로 main 진입 직후에
// HostTracer를 만들어서(=epoch 확정) 워커 스레드보다 먼저 초기화되게 한다.
inline TraceClock::time_point epoch() {
    static const TraceClock::time_point t0 = TraceClock::now();
    return t0;
}

inline double now_us() {
    return std::chrono::duration<double, std::micro>(TraceClock::now() - epoch()).count();
}

struct Span {
    const char *phase;      // "prep"(CPU 전처리) | "infer"(run_async 제출~완료)
    size_t frame_idx;
    double begin_us;
    double end_us;
    double enqueue_ms;      // infer 전용: run_async()가 반환하기까지 걸린 시간(디바이스 큐 backpressure). prep은 -1
    bool ok;
};

// 워커 스레드 하나가 소유하는 이벤트 버퍼.
class ModelTrace {
public:
    explicit ModelTrace(std::string model, size_t reserve_frames)
        : m_model(std::move(model)) {
        m_spans.reserve(reserve_frames * 2);
    }

    void add_prep(size_t frame_idx, double begin_us, double end_us) {
        m_spans.push_back(Span{"prep", frame_idx, begin_us, end_us, -1.0, true});
    }

    void add_infer(size_t frame_idx, double begin_us, double end_us, double enqueue_ms, bool ok) {
        m_spans.push_back(Span{"infer", frame_idx, begin_us, end_us, enqueue_ms, ok});
    }

    const std::string &model() const { return m_model; }
    const std::vector<Span> &spans() const { return m_spans; }

private:
    std::string m_model;
    std::vector<Span> m_spans;
};

class HostTracer {
public:
    HostTracer(std::string dir, std::string label, int run_id, bool enabled)
        : m_dir(std::move(dir)), m_label(std::move(label)), m_run_id(run_id), m_enabled(enabled) {
        (void)epoch();  // 시간 원점 확정
        m_wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (m_enabled) ::mkdir(m_dir.c_str(), 0755);  // 이미 있으면 EEXIST — 무시
    }

    bool enabled() const { return m_enabled; }

    // 비활성화 상태에서도 유효한 포인터를 돌려준다(워커 쪽에 null 체크를 안 넣으려고).
    // 비활성일 땐 flush를 안 하므로 버퍼는 그냥 버려진다.
    ModelTrace *add_model(const std::string &model, size_t reserve_frames) {
        m_traces.push_back(std::unique_ptr<ModelTrace>(
            new ModelTrace(model, m_enabled ? reserve_frames : 0)));
        return m_traces.back().get();
    }

    // events.csv + trace.json 두 개를 쓴다. 성공 시 true.
    bool flush() {
        if (!m_enabled) return true;
        return write_csv() && write_chrome_json();
    }

    std::string csv_path() const { return prefix() + "_events.csv"; }
    std::string json_path() const { return prefix() + "_trace.json"; }

private:
    std::string prefix() const {
        std::ostringstream o;
        o << m_dir << "/" << m_label << "_run" << m_run_id;
        return o.str();
    }

    bool write_csv() {
        std::ofstream f(csv_path());
        if (!f.is_open()) {
            std::fprintf(stderr, "[host_trace] CSV 열기 실패: %s\n", csv_path().c_str());
            return false;
        }
        // 시간 원점을 같이 남겨야 다른 로그(dmesg, hailortcli logs 등)와 맞춰볼 수 있다.
        f << "# epoch_wall_clock_ns=" << m_wall_clock_ns << "\n";
        f << "model,frame_idx,phase,begin_us,end_us,dur_ms,enqueue_ms,ok\n";
        f << std::fixed;
        for (const auto &t : m_traces) {
            for (const auto &s : t->spans()) {
                f << t->model() << "," << s.frame_idx << "," << s.phase << ","
                  << std::setprecision(3) << s.begin_us << "," << s.end_us << ","
                  << (s.end_us - s.begin_us) / 1000.0 << ",";
                if (s.enqueue_ms >= 0.0) f << s.enqueue_ms;
                f << "," << (s.ok ? 1 : 0) << "\n";
            }
        }
        return f.good();
    }

    bool write_chrome_json() {
        std::ofstream f(json_path());
        if (!f.is_open()) {
            std::fprintf(stderr, "[host_trace] JSON 열기 실패: %s\n", json_path().c_str());
            return false;
        }
        f << std::fixed << std::setprecision(3);
        f << "{\"displayTimeUnit\":\"ms\",\n";
        f << " \"otherData\":{\"label\":\"" << m_label << "\",\"run_id\":\"" << m_run_id
          << "\",\"epoch_wall_clock_ns\":\"" << m_wall_clock_ns
          << "\",\"note\":\"host-side spans only; H10 device-side scheduler is not observable from the host\"},\n";
        f << " \"traceEvents\":[\n";

        bool first = true;
        auto comma = [&]() { if (!first) f << ",\n"; first = false; };

        comma();
        f << "  {\"ph\":\"M\",\"pid\":1,\"tid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"hailo10h_sched_bench "
          << m_label << " run" << m_run_id << "\"}}";

        int tid = 1;
        for (const auto &t : m_traces) {
            comma();
            f << "  {\"ph\":\"M\",\"pid\":1,\"tid\":" << tid
              << ",\"name\":\"thread_name\",\"args\":{\"name\":\"" << t->model() << "\"}}";
            for (const auto &s : t->spans()) {
                comma();
                f << "  {\"ph\":\"X\",\"pid\":1,\"tid\":" << tid
                  << ",\"cat\":\"" << s.phase << "\""
                  << ",\"name\":\"" << t->model() << " " << s.phase << " #" << s.frame_idx << "\""
                  << ",\"ts\":" << s.begin_us
                  << ",\"dur\":" << (s.end_us - s.begin_us)
                  << ",\"args\":{\"frame\":" << s.frame_idx << ",\"ok\":" << (s.ok ? "true" : "false");
                if (s.enqueue_ms >= 0.0) f << ",\"enqueue_ms\":" << s.enqueue_ms;
                f << "}}";
            }
            tid++;
        }
        f << "\n ]\n}\n";
        return f.good();
    }

    std::string m_dir;
    std::string m_label;
    int m_run_id;
    bool m_enabled;
    long long m_wall_clock_ns = 0;
    std::vector<std::unique_ptr<ModelTrace>> m_traces;
};

}  // namespace host_trace
