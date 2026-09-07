#pragma once
// npu_monitor.hpp — Hailo-10H 디바이스 NNC 가동률(npu_percent) 수집.
// (2026-08-24 실험 코드의 것을 그대로 재사용. 검증된 코드라 손대지 않았다.)
//
// [왜 8L과 다른 방식인가]
// Hailo-8L에서는 HAILO_MONITOR=1을 주면 libhailort가 /tmp/hmon_files에 ProtoMon 파일을
// 떨궜고, tools/monitoring/hailo_utilization.py가 그걸 1초마다 읽어 npu_log.txt에 쌓은 뒤
// parse_npu_log.py가 평균내서 CSV의 npu_percent를 채웠다.
// H10에서는 스케줄러가 디바이스 안(hailort_server)에 있어서 호스트에 hmon_files가 아예
// 생기지 않는다 — 실측으로 확인함(HAILO_MONITOR=1로 돌려도 디렉토리 미생성).
// 대신 `hailortcli monitor`가 디바이스에 직접 물어본 NNC 가동률을 약 1초 주기로 출력하므로,
// 그 프로세스를 자식으로 띄워 파이프로 읽는다. 우리 프로세스가 VDevice를 잡고 있어도
// 동시 실행에 문제 없음(실측 확인).
//
// 평균 방식은 8L의 parse_npu_log.py와 동일하게 "NNC>0인 샘플만" 평균낸다
// (모델 로딩·종료 구간의 0%가 평균을 끌어내리지 않도록 — 8L과 같은 정의를 유지해야
//  두 보드의 npu_percent를 나란히 비교할 수 있다).
//
// `hailortcli monitor` 한 줄 예시(ANSI 이스케이프 제거 후):
//   pci/0001:01:00.0  HAILO10H  70.8  42.8  11.5  831 / 7221  47.5  800
//   [0]=device id     [1]=arch  [2]=NNC%  [3]=device CPU%  [4]=RAM%  [5..7]=RAM MB  [8]=temp  [9]=mV

#include <signal.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace npu_monitor {

// ESC[ ... <final byte> 형태의 CSI 시퀀스를 걷어낸다(monitor가 화면을 갱신하며 뿌리는 것).
inline std::string strip_ansi(const std::string &in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\x1b' && i + 1 < in.size() && in[i + 1] == '[') {
            i += 2;
            while (i < in.size() && !((in[i] >= '@' && in[i] <= '~'))) ++i;
            continue;  // final byte까지 버림
        }
        out.push_back(in[i]);
    }
    return out;
}

class NpuMonitor {
public:
    // enabled=false면 아무것도 띄우지 않고 샘플 0개 → npu_percent는 NaN이 된다.
    explicit NpuMonitor(bool enabled) : m_enabled(enabled) {}
    ~NpuMonitor() { stop(); }

    bool start() {
        if (!m_enabled || m_running) return false;
        int fds[2];
        if (::pipe(fds) != 0) {
            std::fprintf(stderr, "[npu_monitor] pipe 실패 — npu_percent는 NaN\n");
            return false;
        }
        m_pid = ::fork();
        if (m_pid < 0) {
            ::close(fds[0]); ::close(fds[1]);
            std::fprintf(stderr, "[npu_monitor] fork 실패 — npu_percent는 NaN\n");
            return false;
        }
        if (m_pid == 0) {  // 자식: stdout/stderr를 파이프로
            ::close(fds[0]);
            ::dup2(fds[1], STDOUT_FILENO);
            ::dup2(fds[1], STDERR_FILENO);
            ::close(fds[1]);
            ::execlp("hailortcli", "hailortcli", "monitor", (char *)nullptr);
            ::_exit(127);  // exec 실패 → 부모는 EOF를 보고 샘플 0개로 끝낸다
        }
        ::close(fds[1]);
        m_fp = ::fdopen(fds[0], "r");
        if (!m_fp) { ::close(fds[0]); return false; }
        m_running = true;
        m_reader = std::thread([this]() { read_loop(); });
        return true;
    }

    void stop() {
        if (!m_running) return;
        m_running = false;
        if (m_pid > 0) {
            ::kill(m_pid, SIGTERM);
            int st = 0;
            ::waitpid(m_pid, &st, 0);
            m_pid = -1;
        }
        if (m_reader.joinable()) m_reader.join();  // 자식이 죽으면 EOF로 빠져나옴
        if (m_fp) { ::fclose(m_fp); m_fp = nullptr; }
    }

    // 8L parse_npu_log.py와 동일: NNC>0인 샘플만 평균. 샘플이 없으면 -1(=CSV에서 NaN).
    double avg_npu_percent() const {
        double sum = 0.0;
        size_t n = 0;
        for (double v : m_samples) { if (v > 0.0) { sum += v; ++n; } }
        return n ? sum / n : -1.0;
    }
    size_t sample_count() const { return m_samples.size(); }

private:
    void read_loop() {
        char buf[1024];
        while (m_running && std::fgets(buf, sizeof(buf), m_fp)) {
            std::string line = strip_ansi(buf);
            std::istringstream ss(line);
            std::vector<std::string> tok;
            std::string t;
            while (ss >> t) tok.push_back(t);
            // 디바이스 행만 취한다(헤더/구분선 제외).
            if (tok.size() < 3 || tok[0].rfind("pci/", 0) != 0) continue;
            try {
                m_samples.push_back(std::stod(tok[2]));  // NNC Utilization (%)
            } catch (...) { /* 숫자가 아니면 무시 */ }
        }
    }

    bool m_enabled;
    std::atomic<bool> m_running{false};
    pid_t m_pid = -1;
    FILE *m_fp = nullptr;
    std::thread m_reader;
    std::vector<double> m_samples;  // reader 스레드만 쓰고, join 이후에만 읽는다
};

}  // namespace npu_monitor
