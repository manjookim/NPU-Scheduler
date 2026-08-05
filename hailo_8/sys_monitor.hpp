#pragma once
// sys_monitor.hpp — CPU/메모리/컨텍스트 스위치 측정 유틸 (Linux /proc 기반, Hailo-8L/8 공통 로직)

#include <fstream>
#include <string>
#include <cstdio>

struct CpuStats {
    long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0;
};

inline CpuStats read_cpu_stats() {
    CpuStats s;
    std::ifstream f("/proc/stat");
    std::string line;
    std::getline(f, line);
    sscanf(line.c_str(), "cpu %ld %ld %ld %ld %ld %ld %ld",
           &s.user, &s.nice, &s.system, &s.idle, &s.iowait, &s.irq, &s.softirq);
    return s;
}

inline double calc_cpu_usage(const CpuStats& s1, const CpuStats& s2) {
    long idle1 = s1.idle + s1.iowait;
    long idle2 = s2.idle + s2.iowait;
    long total1 = s1.user + s1.nice + s1.system + s1.idle + s1.iowait + s1.irq + s1.softirq;
    long total2 = s2.user + s2.nice + s2.system + s2.idle + s2.iowait + s2.irq + s2.softirq;
    long dt = total2 - total1;
    if (dt <= 0) return 0.0;
    return 100.0 * (1.0 - (double)(idle2 - idle1) / (double)dt);
}

inline double read_mem_usage() {
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
inline CtxSwitches read_thread_ctx_switches() {
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
