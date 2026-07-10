#!/usr/bin/env python3
import os
import time
import re
import datetime
import sys
from scheduler_mon_pb2 import ProtoMon

class HailoMonitor:
    def __init__(self, update_period=1, directory="/tmp/hmon_files"):
        self.interval = update_period
        self.directory = directory
        self.data = {}

    def get_single_file(self):
        try:
            files = os.listdir(self.directory)
            return os.path.join(self.directory, files[0]) if files else None
        except Exception:
            return None

    def read_stats(self):
        try:
            path = self.get_single_file()
            if path and os.path.exists(path):
                proto = ProtoMon()
                with open(path, "rb") as f:
                    proto.ParseFromString(f.read())
                self.data["utilization"] = proto.device_infos[0].utilization if proto.device_infos else 0.0
            else:
                self.data["utilization"] = 0.0
        except Exception:
            self.data["utilization"] = 0.0
        return self.data

def get_cpu_and_mem_stats():
    stats = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            stats['mem_total_kb'] = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1))
            stats['mem_available_kb'] = int(re.search(r'MemAvailable:\s+(\d+)', meminfo).group(1))
        with open('/proc/stat', 'r') as f:
            cpu_times = [int(x) for x in f.readline().split()[1:]]
            stats['cpu_total_time'] = sum(cpu_times)
            stats['cpu_idle_time'] = cpu_times[3]
    except Exception:
        return None
    return stats

def calculate_cpu_usage(prev, curr):
    if not prev or not curr: return 0.0
    total_diff = curr['cpu_total_time'] - prev['cpu_total_time']
    idle_diff = curr['cpu_idle_time'] - prev['cpu_idle_time']
    return 100.0 * (1 - idle_diff / total_diff) if total_diff else 0.0

if __name__ == "__main__":
    log_path = "/home/rpi1/hailo_cpp_test/npu_log.txt"
    npu_monitor = HailoMonitor()
    prev_stats = get_cpu_and_mem_stats()

    print(f"모니터링 시작... 로그: {log_path}")
    try:
        while True:
            npu_util = npu_monitor.read_stats().get('utilization', 0.0)
            curr_stats = get_cpu_and_mem_stats()
            if not curr_stats:
                time.sleep(1)
                continue

            cpu_util = calculate_cpu_usage(prev_stats, curr_stats)
            mem_used = curr_stats['mem_total_kb'] - curr_stats['mem_available_kb']
            mem_util = 100.0 * (mem_used / curr_stats['mem_total_kb'])

            log_msg = (
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                f"NPU: {npu_util:.2f}%, "
                f"CPU: {cpu_util:.2f}%, "
                f"MEM: {mem_util:.2f}%"
            )

            print(log_msg)
            with open(log_path, "a") as f:
                f.write(log_msg + "\n")

            prev_stats = curr_stats
            time.sleep(1)
    except KeyboardInterrupt:
        print("모니터링 종료")