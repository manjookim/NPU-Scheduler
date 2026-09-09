#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hailo_utilization2.py (2026-09-02) — NPU/CPU/MEM 사용률 폴링 모니터.

기존 tools/monitoring/hailo_utilization.py 의 문제를 고친 판이다.

  1. **로그 경로가 `/home/rpi1/...` 로 하드코딩**돼 있었다. 2026-08-20 계정 이관 후
     그 경로는 존재하지 않는다(저장소에 남은 rpi1 부채 중 하나). -> argv/환경변수로 받는다.
  2. **폴링이 1Hz 고정**이라 fps=0 조건처럼 3~4초에 끝나는 런은 표본이 5개뿐이었다.
     같은 표에 72샘플짜리와 5샘플짜리를 나란히 적으면 안 된다. -> --interval 로 조절.
  3. **`/tmp/hmon_files` 에서 `files[0]` 를 집었다.** 이전 프로세스가 남긴 낡은 ProtoMon
     파일이 있으면 그걸 읽는다(=다른 런의 수치). -> mtime 최신 파일 + 신선도 검사.
  4. **모든 예외를 0.0 으로 삼켰다.** 모니터가 고장난 것과 NPU 가 놀고 있는 것이
     로그상 구분되지 않았다. -> 실패는 nan 으로 기록한다.

사용:
  python3 hailo_utilization2.py <log_path> [--interval 0.2] [--dir /tmp/hmon_files]
  (scheduler_mon_pb2.py 가 있는 디렉터리에서 실행하거나 PYTHONPATH 에 넣을 것)
"""
import argparse
import datetime
import os
import re
import sys
import time

try:
    from scheduler_mon_pb2 import ProtoMon
except Exception as e:                      # import 실패를 조용히 넘기지 않는다
    sys.stderr.write(f"[치명] scheduler_mon_pb2 import 실패: {e}\n"
                     f"       pip install --upgrade protobuf 후 재시도\n")
    raise


class HailoMonitor:
    def __init__(self, directory="/tmp/hmon_files", stale_after=3.0):
        self.directory = directory
        self.stale_after = stale_after      # 이 시간(초)보다 오래된 파일은 '낡음'으로 본다

    def _newest_file(self):
        try:
            entries = [os.path.join(self.directory, f) for f in os.listdir(self.directory)]
            entries = [p for p in entries if os.path.isfile(p)]
            if not entries:
                return None
            return max(entries, key=os.path.getmtime)   # ← files[0] 이 아니라 최신 파일
        except Exception:
            return None

    def read_utilization(self):
        """성공: float(0~100). 실패/낡음: None (= nan 으로 기록)."""
        path = self._newest_file()
        if not path:
            return None
        try:
            if time.time() - os.path.getmtime(path) > self.stale_after:
                return None                  # 이전 런이 남긴 파일 — 이 런의 값이 아니다
            proto = ProtoMon()
            with open(path, "rb") as f:
                proto.ParseFromString(f.read())
            if not proto.device_infos:
                return None
            return float(proto.device_infos[0].utilization)
        except Exception:
            return None


def cpu_mem():
    try:
        with open("/proc/meminfo") as f:
            mi = f.read()
        total = int(re.search(r"MemTotal:\s+(\d+)", mi).group(1))
        avail = int(re.search(r"MemAvailable:\s+(\d+)", mi).group(1))
        with open("/proc/stat") as f:
            t = [int(x) for x in f.readline().split()[1:]]
        return {"mem_total": total, "mem_avail": avail,
                "cpu_total": sum(t), "cpu_idle": t[3] + (t[4] if len(t) > 4 else 0)}
    except Exception:
        return None


def cpu_pct(prev, cur):
    if not prev or not cur:
        return 0.0
    dt = cur["cpu_total"] - prev["cpu_total"]
    di = cur["cpu_idle"] - prev["cpu_idle"]
    return 100.0 * (1 - di / dt) if dt else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", nargs="?",
                    default=os.environ.get("HAILO_UTIL_LOG",
                                           os.path.expanduser("~/hailo_cpp_test/npu_log.txt")))
    ap.add_argument("--interval", type=float,
                    default=float(os.environ.get("HAILO_UTIL_INTERVAL", "0.2")),
                    help="폴링 주기(초). 기본 0.2s = 5Hz — 3초짜리 런에서도 15표본 확보")
    ap.add_argument("--dir", default="/tmp/hmon_files")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.log_path)), exist_ok=True)
    mon = HailoMonitor(a.dir, stale_after=max(3.0, a.interval * 10))
    prev = cpu_mem()
    sys.stderr.write(f"모니터링 시작 (interval={a.interval}s) -> {a.log_path}\n")
    sys.stderr.flush()

    try:
        with open(a.log_path, "a", buffering=1) as f:   # 라인 버퍼링: kill 돼도 직전 줄까지 남는다
            while True:
                u = mon.read_utilization()
                cur = cpu_mem()
                if cur is None:
                    time.sleep(a.interval)
                    continue
                c = cpu_pct(prev, cur)
                m = 100.0 * (cur["mem_total"] - cur["mem_avail"]) / cur["mem_total"]
                us = "nan" if u is None else f"{u:.2f}"
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}, "
                        f"NPU: {us}%, CPU: {c:.2f}%, MEM: {m:.2f}%\n")
                prev = cur
                time.sleep(a.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
