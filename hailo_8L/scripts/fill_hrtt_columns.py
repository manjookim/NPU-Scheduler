#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_hrtt_columns.py
hrtt/batch_priority/*.hrtt 를 직접 파싱해서, results/batch_priority/*.csv 의
HRTT 전용 NaN 컬럼(switches_per_s, idle_time_pct, 모델별 avg_fps/avg_latency/
max_latency/activation)을 채운다. HTML 변환 없이 protobuf에서 바로 뽑음.

파일명 규칙: <ud>D-<us>S-<up>P_b<batch>_<pd>PD-<ps>PS-<pp>PP_run<n>_<시각>.hrtt
"""
import sys, os, re, csv, glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NPUscheduler
sys.path.insert(0, os.path.join(BASE, "tools", "hrtt"))
import profiler_pb2
from parse_hrtt import compute_metrics

HRTT_DIR = os.path.join(BASE, "hrtt", "batch_priority")
CSV_DIR  = os.path.join(BASE, "results", "batch_priority")

FNAME_RE = re.compile(r"(\d)D-(\d)S-(\d)P_b(\d+)_(\d+)PD-(\d+)PS-(\d+)PP_run(\d+)_")

def csv_name(n, batch): return f"results_{n}model_b{batch}.csv"

# ---- CSV 로드 ----
csv_rows = {}      # filename -> (fieldnames, rows[list of dict])
for path in glob.glob(os.path.join(CSV_DIR, "*.csv")):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        csv_rows[os.path.basename(path)] = (r.fieldnames, list(r))

# 각 CSV 내부 (조건 → row) 매핑
keymaps = {}
for name, (fields, rows) in csv_rows.items():
    km = {}
    for row in rows:
        k = (row["use_det"], row["use_seg"], row["use_pose"],
             row["priority_det"], row["priority_seg"], row["priority_pose"], row["run_id"])
        km[k] = row
    keymaps[name] = km

# ---- HRTT 파싱 + 채우기 ----
files = sorted(glob.glob(os.path.join(HRTT_DIR, "*.hrtt")))
matched = unmatched = noevent = skipped = 0
unmatched_list = []

for path in files:
    fn = os.path.basename(path)
    m = FNAME_RE.match(fn)
    if not m:
        skipped += 1
        continue
    ud, us, up, batch, pd, ps, pp, run = (int(x) for x in m.groups())
    n = ud + us + up
    cname = csv_name(n, batch)
    if cname not in keymaps:
        unmatched += 1; unmatched_list.append(fn); continue
    key = (str(ud), str(us), str(up), str(pd), str(ps), str(pp), str(run))
    row = keymaps[cname].get(key)
    if row is None:
        unmatched += 1; unmatched_list.append(fn); continue

    try:
        p = profiler_pb2.ProtoProfiler()
        p.ParseFromString(open(path, "rb").read())
        metrics = compute_metrics(p)
    except Exception:
        noevent += 1; continue
    if metrics is None:
        noevent += 1; continue

    g = metrics["global"]
    row["switches_per_s"] = f"{g['hrtt_switches_per_sec']:.4f}"
    row["idle_time_pct"]  = f"{g['hrtt_idle_time_pct']:.4f}"
    for lbl in ("det", "seg", "pose"):
        mm = metrics["models"].get(lbl)
        if not mm:
            continue
        row[f"avg_fps_{lbl}"]     = f"{mm['avg_fps']:.4f}"
        row[f"avg_latency_{lbl}"] = f"{mm['avg_latency_ms']:.4f}"
        row[f"max_latency_{lbl}"] = f"{mm['max_latency_ms']:.4f}"
        row[f"activation_{lbl}"]  = f"{mm['activation_ms']:.4f}"
    matched += 1

# ---- CSV 다시 쓰기 ----
for name, (fields, rows) in csv_rows.items():
    with open(os.path.join(CSV_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

print(f"총 hrtt: {len(files)}")
print(f"  채움(matched): {matched}")
print(f"  이름규칙 불일치 skip: {skipped}")
print(f"  CSV행 매칭 실패: {unmatched}")
print(f"  이벤트 없음/파싱실패: {noevent}")
if unmatched_list[:5]:
    print("  매칭실패 예시:", unmatched_list[:5])
