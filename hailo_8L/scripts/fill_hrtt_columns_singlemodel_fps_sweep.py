#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_hrtt_columns_singlemodel_fps_sweep.py
2026-08-04_singlemodel_fps_sweep_exp1 실험 전용 (단일모델만, INPUT_FPS={5,10,15,20,25}).
traces/*.hrtt 하나의 폴더에 5개 fps가 다 섞여 있어서, 파일명에서 fps를 뽑아 맞는
results_singlemodel_fps{N}.csv로 라우팅한 뒤 (use_det,use_seg,use_pose,run_id)로 매칭.

파일명 규칙: <ud>D-<us>S-<up>P_fps<N>_run<n>_<시각>.hrtt
"""
import sys, os, re, csv, glob
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hailo_8L
REPO = os.path.dirname(BASE)                                        # NPUscheduler
sys.path.insert(0, os.path.join(REPO, "tools", "hrtt"))
import profiler_pb2  # noqa: F401
from parse_hrtt import compute_metrics

EXP_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE, "experiments", "2026-08-04_singlemodel_fps_sweep_exp1")
TRACES_DIR = os.path.join(EXP_DIR, "traces")
CSV_DIR = os.path.join(EXP_DIR, "csv")

FNAME_RE = re.compile(r"(\d)D-(\d)S-(\d)P_fps(\d+)_run(\d+)_")

# fps -> (fieldnames, rows, keymap)
csv_data = {}
for path in glob.glob(os.path.join(CSV_DIR, "results_singlemodel_fps*.csv")):
    m = re.search(r"fps(\d+)\.csv$", path)
    if not m:
        continue
    fps = m.group(1)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    keymap = {(r["use_det"], r["use_seg"], r["use_pose"], r["run_id"]): r for r in rows}
    csv_data[fps] = {"path": path, "fieldnames": fieldnames, "rows": rows, "keymap": keymap}

files = sorted(glob.glob(os.path.join(TRACES_DIR, "*.hrtt")))
matched = unmatched = noevent = skipped = 0
unmatched_list = []
per_fps_matched = defaultdict(int)

for path in files:
    fn = os.path.basename(path)
    m = FNAME_RE.match(fn)
    if not m:
        skipped += 1
        continue
    ud, us, up, fps, run = m.groups()
    if fps not in csv_data:
        unmatched += 1; unmatched_list.append(fn + " (csv 없음)"); continue
    row = csv_data[fps]["keymap"].get((ud, us, up, run))
    if row is None:
        unmatched += 1; unmatched_list.append(fn); continue

    try:
        p = profiler_pb2.ProtoProfiler()
        p.ParseFromString(open(path, "rb").read())
        metrics = compute_metrics(p)
    except Exception as e:
        print(f"  [!] 파싱 실패: {fn} ({e})")
        noevent += 1
        continue
    if metrics is None:
        noevent += 1
        continue

    g = metrics["global"]
    row["switches_per_s"] = f"{g['hrtt_switches_per_sec']:.4f}"
    row["idle_time_pct"] = f"{g['hrtt_idle_time_pct']:.4f}"
    for lbl in ("det", "seg", "pose"):
        mm = metrics["models"].get(lbl)
        if not mm:
            continue
        row[f"avg_fps_{lbl}"] = f"{mm['avg_fps']:.4f}"
        row[f"avg_latency_{lbl}"] = f"{mm['avg_latency_ms']:.4f}"
        row[f"max_latency_{lbl}"] = f"{mm['max_latency_ms']:.4f}"
        row[f"activation_{lbl}"] = f"{mm['activation_ms']:.4f}"
    matched += 1
    per_fps_matched[fps] += 1

for fps, d in csv_data.items():
    with open(d["path"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=d["fieldnames"])
        w.writeheader()
        w.writerows(d["rows"])

print(f"총 hrtt: {len(files)}")
print(f"  채움(matched): {matched}  (fps별: {dict(per_fps_matched)})")
print(f"  이름규칙 불일치 skip: {skipped}")
print(f"  CSV행 매칭 실패: {unmatched}")
print(f"  이벤트 없음/파싱실패: {noevent}")
if unmatched_list[:5]:
    print("  매칭실패 예시:", unmatched_list[:5])
