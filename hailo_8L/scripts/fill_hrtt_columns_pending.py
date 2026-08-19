#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_hrtt_columns_pending.py
지금까지 HRTT 전용 컬럼(switches_per_s, idle_time_pct, avg_fps_*, avg_latency_*,
max_latency_*, activation_*)이 안 채워졌거나 일부만 채워진 실험 CSV들을 한번에 처리.
(2026-08-05 기준: default_workload_exp2(root), 07-28_default_workload_exp1,
07-29_batch_singlemodel_pp_exp1의 pp_off/pp_on)

각 실험마다 HRTT 파일명 규칙이 달라서 (regex, key_cols, traces_dir, csv_path) 설정을
리스트로 두고 순회. fill_hrtt_columns_batch1_pri0_fps30.py와 동일한 파싱 로직
(tools/hrtt/parse_hrtt.compute_metrics, protobuf 직접 파싱) 재사용.
"""
import sys, os, re, csv, glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hailo_8L
REPO = os.path.dirname(BASE)                                        # NPUscheduler
sys.path.insert(0, os.path.join(REPO, "tools", "hrtt"))
import profiler_pb2  # noqa: F401
from parse_hrtt import compute_metrics

JOBS = [
    {
        "name": "default_workload_exp2 (root, 2026-08-02)",
        "csv": os.path.join(REPO, "2026-08-02_default_workload_exp2", "csv", "results_default_workload.csv"),
        "traces": os.path.join(REPO, "2026-08-02_default_workload_exp2", "hrtt"),
        "regex": re.compile(r"(\d)D-(\d)S-(\d)P_default_run(\d+)_"),
        "key_cols": ["use_det", "use_seg", "use_pose", "run_id"],
    },
    {
        "name": "07-28_default_workload_exp1",
        "csv": os.path.join(BASE, "experiments", "2026-07-28_default_workload_exp1", "csv", "results_default_workload.csv"),
        "traces": os.path.join(BASE, "experiments", "2026-07-28_default_workload_exp1", "traces"),
        "regex": re.compile(r"(\d)D-(\d)S-(\d)P_default_run(\d+)_"),
        "key_cols": ["use_det", "use_seg", "use_pose", "run_id"],
    },
    {
        "name": "07-29_batch_singlemodel_pp_exp1 (pp_off)",
        "csv": os.path.join(BASE, "experiments", "2026-07-29_batch_singlemodel_pp_exp1", "csv", "results_pp.csv"),
        "traces": os.path.join(BASE, "experiments", "2026-07-29_batch_singlemodel_pp_exp1", "traces", "pp_off"),
        "regex": re.compile(r"(\d)D-(\d)S-(\d)P_b(\d+)_ppoff_run(\d+)_"),
        "key_cols": ["use_det", "use_seg", "use_pose", "batch", "run_id"],
    },
    {
        "name": "07-29_batch_singlemodel_pp_exp1 (pp_on)",
        "csv": os.path.join(BASE, "experiments", "2026-07-29_batch_singlemodel_pp_exp1", "csv", "results_pp_on.csv"),
        "traces": os.path.join(BASE, "experiments", "2026-07-29_batch_singlemodel_pp_exp1", "traces", "pp_on"),
        "regex": re.compile(r"(\d)D-(\d)S-(\d)P_b(\d+)_ppon_run(\d+)_"),
        "key_cols": ["use_det", "use_seg", "use_pose", "batch", "run_id"],
    },
]


def run_job(job):
    print(f"\n=== {job['name']} ===")
    csv_path, traces_dir = job["csv"], job["traces"]
    if not os.path.exists(csv_path):
        print(f"  [!] CSV 없음: {csv_path}"); return
    if not os.path.isdir(traces_dir):
        print(f"  [!] traces 폴더 없음: {traces_dir}"); return

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    keymap = {}
    for row in rows:
        key = tuple(row[c] for c in job["key_cols"])
        keymap[key] = row

    files = sorted(glob.glob(os.path.join(traces_dir, "*.hrtt")))
    matched = unmatched = noevent = skipped = 0
    unmatched_list = []

    for path in files:
        fn = os.path.basename(path)
        m = job["regex"].match(fn)
        if not m:
            skipped += 1
            continue
        key = tuple(m.groups())
        row = keymap.get(key)
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

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"  CSV: {csv_path}")
    print(f"  총 hrtt: {len(files)}  채움(matched): {matched}  규칙불일치: {skipped}  CSV매칭실패: {unmatched}  이벤트없음: {noevent}")
    if unmatched_list[:5]:
        print("  매칭실패 예시:", unmatched_list[:5])


if __name__ == "__main__":
    for job in JOBS:
        run_job(job)
