#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_hrtt_columns_default_pri0_fps30_ppoff.py
<날짜>_default_pri0_fps30_ppoff_exp1 실험 전용 (batch=1/threshold=1/timeout=0/
priority=0, INPUT_FPS=30, ENABLE_POSTPROCESS=0 즉 후처리 시간 미측정).
traces/*.hrtt를 직접 파싱해서 csv/results_default_pri0_fps30_ppoff.csv의 HRTT 전용
NaN 컬럼(switches_per_s, idle_time_pct, avg_fps_*, avg_latency_*, max_latency_*,
activation_*)을 채운다. HTML 변환 없이 protobuf에서 바로 뽑음(fill_hrtt_columns.py와
동일 방식이나, 이 실험은 priority가 세 모델 다 고정(0)이라 파일명에 PD/PS/PP가 없고
(use_det,use_seg,use_pose,run_id)만으로 CSV 행과 매칭).

파일명 규칙: <ud>D-<us>S-<up>P_pri0_fps30_ppoff_run<n>_<시각>.hrtt
사용법: python3 fill_hrtt_columns_default_pri0_fps30_ppoff.py <실험폴더>
       (예: hailo_8L/experiments/2026-08-06_default_pri0_fps30_ppoff_exp1)
"""
import sys, os, re, csv, glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hailo_8L
REPO = os.path.dirname(BASE)                                        # NPUscheduler
sys.path.insert(0, os.path.join(REPO, "tools", "hrtt"))
import profiler_pb2  # noqa: F401 (parse_hrtt가 내부에서 사용)
from parse_hrtt import compute_metrics

if len(sys.argv) > 1:
    EXP_DIR = sys.argv[1]
else:
    candidates = sorted(glob.glob(os.path.join(BASE, "experiments", "*_default_pri0_fps30_ppoff_exp*")))
    if not candidates:
        sys.exit("실험 폴더를 찾지 못함 — 경로를 인자로 직접 넘길 것.")
    EXP_DIR = candidates[-1]

TRACES_DIR = os.path.join(EXP_DIR, "traces")
CSV_PATH = os.path.join(EXP_DIR, "csv", "results_default_pri0_fps30_ppoff.csv")

FNAME_RE = re.compile(r"(\d)D-(\d)S-(\d)P_pri0_fps30_ppoff_run(\d+)_")

# ---- CSV 로드 ----
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# (use_det, use_seg, use_pose, run_id) -> row
keymap = {}
for row in rows:
    key = (row["use_det"], row["use_seg"], row["use_pose"], row["run_id"])
    keymap[key] = row

# ---- HRTT 파싱 + 채우기 ----
files = sorted(glob.glob(os.path.join(TRACES_DIR, "*.hrtt")))
matched = unmatched = noevent = skipped = 0
unmatched_list = []

for path in files:
    fn = os.path.basename(path)
    m = FNAME_RE.match(fn)
    if not m:
        skipped += 1
        continue
    ud, us, up, run = m.groups()
    key = (ud, us, up, run)
    row = keymap.get(key)
    if row is None:
        unmatched += 1
        unmatched_list.append(fn)
        continue

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

# ---- CSV 다시 쓰기 ----
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"CSV: {CSV_PATH}")
print(f"총 hrtt: {len(files)}")
print(f"  채움(matched): {matched}")
print(f"  이름규칙 불일치 skip: {skipped}")
print(f"  CSV행 매칭 실패: {unmatched}")
print(f"  이벤트 없음/파싱실패: {noevent}")
if unmatched_list[:5]:
    print("  매칭실패 예시:", unmatched_list[:5])
