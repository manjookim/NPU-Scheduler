"""
add_hrtt_columns.py
기존 results_all.csv에 HRTT/HTML 지표 열을 추가해 results_with_hrtt.csv 생성
추가 열은 모두 비워두며, 사용자가 HTML 파일을 보고 직접 입력

추가 열 구조:
  [글로벌 - 실험당 1개] 상단 바 공통값
    hrtt_networks, hrtt_switches_per_sec, hrtt_idle_time, hrtt_run_time_sec

  [모델별 - det/seg/pose 각각] 우측 패널 개별 모델값
    {m}_avg_fps, {m}_device_usage_pct,
    {m}_sched_threshold_pct, {m}_sched_timeout_pct, {m}_sched_idle_pct,
    {m}_avg_latency_ms, {m}_max_latency_ms, {m}_activation_ms, {m}_deactivation_ms
"""

import csv
import os

SRC_CSV  = os.path.join(os.path.dirname(__file__), "results_all.csv")
DEST_CSV = os.path.join(os.path.dirname(__file__), "results_with_hrtt.csv")

# 글로벌 열 (상단 바 - 실험당 1개)
GLOBAL_COLUMNS = [
    "hrtt_networks",          # Networks 수
    "hrtt_switches_per_sec",  # Switches/s
    "hrtt_idle_time_pct",     # Idle Time (%)
    "hrtt_run_time_sec",      # Run Time (s)
]

# 모델별 열 (우측 패널 - 활성 모델에만 값 입력)
MODEL_METRICS = [
    "avg_fps",               # Avg. FPS
    "device_usage_pct",      # Device Usage (%)
    "sched_threshold_pct",   # Inference Statistics: Threshold (%)
    "sched_timeout_pct",     # Inference Statistics: Timeout (%)
    "sched_idle_pct",        # Inference Statistics: Idle (%)
    "avg_latency_ms",        # Avg. Latency
    "max_latency_ms",        # Max. Latency
    "activation_ms",         # Activation
    "deactivation_ms",       # Deactivation
]

MODELS = ["det", "seg", "pose"]
MODEL_COLUMNS = [f"{m}_{metric}" for m in MODELS for metric in MODEL_METRICS]

NEW_COLUMNS = GLOBAL_COLUMNS + MODEL_COLUMNS

def main():
    if not os.path.exists(SRC_CSV):
        print(f"오류: {SRC_CSV} 파일이 없습니다.")
        return

    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        existing_cols = reader.fieldnames or []

    all_cols = existing_cols + NEW_COLUMNS

    with open(DEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for row in rows:
            for col in NEW_COLUMNS:
                row[col] = ""
            writer.writerow(row)

    print(f"완료: {DEST_CSV}")
    print(f"  기존 열       : {len(existing_cols)}개")
    print(f"  글로벌 추가   : {len(GLOBAL_COLUMNS)}개")
    print(f"  모델별 추가   : {len(MODEL_COLUMNS)}개  ({len(MODELS)}모델 × {len(MODEL_METRICS)}지표)")
    print(f"  총 열         : {len(all_cols)}개 / 총 행: {len(rows)}개")
    print()
    print("[글로벌 열]")
    for col in GLOBAL_COLUMNS:
        print(f"  {col}")
    print("\n[모델별 열]")
    for col in MODEL_COLUMNS:
        print(f"  {col}")

if __name__ == "__main__":
    main()
