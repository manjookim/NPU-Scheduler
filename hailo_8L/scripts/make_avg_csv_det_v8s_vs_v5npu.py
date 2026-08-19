#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_avg_csv_det_v8s_vs_v5npu.py — (Hailo-8L, rpi1) Det 단일모델 v8s_h8l(CPU 후처리) vs
v5-nms_core(NPU 후처리) FPS=60 실험의 3회 반복을 조건(hef_name)별로 평균.

infer_yolov5_hailo8l.cpp::save_csv_single()이 만든 CSV 전용 스키마(hailo_8/infer_yolov5_hailo8.cpp
와 완전히 동일한 스키마):
run_id,hef_name,img_size,batch,threshold,timeout_ms,priority,frame_count,
avg_preprocess_ms,avg_latency_ms,avg_postprocess_ms,avg_total_time_ms,
total_time_s,run_time_s,cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches

hailo_8L/scripts/make_avg_csv.py(3모델 프레임워크용, use_det/seg/pose로 그룹핑)와 같은
방식이되, 이 파일은 단일 모델 1행 CSV라 hef_name 하나로 그룹핑한다는 점만 다르다.

사용법: python3 make_avg_csv_det_v8s_vs_v5npu.py <raw.csv> [출력경로]
       (출력경로 생략 시 <raw>_avg.csv)
"""
import sys, os, csv
from statistics import mean

IDENTITY_COLS = ["hef_name", "img_size", "batch", "threshold", "timeout_ms", "priority"]


def is_nan(v):
    return v is None or v.strip() == "" or v.strip().lower() == "nan"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_avg.csv"

    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    groups = {}
    order = []
    for r in rows:
        key = r["hef_name"]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    other_cols = [c for c in fieldnames if c != "run_id" and c not in IDENTITY_COLS]
    out_header = ["n_runs"] + IDENTITY_COLS + other_cols
    out_rows = [out_header]

    for key in order:
        grp = groups[key]
        row = [str(len(grp))]
        for c in IDENTITY_COLS:
            row.append(grp[0][c])
        for c in other_cols:
            vals = [float(r[c]) for r in grp if not is_nan(r.get(c))]
            row.append(f"{mean(vals):.4f}" if vals else "NaN")
        out_rows.append(row)

    with open(dst, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print(f"완료: {len(order)}개 조건 x 평균 -> {dst}")


if __name__ == "__main__":
    main()
