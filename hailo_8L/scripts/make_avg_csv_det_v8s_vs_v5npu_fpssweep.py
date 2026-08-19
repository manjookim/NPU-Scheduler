#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_avg_csv_det_v8s_vs_v5npu_fpssweep.py — (Hailo-8L, rpi1) Det 단일모델
v8s_h8l(CPU 후처리) vs v5-nms_core(NPU 후처리) x INPUT_FPS{0,30} 실험의 3회 반복을
조건(hef_name, input_fps)별로 평균.

make_avg_csv_det_v8s_vs_v5npu.py(FPS=60 고정판, hef_name 하나로만 그룹핑)의 후속 —
이 스크립트는 FPS도 스윕하므로 (hef_name, input_fps) 조합으로 그룹핑한다는 점만 다르다.

사용법: python3 make_avg_csv_det_v8s_vs_v5npu_fpssweep.py <raw.csv> [출력경로]
       (출력경로 생략 시 <raw>_avg.csv)
"""
import sys, os, csv
from statistics import mean

IDENTITY_COLS = ["hef_name", "input_fps", "img_size", "batch", "threshold", "timeout_ms", "priority"]


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
        key = (r["hef_name"], r["input_fps"])
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
