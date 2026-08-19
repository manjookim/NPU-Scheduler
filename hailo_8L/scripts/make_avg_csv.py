#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_avg_csv.py — 조건(use_det,use_seg,use_pose)별 3회 반복을 평균 낸 요약 CSV 생성.
results_batch1_pri0_fps30_avg.csv와 동일한 포맷: 첫 컬럼은 run_id 대신 n_runs(그룹에
포함된 반복 횟수), use_det/use_seg/use_pose/batch/threshold_*/timeout_ms/priority_*는
정수 그대로(조건 식별용, 그룹 내 항상 동일값), 나머지 측정값 컬럼은 NaN을 제외한 평균을
소수 4자리로 반올림(값이 전부 NaN이면 결과도 NaN).

사용법: python3 make_avg_csv.py <raw.csv> [출력경로]
       (출력경로 생략 시 <raw>_avg.csv)
"""
import sys, os, csv
from statistics import mean

IDENTITY_COLS = ["use_det","use_seg","use_pose","batch",
                  "threshold_det","threshold_seg","threshold_pose","timeout_ms",
                  "priority_det","priority_seg","priority_pose"]

def is_nan(v):
    return v is None or v.strip() == "" or v.strip().lower() == "nan"

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_avg.csv"

    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    key_cols = ("use_det", "use_seg", "use_pose")
    groups = {}
    order = []
    for r in rows:
        key = tuple(r[c] for c in key_cols)
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
