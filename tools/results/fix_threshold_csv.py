# -*- coding: utf-8 -*-
# threshold 실험 CSV 보정:
#  - "=priority"(Excel #NAME? 유발) 제거
#  - 단일 threshold 컬럼 -> 모델별 threshold_det/seg/pose 로 분리
#    (threshold=priority 이므로 각 모델 priority 값을 사용, 비활성 모델은 None)
import sys, csv

src = sys.argv[1] if len(sys.argv) > 1 else "results_threshold.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "results_threshold_fixed.csv"

with open(src, newline="") as f:
    rows = list(csv.reader(f))

header = rows[0]
ti = header.index("threshold")
pd_i = header.index("priority_det")
ps_i = header.index("priority_seg")
pp_i = header.index("priority_pose")

new_header = header[:ti] + ["threshold_det","threshold_seg","threshold_pose"] + header[ti+1:]
out = [new_header]

for r in rows[1:]:
    thr_val = r[ti]
    if thr_val == "=priority":
        # threshold = priority (모델별), 비활성 모델은 priority가 None이라 그대로 None
        thr_det, thr_seg, thr_pose = r[pd_i], r[ps_i], r[pp_i]
    else:
        # 고정 threshold 실험인 경우: 활성 모델만 값, 비활성은 None
        def fix(p): return thr_val if p != "None" else "None"
        thr_det, thr_seg, thr_pose = fix(r[pd_i]), fix(r[ps_i]), fix(r[pp_i])
    out.append(r[:ti] + [thr_det, thr_seg, thr_pose] + r[ti+1:])

with open(dst, "w", newline="") as f:
    csv.writer(f).writerows(out)

print(f"보정 완료: {dst} ({len(out)-1}행)")
print("새 컬럼: threshold_det, threshold_seg, threshold_pose")
