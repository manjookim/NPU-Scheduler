"""
make_excel.py
results_all.csv → Excel 4시트 변환

Sheet1: 1개 모델 추론 결과 (single)
Sheet2: 2개 모델 추론 결과 (multi-2)
Sheet3: 3개 모델 추론 결과 (multi-3)
Sheet4: 각 조건(priority 조합)별 metric 평균

사용법: python3 make_excel.py [results_all.csv] [output.xlsx]
"""

import sys
import pandas as pd
import numpy as np

# ── 인자 처리 ──
csv_path  = sys.argv[1] if len(sys.argv) > 1 else "results_all.csv"
xlsx_path = sys.argv[2] if len(sys.argv) > 2 else "results_summary.xlsx"

# ── CSV 로드 ──
df = pd.read_csv(csv_path)

# use_* 컬럼이 숫자인지 확인 후 정수 변환
for col in ["use_det", "use_seg", "use_pose"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# 모델 수 컬럼 추가
df["model_count"] = df["use_det"] + df["use_seg"] + df["use_pose"]

# ── 시트 분리 ──
df_single = df[df["model_count"] == 1].copy()
df_multi2 = df[df["model_count"] == 2].copy()
df_multi3 = df[df["model_count"] == 3].copy()

# ── 평균 시트 계산 ──
# 조건 키: 모델 조합 + priority 조합
group_keys = [
    "use_det", "use_seg", "use_pose",
    "priority_det", "priority_seg", "priority_pose"
]

# priority 컬럼이 "None" 문자열일 수 있으므로 그대로 groupby
metric_cols = [
    "det_latency_ms", "seg_latency_ms", "pose_latency_ms",
    "cpu_percent", "mem_percent",
    "voluntary_ctx_switches", "nonvoluntary_ctx_switches",
    "npu_percent"
]

# 존재하는 컬럼만 사용
existing_metrics = [c for c in metric_cols if c in df.columns]

# 숫자형 변환
for col in existing_metrics:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df_avg = (
    df.groupby(group_keys, dropna=False)[existing_metrics]
    .agg(["mean", "std"])
    .round(3)
)
# 컬럼 이름 평탄화: (metric, stat) → metric_mean / metric_std
df_avg.columns = [f"{m}_{s}" for m, s in df_avg.columns]
df_avg = df_avg.reset_index()

# ── Excel 출력 ──
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    def write_sheet(df_sheet, sheet_name):
        # model_count 헬퍼 컬럼 제거 후 저장
        out = df_sheet.drop(columns=["model_count"], errors="ignore")
        out.to_excel(writer, sheet_name=sheet_name, index=False)
        # 컬럼 너비 자동 조정
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 30)

    write_sheet(df_single, "Sheet1_Single")
    write_sheet(df_multi2, "Sheet2_Multi2")
    write_sheet(df_multi3, "Sheet3_Multi3")

    # Sheet4: 평균
    df_avg.to_excel(writer, sheet_name="Sheet4_Average", index=False)
    ws4 = writer.sheets["Sheet4_Average"]
    for col_cells in ws4.columns:
        max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws4.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 30)

print(f"저장 완료: {xlsx_path}")
print(f"  Sheet1_Single : {len(df_single)}행")
print(f"  Sheet2_Multi2 : {len(df_multi2)}행")
print(f"  Sheet3_Multi3 : {len(df_multi3)}행")
print(f"  Sheet4_Average: {len(df_avg)}행 (조건별 평균)")
