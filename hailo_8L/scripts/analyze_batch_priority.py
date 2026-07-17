#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch x priority 실험 분석:
 1) 조건별 3회 평균 요약 (요약 시트용)
 2) batch 고정→priority 효과 / priority 고정→batch 효과
 3) 상관관계 + 그래프(PNG)
출력: results/batch_priority/analysis/
"""
import os, glob, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE, "results", "batch_priority")
OUT = os.path.join(CSV_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)

# ---- 로드 ----
frames = []
for path in glob.glob(os.path.join(CSV_DIR, "results_*model_b*.csv")):
    m = re.search(r"results_(\d)model_b(\d+)\.csv", os.path.basename(path))
    if not m: continue
    n, b = int(m.group(1)), int(m.group(2))
    df = pd.read_csv(path, na_values=["NaN","None",""])
    df["model_count"] = n; df["batch"] = b
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)

metrics = ["det_latency_ms","seg_latency_ms","pose_latency_ms",
           "total_time_det_s","total_time_seg_s","total_time_pose_s",
           "avg_fps_det","avg_fps_seg","avg_fps_pose",
           "avg_latency_det","avg_latency_seg","avg_latency_pose",
           "npu_percent","switches_per_s","idle_time_pct","cpu_percent","mem_percent"]
gcols = ["model_count","batch","use_det","use_seg","use_pose",
         "priority_det","priority_seg","priority_pose"]

# ---- 1) 조건별 3회 평균 요약 (전체 + CSV별) ----
summary = raw.groupby(gcols, as_index=False)[metrics].mean().round(3)
summary.to_csv(os.path.join(OUT, "summary_all_avg.csv"), index=False)
# CSV(시트)별로도 저장
for (n,b), g in summary.groupby(["model_count","batch"]):
    g.to_csv(os.path.join(OUT, f"summary_{n}model_b{b}.csv"), index=False)

# ---- 2) 모델별 long 포맷 (own priority / advantage) ----
rows = []
for _, r in raw.iterrows():
    active = {"det":r["use_det"],"seg":r["use_seg"],"pose":r["use_pose"]}
    pri = {"det":r["priority_det"],"seg":r["priority_seg"],"pose":r["priority_pose"]}
    for mdl in ("det","seg","pose"):
        if active[mdl] != 1: continue
        others = [pri[o] for o in ("det","seg","pose") if o!=mdl and active[o]==1]
        max_other = max(others) if others else np.nan
        rows.append(dict(
            model_count=r["model_count"], batch=r["batch"], model=mdl,
            own_pri=pri[mdl], max_other_pri=max_other,
            advantage=(pri[mdl]-max_other) if others else np.nan,
            latency=r[f"{mdl}_latency_ms"], total_time=r[f"total_time_{mdl}_s"],
            fps=r[f"avg_fps_{mdl}"], hrtt_lat=r[f"avg_latency_{mdl}"],
            npu=r["npu_percent"]))
long = pd.DataFrame(rows)
long.to_csv(os.path.join(OUT, "per_model_long.csv"), index=False)

# ---- 3) priority 효과 (3모델, batch 고정): advantage별 평균 ----
multi = long[long["model_count"]>=2].copy()
pri_eff = multi.groupby(["batch","advantage"], as_index=False).agg(
    total_time=("total_time","mean"), latency=("latency","mean"),
    fps=("fps","mean"), n=("total_time","size")).round(3)
pri_eff.to_csv(os.path.join(OUT, "priority_effect.csv"), index=False)

# ---- 4) batch 효과 (priority 동일=starvation 없음): batch별 평균 ----
eq = raw[(raw["priority_det"]==raw["priority_seg"]) & (raw["priority_seg"]==raw["priority_pose"])]
batch_eff = eq.groupby(["model_count","batch"], as_index=False).agg(
    det_lat=("det_latency_ms","mean"), seg_lat=("seg_latency_ms","mean"),
    pose_lat=("pose_latency_ms","mean"),
    tot_det=("total_time_det_s","mean"), npu=("npu_percent","mean"),
    fps_det=("avg_fps_det","mean")).round(3)
batch_eff.to_csv(os.path.join(OUT, "batch_effect.csv"), index=False)

# ---- 5) 상관계수 ----
corr_rows = []
for b, g in multi.groupby("batch"):
    gg = g.dropna(subset=["advantage","total_time"])
    if len(gg)>2:
        corr_rows.append(dict(group=f"batch={b}", pair="advantage vs total_time",
                              corr=round(gg["advantage"].corr(gg["total_time"]),3), n=len(gg)))
        corr_rows.append(dict(group=f"batch={b}", pair="advantage vs latency",
                              corr=round(gg["advantage"].corr(gg["latency"]),3), n=len(gg)))
# batch vs metric (equal-priority)
ge = eq.dropna(subset=["batch"])
for col,label in [("det_latency_ms","batch vs det_latency"),
                  ("total_time_det_s","batch vs total_time_det"),
                  ("npu_percent","batch vs npu"),("avg_fps_det","batch vs fps_det")]:
    gg = ge.dropna(subset=[col])
    if len(gg)>2:
        corr_rows.append(dict(group="equal-priority", pair=label,
                              corr=round(gg["batch"].corr(gg[col]),3), n=len(gg)))
corr = pd.DataFrame(corr_rows)
corr.to_csv(os.path.join(OUT, "correlations.csv"), index=False)

# ================= 그래프 =================
plt.rcParams.update({"figure.dpi":120, "font.size":10})

# (A) priority advantage vs total_time (batch별 라인)
fig, ax = plt.subplots(figsize=(7,4.5))
for b, g in pri_eff.groupby("batch"):
    g = g.dropna(subset=["advantage"]).sort_values("advantage")
    ax.plot(g["advantage"], g["total_time"], marker="o", label=f"batch {b}")
ax.set_xlabel("priority advantage (own - max other)"); ax.set_ylabel("avg total processing time (s)")
ax.set_title("Priority effect: higher advantage -> faster finish (2·3-model)")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig_priority_effect.png")); plt.close(fig)

# (B) batch vs metrics (equal priority)
fig, axs = plt.subplots(2,2, figsize=(10,7))
for mc, g in batch_eff.groupby("model_count"):
    g=g.sort_values("batch")
    axs[0,0].plot(g["batch"], g["det_lat"], marker="o", label=f"{mc}-model")
    axs[0,1].plot(g["batch"], g["tot_det"], marker="o", label=f"{mc}-model")
    axs[1,0].plot(g["batch"], g["npu"],    marker="o", label=f"{mc}-model")
    axs[1,1].plot(g["batch"], g["fps_det"],marker="o", label=f"{mc}-model")
for ax,t,y in [(axs[0,0],"Det latency vs batch","latency (ms)"),
               (axs[0,1],"Det total time vs batch","total time (s)"),
               (axs[1,0],"NPU util vs batch","NPU %"),
               (axs[1,1],"Det FPS vs batch","fps")]:
    ax.set_title(t); ax.set_xlabel("batch size"); ax.set_ylabel(y); ax.legend(); ax.grid(alpha=.3)
fig.suptitle("Batch effect at equal priority (no starvation)"); fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig_batch_effect.png")); plt.close(fig)

# (C) 3모델 heatmap: 특정 batch에서 model=det의 total_time (own_pri x max_other_pri)
for b in [10]:
    sub = long[(long["model_count"]==3)&(long["model"]=="det")&(long["batch"]==b)]
    piv = sub.groupby(["own_pri","max_other_pri"])["total_time"].mean().unstack()
    if not piv.empty:
        fig, ax = plt.subplots(figsize=(5.5,4.5))
        im=ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("max other priority"); ax.set_ylabel("Det own priority")
        ax.set_title(f"Det total time (s), 3-model batch{b}")
        for (i,j),v in np.ndenumerate(piv.values):
            if not np.isnan(v): ax.text(j,i,f"{v:.0f}",ha="center",va="center",color="w",fontsize=8)
        fig.colorbar(im,ax=ax,label="total time (s)"); fig.tight_layout()
        fig.savefig(os.path.join(OUT,f"fig_heatmap_det_b{b}.png")); plt.close(fig)

print("=== 분석 완료 ===")
print("요약 조건 수:", len(summary))
print("\n[priority 효과] advantage vs total_time (batch별 상관):")
print(corr[corr["pair"].str.contains("advantage vs total_time")].to_string(index=False))
print("\n[batch 효과] 상관:")
print(corr[corr["group"]=="equal-priority"].to_string(index=False))
print("\nbatch_effect(동일우선순위) 미리보기:")
print(batch_eff.to_string(index=False))
print("\n생성 그래프:", [f for f in os.listdir(OUT) if f.endswith(".png")])
