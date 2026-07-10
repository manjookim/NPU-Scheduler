# -*- coding: utf-8 -*-
# 재실행(nancheck) 42개 값을 results_threshold_final.csv 의 해당 14조건 42행에 반영.
# base 지표: results_nancheck.csv / HRTT 지표: html/nancheck (idle +1.5%p)
import csv, re, os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from parse_metrics_prithr import parse_html, metrics

FINAL = "results_threshold_final.csv"

def norm(v):
    if v in ('None','NaN','',None): return None
    return str(int(float(v)))  # 15.0 -> 15

# 1) nancheck HRTT 지표 추출: key=(pd,ps,pp,run)
hrtt = {}
for f in glob.glob("html/nancheck/*.html"):
    m = re.match(r"1D-1S-1P_(\d+)PD-(\d+)PS-(\d+)PP_run(\d+)", os.path.basename(f))
    if not m: continue
    pd,ps,pp,run = m.groups()
    pr = parse_html(f)
    mt = metrics(pr) if pr else None
    hrtt[(pd,ps,pp,run)] = mt

# 2) nancheck base 지표: results_nancheck.csv
base = {}
for r in csv.DictReader(open("results_nancheck.csv")):
    base[(norm(r['priority_det']),norm(r['priority_seg']),norm(r['priority_pose']),r['run_id'])] = r

# 3) threshold_final 로드 후 해당 42행 교체
rows = list(csv.DictReader(open(FINAL))); fields = list(rows[0].keys())
replaced = 0
for r in rows:
    if not (r['use_det']=='1' and r['use_seg']=='1' and r['use_pose']=='1'): continue
    key = (norm(r['priority_det']),norm(r['priority_seg']),norm(r['priority_pose']),r['run_id'])
    if key not in base: continue
    b = base[key]
    # base 지표 교체
    for c in ['det_latency_ms','seg_latency_ms','pose_latency_ms','cpu_percent','mem_percent',
              'voluntary_ctx_switches','nonvoluntary_ctx_switches','npu_percent',
              'threshold_det','threshold_seg','threshold_pose']:
        if c in b: r[c] = b[c]
    # HRTT 지표 교체
    mt = hrtt.get((key[0],key[1],key[2],key[3]))
    if mt:
        r['switches_per_s'] = mt['switches_per_s']
        r['idle_time_pct']  = round(mt['idle_time_pct'], 3)  # metrics()에 이미 +1.5 포함
        r['run_time_s']     = mt['run_time_s']
        for m in ('det','seg','pose'):
            mm = mt['models'].get(m)
            if mm:
                r[f'avg_fps_{m}']=mm['avg_fps']; r[f'avg_latency_{m}']=mm['avg_latency']
                r[f'max_latency_{m}']=mm['max_latency']; r[f'activation_{m}']=mm['activation']
    replaced += 1

w = csv.DictWriter(open(FINAL,'w',newline=''), fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"교체된 행: {replaced} (기대 42)")
# NaN 잔여 확인
nan_left = sum(1 for r in rows if any(str(v)=='NaN' for v in r.values()))
print(f"NaN 남은 행: {nan_left}")
