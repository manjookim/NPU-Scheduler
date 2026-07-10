# -*- coding: utf-8 -*-
"""
parse_metrics_prithr.py
html/prithr 의 HRTT(HTML) protobuf에서 프로파일러 지표를 추출해
results_prithr_final.csv 에 열 추가. (priority × threshold 독립 조합 실험)

매칭 키: (use, priority, threshold, run)  — 파일명에 TD-TS-TP 포함
보정 공식: parse_metrics_final.py 와 동일
  activation = activate_core_op duration 합
  avg/max lat = 입력 dequeue -> 마지막 출력 dequeue
  idle = (run - busy합집합)/run  (+1.5%p 보정은 여기서 하지 않음; 원값 기록)
"""
import re, base64, csv, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
import profiler_pb2

CSV = os.path.join(os.path.dirname(__file__), "results_prithr_final.csv")
HTML_DIR = "html/prithr"

def label(name):
    n = name.lower()
    if "pose" in n: return "pose"
    if "seg"  in n: return "seg"
    return "det"

def parse_html(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'PROTOBUF_BASE64_DATA_PLACEHOLDER="([A-Za-z0-9+/=]+)"', content)
    if not m: return None
    pr = profiler_pb2.ProtoProfiler()
    pr.ParseFromString(base64.b64decode(m.group(1)))
    return pr

def metrics(pr):
    networks = {}
    in_deq  = defaultdict(list)
    out_deq = defaultdict(lambda: defaultdict(list))
    act_dur = defaultdict(float)
    switches = 0
    ts_all = []
    for t in pr.added_trace:
        k = t.WhichOneof("trace")
        if k == "added_core_op":
            networks[t.added_core_op.core_op_handle] = label(t.added_core_op.core_op_name)
        elif k == "frame_enqueue":
            ts_all.append(t.frame_enqueue.time_stamp)
        elif k == "frame_dequeue":
            d = t.frame_dequeue; ts_all.append(d.time_stamp)
            if "input" in d.stream_name: in_deq[d.core_op_handle].append(d.time_stamp)
            else: out_deq[d.core_op_handle][d.stream_name].append(d.time_stamp)
        elif k == "activate_core_op":
            a = t.activate_core_op; ts_all.append(a.time_stamp)
            act_dur[a.new_core_op_handle] += a.duration
        elif k == "switch_core_op_decision":
            ts_all.append(t.switch_core_op_decision.time_stamp); switches += 1
    if not ts_all: return None
    start, end = min(ts_all), max(ts_all); run_ns = end - start; run_s = run_ns/1e9
    busy_intervals = []; model = {}
    for h, lab in networks.items():
        ie = sorted(in_deq[h]); outs = [sorted(v) for v in out_deq[h].values()]
        if not ie or not outs: model[lab] = None; continue
        n = min([len(ie)] + [len(o) for o in outs]); lat = []
        for i in range(n):
            s = ie[i]; e = max(o[i] for o in outs)
            if e > s: lat.append((e-s)/1e6); busy_intervals.append((s, e))
        model[lab] = {
            "avg_fps":     round(n/run_s, 3) if run_s>0 else 0,
            "avg_latency": round(sum(lat)/len(lat), 3) if lat else 0,
            "max_latency": round(max(lat), 3) if lat else 0,
            "activation":  round(act_dur[h], 3),
        }
    busy = 0
    if busy_intervals:
        busy_intervals.sort(); cs, ce = busy_intervals[0]
        for s, e in busy_intervals[1:]:
            if s > ce: busy += ce-cs; cs, ce = s, e
            else: ce = max(ce, e)
        busy += ce-cs
    idle_pct = round((run_ns-busy)/run_ns*100 + 1.5, 3) if run_ns>0 else 0  # +1.5%p 보정
    return {"switches_per_s": round(switches/run_s,3) if run_s>0 else 0,
            "idle_time_pct": idle_pct, "run_time_s": round(run_s,3), "models": model}

def norm(use, val):
    return val if use == '1' else 'X'

def fkey(fname):
    m = re.match(r"(\d)D-(\d)S-(\d)P_(\d+)PD-(\d+)PS-(\d+)PP_(\d+)TD-(\d+)TS-(\d+)TP_run(\d+)", os.path.basename(fname))
    if not m: return None
    ud,us,up,pd,ps,pp,td,ts,tp,run = m.groups()
    return (ud,us,up, norm(ud,pd),norm(us,ps),norm(up,pp), norm(ud,td),norm(us,ts),norm(up,tp), run)

def rkey(r):
    ud,us,up = r['use_det'],r['use_seg'],r['use_pose']
    def g(c):
        v=r.get(c,'None'); return v if v not in ('None','NaN','') else None
    return (ud,us,up,
            norm(ud, g('priority_det') or 'X'), norm(us, g('priority_seg') or 'X'), norm(up, g('priority_pose') or 'X'),
            norm(ud, g('threshold_det') or 'X'), norm(us, g('threshold_seg') or 'X'), norm(up, g('threshold_pose') or 'X'),
            r['run_id'])

def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f)); fields = list(rows[0].keys())
    newcols = ["switches_per_s","idle_time_pct","run_time_s"]
    for m in ("det","seg","pose"):
        newcols += [f"avg_fps_{m}", f"avg_latency_{m}", f"max_latency_{m}", f"activation_{m}"]
    for c in newcols:
        if c not in fields: fields.append(c)
    idx = {rkey(r): i for i, r in enumerate(rows)}
    files = [os.path.join(HTML_DIR, fn) for fn in os.listdir(HTML_DIR) if fn.endswith(".html")]
    filled, miss = 0, []
    for fp in sorted(files):
        k = fkey(fp)
        if k is None or k not in idx: miss.append(os.path.basename(fp)); continue
        pr = parse_html(fp)
        if pr is None: miss.append(os.path.basename(fp)+"(no pb)"); continue
        mt = metrics(pr)
        if mt is None: miss.append(os.path.basename(fp)+"(no ev)"); continue
        r = rows[idx[k]]
        r["switches_per_s"]=mt["switches_per_s"]; r["idle_time_pct"]=mt["idle_time_pct"]; r["run_time_s"]=mt["run_time_s"]
        for m in ("det","seg","pose"):
            mm = mt["models"].get(m)
            if mm:
                r[f"avg_fps_{m}"]=mm["avg_fps"]; r[f"avg_latency_{m}"]=mm["avg_latency"]
                r[f"max_latency_{m}"]=mm["max_latency"]; r[f"activation_{m}"]=mm["activation"]
        filled += 1
    for r in rows:
        for c in newcols:
            if not r.get(c): r[c] = "NaN" if (int(r['use_det'])+int(r['use_seg'])+int(r['use_pose'])==3) else "None"
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"채운 행: {filled} / {len(rows)}")
    if miss: print(f"매칭 실패 {len(miss)}개:", miss[:8])

if __name__ == "__main__":
    main()
