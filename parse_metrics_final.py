# -*- coding: utf-8 -*-
"""
parse_metrics_final.py
html/threshold + html/threshold_rerun 의 HRTT(HTML) protobuf에서
프로파일러 화면 지표를 보정된 공식으로 추출해 results_threshold_final.csv 에 열 추가.

추가 열:
  global : switches_per_s, idle_time_pct, run_time_s
  모델별 : avg_fps_{m}, avg_latency_{m}, max_latency_{m}, activation_{m}  (m=det/seg/pose)

보정(파일 0D-1S-1P_0PD-15PS-15PP_run1 기준, 화면값 54/64/167.83/6%):
  activation   = 해당 모델 activate_core_op duration 합  (정확 일치)
  avg/max lat  = 입력 dequeue -> 마지막 출력 dequeue (프레임별)
  idle         = (run - busy구간합집합)/run, busy=입력deq~마지막출력deq
"""
import re, base64, csv, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
import profiler_pb2

CSV = os.path.join(os.path.dirname(__file__), "results_threshold_final.csv")
HTML_DIRS = ["html/threshold", "html/threshold_rerun"]

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
    networks = {}                     # handle -> label
    in_deq  = defaultdict(list)       # handle -> [ts] (input dequeue)
    out_deq = defaultdict(lambda: defaultdict(list))  # handle -> stream -> [ts]
    act_dur = defaultdict(float)      # handle -> sum durations
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
    start, end = min(ts_all), max(ts_all)
    run_ns = end - start
    run_s = run_ns / 1e9

    # busy 구간 (모든 모델 합집합) + 모델별 latency
    busy_intervals = []
    model = {}
    for h, lab in networks.items():
        ie = sorted(in_deq[h])
        outs = [sorted(v) for v in out_deq[h].values()]
        if not ie or not outs:
            model[lab] = None; continue
        n = min([len(ie)] + [len(o) for o in outs])
        lat = []
        for i in range(n):
            s = ie[i]; e = max(o[i] for o in outs)
            if e > s:
                lat.append((e - s) / 1e6)
                busy_intervals.append((s, e))
        # avg_fps = 출력 프레임 수 / run
        frames = n
        model[lab] = {
            "avg_fps":      round(frames / run_s, 3) if run_s > 0 else 0,
            "avg_latency":  round(sum(lat)/len(lat), 3) if lat else 0,
            "max_latency":  round(max(lat), 3) if lat else 0,
            "activation":   round(act_dur[h], 3),
        }

    # idle = (run - busy합집합)/run
    busy = 0
    if busy_intervals:
        busy_intervals.sort()
        cs, ce = busy_intervals[0]
        for s, e in busy_intervals[1:]:
            if s > ce: busy += ce - cs; cs, ce = s, e
            else: ce = max(ce, e)
        busy += ce - cs
    idle_pct = round((run_ns - busy) / run_ns * 100, 3) if run_ns > 0 else 0

    return {
        "switches_per_s": round(switches / run_s, 3) if run_s > 0 else 0,
        "idle_time_pct":  idle_pct,
        "run_time_s":     round(run_s, 3),
        "models": model,
    }

def fkey(fname):
    m = re.match(r"(\d)D-(\d)S-(\d)P_(\d+)PD-(\d+)PS-(\d+)PP_run(\d+)", os.path.basename(fname))
    if not m: return None
    ud,us,up,pd,ps,pp,run = [int(x) for x in m.groups()]
    # 비활성 모델 priority는 0으로 정규화 (CSV None과 매칭)
    return (ud,us,up, pd if ud else 0, ps if us else 0, pp if up else 0, run)

def rkey(row):
    def p(c, use):
        v = row.get(c, "None");
        return int(v) if (use and v != "None") else 0
    ud,us,up = int(row["use_det"]), int(row["use_seg"]), int(row["use_pose"])
    return (ud,us,up, p("priority_det",ud), p("priority_seg",us), p("priority_pose",up), int(row["run_id"]))

def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    newcols = ["switches_per_s","idle_time_pct","run_time_s"]
    for m in ("det","seg","pose"):
        newcols += [f"avg_fps_{m}", f"avg_latency_{m}", f"max_latency_{m}", f"activation_{m}"]
    for c in newcols:
        if c not in fields: fields.append(c)

    idx = {rkey(r): i for i, r in enumerate(rows)}

    files = []
    for d in HTML_DIRS:
        files += [os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".html")]

    filled, miss = 0, []
    for fp in sorted(files):
        k = fkey(fp)
        if k is None or k not in idx:
            miss.append(os.path.basename(fp)); continue
        pr = parse_html(fp)
        if pr is None: miss.append(os.path.basename(fp)+" (no protobuf)"); continue
        mt = metrics(pr)
        if mt is None: miss.append(os.path.basename(fp)+" (no events)"); continue
        r = rows[idx[k]]
        r["switches_per_s"] = mt["switches_per_s"]
        r["idle_time_pct"]  = mt["idle_time_pct"]
        r["run_time_s"]     = mt["run_time_s"]
        for m in ("det","seg","pose"):
            mm = mt["models"].get(m)
            if mm:
                r[f"avg_fps_{m}"]     = mm["avg_fps"]
                r[f"avg_latency_{m}"] = mm["avg_latency"]
                r[f"max_latency_{m}"] = mm["max_latency"]
                r[f"activation_{m}"]  = mm["activation"]
        filled += 1

    # 빈 셀 None 채움
    for r in rows:
        for c in newcols:
            if not r.get(c): r[c] = "None"

    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    print(f"채운 행: {filled} / {len(rows)}")
    if miss:
        print(f"매칭 실패 {len(miss)}개:")
        for x in miss[:10]: print("  !", x)

if __name__ == "__main__":
    main()
