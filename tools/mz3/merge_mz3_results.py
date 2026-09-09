#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_mz3_results.py (2026-09-02) — MZ3 실험 결과 통합기.

인계문서 4-D 의 데이터 정합성 문제 4개를 한 번에 처리한다.

  · CSV 스키마 불일치   : 사전학습 42런(구 24컬럼) -> 신 스키마로 변환 후 병합
  · npu_percent 미병합  : npu_percent_*.csv 를 (tag, run_id, input_fps) 로 조인
  · HRTT 매핑 미검증    : run_mz3_sweep.sh v2 가 만든 **런별 트레이스 디렉터리**를 그대로 신뢰.
                          파일명·시각 순 1:1 추정 매핑을 쓰지 않는다.
  · 분산 미제시         : 반복 3회의 mean/std/n 요약표를 자동 생성

사용:
  python merge_mz3_results.py \
      --old  csv/results_mz3_default.csv csv/results_mz3_default_fps30.csv \
      --new  csv/results_mz3_pretrained.csv csv/results_mz3_pretrained_fps30.csv \
      --npu  csv/npu_percent_pretrained.csv \
      --traces ~/mz3_exp/traces/pretrained \
      --out merged.csv --summary summary.csv
"""
import argparse
import base64
import csv
import os
import re
import statistics
import sys

MODELS = ["ssd", "deeplab", "mnv2"]

# 구 24컬럼 -> 신 스키마 컬럼 대응.  (없는 값은 NaN 으로 채운다 — 지어내지 않는다)
OLD2NEW = {
    "tag": "tag", "run_id": "run_id", "input_fps": "input_fps", "frames": "frames",
    "use_ssd": "use_ssd", "use_deeplab": "use_deeplab", "use_mnv2": "use_mnv2",
    "ssd_latency_ms": "ssd_latency_ms", "deeplab_latency_ms": "deeplab_latency_ms",
    "mnv2_latency_ms": "mnv2_latency_ms",
    "ssd_fps": "avg_fps_ssd", "deeplab_fps": "avg_fps_deeplab", "mnv2_fps": "avg_fps_mnv2",
    "ssd_preprocess_ms": "avg_preprocess_ms_ssd",
    "deeplab_preprocess_ms": "avg_preprocess_ms_deeplab",
    "mnv2_preprocess_ms": "avg_preprocess_ms_mnv2",
    "ssd_total_time_s": "total_time_ssd_s", "deeplab_total_time_s": "total_time_deeplab_s",
    "mnv2_total_time_s": "total_time_mnv2_s",
    "cpu_percent": "cpu_percent", "mem_percent": "mem_percent",
    "voluntary_ctx_switches": "voluntary_ctx_switches",
    "nonvoluntary_ctx_switches": "nonvoluntary_ctx_switches",
    "wall_time_s": "wall_time_s",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_old(rows, source):
    out = []
    for r in rows:
        n = {new: r.get(old, "NaN") for old, new in OLD2NEW.items()}
        n["avg_latency_ssd"] = r.get("ssd_latency_ms", "NaN")
        n["avg_latency_deeplab"] = r.get("deeplab_latency_ms", "NaN")
        n["avg_latency_mnv2"] = r.get("mnv2_latency_ms", "NaN")
        # 구 세트에 없는 것들 — **비어 있음을 명시**한다. 특히 warmup/instr 은
        # 신 세트와 조건이 다르다는 사실 자체가 중요한 정보다.
        n["warmup"] = "0"                 # v1 은 워밍업 개념이 없었다
        n["instr"] = "mon0-trace1-dump30" # 사전학습 세트는 HAILO_MONITOR 없이 돌았다
        n["schema"] = "v1_24col"
        n["source"] = os.path.basename(source)
        out.append(n)
    return out


def tag_new(rows, source):
    for r in rows:
        r["schema"] = "v2"
        r["source"] = os.path.basename(source)
        r.setdefault("instr", "")
    return rows


def set_token(fname):
    """`results_mz3_<set>[_fps30].csv` / `npu_percent_<set>.csv` 에서 세트 이름을 뽑는다."""
    b = os.path.basename(fname)
    b = re.sub(r"\.csv$", "", b)
    b = re.sub(r"^(results_mz3_|npu_percent_)", "", b)
    b = re.sub(r"_fps\d+$", "", b)
    return b


def merge_npu(rows, npu_paths):
    """[중요] (tag, run_id, input_fps) 만으로 조인하면 **다른 세트의 npu_percent 가
    엉뚱한 런에 붙는다**(사전학습 행에 재학습 측정값이 들어가는 사고).
    세트 토큰까지 키에 넣어 세트 간 교차 오염을 막는다."""
    idx = {}
    for p in npu_paths:
        st = set_token(p)
        for r in read_csv(p):
            idx[(st, r["tag"], r["run_id"], r["input_fps"])] = r
    hit = 0
    for r in rows:
        k = (set_token(r.get("source", "")), r.get("tag"), r.get("run_id"), r.get("input_fps"))
        m = idx.get(k)
        if m:
            r["npu_percent"] = m.get("npu_percent", "NaN")
            r["npu_n_samples"] = m.get("n_samples", "NaN")
            r["npu_reliable"] = m.get("reliable", "")
            hit += 1
    print(f"  npu_percent 병합: {hit}/{len(rows)} 행 (세트 토큰 일치 건만)")
    if hit == 0:
        print("    [주의] 0건입니다. 파일명 규칙(results_mz3_<set>.csv / npu_percent_<set>.csv)을 확인하세요.")
    return rows


# ─────────────────────────── HRTT ───────────────────────────
def load_profiler(path):
    """`.hrtt`(raw protobuf) 와 HTML 리포트 양쪽을 모두 받아들인다."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hrtt"))
    import profiler_pb2
    raw = open(path, "rb").read()
    # [2026-09-09 수정] 원래는 raw 의 첫 바이트가 b"\n"/b"<"/b" " 이면 HTML 로 간주했다.
    # 그런데 .hrtt(raw protobuf)의 첫 바이트는 field#1 wiretype#2 = 0x0A = b"\n" 이라
    # **모든 .hrtt 가 HTML 로 오인**되어 placeholder 정규식에 걸리지 않고 None 을 반환했다.
    # (=> hrtt_metrics 가 ts 빈 채로 None, merge_hrtt 가 조용히 0/21)
    # 확장자 + 실제 HTML 서명으로 판별한다.
    _h = raw[:512].lstrip()
    if path.lower().endswith((".html", ".htm")) or _h[:5].lower() == b"<html" or _h[:9].lower() == b"<!doctype":
        m = re.search(rb'PROTOBUF_BASE64_DATA_PLACEHOLDER="([A-Za-z0-9+/=]+)"', raw)
        if not m:
            return None
        raw = base64.b64decode(m.group(1))
    p = profiler_pb2.ProtoProfiler()
    try:
        p.ParseFromString(raw)
    except Exception:
        return None
    return p


def hrtt_key(core_op_name):
    n = core_op_name.lower()
    if "ssd" in n:
        return "ssd"
    if "deeplab" in n:
        return "deeplab"
    if "mobilenet" in n or "mnv2" in n:
        return "mnv2"
    return None


def hrtt_metrics(trace_dir):
    """런 디렉터리 하나 -> 8L 기존 실험과 **동일한 공식**의 HRTT 지표.

    [2026-09-09] 공식을 tools/hrtt/parse_metrics_final.py 와 완전히 일치시켰다.
    그 스크립트가 results_singlemodel_fps5to25.xlsx 의 HRTT 열을 만든 원본이므로,
    같은 정의를 쓰지 않으면 8L 기존 실험과 숫자를 나란히 놓을 수 없다.

      switches_per_s = switch_core_op_decision 개수 / run_time_s
                       (activate_core_op 개수가 아니다)
      idle_time_pct  = (run - busy 구간 합집합) / run,
                       busy = 입력 dequeue ~ 그 프레임의 마지막 출력 dequeue
      avg_latency_/max_latency_ = 프레임별 (마지막 출력 deq - 입력 deq) ms
      avg_fps_       = 완료 프레임 수 / run_time_s
      activation_    = 해당 core_op 의 activate_core_op duration 합 (ms)

    조건 라벨은 **디렉터리 경로**에서 오므로 파일명 추정이 필요 없다."""
    from collections import defaultdict
    files = [os.path.join(trace_dir, f) for f in sorted(os.listdir(trace_dir))
             if f.endswith((".hrtt", ".html"))]
    if not files:
        return None
    names = {}
    in_deq = defaultdict(list)
    out_deq = defaultdict(lambda: defaultdict(list))
    act_dur, act_n, d2h = defaultdict(float), defaultdict(int), defaultdict(int)
    switches, ts = 0, []
    for fp in files:
        pr = load_profiler(fp)
        if pr is None:
            continue
        for t in pr.added_trace:
            k = t.WhichOneof("trace")
            if k == "added_core_op":
                names[t.added_core_op.core_op_handle] = t.added_core_op.core_op_name
            elif k == "frame_enqueue":
                ts.append(t.frame_enqueue.time_stamp)
            elif k == "frame_dequeue":
                d = t.frame_dequeue
                ts.append(d.time_stamp)
                if "input" in d.stream_name:
                    in_deq[d.core_op_handle].append(d.time_stamp)
                else:
                    out_deq[d.core_op_handle][d.stream_name].append(d.time_stamp)
                if d.direction == 1:
                    d2h[d.core_op_handle] += 1
            elif k == "activate_core_op":
                a = t.activate_core_op
                ts.append(a.time_stamp)
                act_dur[a.new_core_op_handle] += a.duration   # proto 필드명 주의
                act_n[a.new_core_op_handle] += 1
            elif k == "switch_core_op_decision":
                ts.append(t.switch_core_op_decision.time_stamp)
                switches += 1
    if not ts:
        return None
    run_ns = max(ts) - min(ts)
    run_s = run_ns / 1e9
    out = {"run_time_s": round(run_s, 3),
           "switches_per_s": round(switches / run_s, 3) if run_s > 0 else float("nan"),
           "hrtt_switches": switches,
           "hrtt_n_files": len(files)}
    busy = []
    for h, name in names.items():
        key = hrtt_key(name)
        if not key:
            continue
        out[f"activation_{key}"] = round(act_dur[h], 3)
        out[f"activations_{key}"] = act_n[h]
        out[f"hrtt_d2h_{key}"] = d2h.get(h, 0)
        # batch=auto 의 '실효 배치' = 활성화 1회당 처리 프레임 수.
        out[f"frames_per_activation_{key}"] = (round(d2h.get(h, 0) / act_n[h], 3)
                                               if act_n[h] else float("nan"))
        ie = sorted(in_deq[h])
        outs = [sorted(v) for v in out_deq[h].values()]
        if not ie or not outs:
            continue
        n = min([len(ie)] + [len(o) for o in outs])
        lat = []
        for i in range(n):
            s0 = ie[i]
            e0 = max(o[i] for o in outs)
            if e0 > s0:
                lat.append((e0 - s0) / 1e6)
                busy.append((s0, e0))
        out[f"avg_fps_{key}"] = round(n / run_s, 3) if run_s > 0 else 0.0
        out[f"avg_latency_{key}"] = round(sum(lat) / len(lat), 3) if lat else 0.0
        out[f"max_latency_{key}"] = round(max(lat), 3) if lat else 0.0
        out[f"hrtt_frames_{key}"] = n
    b = 0
    if busy:
        busy.sort()
        cs, ce = busy[0]
        for s0, e0 in busy[1:]:
            if s0 > ce:
                b += ce - cs
                cs, ce = s0, e0
            else:
                ce = max(ce, e0)
        b += ce - cs
    out["idle_time_pct"] = round((run_ns - b) / run_ns * 100, 3) if run_ns > 0 else 0.0
    return out


def merge_hrtt(rows, traces_root):
    hit = 0
    for r in rows:
        fps, tag, run = r.get("input_fps"), r.get("tag"), r.get("run_id")
        d = os.path.join(traces_root, f"fps{fps}", str(tag), f"run{run}")
        if not os.path.isdir(d):
            continue
        m = hrtt_metrics(d)
        if not m:
            continue
        for k, v in m.items():
            r[f"{k}" if k.startswith(("hrtt_", "activation", "frames_per")) else k] = v
        hit += 1
    print(f"  HRTT 병합: {hit}/{len(rows)} 행  (경로 규칙: <traces>/fps<F>/<tag>/run<N>/)")
    return rows


# ─────────────────────────── 요약 ───────────────────────────
def fnum(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def summarize(rows, cols):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r.get("source", ""), r.get("input_fps"), r.get("tag"))].append(r)
    out = []
    for (src, fps, tag), rs in sorted(g.items()):
        rec = {"source": src, "input_fps": fps, "tag": tag, "n_runs": len(rs)}
        for c in cols:
            vals = [v for v in (fnum(r.get(c)) for r in rs) if v is not None]
            if not vals:
                continue
            rec[f"{c}_mean"] = round(statistics.mean(vals), 4)
            rec[f"{c}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
            rec[f"{c}_n"] = len(vals)
        out.append(rec)
    return out


def write_csv(path, rows):
    if not rows:
        print(f"  (빈 결과 — {path} 생략)")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="NaN")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}  ({len(rows)}행 x {len(keys)}열)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", nargs="*", default=[], help="구 24컬럼 CSV")
    ap.add_argument("--new", nargs="*", default=[], help="신 스키마 CSV")
    ap.add_argument("--npu", nargs="*", default=[])
    ap.add_argument("--traces", help="run_mz3_sweep.sh v2 의 traces/<set> 루트")
    ap.add_argument("--out", default="merged_mz3.csv")
    ap.add_argument("--summary", default="summary_mz3.csv")
    a = ap.parse_args()

    rows = []
    for p in a.old:
        r = convert_old(read_csv(p), p)
        print(f"  구 스키마 {p}: {len(r)}행 변환")
        rows += r
    for p in a.new:
        r = tag_new(read_csv(p), p)
        print(f"  신 스키마 {p}: {len(r)}행")
        rows += r
    if not rows:
        raise SystemExit("입력 CSV 가 없습니다.")
    if a.npu:
        rows = merge_npu(rows, a.npu)
    if a.traces:
        rows = merge_hrtt(rows, a.traces)

    write_csv(a.out, rows)
    cols = ([f"{m}_latency_ms" for m in MODELS] + [f"avg_fps_{m}" for m in MODELS]
            + [f"coex_latency_{m}" for m in MODELS] + [f"wblock_ms_{m}" for m in MODELS]
            + [f"device_ms_{m}" for m in MODELS] + [f"frames_per_activation_{m}" for m in MODELS]
            + ["cpu_percent", "npu_percent", "switches_per_s"])
    write_csv(a.summary, summarize(rows, cols))
    print("\n[주의] source/instr 컬럼이 다른 행끼리 직접 비교하지 마십시오. "
          "계측 조건(HAILO_MONITOR 등)이 다르면 그 자체가 교란변수입니다.")


if __name__ == "__main__":
    main()
