"""
parse_hrtt.py
HTML(HRTT) 파일에서 protobuf 데이터를 파싱해 results_with_hrtt.csv를 자동 채움
"""

import re, base64, csv, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import profiler_pb2

HTML_DIR  = os.path.join(os.path.dirname(__file__), "html")
SRC_CSV   = os.path.join(os.path.dirname(__file__), "results_with_hrtt.csv")
DEST_CSV  = os.path.join(os.path.dirname(__file__), "results_with_hrtt_parsed.csv")

# 네트워크 이름 → 모델 레이블 매핑
def name_to_label(name):
    n = name.lower()
    if "pose" in n: return "pose"
    if "seg"  in n: return "seg"
    return "det"


def parse_html(html_path):
    with open(html_path, encoding="utf-8") as f:
        content = f.read()

    m = re.search(r'PROTOBUF_BASE64_DATA_PLACEHOLDER="([A-Za-z0-9+/=]+)"', content)
    if not m:
        return None
    raw = base64.b64decode(m.group(1))
    profiler = profiler_pb2.ProtoProfiler()
    profiler.ParseFromString(raw)
    return profiler


def compute_metrics(profiler):
    # ── 네트워크 목록 ──────────────────────────────────────────
    networks = {}   # handle → name
    for t in profiler.added_trace:
        if t.WhichOneof("trace") == "added_core_op":
            networks[t.added_core_op.core_op_handle] = t.added_core_op.core_op_name

    # ── 이벤트 분류 ──────────────────────────────────────────
    d2h_ts   = defaultdict(list)   # handle → [timestamps]
    act_dur  = defaultdict(list)   # handle → [durations ms]
    deact_dur= defaultdict(list)
    act_ts   = defaultdict(list)   # handle → [(start_ns, duration_ns)]
    switch_ev= defaultdict(lambda: {"threshold": 0, "timeout": 0, "idle": 0, "total": 0})

    all_ts = []

    for t in profiler.added_trace:
        kind = t.WhichOneof("trace")

        if kind == "frame_dequeue":
            fd = t.frame_dequeue
            all_ts.append(fd.time_stamp)
            if fd.direction == 1:   # D2H
                d2h_ts[fd.core_op_handle].append((fd.time_stamp, fd.stream_name))

        elif kind == "activate_core_op":
            ev = t.activate_core_op
            all_ts.append(ev.time_stamp)
            act_dur[ev.new_core_op_handle].append(ev.duration)
            act_ts[ev.new_core_op_handle].append(ev.time_stamp)

        elif kind == "deactivate_core_op":
            ev = t.deactivate_core_op
            all_ts.append(ev.time_stamp)
            deact_dur[ev.core_op_handle].append(ev.duration)

        elif kind == "switch_core_op_decision":
            ev = t.switch_core_op_decision
            all_ts.append(ev.time_stamp)
            sw = switch_ev[ev.core_op_handle]
            sw["total"] += 1
            if ev.over_threshold:     sw["threshold"] += 1
            if ev.over_timeout:       sw["timeout"]   += 1
            if ev.switch_because_idle: sw["idle"]     += 1

    if not all_ts:
        return None

    start_ns = min(all_ts)
    end_ns   = max(all_ts)
    run_time_s = (end_ns - start_ns) / 1e9

    # ── 글로벌 지표 ──────────────────────────────────────────
    total_switches = sum(sw["total"] for sw in switch_ev.values())
    switches_per_sec = total_switches / run_time_s if run_time_s > 0 else 0

    # 모든 activation 이벤트를 시간순 정렬
    # 스케줄러 모드에서는 deactivate 이벤트가 없고, 다음 activate가 이전 모델의 종료를 의미함
    all_acts_sorted = sorted(
        [(ts, h) for h, times in act_ts.items() for ts in times]
    )

    # 핸들별 active_ns 계산: 현재 activate → 다음 activate(어느 모델이든) 간격
    handle_active_ns = defaultdict(int)
    for i, (ts, h) in enumerate(all_acts_sorted):
        next_ts = all_acts_sorted[i + 1][0] if i + 1 < len(all_acts_sorted) else end_ns
        handle_active_ns[h] += (next_ts - ts)

    # 전체 active time = 첫 activation ~ end_ns
    first_act_ts = all_acts_sorted[0][0] if all_acts_sorted else end_ns
    total_active_ns = end_ns - first_act_ts
    idle_pct = max(0.0, (first_act_ts - start_ns) / (end_ns - start_ns) * 100) if end_ns > start_ns else 0

    # ── 모델별 지표 ──────────────────────────────────────────
    model_metrics = {}

    for handle, name in networks.items():
        label = name_to_label(name)

        # D2H 타임스탬프 → 스트림별 그룹
        streams = defaultdict(list)
        for ts, sname in d2h_ts[handle]:
            streams[sname].append(ts)

        # 하나의 스트림을 대표로 사용 (알파벳 첫 번째)
        if not streams:
            model_metrics[label] = None
            continue

        rep_stream = sorted(streams.keys())[0]
        rep_ts = sorted(streams[rep_stream])
        frame_count = len(rep_ts)

        avg_fps = frame_count / run_time_s if run_time_s > 0 else 0

        # H2D 입력 타임스탬프 (input enqueue)
        h2d_ts = sorted([ts for ts, sn in d2h_ts[handle] if sn not in streams])
        # d2h_ts[handle]는 D2H만 담겨 있으므로, H2D는 별도 수집
        h2d_list = sorted([t.frame_dequeue.time_stamp
                           for t in profiler.added_trace
                           if t.WhichOneof("trace") == "frame_dequeue"
                           and t.frame_dequeue.core_op_handle == handle
                           and t.frame_dequeue.direction == 0])

        # H2D → D2H 페어링으로 실제 latency 계산
        pair_count = min(len(h2d_list), len(rep_ts))
        latencies_ms = [(rep_ts[i] - h2d_list[i]) / 1e6
                        for i in range(pair_count) if rep_ts[i] > h2d_list[i]]
        avg_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
        max_lat = max(latencies_ms) if latencies_ms else 0

        # device usage: handle_active_ns는 글로벌에서 이미 계산됨
        active_ns = handle_active_ns.get(handle, 0)
        device_usage_pct = active_ns / (run_time_s * 1e9) * 100 if run_time_s > 0 else 0

        # 스케줄러 통계
        sw = switch_ev[handle]
        sw_total = sw["total"]
        def pct(x): return round(x / sw_total * 100, 2) if sw_total > 0 else 0

        # activation / deactivation duration
        avg_act   = sum(act_dur[handle])   / len(act_dur[handle])   if act_dur[handle]   else 0
        avg_deact = sum(deact_dur[handle]) / len(deact_dur[handle]) if deact_dur[handle] else 0

        model_metrics[label] = {
            "avg_fps":          round(avg_fps, 3),
            "device_usage_pct": round(device_usage_pct, 3),
            "sched_threshold_pct": pct(sw["threshold"]),
            "sched_timeout_pct":   pct(sw["timeout"]),
            "sched_idle_pct":      pct(sw["idle"]),
            "avg_latency_ms":   round(avg_lat, 3),
            "max_latency_ms":   round(max_lat, 3),
            "activation_ms":    round(avg_act, 3),
            "deactivation_ms":  round(avg_deact, 3),
        }

    return {
        "global": {
            "hrtt_networks":       len(networks),
            "hrtt_switches_per_sec": round(switches_per_sec, 3),
            "hrtt_idle_time_pct":  round(idle_pct, 3),
            "hrtt_run_time_sec":   round(run_time_s, 3),
        },
        "models": model_metrics,
    }


def filename_to_key(fname):
    """파일명에서 (use_det, use_seg, use_pose, pri_det, pri_seg, pri_pose) 파싱"""
    m = re.match(r"(\d+)D-(\d+)S-(\d+)P_(\d+)PD-(\d+)PS-(\d+)PP", os.path.basename(fname))
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def row_to_key(row):
    def v(col):
        s = row.get(col, "None")
        return None if s == "None" else int(s)
    ud = int(row["use_det"])
    us = int(row["use_seg"])
    up = int(row["use_pose"])
    pd_ = v("priority_det")  if ud else 0
    ps  = v("priority_seg")  if us else 0
    pp  = v("priority_pose") if up else 0
    return (ud, us, up, pd_ or 0, ps or 0, pp or 0)


def main():
    # CSV 로드
    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # HTML 파일 목록 (서브폴더 포함)
    html_files = []
    for dirpath, _, filenames in os.walk(HTML_DIR):
        for fn in filenames:
            if fn.endswith(".html"):
                html_files.append(os.path.join(dirpath, fn))
    print(f"HTML 파일: {len(html_files)}개")

    # key → row index 매핑
    key_to_idx = {}
    for i, row in enumerate(rows):
        key_to_idx[row_to_key(row)] = i

    filled = 0
    errors = []

    for html_path in sorted(html_files):
        key = filename_to_key(html_path)
        if key is None:
            errors.append(f"파일명 파싱 실패: {os.path.basename(html_path)}")
            continue

        idx = key_to_idx.get(key)
        if idx is None:
            errors.append(f"CSV 행 없음: {os.path.basename(html_path)} key={key}")
            continue

        print(f"  처리중: {os.path.basename(html_path)}", end=" ")
        profiler = parse_html(html_path)
        if profiler is None:
            print("→ 데이터 없음")
            errors.append(f"base64 없음: {os.path.basename(html_path)}")
            continue

        metrics = compute_metrics(profiler)
        if metrics is None:
            print("→ 이벤트 없음")
            continue

        row = rows[idx]
        g = metrics["global"]
        row["hrtt_networks"]       = g["hrtt_networks"]
        row["hrtt_switches_per_sec"] = g["hrtt_switches_per_sec"]
        row["hrtt_idle_time_pct"]  = g["hrtt_idle_time_pct"]
        row["hrtt_run_time_sec"]   = g["hrtt_run_time_sec"]

        for model_label in ("det", "seg", "pose"):
            mm = metrics["models"].get(model_label)
            if mm is None:
                continue
            for metric_name, val in mm.items():
                col = f"{model_label}_{metric_name}"
                if col in row:
                    row[col] = val

        det_mm = metrics['models'].get('det') or {}
        print(f"→ OK (networks={g['hrtt_networks']}, fps_det={det_mm.get('avg_fps', '-')})")
        filled += 1

    # CSV 저장
    with open(DEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n완료: {filled}개 행 채움, 오류: {len(errors)}개")
    for e in errors:
        print(f"  ! {e}")


if __name__ == "__main__":
    main()
