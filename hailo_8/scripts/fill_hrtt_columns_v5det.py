#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_hrtt_columns_v5det.py
hailo_8 v5det(NPU/CPU 후처리 비교) 실험 전용 HRTT 채움 스크립트.
hailo_8L/scripts/fill_hrtt_columns_batch1_pri0_fps30.py를 이식한 것 — 파일명 규칙만
run_v5det_seg_pose.sh / run_v5det_cpu_seg_pose.sh가 만드는 이름에 맞게 바꿈:

    1D-<us>S-<up>P_0PD-0PS-0PP_<engine>_<combo>_run<n>_<시각>.hrtt

CSV 행 매칭 키는 (use_det=1 고정, use_seg, use_pose, run_id) — engine/combo 라벨은
파일명에서 참고용으로만 쓰고 매칭엔 안 쓴다(NPU/CPU 두 실험은 애초에 CSV/traces
폴더가 분리돼 있으므로 엔진 구분이 필요 없음).

채우는 컬럼:
  - switches_per_s, idle_time_pct           (global, 트레이스 전체 1개 값)
  - avg_fps_det/seg/pose, avg_latency_det/seg/pose,
    max_latency_det/seg/pose, activation_det/seg/pose  (모델별)
  - npu_percent  ← [2026-08-08 신규] hailo_utilization_hailo8.py(백그라운드 폴링)가
    이 실험에서 계속 샘플을 못 얻어 NaN으로 남던 컬럼. 트레이스 자체에 이미 모델별
    device_usage_pct(해당 core_op가 활성 상태였던 시간 비율)가 있으므로, 활성 모델들의
    device_usage_pct 합을 "NPU가 어떤 모델이든 처리 중이었던 비율" 추정치로 대신 채운다.
    외부 폴링 스크립트에 의존하지 않아 더 신뢰할 수 있음(단, "칩 사용률"의 정의가
    hailo_utilization_hailo8.py의 ProtoMon.utilization과 완전히 같지는 않다는 점은
    CSV 컬럼 설명에 남겨둘 것).

[중요 — 08-08 리뷰에서 발견] 이 스크립트로 det+seg+pose 조합 hrtt 3개를 실제로 파싱해보면
avg_latency_det(hrtt, ~26ms)가 같은 행의 det_latency_ms(앱 자체 측정, ~1316ms)와 50배
가까이 차이난다. det_latency_ms는 write() 블로킹 대기시간까지 포함한 "앱이 관측한
enqueue~dequeue 왕복시간"이고, avg_latency_det(hrtt)는 HailoRT 트레이스의 H2D~D2H
타임스탬프 기준 순수 디바이스측 latency라 정의 자체가 다르다 — 어느 쪽이 "틀렸다"가
아니라 서로 다른 걸 재는 두 지표이니, 둘 다 CSV에 남겨서 비교할 수 있게 하는 게 맞다
(det_latency_ms를 이 스크립트가 덮어쓰지 않는 이유).

사용법:
    python3 fill_hrtt_columns_v5det.py <실험폴더> <csv파일이름>
    예) python3 fill_hrtt_columns_v5det.py \
            hailo_8/experiments/2026-08-07_v5det_seg_pose_exp1 results_v5det_seg_pose.csv

    실험폴더 밑에 csv/<csv파일이름> 과 traces/*.hrtt 가 있다고 가정.
    --dry-run 을 마지막 인자로 주면 파일을 덮어쓰지 않고 매칭 결과만 출력.

[2026-08-08 수정 — 실기에서 경로 불일치로 막힘] 원래는 tools/hrtt/parse_hrtt.py의
compute_metrics()를 import해서 썼는데, 그러려면 로컬 git 저장소와 똑같은 중첩 구조
(<repo>/hailo_8/scripts/이 파일, <repo>/tools/hrtt/profiler_pb2.py)가 보드에도 그대로
있어야 했음. 그런데 실제 rpi4 보드는 ~/hailo_cpp_test/ 밑이 평평한 구조(scripts/, 리소스,
.cpp 전부 한 depth)라 경로 계산이 어긋남. 그래서 compute_metrics()/name_to_label()을 이
파일 안에 그대로 복사해 넣어 tools/hrtt에 대한 의존을 없앴고, profiler_pb2.py 하나만
있으면 되게 줄임 — **이 파일과 같은 폴더(scripts/)에 profiler_pb2.py를 같이 복사해두면
어떤 디렉토리 구조에서도 동작함** (그 외에 <이 폴더>/../tools/hrtt, <이 폴더>/../../tools/hrtt
위치도 순서대로 찾아봄 — 로컬 저장소 구조에서 바로 돌릴 때 대비).
"""
import sys, os, re, csv, glob
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    SCRIPT_DIR,                                                          # 이 파일과 같은 폴더
    os.path.join(os.path.dirname(SCRIPT_DIR), "tools", "hrtt"),          # <base>/tools/hrtt
    os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "tools", "hrtt"),  # repo 구조
]
for _c in _CANDIDATES:
    if os.path.isfile(os.path.join(_c, "profiler_pb2.py")):
        sys.path.insert(0, _c)
        break
else:
    print("[오류] profiler_pb2.py를 못 찾음. 다음 위치 중 하나에 복사해두세요:")
    for _c in _CANDIDATES:
        print(f"    {_c}/profiler_pb2.py")
    sys.exit(1)
import profiler_pb2

FNAME_RE = re.compile(r"(\d)D-(\d)S-(\d)P_\d+PD-\d+PS-\d+PP_(npu|cpu)_([a-z_]+)_run(\d+)_")


# ── tools/hrtt/parse_hrtt.py의 compute_metrics()를 그대로 복사(의존성 제거용) ──
def _name_to_label(name):
    n = name.lower()
    if "pose" in n: return "pose"
    if "seg" in n: return "seg"
    return "det"


def compute_metrics(profiler):
    networks = {}
    for t in profiler.added_trace:
        if t.WhichOneof("trace") == "added_core_op":
            networks[t.added_core_op.core_op_handle] = t.added_core_op.core_op_name

    d2h_ts = defaultdict(list)
    act_dur = defaultdict(list)
    deact_dur = defaultdict(list)
    act_ts = defaultdict(list)
    switch_ev = defaultdict(lambda: {"threshold": 0, "timeout": 0, "idle": 0, "total": 0})
    all_ts = []

    for t in profiler.added_trace:
        kind = t.WhichOneof("trace")
        if kind == "frame_dequeue":
            fd = t.frame_dequeue
            all_ts.append(fd.time_stamp)
            if fd.direction == 1:
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
            if ev.over_threshold: sw["threshold"] += 1
            if ev.over_timeout: sw["timeout"] += 1
            if ev.switch_because_idle: sw["idle"] += 1

    if not all_ts:
        return None

    start_ns = min(all_ts)
    end_ns = max(all_ts)
    run_time_s = (end_ns - start_ns) / 1e9

    total_switches = sum(sw["total"] for sw in switch_ev.values())
    switches_per_sec = total_switches / run_time_s if run_time_s > 0 else 0

    all_acts_sorted = sorted([(ts, h) for h, times in act_ts.items() for ts in times])
    handle_active_ns = defaultdict(int)
    for i, (ts, h) in enumerate(all_acts_sorted):
        next_ts = all_acts_sorted[i + 1][0] if i + 1 < len(all_acts_sorted) else end_ns
        handle_active_ns[h] += (next_ts - ts)

    first_act_ts = all_acts_sorted[0][0] if all_acts_sorted else end_ns
    idle_pct = max(0.0, (first_act_ts - start_ns) / (end_ns - start_ns) * 100) if end_ns > start_ns else 0

    model_metrics = {}
    for handle, name in networks.items():
        label = _name_to_label(name)
        streams = defaultdict(list)
        for ts, sname in d2h_ts[handle]:
            streams[sname].append(ts)
        if not streams:
            model_metrics[label] = None
            continue

        rep_stream = sorted(streams.keys())[0]
        rep_ts = sorted(streams[rep_stream])
        frame_count = len(rep_ts)
        avg_fps = frame_count / run_time_s if run_time_s > 0 else 0

        h2d_list = sorted([t.frame_dequeue.time_stamp
                           for t in profiler.added_trace
                           if t.WhichOneof("trace") == "frame_dequeue"
                           and t.frame_dequeue.core_op_handle == handle
                           and t.frame_dequeue.direction == 0])

        pair_count = min(len(h2d_list), len(rep_ts))
        latencies_ms = [(rep_ts[i] - h2d_list[i]) / 1e6
                        for i in range(pair_count) if rep_ts[i] > h2d_list[i]]
        avg_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
        max_lat = max(latencies_ms) if latencies_ms else 0

        active_ns = handle_active_ns.get(handle, 0)
        device_usage_pct = active_ns / (run_time_s * 1e9) * 100 if run_time_s > 0 else 0

        avg_act = sum(act_dur[handle]) / len(act_dur[handle]) if act_dur[handle] else 0
        avg_deact = sum(deact_dur[handle]) / len(deact_dur[handle]) if deact_dur[handle] else 0

        model_metrics[label] = {
            "avg_fps": round(avg_fps, 3),
            "device_usage_pct": round(device_usage_pct, 3),
            "avg_latency_ms": round(avg_lat, 3),
            "max_latency_ms": round(max_lat, 3),
            "activation_ms": round(avg_act, 3),
            "deactivation_ms": round(avg_deact, 3),
        }

    return {
        "global": {
            "hrtt_networks": len(networks),
            "hrtt_switches_per_sec": round(switches_per_sec, 3),
            "hrtt_idle_time_pct": round(idle_pct, 3),
            "hrtt_run_time_sec": round(run_time_s, 3),
        },
        "models": model_metrics,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    exp_dir = sys.argv[1]
    csv_name = sys.argv[2]
    dry_run = (len(sys.argv) > 3 and sys.argv[3] == "--dry-run")

    csv_path = os.path.join(exp_dir, "csv", csv_name)
    traces_dir = os.path.join(exp_dir, "traces")

    if not os.path.isfile(csv_path):
        print(f"[오류] CSV 없음: {csv_path}")
        sys.exit(1)
    if not os.path.isdir(traces_dir):
        print(f"[오류] traces 폴더 없음: {traces_dir}")
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # (use_seg, use_pose, run_id) -> row  (use_det은 이 실험에서 항상 1이라 키에서 뺌)
    keymap = {}
    dup_keys = set()
    for row in rows:
        key = (row["use_seg"], row["use_pose"], row["run_id"])
        if key in keymap:
            dup_keys.add(key)
        keymap[key] = row

    if dup_keys:
        print(f"[경고] CSV 안에서 (use_seg,use_pose,run_id) 중복 키 {len(dup_keys)}개 발견 "
              f"— run_id 채번 버그로 마지막 행만 채워짐: {sorted(dup_keys)}")

    files = sorted(glob.glob(os.path.join(traces_dir, "*.hrtt")))
    matched = unmatched = noevent = skipped = 0
    unmatched_list = []

    for path in files:
        fn = os.path.basename(path)
        m = FNAME_RE.match(fn)
        if not m:
            skipped += 1
            print(f"  [skip] 파일명 규칙 불일치: {fn}")
            continue
        _ud, us, up, _engine, _combo, run = m.groups()
        key = (us, up, run)
        row = keymap.get(key)
        if row is None:
            unmatched += 1
            unmatched_list.append(fn)
            continue

        try:
            p = profiler_pb2.ProtoProfiler()
            p.ParseFromString(open(path, "rb").read())
            metrics = compute_metrics(p)
        except Exception as e:
            print(f"  [!] 파싱 실패: {fn} ({e})")
            noevent += 1
            continue
        if metrics is None:
            noevent += 1
            continue

        g = metrics["global"]
        row["switches_per_s"] = f"{g['hrtt_switches_per_sec']:.4f}"
        row["idle_time_pct"] = f"{g['hrtt_idle_time_pct']:.4f}"

        npu_pct_sum = 0.0
        for lbl in ("det", "seg", "pose"):
            mm = metrics["models"].get(lbl)
            if not mm:
                continue
            row[f"avg_fps_{lbl}"] = f"{mm['avg_fps']:.4f}"
            row[f"avg_latency_{lbl}"] = f"{mm['avg_latency_ms']:.4f}"
            row[f"max_latency_{lbl}"] = f"{mm['max_latency_ms']:.4f}"
            row[f"activation_{lbl}"] = f"{mm['activation_ms']:.4f}"
            npu_pct_sum += mm.get("device_usage_pct", 0.0)

        # npu_percent가 아직 안 채워져 있을 때만(외부 모니터가 이미 성공적으로 채웠으면 안 덮어씀)
        if row.get("npu_percent", "NaN") in ("NaN", "", None):
            row["npu_percent"] = f"{min(npu_pct_sum, 100.0):.4f}"

        matched += 1
        print(f"  [ok] {fn} -> use_seg={us} use_pose={up} run_id={run} "
              f"(det_avg_lat_hrtt={metrics['models'].get('det', {}).get('avg_latency_ms')}ms)")

    if not dry_run:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"\nCSV: {csv_path}{'  (dry-run, 저장 안 함)' if dry_run else ''}")
    print(f"총 hrtt: {len(files)}")
    print(f"  채움(matched): {matched}")
    print(f"  이름규칙 불일치 skip: {skipped}")
    print(f"  CSV행 매칭 실패: {unmatched}")
    print(f"  이벤트 없음/파싱실패: {noevent}")
    if unmatched_list[:5]:
        print("  매칭실패 예시:", unmatched_list[:5])


if __name__ == "__main__":
    main()
