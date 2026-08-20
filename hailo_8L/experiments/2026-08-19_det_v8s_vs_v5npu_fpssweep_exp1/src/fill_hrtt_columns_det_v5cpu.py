#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_hrtt_columns_det_v5cpu.py
results_det_v5cpu_fpssweep.csv 의 HRTT 전용 NaN 컬럼(switches_per_s,
idle_time_pct, avg_fps_hrtt, avg_latency_hrtt, max_latency_hrtt, activation_hrtt)을
같은 실험의 .hrtt 파일들을 직접 파싱해서 채운다. HTML 변환 불필요 (protobuf 원본을
바로 읽음 — tools/hrtt/parse_hrtt.py::compute_metrics 재사용).

fill_hrtt_columns_det_v8s_vs_v5npu.py를 v5-CPU 단일 조건 실험용으로 축소한 버전
(태그가 v5cpu 하나뿐이라 TAG_TO_HEF_NAME 매핑도 1개).

파일명 규칙: det_v5cpu_fps{0|30}_run{1..3}_{timestamp}.hrtt
CSV 매칭 키: (hef_name, input_fps) 그룹 내에서 run_id 오름차순 = hrtt run{n} 오름차순
             (run_det_v5cpu_fpssweep.sh가 그 순서로 만들었으므로 순서 매칭 가능)

사용법: python fill_hrtt_columns_det_v5cpu.py <csv_path> <hrtt_dir> [출력_csv_path]
        출력_csv_path 생략 시 입력 csv를 그대로 덮어씀.
"""
import sys, os, re, csv, glob

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NPUscheduler
sys.path.insert(0, os.path.join(BASE, "tools", "hrtt"))
import profiler_pb2
from parse_hrtt import compute_metrics

TAG_TO_HEF_NAME = {
    "v5cpu": "YOLOv5-CPU-baseline",
}

FNAME_RE = re.compile(r"det_(v5cpu)_fps(\d+)_run(\d+)_")


def main():
    if len(sys.argv) < 3:
        print(f"사용법: {sys.argv[0]} <csv_path> <hrtt_dir> [출력_csv_path]")
        sys.exit(1)
    csv_path = sys.argv[1]
    hrtt_dir = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else csv_path

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # (hef_name, input_fps) -> [row indices] 오름차순(run_id 순, 파일 안에 이미 순서대로 들어있음)
    group_rows = {}
    for i, row in enumerate(rows):
        key = (row["hef_name"], row["input_fps"])
        group_rows.setdefault(key, []).append(i)

    # (hef_name, input_fps) -> [(run_local, hrtt_path)] 오름차순
    group_files = {}
    for path in glob.glob(os.path.join(hrtt_dir, "*.hrtt")):
        fn = os.path.basename(path)
        m = FNAME_RE.match(fn)
        if not m:
            print(f"  [건너뜀] 파일명 패턴 불일치: {fn}")
            continue
        tag, fps, run_local = m.group(1), m.group(2), int(m.group(3))
        hef_name = TAG_TO_HEF_NAME[tag]
        key = (hef_name, fps)
        group_files.setdefault(key, []).append((run_local, path))

    filled = 0
    errors = []

    for key, idx_list in group_rows.items():
        files = sorted(group_files.get(key, []))
        if len(files) != len(idx_list):
            errors.append(f"개수 불일치: {key} → CSV행 {len(idx_list)}개, hrtt파일 {len(files)}개")
        for (run_local, path), row_idx in zip(files, idx_list):
            row = rows[row_idx]
            print(f"  처리중: run_id={row['run_id']} ({key[0]}, fps={key[1]}) <- {os.path.basename(path)}", end=" ")
            try:
                p = profiler_pb2.ProtoProfiler()
                p.ParseFromString(open(path, "rb").read())
                metrics = compute_metrics(p)
            except Exception as e:
                print(f"→ 파싱 실패 ({e})")
                errors.append(f"파싱 실패: {path} ({e})")
                continue
            if metrics is None:
                print("→ 이벤트 없음")
                errors.append(f"이벤트 없음: {path}")
                continue

            g = metrics["global"]
            mm = metrics["models"].get("det")
            row["switches_per_s"] = g["hrtt_switches_per_sec"]
            row["idle_time_pct"]  = g["hrtt_idle_time_pct"]
            if mm:
                row["avg_fps_hrtt"]     = mm["avg_fps"]
                row["avg_latency_hrtt"] = mm["avg_latency_ms"]
                row["max_latency_hrtt"] = mm["max_latency_ms"]
                row["activation_hrtt"]  = mm["activation_ms"]
                print(f"→ OK (avg_fps_hrtt={mm['avg_fps']}, avg_latency_hrtt={mm['avg_latency_ms']}ms)")
            else:
                print("→ OK (global만, det 모델 지표 없음)")
            filled += 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n완료: {filled}/{len(rows)}행 채움 -> {out_path}")
    if errors:
        print(f"오류 {len(errors)}건:")
        for e in errors:
            print(f"  ! {e}")


if __name__ == "__main__":
    main()
