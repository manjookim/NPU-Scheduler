#!/usr/bin/env python3
import re
import csv
import os

def parse_npu_log(log_path):
    npu_values = []
    cpu_values = []
    mem_values = []

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(
                r'NPU: ([\d.]+)%, CPU: ([\d.]+)%, MEM: ([\d.]+)%', line)
            if match:
                npu = float(match.group(1))
                cpu = float(match.group(2))
                mem = float(match.group(3))
                if npu > 0:
                    npu_values.append(npu)
                    cpu_values.append(cpu)
                    mem_values.append(mem)

    if not npu_values:
        print(f"경고: NPU 활성 구간을 찾을 수 없습니다.")
        return None

    return {
        'npu_avg': sum(npu_values) / len(npu_values),
        'cpu_avg': sum(cpu_values) / len(cpu_values),
        'mem_avg': sum(mem_values) / len(mem_values),
        'samples': len(npu_values)
    }

def get_params_from_cpp(cpp_path):
    with open(cpp_path, 'r') as f:
        content = f.read()
    batch = int(re.search(r'#define BATCH_SIZE\s+(\d+)', content).group(1))
    threshold = int(re.search(r'#define THRESHOLD\s+(\d+)', content).group(1))
    timeout = int(re.search(r'#define TIMEOUT_MS\s+(\d+)', content).group(1))
    use_det = int(re.search(r'#define USE_DET\s+(\d+)', content).group(1))
    use_seg = int(re.search(r'#define USE_SEG\s+(\d+)', content).group(1))
    use_pose = int(re.search(r'#define USE_POSE\s+(\d+)', content).group(1))
    priority_det = int(re.search(r'#define PRIORITY_DET\s+(\d+)', content).group(1))
    priority_seg = int(re.search(r'#define PRIORITY_SEG\s+(\d+)', content).group(1))
    priority_pose = int(re.search(r'#define PRIORITY_POSE\s+(\d+)', content).group(1))
    return batch, threshold, timeout, use_det, use_seg, use_pose, priority_det, priority_seg, priority_pose

def update_csv(csv_path, log_path, batch, threshold, timeout,
               use_det, use_seg, use_pose,
               priority_det, priority_seg, priority_pose):
    stats = parse_npu_log(log_path)
    if not stats:
        return

    print(f"\n추출된 NPU 로그 통계:")
    print(f"  use: Det={use_det}, Seg={use_seg}, Pose={use_pose}")
    print(f"  priority: Det={priority_det}, Seg={priority_seg}, Pose={priority_pose}")
    print(f"  NPU 평균: {stats['npu_avg']:.2f}%")
    print(f"  CPU 평균: {stats['cpu_avg']:.2f}%")
    print(f"  MEM 평균: {stats['mem_avg']:.2f}%")
    print(f"  샘플 수: {stats['samples']}개")

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if 'npu_percent' not in fieldnames:
        fieldnames = list(fieldnames) + ['npu_percent']

    p_det = str(priority_det) if use_det else 'None'
    p_seg = str(priority_seg) if use_seg else 'None'
    p_pose = str(priority_pose) if use_pose else 'None'

    updated = False
    for row in rows:
        if (row['use_det'] == str(use_det) and
            row['use_seg'] == str(use_seg) and
            row['use_pose'] == str(use_pose) and
            row['priority_det'] == p_det and
            row['priority_seg'] == p_seg and
            row['priority_pose'] == p_pose and
            row.get('npu_percent', '').strip() == ''):
            row['npu_percent'] = f"{stats['npu_avg']:.4f}"
            updated = True
            print(f"CSV 업데이트 완료!")
            break

    if not updated:
        print(f"경고: 해당 파라미터 행을 찾을 수 없습니다.")
        return

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV 저장 완료: {csv_path}")

if __name__ == "__main__":
    import sys
    # argv[1]: csv_path (optional, 기본값 사용)
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/rpi1/hailo_cpp_test/results_all.csv"
    LOG_PATH = "/home/rpi1/hailo_cpp_test/npu_log.txt"
    CPP_PATH = "/home/rpi1/hailo_cpp_test/infer_scheduler.cpp"

    batch, threshold, timeout, use_det, use_seg, use_pose, \
        priority_det, priority_seg, priority_pose = get_params_from_cpp(CPP_PATH)

    print(f"파라미터: batch={batch}, threshold={threshold}, timeout={timeout}")
    print(f"use: Det={use_det}, Seg={use_seg}, Pose={use_pose}")
    print(f"priority: Det={priority_det}, Seg={priority_seg}, Pose={priority_pose}")

    update_csv(CSV_PATH, LOG_PATH, batch, threshold, timeout,
               use_det, use_seg, use_pose,
               priority_det, priority_seg, priority_pose)