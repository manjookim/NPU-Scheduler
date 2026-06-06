#!/usr/bin/env python3
import re
import csv
import os

def parse_npu_log(log_path):
    """NPU 로그에서 추론 중인 구간(NPU != 0.00%)만 필터링해서 평균값 추출"""
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
        print(f"경고: {log_path}에서 NPU 활성 구간을 찾을 수 없습니다.")
        return None

    return {
        'npu_avg': sum(npu_values) / len(npu_values),
        'cpu_avg': sum(cpu_values) / len(cpu_values),
        'mem_avg': sum(mem_values) / len(mem_values),
        'samples': len(npu_values)
    }

def get_params_from_cpp(cpp_path):
    """infer_scheduler.cpp에서 파라미터 값 자동으로 읽기"""
    with open(cpp_path, 'r') as f:
        content = f.read()
    batch = int(re.search(r'#define BATCH_SIZE\s+(\d+)', content).group(1))
    threshold = int(re.search(r'#define THRESHOLD\s+(\d+)', content).group(1))
    timeout = int(re.search(r'#define TIMEOUT_MS\s+(\d+)', content).group(1))
    priority = int(re.search(r'#define PRIORITY\s+(\d+)', content).group(1))
    return batch, threshold, timeout, priority

def update_csv(csv_path, log_path, batch, threshold, timeout, priority):
    """CSV 파일의 해당 실험 행에 NPU 평균값 추가"""
    stats = parse_npu_log(log_path)
    if not stats:
        return

    print(f"\n추출된 NPU 로그 통계 (batch={batch}, threshold={threshold}, timeout={timeout}, priority={priority}):")
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

    updated = False
    for row in rows:
        if (int(row['batch']) == batch and
            int(row['threshold']) == threshold and
            int(row['timeout_ms']) == timeout and
            int(row['priority']) == priority):
            row['npu_percent'] = f"{stats['npu_avg']:.4f}"
            updated = True
            print(f"CSV 업데이트 완료: batch={batch}, threshold={threshold}, timeout={timeout}, priority={priority}")

    if not updated:
        print(f"경고: 해당 파라미터 행을 찾을 수 없습니다.")
        return

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV 저장 완료: {csv_path}")

if __name__ == "__main__":
    CSV_PATH = "/home/rpi1/hailo_cpp_test/results.csv"
    LOG_PATH = "/home/rpi1/hailo_cpp_test/npu_log.txt"
    CPP_PATH = "/home/rpi1/hailo_cpp_test/infer_scheduler.cpp"

    batch, threshold, timeout, priority = get_params_from_cpp(CPP_PATH)
    print(f"파라미터 자동 읽기: batch={batch}, threshold={threshold}, timeout={timeout}, priority={priority}")

    update_csv(CSV_PATH, LOG_PATH, batch, threshold, timeout, priority)
