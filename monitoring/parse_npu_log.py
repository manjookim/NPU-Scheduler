#!/usr/bin/env python3
import re
import csv
import sys
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
                # 추론 중인 구간만 (NPU > 0)
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

def update_csv(csv_path, log_path, batch, threshold):
    """CSV 파일의 해당 실험 행에 NPU 평균값 추가"""
    stats = parse_npu_log(log_path)
    if not stats:
        return

    print(f"\n추출된 NPU 로그 통계 (batch={batch}, threshold={threshold}):")
    print(f"  NPU 평균: {stats['npu_avg']:.2f}%")
    print(f"  CPU 평균: {stats['cpu_avg']:.2f}%")
    print(f"  MEM 평균: {stats['mem_avg']:.2f}%")
    print(f"  샘플 수: {stats['samples']}개")

    # CSV 읽기
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # 헤더에 npu_percent 추가
    if 'npu_percent' not in fieldnames:
        fieldnames = list(fieldnames) + ['npu_percent']

    # 해당 행 업데이트
    updated = False
    for row in rows:
        if int(row['batch']) == batch and int(row['threshold']) == threshold:
            row['npu_percent'] = f"{stats['npu_avg']:.4f}"
            updated = True
            print(f"CSV 업데이트 완료: batch={batch}, threshold={threshold}")

    if not updated:
        print(f"경고: batch={batch}, threshold={threshold} 행을 찾을 수 없습니다.")
        return

    # CSV 저장
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV 저장 완료: {csv_path}")

if __name__ == "__main__":
    CSV_PATH = "/home/rpi1/hailo_cpp_test/results.csv"
    LOG_PATH = "/home/rpi1/hailo_cpp_test/npu_log.txt"
    BATCH = 1
    THRESHOLD = 0

    update_csv(CSV_PATH, LOG_PATH, BATCH, THRESHOLD)