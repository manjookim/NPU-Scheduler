# -*- coding: utf-8 -*-
"""H10 워크로드 스윕(전체 컬럼판) 원본 CSV를 조건별 평균으로 정리한다.

입력  csv/results_det_seg_pose_workload_full.csv   (7조건 x 3회 = 21행)
출력  csv/results_det_seg_pose_workload_full_avg.csv

컬럼 구성은 hailo_8L/csv_writer.hpp와 같다(HRTT 전용 컬럼 제외). 자세한 정의는
HRTT_ON_HAILO10H.md 5-1절 참고.

데이터 출처 (2026-08-31):
  src/run_workload_sweep.sh 로 7조건 x 3회. 18번째 런(seg_pose run3, 10:01) 직후 npu-rpi5가
  리부팅돼(uptime/-/tmp 초기화로 확인, 원인 로그는 persistent journal이 없어 확인 불가) 스윕이
  끊겼고, 마지막 조건 det_seg_pose 3회만 같은 바이너리로 이어서 실행해 21행을 채웠다.
  그래서 logs_full/det_seg_pose_run*.log 3개만 끝부분(tail -6)만 담겨 있다 — CSV 값 자체는
  나머지 18런과 동일한 방식으로 기록된 것이라 차이 없음.

기존 make_avg_csv_h10_workload.py와의 차이:
  - 그쪽은 예전 스키마(det_avg_latency_ms / det_fps)와 58행(A/B/C 3세트가 append로 누적된
    파일)을 전제로 하고, 세트 구분 행 범위가 하드코딩돼 있다. 이 스크립트는 새 스키마의
    깨끗한 21행 파일 하나만 다룬다.
  - 평균을 낼 컬럼을 나열하지 않고, 원본 컬럼 순서를 그대로 유지한 채 수치 컬럼만 평균낸다.
    실행 구성 컬럼(batch/threshold/timeout/priority/use_*)은 런마다 같은 값이므로 평균이
    아니라 첫 런의 값을 그대로 옮기고, 세 런의 값이 다르면 경고를 띄운다.
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_full.csv')
AVG = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_full_avg.csv')

CONDITIONS = ['det', 'seg', 'pose', 'det_seg', 'det_pose', 'seg_pose', 'det_seg_pose']
MODELS = ['det', 'seg', 'pose']

# 런마다 값이 같아야 하는(= 평균이 의미 없는) 실행 구성 컬럼.
CONFIG_COLS = {
    'use_det', 'use_seg', 'use_pose', 'batch',
    'threshold_det', 'threshold_seg', 'threshold_pose', 'timeout_ms',
    'priority_det', 'priority_seg', 'priority_pose',
}
# 평균 대상에서 빼는 컬럼.
SKIP_COLS = {'label', 'run_id'}


def num(v):
    """'NaN'/빈칸/비활성 모델 칸은 None으로. 8L csv_writer.hpp와 같은 규약."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN 제외


def mean(vals):
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else None


def fmt(v, p=4):
    return 'NaN' if v is None else ('%.*f' % (p, v))


def main():
    with open(SRC, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        hdr = rd.fieldnames
        rows = list(rd)

    print('%s: %d행' % (os.path.basename(SRC), len(rows)))

    # 평균낼 컬럼 = 원본 순서 그대로, 설정/식별 컬럼 제외
    value_cols = [c for c in hdr if c not in CONFIG_COLS and c not in SKIP_COLS]
    config_cols = [c for c in hdr if c in CONFIG_COLS]

    out_hdr = ['label', 'n_runs'] + config_cols + value_cols + ['total_fps']

    with open(AVG, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(out_hdr)
        for cond in CONDITIONS:
            rs = [r for r in rows if r['label'] == cond]
            if not rs:
                print('  [경고] 조건 없음: %s' % cond)
                continue

            line = [cond, len(rs)]
            for c in config_cols:
                vals = {r[c] for r in rs}
                if len(vals) > 1:
                    print('  [경고] %s 조건에서 %s 값이 런마다 다름: %s' % (cond, c, sorted(vals)))
                line.append(rs[0][c])
            for c in value_cols:
                line.append(fmt(mean([num(r[c]) for r in rs])))

            # 조건 전체 처리량 = 활성 모델 FPS 합
            total_fps = 0.0
            for m in MODELS:
                v = mean([num(r['avg_fps_%s' % m]) for r in rs])
                if v:
                    total_fps += v
            line.append(fmt(total_fps))
            w.writerow(line)
            print('  %-13s n=%d' % (cond, len(rs)))

    print('avg ->', AVG)


if __name__ == '__main__':
    main()
