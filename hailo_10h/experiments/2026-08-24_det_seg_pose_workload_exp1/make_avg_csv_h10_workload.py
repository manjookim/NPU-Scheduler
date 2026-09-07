# -*- coding: utf-8 -*-
"""H10 default_workload 스윕 원본 CSV(58행, 3세트가 append로 누적)를 정리한다.

원본은 그대로 두고 파생 파일을 만든다.
  results_det_seg_pose_workload_labeled.csv : 58행 + sweep_set(A/B/C) 컬럼
  results_det_seg_pose_workload_21runs.csv  : ★본 실험 21행(7조건 x 3회) — 세트 C
  results_det_seg_pose_workload_setA_21runs.csv : 재현성 확인용 21행 — 세트 A
  results_det_seg_pose_workload_avg.csv     : 완주 세트(A, C)의 조건별 3회 평균

세트 구분 근거 (2026-08-28 로그 대조):
  A = 1~21행  : logs/sweep_full.log (nohup 실행, 7조건x3 완주)
  B = 22~37행 : 16행에서 중단됨 (seg_pose run1까지, det_seg_pose 없음) -> 분석 제외
  C = 38~58행 : logs/<label>_run<N>.log 21개와 값이 일치 (7조건x3 완주)
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload.csv')
LABELED = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_labeled.csv')
AVG = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_avg.csv')
RUNS21 = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_21runs.csv')
RUNS21_A = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_setA_21runs.csv')

# 본 실험으로 삼는 세트. C는 개별 run 로그(logs/<label>_run<N>.log)와 값이 1:1로 대응돼
# 추적이 가능하므로 기본으로 쓴다. A는 동일 조건 재현성 확인용.
PRIMARY = 'C'

# 1-indexed 데이터 행 범위 (끝 포함)
SETS = [('A', 1, 21), ('B', 22, 37), ('C', 38, 58)]
COMPLETE = ('A', 'C')
CONDITIONS = ['det', 'seg', 'pose', 'det_seg', 'det_pose', 'seg_pose', 'det_seg_pose']
MODELS = ['det', 'seg', 'pose']


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else None


def fmt(v, p=4):
    return '' if v is None else ('%.*f' % (p, v))


def main():
    with open(SRC, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        hdr = rd.fieldnames
        rows = list(rd)

    for name, lo, hi in SETS:
        for r in rows[lo - 1:hi]:
            r['sweep_set'] = name

    with open(LABELED, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sweep_set'] + hdr)
        w.writeheader()
        w.writerows(rows)

    # 조건 순서를 CONDITIONS 순으로 정렬해서 21행 파일을 뽑는다.
    order = {c: i for i, c in enumerate(CONDITIONS)}
    for s, path in ((PRIMARY, RUNS21), ('A', RUNS21_A)):
        picked = sorted((r for r in rows if r.get('sweep_set') == s),
                        key=lambda r: (order[r['label']], int(r['run_id'])))
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=hdr, extrasaction='ignore')
            w.writeheader()
            w.writerows(picked)
        print('%d runs (세트 %s) -> %s' % (len(picked), s, path))

    out_hdr = ['sweep_set', 'label', 'n_runs']
    for m in MODELS:
        out_hdr += ['%s_avg_latency_ms' % m, '%s_fps' % m]
    out_hdr += ['cpu_percent', 'mem_percent', 'total_fps']

    with open(AVG, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(out_hdr)
        for s in COMPLETE:
            for cond in CONDITIONS:
                rs = [r for r in rows
                      if r.get('sweep_set') == s and r['label'] == cond]
                if not rs:
                    continue
                line = [s, cond, len(rs)]
                total_fps = 0.0
                for m in MODELS:
                    lat = mean([num(r['%s_avg_latency_ms' % m]) for r in rs])
                    fps = mean([num(r['%s_fps' % m]) for r in rs])
                    line += [fmt(lat), fmt(fps)]
                    if fps:
                        total_fps += fps
                line += [fmt(mean([num(r['cpu_percent']) for r in rs])),
                         fmt(mean([num(r['mem_percent']) for r in rs])),
                         fmt(total_fps)]
                w.writerow(line)

    print('labeled ->', LABELED)
    print('avg     ->', AVG)


if __name__ == '__main__':
    main()
