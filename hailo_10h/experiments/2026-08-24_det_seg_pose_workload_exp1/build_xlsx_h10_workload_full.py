#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_xlsx_h10_workload_full.py — (Hailo-10H, npu-rpi5) det/seg/pose 워크로드 조합 실험
결과(전체 컬럼판)를 컬럼설명/전체_21회/조건별_3회평균/그래프 4개 시트짜리 xlsx로 합친다.

기존 build_xlsx_h10_workload.py와의 차이:
  - 예전 스키마(det_avg_latency_ms / det_fps)가 아니라 hailo_8L/csv_writer.hpp와 같은
    구성의 새 스키마(det_latency_ms / avg_fps_det / max_latency_det / npu_percent ...)를 읽는다.
  - 평균을 여기서 다시 계산하지 않고 make_avg_csv_h10_workload_full.py가 만든 avg CSV를
    그대로 읽는다(평균 정의가 두 군데로 갈라지지 않게).
  - npu_percent / max_latency / context switch 차트가 추가됐다.

사용법: python build_xlsx_h10_workload_full.py [full.csv] [full_avg.csv] [출력.xlsx]
        (인자 없으면 csv/ 안의 기본 경로를 쓴다)
"""
import csv
import os
import sys

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_RAW = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_full.csv')
DEF_AVG = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_full_avg.csv')
DEF_OUT = os.path.join(HERE, 'csv', 'results_det_seg_pose_workload_full_combined.xlsx')

CONDITIONS = ['det', 'seg', 'pose', 'det_seg', 'det_pose', 'seg_pose', 'det_seg_pose']
MODELS = ['det', 'seg', 'pose']
# 모델 정체성 = 색상 (hailo_8L 실험 시리즈 카테고리 팔레트와 동일 계열)
MODEL_COLOR = {'det': '2A78D6', 'seg': 'EB6834', 'pose': '2CA02C'}
NPU_COLOR = 'D62728'
CPU_COLOR = '7F5FC4'

HDR_FILL = PatternFill('solid', fgColor='1F4E79')
HDR_FONT = Font(color='FFFFFF', bold=True)
SEC_FONT = Font(bold=True, size=12, color='1F4E79')

COLUMN_DESC = [
    ("실험 조건",
     "Hailo-10H(npu-rpi5, RPi5, hailort 5.3.0, PCIe Gen3 x1). det=yolov8s / seg=yolov8s_seg / "
     "pose=yolov8s_pose 3개 모델의 조합 7가지(단독 3 + 쌍 3 + 트리플 1)를 각 3회씩 "
     "= 21회 실행. **스케줄러 파라미터(priority/threshold/timeout/batch)는 전혀 설정하지 "
     "않고 HailoRT 기본값 그대로 둔 채 워크로드 조합만 바꿨다.** 데이터셋은 8L 실험과 "
     "동일한 sampled_val2017 673장. 모델별로 독립 스레드가 하나씩 붙어 "
     "'전처리 → run_async → wait'를 673회 반복한다."),
    ("컬럼 구성",
     "hailo_8L/csv_writer.hpp와 같은 구성으로 맞췄다. 8L에 있던 컬럼 중 **HRTT에서만 "
     "얻을 수 있는 3종(switches_per_s / idle_time_pct / activation_*)만 빠져 있다** — "
     "H10은 추론 스택이 디바이스 내부 hailort_server에 있어 호스트 .hrtt가 항상 0바이트다"
     "(HRTT_ON_HAILO10H.md 참고). 8L에서 HRTT가 채워주던 avg_fps_* / max_latency_*는 "
     "호스트 실측으로 대체했다."),
    ("데이터 출처 (2026-08-31)",
     "src/run_workload_sweep.sh 로 7조건 x 3회. 18번째 런(seg_pose run3, 10:01) 직후 "
     "npu-rpi5가 리부팅돼 스윕이 끊겼고(uptime/-tmp 초기화로 확인, persistent journal이 "
     "없어 원인은 확인 불가), 마지막 조건 det_seg_pose 3회만 같은 바이너리로 이어서 "
     "실행해 21행을 채웠다. CSV 기록 방식은 나머지 18런과 동일하다. 다만 "
     "logs_full/det_seg_pose_run*.log 3개는 끝부분(tail -6)만 담겨 있다. "
     "프레임 단위 타임라인은 traces/ 에 42개(events.csv + trace.json) 있다."),
    ("label", "워크로드 조합. det / seg / pose / det_seg / det_pose / seg_pose / det_seg_pose"),
    ("run_id", "같은 조건의 반복 순번(1~3)"),
    ("use_det / use_seg / use_pose", "해당 실행에서 그 모델이 활성인지 여부(1=활성, 0=비활성)"),
    ("batch / threshold_* / timeout_ms / priority_*",
     "적용된 스케줄러 파라미터. 이 실험은 옵션을 주지 않았으므로 HailoRT 기본값이 그대로 "
     "적혀 있다(batch=0=자동, threshold=1, timeout=0ms, priority=16=NORMAL). 벤치는 옵션을 "
     "준 경우에만 setter를 호출한다 — H10에서도 RPC로 디바이스 스케줄러에 전달돼 실제로 먹는다."),
    ("<model>_frame_count", "모델별로 정상 처리된 프레임 수(전 조건 673 = 데이터셋 전량)"),
    ("<model>_latency_ms",
     "장당 평균 추론 시간(ms) — run_async() 제출 ~ wait() 완료 왕복. **전처리는 빠져 "
     "있다.** 비활성 모델은 NaN."),
    ("max_latency_<model>",
     "그 모델의 프레임별 latency 중 최댓값(ms). 8L에서는 HRTT가 주던 값이고 H10에서는 "
     "호스트 실측이다. 평균값만 보면 안 보이는 스케줄링 지연의 꼬리를 보여준다."),
    ("avg_fps_<model>",
     "모델별 처리량 = 프레임 수 ÷ 그 모델 워커의 전체 루프 시간. **전처리가 포함된다.** "
     "그래서 fps != 1000/latency_ms 이다(전처리가 latency에는 빠지고 FPS에는 들어감)."),
    ("total_fps (평균 시트/그래프에만)",
     "활성 모델들의 fps 합 = 그 조건에서 디바이스가 낸 전체 처리량. 조합 간 성능을 비교할 "
     "때 가장 안전한 지표다."),
    ("npu_percent",
     "**디바이스(NPU) NNC 가동률(%).** 8L은 HAILO_MONITOR=1이 떨구는 /tmp/hmon_files를 "
     "읽었지만 H10에서는 그 경로가 아예 생기지 않는다(스케줄러가 디바이스 안에 있어서). "
     "그래서 벤치가 hailortcli monitor를 자식 프로세스로 띄워 약 1Hz로 받아 파싱한다. "
     "평균 정의는 8L parse_npu_log.py와 같게 **NNC>0인 샘플만** 평균낸다. "
     "디바이스 전체 합계라 모델별 분해는 안 된다."),
    ("cpu_percent / mem_percent",
     "실행 중 **호스트(RPi5) 시스템 전체** 사용률(%). /proc/stat, /proc/meminfo를 300ms "
     "간격으로 샘플링해 평균낸 값. npu_percent와 달리 이건 호스트 쪽 값이다."),
    ("voluntary / nonvoluntary_ctx_switches",
     "각 모델 워커 스레드의 컨텍스트 스위치 증가분 합(/proc/thread-self/status). "
     "nonvoluntary 쪽이 CPU 경합의 지표다 — 1모델 대비 3모델에서 크게 뛴다."),
    ("run_time_s", "워커 스레드 전체 구간의 wall-time(초). 모델 로딩·configure는 제외."),
    ("total_time_<model>_s", "그 모델 워커 루프 하나의 총 시간(초)."),
    ("avg_preprocess_ms_<model>",
     "장당 평균 전처리 시간(ms) = imread + cvtColor + resize + memcpy."),
    ("postprocess_ms_<model>",
     "**전 행 NaN.** 이 벤치는 bbox/mask/keypoint 디코딩과 NMS를 하지 않고 raw output "
     "버퍼만 받아 타이밍을 잰다(8L의 ppoff/noppt 조건과 같은 상태)."),
    ("total_time_ms_<model> / total_time_ms_nopp_<model>",
     "장당 전체시간 = 전처리 + latency + 후처리(측정한 경우만). 이 실험은 후처리를 재지 "
     "않으므로 두 컬럼이 같은 값이다. 8L csv_writer.hpp와 같은 규약."),
    ("n_runs (평균 시트에만)", "그룹에 포함된 반복 횟수(=3)"),
    ("이 실험에서 얻을 수 없는 것",
     "컨텍스트 스위치 횟수/사유, 모델별 activation 시간, 동시 실행 시 모델별 디바이스 "
     "점유율 — 전부 디바이스 내부 스케줄러 정보라 호스트에서 관측 불가. 전력도 보드에 "
     "INA 센서가 없어 측정 불가. 근거와 전수 조사 결과는 HRTT_ON_HAILO10H.md 참고."),
]


def read_dicts(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def num(v):
    """'NaN'/빈칸/비활성 모델 칸은 None으로. 8L csv_writer.hpp와 같은 규약."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def style_header(ws, row=1):
    for c in ws[row]:
        if c.value is not None:
            c.fill = HDR_FILL
            c.font = HDR_FONT
            c.alignment = Alignment(horizontal='center', vertical='center')


def autosize(ws, min_w=10, max_w=26):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        longest = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[letter].width = min(max(longest + 2, min_w), max_w)


def sheet_desc(wb):
    ws = wb.create_sheet('컬럼설명')
    ws.append(['항목', '설명'])
    style_header(ws)
    for k, v in COLUMN_DESC:
        ws.append([k, v])
    ws.column_dimensions['A'].width = 34
    ws.column_dimensions['B'].width = 110
    for r in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        r[0].alignment = Alignment(wrap_text=True, vertical='top')
    for r in ws.iter_rows(min_row=2, min_col=1, max_col=1):
        r[0].alignment = Alignment(vertical='top')
        r[0].font = Font(bold=True)
    ws.freeze_panes = 'A2'
    return ws


def sheet_table(wb, title, rows, freeze):
    ws = wb.create_sheet(title)
    hdr = list(rows[0].keys())
    ws.append(hdr)
    style_header(ws)
    for r in rows:
        # 숫자는 숫자로, 'NaN'은 빈칸으로 넣어야 엑셀 차트/집계가 제대로 먹는다.
        ws.append([num(r[h]) if num(r[h]) is not None else (None if r[h] == 'NaN' else r[h])
                   for h in hdr])
    ws.freeze_panes = freeze
    autosize(ws)
    return ws


def write_block(ws, top, title, header, rows):
    """그래프 시트에 차트용 데이터 블록을 쓰고 (헤더행, 마지막행)을 돌려준다."""
    ws.cell(row=top, column=1, value=title).font = SEC_FONT
    hdr_row = top + 1
    for j, h in enumerate(header, start=1):
        c = ws.cell(row=hdr_row, column=j, value=h)
        c.fill = HDR_FILL
        c.font = HDR_FONT
        c.alignment = Alignment(horizontal='center')
    for i, row in enumerate(rows, start=hdr_row + 1):
        for j, v in enumerate(row, start=1):
            ws.cell(row=i, column=j, value=v)
    return hdr_row, hdr_row + len(rows)


def color_series(chart, colors):
    for s, hexc in zip(chart.series, colors):
        s.graphicalProperties.solidFill = hexc
        s.graphicalProperties.line.noFill = True


def bar(ws, title, ytitle, hdr_row, last_row, n_series, stacked=False):
    ch = BarChart()
    ch.type = 'col'
    ch.grouping = 'stacked' if stacked else 'clustered'
    if stacked:
        ch.overlap = 100
    ch.title = title
    ch.y_axis.title = ytitle
    ch.x_axis.title = '워크로드 조합'
    ch.height, ch.width = 9, 20
    ch.add_data(Reference(ws, min_col=2, max_col=1 + n_series,
                          min_row=hdr_row, max_row=last_row), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, max_col=1,
                                min_row=hdr_row + 1, max_row=last_row))
    return ch


def sheet_chart(wb, avg):
    ws = wb.create_sheet('그래프')
    g = lambda r, k: num(r.get(k))  # noqa: E731

    # --- 차트 A: 조건별 모델별 평균 Latency ---
    rows_a = [[r['label']] + [g(r, '%s_latency_ms' % m) for m in MODELS] for r in avg]
    hdr_a, last_a = write_block(ws, 1, '차트 A 데이터 — 조건별 모델별 평균 Latency (ms)',
                                ['조건'] + MODELS, rows_a)
    ch_a = bar(ws, 'Hailo-10H 조건별 평균 Latency (ms, 3회 평균)', 'latency_ms',
               hdr_a, last_a, len(MODELS))
    color_series(ch_a, [MODEL_COLOR[m] for m in MODELS])
    ws.add_chart(ch_a, 'H2')

    # --- 차트 B: 조건별 처리량 FPS (누적 = 쌓인 높이가 합계) ---
    top_b = last_a + 2
    rows_b = [[r['label']] + [g(r, 'avg_fps_%s' % m) for m in MODELS] + [g(r, 'total_fps')]
              for r in avg]
    hdr_b, last_b = write_block(ws, top_b, '차트 B 데이터 — 조건별 처리량 FPS (합계 = 누적 높이)',
                                ['조건'] + MODELS + ['합계FPS(참고)'], rows_b)
    ch_b = bar(ws, 'Hailo-10H 조건별 처리량 (FPS, 모델별 누적 = 합계)', 'FPS',
               hdr_b, last_b, len(MODELS), stacked=True)
    color_series(ch_b, [MODEL_COLOR[m] for m in MODELS])
    ch_b.dLbls = DataLabelList()
    ch_b.dLbls.showVal = True
    ws.add_chart(ch_b, 'H21')

    # --- 차트 C: NPU 가동률 vs 호스트 CPU (이번 컬럼 추가의 핵심) ---
    top_c = last_b + 2
    rows_c = [[r['label'], g(r, 'npu_percent'), g(r, 'cpu_percent')] for r in avg]
    hdr_c, last_c = write_block(ws, top_c,
                                '차트 C 데이터 — NPU(NNC) 가동률 vs 호스트 CPU 사용률 (%)',
                                ['조건', 'npu_percent', 'cpu_percent'], rows_c)
    ch_c = bar(ws, 'NPU(NNC) 가동률 vs 호스트 CPU 사용률 (%, 3회 평균)', '%',
               hdr_c, last_c, 2)
    color_series(ch_c, [NPU_COLOR, CPU_COLOR])
    ws.add_chart(ch_c, 'H40')

    # --- 차트 D: 평균 vs 최대 Latency (스케줄링 지연 꼬리) ---
    top_d = last_c + 2
    rows_d = [[r['label']] + [g(r, 'max_latency_%s' % m) for m in MODELS] for r in avg]
    hdr_d, last_d = write_block(ws, top_d, '차트 D 데이터 — 조건별 모델별 최대 Latency (ms)',
                                ['조건'] + MODELS, rows_d)
    ch_d = bar(ws, 'Hailo-10H 조건별 최대 Latency (ms, 3회 평균) — 지연 꼬리', 'max_latency_ms',
               hdr_d, last_d, len(MODELS))
    color_series(ch_d, [MODEL_COLOR[m] for m in MODELS])
    ws.add_chart(ch_d, 'H59')

    ws.column_dimensions['A'].width = 18
    for j in range(2, 7):
        ws.column_dimensions[get_column_letter(j)].width = 14
    return ws


def main():
    raw_p = sys.argv[1] if len(sys.argv) > 1 else DEF_RAW
    avg_p = sys.argv[2] if len(sys.argv) > 2 else DEF_AVG
    out_p = sys.argv[3] if len(sys.argv) > 3 else DEF_OUT

    raw = read_dicts(raw_p)
    avg = read_dicts(avg_p)
    if len(raw) != 21:
        print('경고: 원본이 21행이 아니다 (%d행) — %s' % (len(raw), raw_p))
    missing = [c for c in CONDITIONS if c not in {r['label'] for r in avg}]
    if missing:
        print('경고: 평균에 빠진 조건 — %s' % ', '.join(missing))
    avg = sorted(avg, key=lambda r: CONDITIONS.index(r['label']))

    wb = Workbook()
    wb.remove(wb.active)
    sheet_desc(wb)
    sheet_table(wb, '전체_21회', raw, freeze='C2')
    sheet_table(wb, '조건별_3회평균', avg, freeze='B2')
    sheet_chart(wb, avg)
    wb.save(out_p)

    print('시트: %s' % ', '.join(wb.sheetnames))
    print('xlsx -> %s' % out_p)


if __name__ == '__main__':
    main()
