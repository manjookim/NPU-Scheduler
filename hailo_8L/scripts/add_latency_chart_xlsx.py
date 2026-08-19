#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_latency_chart_xlsx.py
results_det_v8s_vs_v5npu_fpssweep_combined.xlsx에 "그래프" 시트를 추가하고, 엑셀
네이티브 막대그래프 2개를 삽입한다(조교님께 전달할 때 엑셀에서 바로 보이고, 값도
클릭하면 수정 가능한 진짜 차트 객체로 남기기 위함 — 이미지 붙여넣기 아님).

  차트 A: 조건별(hef_name x input_fps) 3회 평균 latency 비교
  차트 B: 조건별 개별 실행(run 1~3) latency 비교 — "각 케이스 테스트별" 요청 반영

색상은 hailo_8L 실험 시리즈에서 쓰는 카테고리 팔레트와 동일하게 고정:
  v5-NPU 후처리 = blue(#2a78d6), v8s-CPU 후처리 = orange(#eb6834)
  (FPS=30 조건은 같은 계열의 밝은 톤으로 구분 — 모델 정체성은 색상, FPS는 명도로 구분)

사용법: python add_latency_chart_xlsx.py <xlsx_path>  (같은 파일에 시트 추가 후 저장)
"""
import sys
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList

BLUE_DARK   = "2A78D6"   # v5-NPU, FPS 0
BLUE_LIGHT  = "86B6EF"   # v5-NPU, FPS 30
ORANGE_DARK = "EB6834"   # v8s-CPU, FPS 0
ORANGE_LIGHT= "F3A679"   # v8s-CPU, FPS 30

MEAN_ROWS = [
    ("FPS 0",  491.005, 419.770),
    ("FPS 30", 486.081, 392.400),
]

RUN_ROWS = [
    ("Run 1", 490.959, 419.802, 486.072, 392.229),
    ("Run 2", 491.165, 419.639, 486.037, 392.305),
    ("Run 3", 490.891, 419.869, 486.134, 392.666),
]


def style_series(series, hex_color):
    series.graphicalProperties.solidFill = hex_color
    series.graphicalProperties.line.noFill = True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]

    wb = load_workbook(path)
    if "그래프" in wb.sheetnames:
        del wb["그래프"]
    ws = wb.create_sheet("그래프")

    # ---------- 표 A: 조건별 평균 ----------
    ws["A1"] = "조건별 3회 평균 avg_latency_ms (ms)"
    ws["A2"] = ""
    ws["B2"] = "YOLOv5xs-nms_core (NPU후처리)"
    ws["C2"] = "YOLOv8s_h8l (CPU후처리)"
    for i, (label, v5, v8) in enumerate(MEAN_ROWS, start=3):
        ws.cell(row=i, column=1, value=label)
        ws.cell(row=i, column=2, value=v5)
        ws.cell(row=i, column=3, value=v8)

    chart_a = BarChart()
    chart_a.type = "col"
    chart_a.grouping = "clustered"
    chart_a.title = "조건별 평균 Latency (ms)"
    chart_a.y_axis.title = "avg_latency_ms"
    chart_a.x_axis.title = "INPUT_FPS"
    chart_a.style = 2
    chart_a.height = 9
    chart_a.width = 16

    data_a = Reference(ws, min_col=2, max_col=3, min_row=2, max_row=2 + len(MEAN_ROWS))
    cats_a = Reference(ws, min_col=1, max_col=1, min_row=3, max_row=2 + len(MEAN_ROWS))
    chart_a.add_data(data_a, titles_from_data=True)
    chart_a.set_categories(cats_a)
    style_series(chart_a.series[0], BLUE_DARK)
    style_series(chart_a.series[1], ORANGE_DARK)
    chart_a.dataLabels = DataLabelList()
    chart_a.dataLabels.showVal = True
    ws.add_chart(chart_a, "A8")

    # ---------- 표 B: 조건별 개별 실행(run 1~3) ----------
    base_row = 8 + 18  # 차트 A 아래로 표 B 배치
    ws.cell(row=base_row - 1, column=1, value="조건별 개별 실행(run) avg_latency_ms (ms)")
    hdr_row = base_row
    ws.cell(row=hdr_row, column=1, value="")
    ws.cell(row=hdr_row, column=2, value="v5-NPU · FPS0")
    ws.cell(row=hdr_row, column=3, value="v8s-CPU · FPS0")
    ws.cell(row=hdr_row, column=4, value="v5-NPU · FPS30")
    ws.cell(row=hdr_row, column=5, value="v8s-CPU · FPS30")
    for i, (label, a, b, c, d) in enumerate(RUN_ROWS, start=hdr_row + 1):
        ws.cell(row=i, column=1, value=label)
        ws.cell(row=i, column=2, value=a)
        ws.cell(row=i, column=3, value=b)
        ws.cell(row=i, column=4, value=c)
        ws.cell(row=i, column=5, value=d)

    chart_b = BarChart()
    chart_b.type = "col"
    chart_b.grouping = "clustered"
    chart_b.title = "케이스(반복 실행)별 Latency 비교 (ms)"
    chart_b.y_axis.title = "avg_latency_ms"
    chart_b.x_axis.title = "반복 실행"
    chart_b.style = 2
    chart_b.height = 9
    chart_b.width = 16

    last_row = hdr_row + len(RUN_ROWS)
    data_b = Reference(ws, min_col=2, max_col=5, min_row=hdr_row, max_row=last_row)
    cats_b = Reference(ws, min_col=1, max_col=1, min_row=hdr_row + 1, max_row=last_row)
    chart_b.add_data(data_b, titles_from_data=True)
    chart_b.set_categories(cats_b)
    style_series(chart_b.series[0], BLUE_DARK)
    style_series(chart_b.series[1], ORANGE_DARK)
    style_series(chart_b.series[2], BLUE_LIGHT)
    style_series(chart_b.series[3], ORANGE_LIGHT)
    chart_b.dataLabels = DataLabelList()
    chart_b.dataLabels.showVal = True
    ws.add_chart(chart_b, f"A{last_row + 3}")

    for col, w in (("A", 20), ("B", 24), ("C", 24), ("D", 18), ("E", 18)):
        ws.column_dimensions[col].width = w

    wb.save(path)
    print(f"완료: '그래프' 시트(+차트 2개) 추가 -> {path}")


if __name__ == "__main__":
    main()
