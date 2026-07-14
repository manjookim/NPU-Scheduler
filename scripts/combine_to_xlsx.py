#!/usr/bin/env python3
"""
combine_to_xlsx.py
사용모델수(1/2/3) × batch(1/10/50/63) = 12개 시트 CSV 를 시트 12개짜리 xlsx 하나로 합친다.
auto_experiment_batch_priority.sh 가 만든 csv 폴더를 입력으로 준다.

사용법:
  python3 combine_to_xlsx.py <csv_dir> [출력.xlsx]
  예) python3 combine_to_xlsx.py experiments/2026-07-12_batch_priority_exp1/csv
"""
import sys, os, csv
from openpyxl import Workbook

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    csv_dir = sys.argv[1]
    xlsx_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(csv_dir.rstrip("/")) or ".", "results_batch_priority.xlsx")

    wb = Workbook(); wb.remove(wb.active)
    made = 0
    for n in (1, 2, 3):                       # 사용 모델 수
        for b in (1, 10, 50, 63):             # batch
            path = os.path.join(csv_dir, f"results_{n}model_b{b}.csv")
            ws = wb.create_sheet(title=f"{n}model_b{b}")
            if os.path.exists(path):
                for row in csv.reader(open(path)):
                    ws.append(row)
                made += 1
            else:
                print(f"  [!] 없음: {os.path.basename(path)} (빈 시트 생성)")
    wb.save(xlsx_path)
    print(f"완료: 시트 12개 (데이터 채워진 시트 {made}개) → {xlsx_path}")

if __name__ == "__main__":
    main()
