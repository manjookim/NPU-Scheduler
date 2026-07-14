# -*- coding: utf-8 -*-
# html/prithr 파싱을 여러 번에 나눠 실행 (이미 처리한 파일은 건너뜀).
# 결과를 metrics_prithr.json 에 누적 저장. 시간예산 초과 전 저장 후 종료.
import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(__file__))
from parse_metrics_prithr import parse_html, metrics

OUT = "metrics_prithr.json"
BUDGET = 38  # 초

data = {}
if os.path.exists(OUT):
    data = json.load(open(OUT))

files = sorted(glob.glob("html/prithr/*.html"))
todo = [f for f in files if os.path.basename(f) not in data]
print(f"전체 {len(files)} / 이미완료 {len(data)} / 남음 {len(todo)}")

t0 = time.time(); done = 0
for f in todo:
    if time.time() - t0 > BUDGET:
        break
    pr = parse_html(f)
    if pr is None:
        data[os.path.basename(f)] = None; done += 1; continue
    data[os.path.basename(f)] = metrics(pr)
    done += 1

json.dump(data, open(OUT, "w"))
print(f"이번 처리 {done}개, 누적 {len(data)}/{len(files)}")
print("ALL DONE" if len(data) >= len(files) else "MORE")
