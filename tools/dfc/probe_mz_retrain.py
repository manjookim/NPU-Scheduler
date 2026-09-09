#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_mz_retrain.py (2026-09-02) — "Hailo 깃허브에 재학습 방법이 있다"를 실제로 확인한다.

설치된 hailo_model_zoo 2.17.0 에서 다음을 직접 읽어 판정한다.
  1. training/ 디렉터리에 어떤 모델의 재학습 Docker 가 있는가 (deeplab 이 있는가?)
  2. 재학습 관련 문서(docs/) 목록
  3. Cityscapes 를 쓰는 네트워크가 무엇인가 (deeplab 말고 대안이 있는가)
  4. segmentation 계열 네트워크 전체와 각각의 dataset

실행:  python tools/dfc/probe_mz_retrain.py
결과:  tools/dfc/mz_retrain_report.txt
"""
import io
import os
import re

BUF = io.StringIO()
HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "mz_retrain_report.txt")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    BUF.write(s + "\n")


def sec(t):
    P("\n" + "=" * 78); P("== " + t); P("=" * 78)


import hailo_model_zoo
MZ = os.path.dirname(hailo_model_zoo.__file__)
ROOT = os.path.dirname(MZ)
P(f"hailo_model_zoo {getattr(hailo_model_zoo,'__version__','?')}  @ {ROOT}")

# 1. training 디렉터리
sec("1. 재학습(training) 지원 모델 — Hailo 공식 Docker 목록")
cands = [os.path.join(ROOT, "training"), os.path.join(MZ, "training")]
tdir = next((d for d in cands if os.path.isdir(d)), None)
if not tdir:
    P("  training/ 디렉터리 없음 (경로:", cands, ")")
else:
    P(f"  {tdir}")
    names = sorted(os.listdir(tdir))
    P(f"  총 {len(names)}개:")
    for n in names:
        has_df = os.path.exists(os.path.join(tdir, n, "Dockerfile"))
        P(f"    {'[Dockerfile]' if has_df else '[          ]'} {n}")
    P("\n  >>> deeplab 관련:", [n for n in names if "deeplab" in n.lower() or "lab" in n.lower()])
    P("  >>> segmentation 관련:", [n for n in names
                                  if re.search(r"seg|fcn|unet|stdc|yolact", n, re.I)])

# 2. 문서
sec("2. 재학습 관련 문서")
for d in [os.path.join(ROOT, "docs"), os.path.join(MZ, "docs")]:
    if not os.path.isdir(d):
        continue
    for dp, _, fns in os.walk(d):
        for fn in sorted(fns):
            if re.search(r"retrain|train|custom", fn, re.I):
                P("  ", os.path.join(dp, fn))
# 루트의 README/RETRAIN 문서
for fn in sorted(os.listdir(ROOT)):
    if re.search(r"retrain|train|README", fn, re.I):
        P("  ", os.path.join(ROOT, fn))

# 3. Cityscapes 를 쓰는 네트워크
sec("3. Cityscapes 를 쓰는 네트워크 (deeplab 대안이 있는지)")
netdir = os.path.join(MZ, "cfg", "networks")
basedir = os.path.join(MZ, "cfg", "base")
if os.path.isdir(basedir):
    P("  base yaml 중 cityscapes:")
    for fn in sorted(os.listdir(basedir)):
        p = os.path.join(basedir, fn)
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if "cityscapes" in t.lower():
            P(f"    {fn}")
hits = []
for fn in sorted(os.listdir(netdir)):
    p = os.path.join(netdir, fn)
    try:
        t = open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    if "cityscapes" in t.lower():
        hits.append(fn)
P("  networks yaml 중 cityscapes:", hits if hits else "없음")
for fn in hits:
    P(f"\n  === {fn} ===")
    P(open(os.path.join(netdir, fn), encoding="utf-8", errors="ignore").read()[:1800])

# 4. segmentation 네트워크 전체
sec("4. segmentation 계열 네트워크와 데이터셋")
rows = []
for fn in sorted(os.listdir(netdir)):
    p = os.path.join(netdir, fn)
    try:
        t = open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    if "segmentation" not in t.lower():
        continue
    ds = re.search(r"training_data:\s*(.+)", t)
    sh = re.search(r"input_shape:\s*(.+)", t)
    fp = re.search(r"full_precision_result:\s*(.+)", t)
    rows.append((fn.replace(".yaml", ""), (ds.group(1).strip() if ds else "-"),
                 (sh.group(1).strip() if sh else "-"), (fp.group(1).strip() if fp else "-")))
P(f"  {'network':<45}{'training_data':<26}{'input':<14}{'result'}")
for r in rows:
    P(f"  {r[0]:<45}{r[1]:<26}{r[2]:<14}{r[3]}")

open(REPORT, "w", encoding="utf-8").write(BUF.getvalue())
print(f"\n리포트 저장: {REPORT}")
