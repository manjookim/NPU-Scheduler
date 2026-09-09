#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_dfc_grammar.py (2026-09-02) — 설치된 DFC 에서 model script(alls) 문법과
device pre/post layer 적용 경로를 **직접 캐낸다**. 더 이상 철자를 추측하지 않는다.

WSL 의 hailo_venv 에서 실행:
    source ~/hailo_venv/bin/activate
    python tools/dfc/probe_dfc_grammar.py

결과는 tools/dfc/dfc_grammar_report.txt 에 저장된다(저장소 안이라 그대로 공유 가능).
"""
import importlib
import io
import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "dfc_grammar_report.txt")
BUF = io.StringIO()


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    BUF.write(s + "\n")


def section(t):
    P("\n" + "=" * 78)
    P("== " + t)
    P("=" * 78)


# ─────────────────────────── 1. 설치 현황 ───────────────────────────
section("1. 설치된 Hailo 패키지")
PKGS = {}
for name in ["hailo_sdk_client", "hailo_sdk_common", "hailo_model_zoo",
             "hailo_platform", "hailo_tutorials"]:
    try:
        m = importlib.import_module(name)
        d = os.path.dirname(m.__file__)
        PKGS[name] = d
        P(f"  {name:20s} {getattr(m, '__version__', '?'):12s} {d}")
    except Exception as e:
        P(f"  {name:20s} 없음 ({type(e).__name__})")

P(f"  python {sys.version.split()[0]}")

# ─────────────────────────── 2. alls 문법 파일 ───────────────────────────
section("2. model script(alls) 문법 파일 탐색")
GRAMMAR_EXT = (".lark", ".g", ".g4", ".ebnf", ".bnf", ".grammar")
grammar_files = []
for name, root in PKGS.items():
    if not root:
        continue
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(GRAMMAR_EXT) or "grammar" in fn.lower() or "alls" in fn.lower():
                grammar_files.append(os.path.join(dp, fn))
for f in grammar_files[:40]:
    P("  ", f)
P(f"  (총 {len(grammar_files)}개)")

# 문법 파일에서 명령 이름으로 보이는 토큰을 뽑는다
CMDS = set()
for f in grammar_files:
    try:
        txt = open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    CMDS |= set(re.findall(r'"([a-z][a-z0-9_]{3,40})"\s*(?:\(|\s|$)', txt))
    CMDS |= set(re.findall(r"'([a-z][a-z0-9_]{3,40})'", txt))
if CMDS:
    P("\n  문법 파일에서 추출한 명령 후보 (알파벳순):")
    for c in sorted(CMDS):
        P("   ", c)

# ─────────────────────────── 3. 관심 토큰 grep ───────────────────────────
section("3. 관심 토큰이 등장하는 위치 (argmax / bilinear / logits / device_pre_post / resize)")
TOKENS = ["device_pre_post", "logits_layer", "argmax", "bilinear",
          "resize_bilinear", "post_process", "postprocess", "max_finder"]
hits = {t: [] for t in TOKENS}
for name, root in PKGS.items():
    if not root:
        continue
    for dp, _, fns in os.walk(root):
        if "__pycache__" in dp or "/test" in dp:
            continue
        for fn in fns:
            if not fn.endswith((".py", ".json", ".lark", ".yaml", ".yml", ".txt")):
                continue
            p = os.path.join(dp, fn)
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for t in TOKENS:
                if t in txt:
                    for i, line in enumerate(txt.splitlines(), 1):
                        if t in line and len(hits[t]) < 25:
                            hits[t].append(f"{p}:{i}: {line.strip()[:170]}")
for t in TOKENS:
    P(f"\n  --- '{t}' ({len(hits[t])}건, 최대 25개) ---")
    for h in hits[t]:
        P("   ", h)

# ─────────────────────────── 4. 전체 파싱 에러 메시지 ───────────────────────────
section("4. load_model_script 전체 에러 메시지 (기대 토큰이 여기 들어 있다)")
BASE = ("normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])\n"
        "model_optimization_config(calibration, batch_size=1, calibset_size=16)\n"
        "pre_quantization_optimization(equalization, policy=disabled)\n")
CANDS = [
    "logits_layer(argmax)\n",
    "argmax1 = argmax(conv_last)\n",
    "resize1 = resize(resize_shapes=[513,513], resize_method=bilinear)\n",
    "device_pre_post_layers(bilinear=true, argmax=true)\n",
    "post_process(argmax)\n",
    "zzz_definitely_not_a_command(1)\n",   # ← 대조군: 문법이 어떤 형태로 에러를 내는지 확인용
]
try:
    from hailo_sdk_client import ClientRunner
    har = None
    for cand in ["deeplab_v3_mnv2_wo_dilation_cityscapes_parsed.har",
                 "../hailo_8L/experiments/2026-08-31_mz3_retrain/artifacts/"
                 "deeplab_v3_mnv2_wo_dilation_cityscapes_parsed.har",
                 os.path.join(HERE, "..", "..", "hailo_8L", "experiments",
                              "2026-08-31_mz3_retrain", "artifacts",
                              "deeplab_v3_mnv2_wo_dilation_cityscapes_parsed.har")]:
        if os.path.exists(cand):
            har = cand
            break
    runner = ClientRunner(har=har) if har else ClientRunner(hw_arch="hailo8l")
    P(f"  runner: {'HAR=' + har if har else 'hw_arch=hailo8l (HN 없음)'}")
    for c in CANDS:
        P(f"\n  ---- 시도: {c.strip()}")
        try:
            runner.load_model_script(BASE + c)
            P("       [OK] 이 명령은 받아들여집니다.")
        except Exception as e:
            P("       [실패] 전체 메시지:")
            for line in str(e).splitlines()[:25]:
                P("        | " + line[:200])
except Exception:
    P("  ClientRunner 사용 불가:")
    P(traceback.format_exc()[:2000])

# ─────────────────────────── 5. ClientRunner API ───────────────────────────
section("5. ClientRunner 공개 메서드 (postprocess/hn 관련)")
try:
    from hailo_sdk_client import ClientRunner as CR
    ms = [m for m in dir(CR) if not m.startswith("_")]
    P("  전체:", ", ".join(ms))
    P("\n  관심:", ", ".join(m for m in ms if re.search(
        r"post|hn|script|output|layer|nms|infer|save|load|translate|optimi|compil", m, re.I)))
except Exception as e:
    P("  실패:", e)

# ─────────────────────────── 6. HN 스키마의 레이어 타입 ───────────────────────────
section("6. HN 이 지원하는 레이어 타입 (argmax/resize 가 있는지)")
try:
    root = PKGS.get("hailo_sdk_common") or PKGS.get("hailo_sdk_client")
    found = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".py") and ("layer" in fn.lower() or "hn" in fn.lower()):
                p = os.path.join(dp, fn)
                txt = open(p, encoding="utf-8", errors="ignore").read()
                for m in re.finditer(r"^\s*([A-Z_]{3,30})\s*=\s*['\"]([a-z_]{3,30})['\"]", txt, re.M):
                    if re.search(r"argmax|resize|logit|bilinear|post", m.group(2)):
                        found.append(f"{os.path.basename(p)}: {m.group(1)} = '{m.group(2)}'")
    for f in sorted(set(found))[:60]:
        P("  ", f)
    if not found:
        P("  (해당 없음)")
except Exception as e:
    P("  실패:", e)

# ─────────────────────────── 7. Model Zoo 의 device_pre_post 구현 ───────────────────────────
section("7. hailo_model_zoo 가 device_pre_post_layers 를 어떻게 적용하는가 (경로 A의 핵심)")
mz = PKGS.get("hailo_model_zoo")
if not mz:
    P("  hailo_model_zoo 미설치. 경로 A 를 쓰려면 v2.19.0 이하를 설치해야 합니다:")
    P("    git clone -b v2.19.0 https://github.com/hailo-ai/hailo_model_zoo.git")
    P("    cd hailo_model_zoo && pip install -e .")
else:
    for dp, _, fns in os.walk(mz):
        if "__pycache__" in dp:
            continue
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            txt = open(p, encoding="utf-8", errors="ignore").read()
            if "device_pre_post" in txt:
                P(f"\n  --- {p}")
                lines = txt.splitlines()
                for i, line in enumerate(lines):
                    if "device_pre_post" in line:
                        for j in range(max(0, i - 6), min(len(lines), i + 22)):
                            P(f"   {j+1:5d}| {lines[j][:180]}")
                        P("   " + "-" * 60)
    # 공식 deeplab yaml/alls 원본도 있으면 찍는다
    for dp, _, fns in os.walk(mz):
        for fn in fns:
            if "deeplab" in fn and fn.endswith((".yaml", ".alls")):
                p = os.path.join(dp, fn)
                P(f"\n  === {p} ===")
                P(open(p, encoding="utf-8", errors="ignore").read()[:2500])

open(REPORT, "w", encoding="utf-8").write(BUF.getvalue())
print(f"\n\n리포트 저장: {REPORT}")
