#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_alloc_grammar.py (2026-09-02) — resize1 을 칩에 앉히기 위한 할당기 옵션 문법 확보.

[여기까지 확정된 것]
  · 우리 HN 은 공식과 동일하게 resize(513x513x19) -> argmax(513x513x1) -> output 으로 나온다.
    모델/ONNX/파싱 전부 정상. Cast 는 무관(A/B HN 동일).
  · 실패 지점은 오직 할당: `microcode exceeded size for resize1_sd0`.
  · 공식 HEF 도 같은 HN 인데 컴파일에 성공했고, 그 base alls 에는
    allocator_param / resources_param / performance_param / defuse / format_conversion 이 들어 있다.

[이 스크립트가 하는 일 — 전부 파싱만, 컴파일 안 함(수십 초)]
  1. MZ 의 deeplab alls 를 전수 나열. hailo8l/base/..._wo_dilation.alls 가 있으면
     그게 공식 HEF 를 만든 할당 스크립트 '정답지'다.
  2. DFC 내부에서 resize / allocator_param / performance_param / defuse 의 실제 파라미터 이름을 뽑는다.
  3. 후보 alls 를 load_model_script 로 파싱만 시켜 문법 적합 여부를 즉시 판정한다.
  4. HN 의 실제 레이어 이름을 찍는다 (defuse 대상 지정에 필요).

실행:  python tools/dfc/probe_alloc_grammar.py
결과:  tools/dfc/alloc_grammar_report.txt
"""
import io
import os
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(REPO, "hailo_8L", "experiments", "2026-08-31_mz3_retrain", "artifacts")
REPORT = os.path.join(HERE, "alloc_grammar_report.txt")
BUF = io.StringIO()


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    BUF.write(s + "\n")


def sec(t):
    P("\n" + "=" * 78); P("== " + t); P("=" * 78)


# ─────────── 1. MZ 의 deeplab alls 전수 ───────────
sec("1. Model Zoo 의 deeplab 관련 파일 전수 (hailo8l/base 에 wo_dilation 이 있으면 그게 정답지)")
try:
    import hailo_model_zoo
    mzroot = os.path.dirname(hailo_model_zoo.__file__)
    found = []
    for dp, _, fns in os.walk(os.path.join(mzroot, "cfg")):
        for fn in fns:
            if "deeplab" in fn:
                found.append(os.path.join(dp, fn))
    for f in sorted(found):
        P("  ", f)
    for f in sorted(found):
        if f.endswith(".alls") and ("wo_dilation" in f or "/base/" in f):
            P(f"\n  === 전문: {f} ===")
            P(open(f, encoding="utf-8", errors="ignore").read()[:6000])
    b = os.path.join(mzroot, "cfg", "alls", "hailo8l", "base")
    if os.path.isdir(b):
        P(f"\n  hailo8l/base 파일 수: {len(os.listdir(b))}")
        P("  deeplab 관련:", [x for x in os.listdir(b) if "deeplab" in x])
except Exception as e:
    P("  실패:", e)

# ─────────── 2. DFC 안의 파라미터 이름 grep ───────────
sec("2. DFC 내부에서 뽑은 실제 파라미터 이름")
roots = []
try:
    import hailo_sdk_client, hailo_sdk_common
    roots = [os.path.dirname(hailo_sdk_client.__file__),
             os.path.dirname(hailo_sdk_common.__file__)]
except Exception as e:
    P("  import 실패:", e)

KEYS = ["resize_shapes", "resize_method", "compiler_optimization_level",
        "defuse_type", "format_conversion", "max_utilization",
        "automatic_resize_reshapes", "enable_auto_spatial", "width_splitter_defuse"]
for k in KEYS:
    P(f"\n  --- '{k}' ---")
    n = 0
    for root in roots:
        for dp, _, fns in os.walk(root):
            if "__pycache__" in dp:
                continue
            for fn in fns:
                if not fn.endswith((".py", ".lark", ".json", ".txt")):
                    continue
                p = os.path.join(dp, fn)
                try:
                    txt = open(p, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                if k not in txt:
                    continue
                for i, line in enumerate(txt.splitlines(), 1):
                    if k in line and n < 12:
                        P(f"    {os.path.basename(p)}:{i}: {line.strip()[:180]}")
                        n += 1
    if n == 0:
        P("    (없음)")

# ─────────── 3. 후보 alls 문법 판정 (파싱만) ───────────
sec("3. 후보 alls 파싱 판정 (컴파일 안 함)")
BASE = ("normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])\n"
        "model_optimization_config(calibration, batch_size=1, calibset_size=16)\n"
        "pre_quantization_optimization(equalization, policy=disabled)\n")

CANDS = [
    ("P1 perf level max",       "performance_param(compiler_optimization_level=max)\n"),
    ("P2 perf level 1",         "performance_param(compiler_optimization_level=1)\n"),
    ("P3 perf level 2",         "performance_param(compiler_optimization_level=2)\n"),
    ("R1 resources 0.95",       "resources_param(max_utilization=0.95)\n"),
    ("R2 resources full",       "resources_param(max_compute_utilization=0.95, max_memory_utilization=0.9, "
                                "max_control_utilization=0.95)\n"),
    ("A1 auto resize reshapes", "allocator_param(automatic_resize_reshapes=enabled)\n"),
    ("A2 auto reshapes",        "allocator_param(automatic_reshapes=enabled)\n"),
    ("A3 spatial reshapes",     "allocator_param(enable_auto_spatial_reshapes=True)\n"),
    ("A4 timeout up",           "allocator_param(timeout=3600, cluster_timeout=1800, splitter_timeout=1800)\n"),
    ("A5 width splitter",       "allocator_param(width_splitter_defuse=enabled)\n"),
    ("D1 defuse INPUT_FEATURES","resize1_fs, resize1_d0, resize1_d1, resize1_dc = "
                                "defuse(resize1, 2, defuse_type=INPUT_FEATURES)\n"),
    ("D2 defuse COMPUTE_LANES", "resize1_fs, resize1_2, resize1_c = "
                                "defuse(resize1, 2, defuse_type=COMPUTE_LANES, axis=FEATURES)\n"),
    ("Z  대조군(가짜 명령)",      "zzz_not_a_command(1)\n"),
]

har = os.path.join(ART, "probe_A.har")
if not os.path.exists(har):
    har = os.path.join(ART, "deeplab_v3_mnv2_wo_dilation_cityscapes_parsed.har")
try:
    from hailo_sdk_client import ClientRunner
    P(f"  HAR: {har} ({'있음' if os.path.exists(har) else '없음 -> hw_arch 로 대체'})")
    ok_list = []
    for label, snip in CANDS:
        try:
            r = ClientRunner(har=har) if os.path.exists(har) else ClientRunner(hw_arch="hailo8l")
            r.load_model_script(BASE + snip)
            P(f"  [OK  ] {label:26s} {snip.strip()[:110]}")
            ok_list.append(label)
        except Exception as e:
            msg = str(e).replace("\n", " ")[:200]
            P(f"  [실패] {label:26s} {msg}")
    P("\n  파싱 통과:", ", ".join(ok_list) if ok_list else "없음")
except Exception:
    P(traceback.format_exc()[:2000])

# ─────────── 4. HN 의 실제 레이어 이름 ───────────
sec("4. HN 레이어 이름 (defuse/format_conversion 에서 이 이름을 써야 한다)")
try:
    from hailo_sdk_client import ClientRunner
    if os.path.exists(har):
        r = ClientRunner(har=har)
        hn = r.get_hn()
        P("  net name:", hn.get("name"))
        names = list(hn["layers"].keys())
        P(f"  총 {len(names)}개. 꼬리 8개:")
        for n in names[-8:]:
            v = hn["layers"][n]
            P(f"    {v.get('type','?'):16s} {n}  out={v.get('output_shapes')}")
    else:
        P("  HAR 없음")
except Exception as e:
    P("  실패:", e)

open(REPORT, "w", encoding="utf-8").write(BUF.getvalue())
print(f"\n리포트 저장: {REPORT}")
