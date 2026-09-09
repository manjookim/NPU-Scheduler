#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_deeplab_tail.py (2026-09-02) — deeplab 컴파일 실패의 진짜 원인 판정 (약 2분).

[리포트로 확정된 사실]
  · DFC 3.33 이 받아들이는 명령은 `resize(resize_shapes=[...], resize_method=bilinear)` 하나뿐.
    logits_layer / device_pre_post_layers / post_process 는 **존재하지 않는 명령**이다.
  · yaml 의 `device_pre_post_layers` 는 컴파일러 지시가 아니라 **MZ 평가 콜백용 플래그**다
    (hailo_model_zoo/core/main_utils.py::get_postprocessing_callback).
    -> "yaml 을 안 써서 device layer 가 안 붙었다"는 내 앞선 진단은 틀렸다.
  · 공식 wo_dilation alls 는 3줄뿐이고 argmax 명령이 없다. 즉 argmax/bilinear 는
    **ONNX 그래프의 Resize+ArgMax 를 DFC 파서가 인식해서** 처리한다.
    (공식 yaml parser.nodes = [MobilenetV2/Conv/Conv2D, ArgMax])

[그렇다면 왜 우리 것만 실패했나 — 검증할 가설]
  H1 (유력): 우리 ONNX 는 꼬리가 Resize -> ArgMax -> **Cast** 다. `.to(torch.int32)` 가 만든
      Cast 가 붙어서 DFC 의 "Resize+ArgMax 꼬리" 패턴 매칭이 깨졌고, resize 가 평범한
      NN core 레이어(`resize1_sd0`)로 배치돼 microcode 한계를 넘었다.
  H2: Cast 와 무관하게 우리 그래프에서는 애초에 접히지 않는다 -> 할당기(allocator) 문제.
      공식 hailo8/base/deeplab_v3_mobilenet_v2.alls 에 defuse/format_conversion/
      allocator_param/performance_param 이 잔뜩 들어 있는 것이 그 증거다.

판정법: 세 ONNX 를 파싱만 해서 HN 레이어 목록을 비교한다.
  · 출력이 [513,513,1] 이고 resize 레이어가 없다 -> 꼬리가 접혔다(성공 경로)
  · resize 레이어가 보인다               -> 접히지 않았다(H2, 할당기 튜닝 필요)

실행:  python tools/dfc/probe_deeplab_tail.py
결과:  tools/dfc/deeplab_tail_report.txt
"""
import io
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(REPO, "hailo_8L", "experiments", "2026-08-31_mz3_retrain", "artifacts")
REPORT = os.path.join(HERE, "deeplab_tail_report.txt")
BUF = io.StringIO()


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    BUF.write(s + "\n")


CASES = [
    ("A_resize_argmax  (Cast 제거, /ArgMax 까지)",
     os.path.join(ART, "deeplab_cs_resize_argmax.onnx"), "/ArgMax"),
    ("B_with_cast      (원본 sim, Cast 포함)",
     os.path.join(REPO, "deeplab_cs_513_sim.onnx"), None),          # end node 지정 안 함 = 전체
    ("C_logits_only    (/classifier/Conv 까지)",
     os.path.join(ART, "deeplab_cs_logits.onnx"), "/classifier/Conv"),
]

START = "/backbone/backbone.0/backbone.0.0/Conv"

from hailo_sdk_client import ClientRunner  # noqa: E402

for label, path, end in CASES:
    P("\n" + "=" * 78)
    P(f"== {label}")
    P(f"   {path}")
    P("=" * 78)
    if not os.path.exists(path):
        P("   [없음] 건너뜀")
        continue
    try:
        r = ClientRunner(hw_arch="hailo8l")
        kw = {"start_node_names": [START]}
        if end:
            kw["end_node_names"] = [end]
        r.translate_onnx_model(path, "probe_net", **kw)
        hn = r.get_hn()
        layers = hn["layers"]
        P(f"   총 HN 레이어 {len(layers)}개")

        # 꼬리 쪽 관심 레이어만
        interesting = [(n, v) for n, v in layers.items()
                       if v.get("type") not in ("conv", "dw", "activation", "input_layer",
                                                "normalization", "ew_add", "const_input")]
        P("\n   [입출력/특수 레이어]")
        for n, v in interesting:
            P(f"     {v.get('type','?'):22s} {n}")
            P(f"       out={v.get('output_shapes')}  engine={v.get('engine','-')}")

        outs = [(n, v.get("output_shapes")) for n, v in layers.items()
                if v.get("type") == "output_layer"]
        P(f"\n   >>> 출력 레이어: {outs}")

        has_resize = [n for n, v in layers.items() if "resize" in str(v.get("type", "")).lower()
                      or "resize" in n.lower()]
        has_argmax = [n for n, v in layers.items() if "argmax" in str(v.get("type", "")).lower()
                      or "argmax" in n.lower()]
        P(f"   >>> resize 레이어: {has_resize if has_resize else '없음 (= 접혔거나 그래프에 없음)'}")
        P(f"   >>> argmax 레이어: {has_argmax if has_argmax else '없음'}")

        # 판정
        shp = outs[0][1] if outs else None
        if shp and shp[0][-3:] == [513, 513, 1] and not has_resize:
            P("\n   ★ 판정: 꼬리가 device post layer 로 접혔다. 이대로 컴파일하면 성공 가능성 높음.")
        elif has_resize:
            P("\n   ★ 판정: resize 가 NN core 레이어로 남았다. 이대로 컴파일하면 microcode 초과 재현.")
        else:
            P("\n   ★ 판정: 출력이 513x513x1 이 아님 — 호스트 후처리가 생긴다(비교 불가).")

        r.save_har(os.path.join(ART, f"probe_{label.split('_')[0]}.har"))
    except Exception:
        P("   [실패]")
        P(traceback.format_exc()[:3000])

P("\n\n" + "=" * 78)
P("== 참고: 공식 alls 가 쓰는 할당기 명령 (hailo8/base/deeplab_v3_mobilenet_v2.alls)")
P("=" * 78)
P("""  allocator_param(...), resources_param(...), performance_param(compiler_optimization_level=1),
  defuse(<layer>, N, defuse_type=COMPUTE_LANES|INPUT_FEATURES, axis=FEATURES),
  <x>_input_reshape = format_conversion(<prev>, <resize>, hailo_rgb_to_f8cr, ...)
  -> 공식 HEF 자체가 이런 할당 스크립트로 만들어졌다. 즉 '기본값 조건'을 지킨다고
     performance_param 을 금지하면 오히려 공식 HEF 와 조건이 달라진다.""")

open(REPORT, "w", encoding="utf-8").write(BUF.getvalue())
print(f"\n리포트 저장: {REPORT}")
