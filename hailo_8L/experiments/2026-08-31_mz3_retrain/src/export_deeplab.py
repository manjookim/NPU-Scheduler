#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_deeplab.py (2026-09-02) — DeepLabV3+MNv2(Cityscapes) 를 Hailo-8L 컴파일 가능한
ONNX 로 내보내고, **배포 모델과 동일한 forward 로** mIoU 를 재측정한다.

train_deeplab_cityscapes.py 를 모듈로 import 해서 쓴다(학습 코드는 건드리지 않는다).

─────────────────────────────────────────────────────────────────────────────
왜 이 파일이 필요한가 — 인계문서 4-A 의 "다음 수정안"은 쓰면 안 된다
─────────────────────────────────────────────────────────────────────────────
인계문서는 컴파일 실패(microcode exceeded for resize1_sd0)를 피하려고
  argmax(17x17x19) -> nearest upsample(513) 로 **순서를 바꾸자**고 제안한다.
이건 두 가지 이유로 실험을 망친다.

  (1) 정확도: 17x17 에서 argmax 를 하면 라벨맵이 30x30 픽셀짜리 블록이 된다.
      공식 모델은 logits 를 513x513 으로 bilinear 업샘플한 **뒤** argmax 한다.
      두 연산은 교환법칙이 성립하지 않는다. mIoU 47.72 는 그대로 유지되지 않는다.
      (--mode argmax_first --eval 로 직접 측정해 보라. 이 파일에 그 경로도 넣어 뒀다.)
  (2) 더 치명적 — 비교 대상이 아니게 된다: 공식은 21채널 bilinear resize 를 쓴다.
      argmax 선행안은 1채널 nearest resize 를 돌린다. 연산 배치가 다르면
      latency/컨텍스트 수/스케줄링 거동이 달라져서, "공식 구조 재현"이라는 전제가 깨진다.

─────────────────────────────────────────────────────────────────────────────
[2026-09-09 전면 정정] 이 파일의 이전 진단은 틀렸다
─────────────────────────────────────────────────────────────────────────────
이전 버전은 원인을 "device_pre_post_layers 가 적용되지 않은 것"으로 적었다. **오진이다.**
  · MZ 소스 확인: 이 키는 get_postprocessing_callback()/postprocessing_callback()
    에서만 쓰인다 = 평가 시 호스트 후처리를 건너뛸지 정하는 용도이고,
    그래프에 레이어를 추가하지 않는다.
  · 공식 yaml 의 parser.nodes 는 [MobilenetV2/Conv/Conv2D, **ArgMax**] 로,
    공식도 우리처럼 꼬리를 ONNX 그래프에서 가져온다. logits 에서 끊는 게 아니다.

진짜 원인은 **F.interpolate 의 align_corners 인자** 하나였다. 공식 HEF 와 우리 HN 을
직접 뜯어 비교한 결과:

    공식 resize2 : mode='align_corners', ratio_list=[15.0, 2.0117...]   <- 2단 분해
    우리 resize1 : mode='half_pixels',   ratio_list=[30.176...]         <- 단일 단계

align_corners 는 src = dst/32 라 17->255->513 으로 쪼개도 합성이 정확히 일치해서
DFC 가 자유롭게 분해할 수 있다. half_pixel 은 ±0.5 오프셋 때문에 분해가 불가능해
x30.176 을 한 방에 처리해야 하고, gcd(17,513)=1 이라 보간 가중치가 513칸 내내
반복되지 않아 주소생성 microcode 가 폭발한다. -> `resize1_sd0` 초과.

또한 헤드 구조도 공식과 달랐다(구 --head simple). 공식은 백본을 320ch 에서 자르고
image pooling 브랜치 + concat projection 을 갖는다. train_deeplab_cityscapes.py 의
--head official 로 복원했다. 자세한 내용은 그 파일의 OfficialHead 참조.

─────────────────────────────────────────────────────────────────────────────
사용
─────────────────────────────────────────────────────────────────────────────
  # 1) 배포와 동일한 forward 로 mIoU 재측정 (인계문서 Next Action #5)
  python export_deeplab.py --out ./runs/dlv3_mnv2 --mode logits --eval --data D:/datasets/Cityscapes

  # 2) ONNX export (권장: logits 에서 끊고 bilinear+argmax 는 칩에 맡긴다)
  python export_deeplab.py --out ./runs/dlv3_mnv2 --mode logits --onnx deeplab_cs_logits.onnx

  # 3) 공식과 동일하게 그래프에 Resize+ArgMax 까지 넣는 판 (파서에게 device layer 로 넘길 때)
  python export_deeplab.py --out ./runs/dlv3_mnv2 --mode resize_argmax --onnx deeplab_cs_513.onnx
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_deeplab_cityscapes as T   # noqa: E402

IMG = T.IMG
NUM_CLASSES = T.NUM_CLASSES

# 공식 deeplab_v3_mobilenet_v2_wo_dilation = 2.10M params / 3.21 GOPs.
# [2026-09-09] --head official 복원판 = 2,113,043 (2.113M). 공식 2.10M 대비 1.01배.
#   백본 features[:18](1,811,712) + OfficialHead(296,448) + logits 19cls(4,883).
# 구 --head simple 은 2,248,211 (2.25M, 공식 대비 1.07배) 였다.
OFFICIAL_PARAMS = 2.10e6


class ExportWrapper(nn.Module):
    """학습된 DeepLabV3MNv2 를 감싸서 export 모드별 forward 를 만든다.

    mode = "logits"        : 백본 -> 1x1 conv 까지. 출력 (1,19,17,17).
                             업샘플+argmax 는 HEF 의 device pre/post layer 가 처리한다(공식과 동일).
    mode = "resize_argmax" : 공식 TF 그래프와 같은 순서. Resize(bilinear,513) -> ArgMax.
                             출력 (1,1,513,513). 파서가 이 꼬리를 device layer 로 접어야 한다.
    mode = "argmax_first"  : 인계문서가 제안한 안. ArgMax(17x17) -> Resize(nearest,513).
                             **정확도가 무너진다. 비교용으로만 남겨 둔다.**
    """

    def __init__(self, base, mode):
        super().__init__()
        self.backbone = base.backbone
        self.aspp = base.aspp
        self.classifier = base.classifier
        self.mode = mode

    def logits(self, x):
        f = self.backbone(x)
        if self.aspp is not None:
            f = self.aspp(f)
        return self.classifier(f)                      # (N,19,17,17)

    def forward(self, x):
        z = self.logits(x)
        if self.mode == "logits":
            return z
        if self.mode == "resize_argmax":
            # ★★★ [2026-09-09 수정] align_corners=False -> True ★★★
            # microcode exceeded size for resize1_sd0 의 **진짜 원인**이 이 인자였다.
            # 공식 HEF 와 우리 것의 파싱된 HN 을 나란히 비교한 결과:
            #
            #   공식 resize2 : mode='align_corners', ratio_list=[15.0, 2.0117...]  <- 2단 분해됨
            #   우리 resize1 : mode='half_pixels',   ratio_list=[30.176...]        <- 단일 단계
            #
            # align_corners 는 src = dst*(17-1)/(513-1) = dst/32 라서, 17->255->513 으로
            # 쪼개도 합성이 **정확히** 원래 매핑과 같아진다(i*127/256 * 16/254 = i/32).
            # 그래서 DFC 가 큰 업샘플을 두 단계로 자유롭게 분해할 수 있고, 각 단계의
            # microcode 가 작아져서 3줄짜리 alls 로도 컴파일된다.
            #
            # 반면 half_pixel 은 src = (dst+0.5)*17/513 - 0.5 로 ±0.5 오프셋 때문에
            # 단계 합성이 성립하지 않는다 -> DFC 가 분해하지 못하고 x30.176 을 한 방에
            # 처리해야 한다. gcd(17,513)=1 이라 보간 가중치가 513칸 내내 반복되지 않아
            # 주소생성 microcode 가 폭발한다. 이것이 5번의 컴파일 실패 원인이다.
            #
            # [주의] 학습/평가는 align_corners=False 로 했으므로 배포 mIoU 를 다시 재야 한다.
            #        아래 predict_labels 도 True 로 맞춰 두었으니 --eval 로 재측정할 것.
            z = F.interpolate(z, size=(IMG, IMG), mode="bilinear", align_corners=True)
            # [수정] v1 은 여기서 .to(torch.int32) 를 붙였다. 그러면 ONNX 에 ArgMax 뒤로
            # Cast 노드가 하나 더 생겨서 --end-node-names ArgMax 가 실제 그래프 끝과 어긋난다.
            return torch.argmax(z, dim=1, keepdim=True)
        if self.mode == "argmax_first":
            z = torch.argmax(z, dim=1, keepdim=True).float()
            return F.interpolate(z, size=(IMG, IMG), mode="nearest")
        raise ValueError(self.mode)

    @torch.no_grad()
    def predict_labels(self, x):
        """평가용 — **export forward 와 수치적으로 동일한** 라벨맵 (N,513,513)."""
        if self.mode == "logits":
            # [2026-09-09] 칩의 device resize 는 align_corners 규약을 쓴다(공식 HN 확인).
            # 배포 기준 mIoU 를 재려면 여기도 True 여야 한다.
            z = F.interpolate(self.logits(x), size=(IMG, IMG), mode="bilinear", align_corners=True)
            return z.argmax(1)                          # 칩의 bilinear->argmax 와 같은 순서
        return self.forward(x).squeeze(1).long()


def load_model(out_dir, head, ckpt_name=None):
    path = None
    for cand in ([ckpt_name] if ckpt_name else []) + ["best.pt", "last.pt"]:
        p = cand if (cand and os.path.isabs(cand)) else os.path.join(out_dir, cand)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise SystemExit(f"[에러] 체크포인트가 없습니다: {out_dir}/best.pt 또는 last.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = ck.get("head", head)               # v1 은 head 를 저장하지 않아 resume 시 shape 불일치가 났다
    base = T.DeepLabV3MNv2(pretrained=False, export=False, head=head)
    base.load_state_dict(ck["model"])
    base.eval()
    n = sum(p.numel() for p in base.parameters())
    miou = ck.get("miou")
    print(f"체크포인트: {path}  head={head}  "
          f"학습시 기록 mIoU={'%.2f' % miou if isinstance(miou, (int, float)) else 'n/a'}")
    print(f"파라미터: {n:,} ({n/1e6:.3f}M)  / 공식 2.10M 대비 {n/OFFICIAL_PARAMS:.2f}배")
    if n > 4e6:
        raise SystemExit("[중단] 공식 대비 파라미터가 2배 이상입니다. latency 비교가 성립하지 않습니다. "
                         "--head official 로 재학습한 체크포인트를 쓰세요.")
    return base, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs/dlv3_mnv2")
    ap.add_argument("--ckpt")
    ap.add_argument("--head", choices=["official", "simple", "aspp"], default="official")
    ap.add_argument("--mode", choices=["logits", "resize_argmax", "argmax_first"], default="logits")
    ap.add_argument("--onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--eval", action="store_true", help="배포 forward 와 동일한 경로로 mIoU 재측정")
    ap.add_argument("--data", help="--eval 시 Cityscapes 루트")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    base, head = load_model(a.out, a.head, a.ckpt)
    model = ExportWrapper(base, a.mode).eval()
    if a.mode == "argmax_first":
        print("\n[경고] argmax_first 는 17x17 에서 argmax 를 한 뒤 nearest 로 키웁니다.\n"
              "       공식 모델(bilinear 후 argmax)과 연산 순서가 다르고 정확도가 떨어집니다.\n"
              "       실험 비교군으로 쓰지 마세요. --eval 로 손실을 확인만 하십시오.\n")

    if a.eval:
        if not a.data:
            raise SystemExit("--eval 에는 --data 가 필요합니다.")
        from torch.utils.data import DataLoader
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        va = T.Cityscapes(a.data, "val", train=False)
        dl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)
        cm = T.ConfMat()
        with torch.no_grad():
            for x, y in dl:
                pred = model.predict_labels(x.to(dev, non_blocking=True))
                cm.update(pred.cpu(), y)
        miou, per = cm.miou()
        print(f"\n[mode={a.mode}] 배포 forward 기준 mIoU = {miou:.2f}")
        for n_, v_ in zip(T.CLASS_NAMES, per):
            print(f"  {n_:14s} {v_:5.1f}")
        print("\n※ 이 값은 513x513 리사이즈 기준입니다. Cityscapes 공식 프로토콜(1024x2048 원해상도)"
              "\n   과는 다르므로 논문 표에는 측정 조건을 함께 적으십시오.")

    if a.onnx:
        model.cpu().eval()
        dummy = torch.randn(1, 3, IMG, IMG)
        out_name = {"logits": "logits", "resize_argmax": "ArgMax", "argmax_first": "ArgMax"}[a.mode]
        torch.onnx.export(model, dummy, a.onnx,
                          input_names=["input"], output_names=[out_name],
                          opset_version=a.opset, do_constant_folding=True, dynamic_axes=None)
        print(f"\nONNX 저장: {a.onnx}")
        try:
            import onnx
            m = onnx.load(a.onnx)
            print("  그래프 꼬리 노드 (--end-node-names 에 쓸 **실제 이름**):")
            for nd in m.graph.node[-6:]:
                print(f"    {nd.op_type:12s} name='{nd.name}'  out={list(nd.output)}")
            print("  그래프 입력 노드:")
            for nd in m.graph.node[:2]:
                print(f"    {nd.op_type:12s} name='{nd.name}'")
            for o in m.graph.output:
                dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
                print(f"  출력 '{o.name}' shape={dims}")
        except ImportError:
            print("  (onnx 미설치 — 노드 이름 확인 생략. pip install onnx)")
        print("\n다음 단계:")
        print(f"  python -m onnxsim {a.onnx} {os.path.splitext(a.onnx)[0]}_sim.onnx")
        print(f"  python compile_deeplab.py --onnx {os.path.splitext(a.onnx)[0]}_sim.onnx "
              f"--mode {a.mode} --stop-after parse    # ← 먼저 파싱만 (1분). 10분 낭비 방지")


if __name__ == "__main__":
    main()
