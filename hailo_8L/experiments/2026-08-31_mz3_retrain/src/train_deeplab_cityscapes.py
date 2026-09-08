#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabV3 + MobileNetV2 (dilation 없음 = wo_dilation) 를 Cityscapes 19클래스로 학습한다.
workload2.xlsx SMART TRAFFIC 시나리오 2 의 "차선/도로 영역 분할" 워크로드용.

[왜 PyTorch인가]
Hailo Model Zoo 의 deeplab_v3_mobilenet_v2_wo_dilation 원본은 TensorFlow research/deeplab
(TF1 전용) 이다. 보유 GPU(RTX 5060 = Blackwell sm_120)는 TF1 마지막 컨테이너
(NGC 23.03-tf1-py3, CUDA 12.1, Hopper까지)가 지원하지 않아 TF1 경로가 막힌다.
그래서 동일 구조를 PyTorch 로 재구현한다. (PyTorch 2.11+cu128 은 sm_120 정상 동작 확인)

[wo_dilation 이 구조적으로 보장되는 이유]
torchvision MobileNetV2 의 features 는 원래 dilation 이 없고 stride 로만 다운샘플해서
출력이 input/32 (output_stride=32) 다. Hailo 의 wo_dilation 변형이 노리는 바로 그 구성이라
따로 atrous rate 를 끄는 조작이 필요 없다. (dilated 버전은 OS=16 + atrous conv)

[Hailo 입출력 규약 — 반드시 지킬 것]
  · 입력: HEF 는 uint8[0,255] 를 받고, .alls 의
      normalization1 = normalization([127.5]*3, [127.5]*3)
    가 칩에서 (x-127.5)/127.5 = [-1,1] 로 바꿔준다.
    따라서 이 모델은 **[-1,1] 입력을 기대**하도록 학습하고, 그래프 안에 정규화 층을 두지 않는다.
    (mobilenet_v2 GTSRB 재학습 때와 동일한 규약)
  · 출력: 원본 yaml 의 device_pre_post_layers 가 {bilinear: true, argmax: true} 라
    업샘플(513x513)과 argmax 가 칩에서 처리되고 최종 출력이 513x513x1 (라벨 인덱스 맵)이다.
    그래서 --export-onnx 시 Resize(bilinear) + ArgMax 까지 그래프에 포함시킨다.
    ArgMax 를 빼고 자르면 출력이 513x513x19 float 이 되어 후처리가 호스트로 넘어오고
    latency 비교가 무의미해진다.

[데이터셋]
  Cityscapes gtFine_trainvaltest.zip (253MB) + leftImg8bit_trainvaltest.zip (11GB)
  압축 해제 후 구조:
    <root>/gtFine/{train,val}/<city>/*_gtFine_labelIds.png
    <root>/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
  라벨은 labelIds(0~33) 로 저장돼 있으므로 코드 안에서 trainId(0~18, ignore=255) 로 매핑한다.
  → cityscapesscripts 로 별도 변환할 필요 없음.

사용:
  # 학습 (중단 시 --resume 으로 이어서)
  python train_deeplab_cityscapes.py --data D:/datasets/Cityscapes --out ./runs/dlv3_mnv2 \
      --epochs 40 --batch 4

  # 평가만
  python train_deeplab_cityscapes.py --data ... --out ... --eval-only --resume

  # ONNX export (Resize+ArgMax 포함, 513x513 고정 입력)
  python train_deeplab_cityscapes.py --out ... --resume --export-onnx deeplab_cs_513.onnx
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMG = 513
NUM_CLASSES = 19
IGNORE = 255

# ───────────── Cityscapes labelId(0~33) → trainId(0~18, 그 외 255) ─────────────
# 출처: cityscapesScripts/helpers/labels.py 의 trainId 정의
_LABELID_TO_TRAINID = np.full(256, IGNORE, dtype=np.uint8)
for _lid, _tid in {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}.items():
    _LABELID_TO_TRAINID[_lid] = _tid

CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
               "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
               "truck", "bus", "train", "motorcycle", "bicycle"]


# ───────────────────────────── 데이터셋 ─────────────────────────────
class Cityscapes(Dataset):
    """gtFine + leftImg8bit 를 직접 읽는다 (torchvision.datasets.Cityscapes 의존 없음)."""

    def __init__(self, root, split="train", train=True):
        self.train = train
        img_root = os.path.join(root, "leftImg8bit", split)
        lbl_root = os.path.join(root, "gtFine", split)
        if not os.path.isdir(img_root):
            raise SystemExit(f"[에러] 경로 없음: {img_root}\n"
                             f"       <root>/leftImg8bit/{split}/<city>/ 구조인지 확인하세요.")
        self.items = []
        for city in sorted(os.listdir(img_root)):
            cdir = os.path.join(img_root, city)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if not fn.endswith("_leftImg8bit.png"):
                    continue
                base = fn[: -len("_leftImg8bit.png")]
                lbl = os.path.join(lbl_root, city, base + "_gtFine_labelIds.png")
                if os.path.exists(lbl):
                    self.items.append((os.path.join(cdir, fn), lbl))
        if not self.items:
            raise SystemExit(f"[에러] {split} 샘플이 0개입니다. gtFine/leftImg8bit 둘 다 풀었는지 확인하세요.")
        print(f"  [{split}] {len(self.items)}장")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, lp = self.items[i]
        img = Image.open(ip).convert("RGB")
        lbl = Image.open(lp)

        if self.train:
            # 원본 1024x2048 → 랜덤 스케일 후 513x513 크롭 (DeepLab 표준 증강)
            s = np.random.uniform(0.5, 1.3)
            nw, nh = int(2048 * s * 0.5), int(1024 * s * 0.5)   # 0.5배 기준 스케일
            nw, nh = max(nw, IMG), max(nh, IMG)
            img = img.resize((nw, nh), Image.BILINEAR)
            lbl = lbl.resize((nw, nh), Image.NEAREST)
            x0 = np.random.randint(0, nw - IMG + 1)
            y0 = np.random.randint(0, nh - IMG + 1)
            img = img.crop((x0, y0, x0 + IMG, y0 + IMG))
            lbl = lbl.crop((x0, y0, x0 + IMG, y0 + IMG))
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # 평가는 배포 조건과 동일하게 513x513 고정 입력
            img = img.resize((IMG, IMG), Image.BILINEAR)
            lbl = lbl.resize((IMG, IMG), Image.NEAREST)

        a = np.asarray(img, dtype=np.float32)
        a = a / 127.5 - 1.0                       # ← [-1,1]. alls 의 normalization 과 동일 규약
        x = torch.from_numpy(a).permute(2, 0, 1)  # HWC → CHW
        y = torch.from_numpy(_LABELID_TO_TRAINID[np.asarray(lbl, dtype=np.uint8)]).long()
        return x, y


# ───────────────────────────── 모델 ─────────────────────────────
class ASPP(nn.Module):
    """DeepLabV3 ASPP. output_stride=32 이므로 atrous rate 를 원본(6/12/18)의 절반으로 둔다.
    OS=16 기준 rate 12 가 OS=32 에서는 rate 6 과 같은 수용영역을 갖는다."""

    def __init__(self, cin, cout=256, rates=(3, 6, 9)):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.bs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=r, dilation=r, bias=False),
                          nn.BatchNorm2d(cout), nn.ReLU(inplace=True)) for r in rates])
        # 원본 ASPP 의 image pooling 은 AdaptiveAvgPool + Resize 조합이라
        # DFC 파싱에서 dynamic shape 문제를 일으킬 수 있어 제외한다(정확도 영향 미미).
        n = 1 + len(rates)
        self.proj = nn.Sequential(nn.Conv2d(cout * n, cout, 1, bias=False),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                  nn.Dropout(0.1))

    def forward(self, x):
        return self.proj(torch.cat([self.b0(x)] + [b(x) for b in self.bs], dim=1))


class OfficialHead(nn.Module):
    """[2026-09-09 신규] 공식 deeplab_v3_mobilenet_v2_wo_dilation 의 헤드를 그대로 복원.

    추측이 아니라 **공식 HEF 를 파싱한 HN 에서 실측한 값**이다
    (artifacts/pathA/control/deeplab_v3_mobilenet_v2_wo_dilation.har):

        conv34   1x1 conv 960->320, linear      <- 백본 마지막 (features[17] 의 projection)
        ├─ conv35   1x1 conv 320->256, relu                  (1x1 브랜치)
        └─ avgpool1 17x17 global avg -> 1x1x320
           fc1      320->256 dense, relu                     (image pooling 브랜치)
           resize1  1x1 -> 17x17, nearest_neighbor
        concat1  [resize1, conv35] -> 17x17x512              <- 순서 주의: pooling 이 앞
        conv36   1x1 conv 512->256, relu                     (concat projection)
        conv37   1x1 conv 256->21,  linear                   (logits)

    HN 의 batch_norm=False 는 BN 이 없다는 뜻이 아니라 DFC 가 conv 에 **접어 넣은**
    결과다. 학습 시에는 BN 이 있어야 하고, export 시 자동으로 접힌다.

    3x3 atrous 브랜치는 없다 — wo_dilation 이라 atrous_rates 가 비어 있기 때문이다.
    """

    def __init__(self, cin=320, cout=256):
        super().__init__()
        # conv35 — 1x1 브랜치
        self.branch = nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        # avgpool1 + fc1 — image pooling 브랜치
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        # conv36 — concat projection
        self.project = nn.Sequential(
            nn.Conv2d(2 * cout, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.shape[-2:]
        b = self.branch(x)
        p = self.pool(x)
        # 공식 HN: method=nearest_neighbor (1x1 -> 17x17). bilinear 로 하면 DFC 가
        # 다른 레이어로 파싱해서 구조가 어긋난다.
        p = F.interpolate(p, size=(h, w), mode="nearest")
        # 공식 concat1 의 입력 순서가 [resize1, conv35] 이므로 pooling 을 앞에 둔다.
        return self.project(torch.cat([p, b], dim=1))


class DeepLabV3MNv2(nn.Module):
    """MobileNetV2(OS=32, dilation 없음) + 헤드 + bilinear 업샘플 (+ export 시 argmax).

    [헤드 선택 — head 인자]
      "official" (기본, 2026-09-09 신규):
          백본을 **features[:18] (320ch) 에서 자르고** OfficialHead 를 붙인다.
          공식 HEF 의 HN 을 실측해서 1:1 로 복원한 구성 -> 약 2.11M 파라미터로
          공식 2.10M 과 사실상 동일(차이는 클래스 수 19 vs 21 뿐).
          latency/컨텍스트 비교가 성립하는 유일한 구성이다.

      "simple" (구버전 기본 — **공식 구성이 아니다**):
          백본 전체(1280ch) -> Conv2d(1280, num_classes, 1) 하나. 약 2.25M.
          [2026-09-09 정정] 이전 주석은 이 구성을 "공식 구성"이라고 적었으나 사실과
          다르다. 공식 HN 을 직접 뜯어보니 공식은 (a) 백본을 320ch 에서 자르고
          (b) image pooling 브랜치 + concat projection 을 갖는다. simple 은 둘 다 없고
          대신 320->1280 conv(17x17 에서 약 118M MAC)를 그대로 안고 간다.
          꼬리 연산량이 공식보다 무거워 동등 비교가 성립하지 않는다.
          기존 체크포인트 호환용으로만 남겨 둔다.

      "aspp":
          3x3 atrous 브랜치 3개를 단 무거운 변형. 약 11.7M 로 공식 대비 5.6배.
          스케줄링 비교에 쓰면 안 된다. 참고용.

    export=True 로 두면 bilinear 업샘플 + argmax 까지 그래프에 포함한다(Hailo 규약).

    [주의] head 를 바꾸면 가중치 shape 이 달라져 기존 체크포인트를 못 읽는다.
           official 로 가려면 처음부터 재학습해야 한다.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, export=False, head="official"):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        w = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        feats = mobilenet_v2(weights=w).features
        self.head = head
        if head == "official":
            # features[18] = ConvBNReLU(320->1280). 공식은 이걸 쓰지 않는다.
            self.backbone = feats[:18]                     # 출력 320ch, stride 32
            self.aspp = OfficialHead(320, 256)
            self.classifier = nn.Conv2d(256, num_classes, 1)
        elif head == "aspp":
            self.backbone = feats
            self.aspp = ASPP(1280)
            self.classifier = nn.Conv2d(256, num_classes, 1)
        else:                                              # "simple" (구버전)
            self.backbone = feats
            self.aspp = None
            self.classifier = nn.Conv2d(1280, num_classes, 1)
        self.export = export

    def forward(self, x):
        f = self.backbone(x)
        if self.aspp is not None:
            f = self.aspp(f)
        logits = self.classifier(f)
        # [2026-09-09] align_corners=False -> True.
        # False(=ONNX half_pixel)면 배율 513/17=30.176 이 한 단계로 남아 DFC 가 분해하지
        # 못하고 microcode 초과로 컴파일이 실패한다. 공식 HN 은 align_corners 이고
        # DFC 가 [15.0, 2.0118] 두 단계로 분해한다.
        logits = F.interpolate(logits, size=(IMG, IMG), mode="bilinear", align_corners=True)
        if self.export:
            return torch.argmax(logits, dim=1, keepdim=True).to(torch.int32)  # N,1,513,513
        return logits


# ───────────────────────────── mIoU ─────────────────────────────
class ConfMat:
    def __init__(self, n=NUM_CLASSES):
        self.n = n
        self.m = torch.zeros((n, n), dtype=torch.int64)

    def update(self, pred, gt):
        k = (gt >= 0) & (gt < self.n)
        idx = self.n * gt[k].to(torch.int64) + pred[k].to(torch.int64)
        self.m += torch.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n).cpu()

    def miou(self):
        m = self.m.float()
        inter = m.diag()
        union = m.sum(0) + m.sum(1) - inter
        iou = inter / union.clamp(min=1)
        valid = union > 0
        return iou[valid].mean().item() * 100, (iou * 100).tolist()


# ───────────────────────────── 학습 / 평가 ─────────────────────────────
@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    cm = ConfMat()
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        cm.update(out.float().argmax(1).cpu(), y)
    miou, per = cm.miou()
    return miou, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Cityscapes 루트 (gtFine/ 와 leftImg8bit/ 의 상위 폴더)")
    ap.add_argument("--out", default="./runs/dlv3_mnv2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4, help="RTX 5060 8GB 기준 4~6 권장")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--head", choices=["official", "simple", "aspp"], default="official",
                    help="official=공식 HN 복원(~2.11M, 공식 2.10M과 동등 — 이걸 쓸 것) / "
                         "simple=구버전(~2.25M, 공식 구성 아님) / aspp=무거운 변형(~11.7M)")
    ap.add_argument("--resume", action="store_true", help="out/last.pt 에서 이어서")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--export-onnx", metavar="PATH")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"CC={torch.cuda.get_device_capability(0)}  torch={torch.__version__}")

    # ── ONNX export 전용 경로 (데이터셋 불필요) ──
    if a.export_onnx:
        model = DeepLabV3MNv2(pretrained=False, export=True, head=a.head)
        ck = torch.load(os.path.join(a.out, "best.pt"), map_location="cpu")
        model.load_state_dict(ck["model"])
        model.eval()
        dummy = torch.randn(1, 3, IMG, IMG)
        torch.onnx.export(
            model, dummy, a.export_onnx,
            input_names=["input"], output_names=["ArgMax"],
            opset_version=13, do_constant_folding=True,
            dynamic_axes=None)            # 고정 배치/해상도 — Hailo 는 동적 shape 미지원
        print(f"ONNX 저장: {a.export_onnx}  (mIoU={ck.get('miou'):.2f} 체크포인트)")
        print("\n다음 단계 (WSL, DFC 3.33):")
        print(f"  python -m onnxsim {a.export_onnx} deeplab_cs_513_sim.onnx")
        print("  python compile_hailo.py --model deeplab_cs_513_sim.onnx --size 513 \\")
        print("      --arch hailo8l --name deeplab_v3_mnv2_wo_dilation_cityscapes \\")
        print("      --calib-dir <Cityscapes>/leftImg8bit/train --calib-n 512 \\")
        print("      --alls deeplab.alls --out deeplab_v3_mnv2_wo_dilation_cityscapes.hef")
        return

    if not a.data:
        raise SystemExit("--data 가 필요합니다.")

    print("데이터셋 로딩")
    tr = Cityscapes(a.data, "train", train=True)
    va = Cityscapes(a.data, "val", train=False)
    trl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                     pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    val = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                     pin_memory=True)

    model = DeepLabV3MNv2(pretrained=True, head=a.head).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"헤드={a.head}  파라미터={n_par/1e6:.2f}M  "
          f"(공식 deeplab_v3_mobilenet_v2_wo_dilation = 2.10M)")
    if n_par > 4e6:
        print("  [경고] 공식 대비 파라미터가 크게 많습니다 — latency/컨텍스트 비교가 성립하지 않습니다.")
    # 백본은 ImageNet 사전학습이므로 lr 을 1/10 로
    opt = torch.optim.SGD([
        {"params": model.backbone.parameters(), "lr": a.lr * 0.1},
        {"params": (list(model.aspp.parameters()) if model.aspp is not None else [])
                   + list(model.classifier.parameters()),
         "lr": a.lr},
    ], momentum=0.9, weight_decay=1e-4, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE)

    start_ep, best = 0, 0.0
    last_p = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start_ep = ck["epoch"] + 1; best = ck["best"]
        print(f"재개: epoch {start_ep} 부터, best mIoU={best:.2f}")

    if a.eval_only:
        miou, per = evaluate(model, val, dev)
        print(f"mIoU = {miou:.2f}")
        for n, v in zip(CLASS_NAMES, per):
            print(f"  {n:14s} {v:5.1f}")
        return

    total_steps = a.epochs * len(trl)
    print(f"학습 시작: {a.epochs} epoch x {len(trl)} step = {total_steps} step, batch={a.batch}")
    for ep in range(start_ep, a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, y) in enumerate(trl):
            step = ep * len(trl) + i
            lr_scale = (1 - step / total_steps) ** 0.9          # poly decay
            for g, base in zip(opt.param_groups, [a.lr * 0.1, a.lr]):
                g["lr"] = base * lr_scale
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item()
            if i % 50 == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(trl) - i - 1)
                print(f"  ep{ep+1}/{a.epochs} {i+1}/{len(trl)} "
                      f"loss={run/(i+1):.4f} lr={opt.param_groups[1]['lr']:.5f} "
                      f"eta={eta/60:.1f}m", flush=True)

        miou, _ = evaluate(model, val, dev)
        print(f"[epoch {ep+1}] loss={run/len(trl):.4f}  val mIoU={miou:.2f}  "
              f"({(time.time()-t0)/60:.1f}분)")
        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "scaler": scaler.state_dict(), "epoch": ep, "best": max(best, miou),
              "miou": miou}
        torch.save(ck, last_p)
        if miou > best:
            best = miou
            torch.save(ck, os.path.join(a.out, "best.pt"))
            print(f"  -> best 갱신: {best:.2f}")

    print(f"\n완료. best mIoU = {best:.2f}")
    print(f"ONNX export: python {os.path.basename(__file__)} --out {a.out} "
          f"--export-onnx deeplab_cs_513.onnx")


if __name__ == "__main__":
    main()
