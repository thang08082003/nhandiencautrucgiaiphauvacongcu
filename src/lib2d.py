"""Dùng chung cho CAMUS và CholecSeg8k: nạp cache, dựng mô hình, suy luận, đánh giá."""
import json

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from common import CACHE, eval_case, rows_df

ENC = {"camus": "resnet18", "cholec": "mobilenet_v2"}
RDIR = {"camus": "camus", "cholec": "cholecseg8k"}  # tên thư mục trong results/


def load_cache(name, split):
    d = np.load(CACHE / name / f"{split}.npz")
    return d["X"], d["Y"]


def meta(name):
    return json.load(open(CACHE / name / "meta.json"))


def build(name, pretrained=True):
    m = meta(name)
    return smp.Unet(ENC[name], encoder_weights="imagenet" if pretrained else None,
                    in_channels=m["in_ch"], classes=len(m["names"]))


def prep(xb):
    """uint8 (B,H,W) hoặc (B,H,W,3) -> float (B,C,H,W), chuẩn hoá cố định."""
    x = xb.float() / 255
    x = x[:, None] if x.ndim == 3 else x.permute(0, 3, 1, 2)
    return (x - 0.5) / 0.25


@torch.no_grad()
def predict(model, X, scale=1.0, amp=False, bs=32):
    """Resize đầu vào theo `scale`, đưa logits về kích thước gốc rồi argmax."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = prep(torch.from_numpy(X[i:i + bs]).to(dev))
        H, W = x.shape[-2:]
        if scale != 1:
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        with torch.autocast("cuda", enabled=amp and dev.type == "cuda"):
            lo = model(x)
        if scale != 1:
            lo = F.interpolate(lo.float(), size=(H, W), mode="bilinear", align_corners=False)
        out.append(lo.argmax(1).byte().cpu().numpy())
    return np.concatenate(out)


def evaluate(pred, Y, names, hd=True):
    """Chỉ số theo từng ảnh và từng nhãn (bỏ nhãn 0 = nền). HD95 tính bằng pixel ở 256x256."""
    nm = {k: n for k, n in enumerate(names) if k > 0}
    return rows_df({i: eval_case(p, g, nm, (1, 1), hd) for i, (p, g) in enumerate(zip(pred, Y))})


def val_dice(pred, Y, k):
    """Dice gộp toàn tập (nhanh) cho theo dõi lúc huấn luyện; trung bình các nhãn có trong GT."""
    ds = []
    for c in range(1, k):
        g = Y == c
        if g.any():
            p = pred == c
            ds.append(2 * (p & g).sum() / (p.sum() + g.sum()))
    return float(np.mean(ds)) if ds else float("nan")
