"""Bước 3 (CAMUS / CholecSeg8k): U-Net 2D (segmentation-models-pytorch), dừng theo ngân sách thời gian.
python train_2d.py --name camus --budget-min 30
"""
import argparse
import json
import math
import time

import matplotlib
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CACHE, RESULTS, SEED, metrics_table, overlay, seed_all, show_rows, start
from lib2d import ENC, RDIR, build, evaluate, load_cache, meta, predict, prep, val_dice

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AUG = {"none": None, "affine": (10, .2, .1), "strong": (20, .4, .2)}  # (xoay ±độ, scale ±, dịch ±)


def augment(x, y, level="affine"):
    """Affine ngẫu nhiên trên GPU (mặc định xoay ±10°, scale ±10%, dịch ±5%) + đổi độ sáng/tương phản."""
    if AUG[level] is None:
        return x, y
    rot, sc, tr = AUG[level]
    B, d = x.shape[0], x.device
    a = (torch.rand(B, device=d) - .5) * 2 * math.radians(rot)
    s = 1 + (torch.rand(B, device=d) - .5) * sc
    th = torch.zeros(B, 2, 3, device=d)
    th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1] = torch.cos(a) / s, -torch.sin(a) / s, torch.sin(a) / s, torch.cos(a) / s
    th[:, :, 2] = (torch.rand(B, 2, device=d) - .5) * tr
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, align_corners=False)
    y = F.grid_sample(y[:, None].float(), g, mode="nearest", align_corners=False)[:, 0].long()
    x = x * (1 + (torch.rand(B, 1, 1, 1, device=d) - .5) * .4) + (torch.rand(B, 1, 1, 1, device=d) - .5) * .4
    return x, y


def make_loss(kind):
    """Trả về hàm loss(logits, y). dice_ce là cấu hình tuần 2; dice_focal/ce dùng cho ablation."""
    dice = smp.losses.DiceLoss("multiclass", from_logits=True)
    if kind == "dice_ce":
        ce = nn.CrossEntropyLoss()
        return lambda o, y: dice(o, y) + ce(o.float(), y)
    if kind == "dice_focal":
        foc = smp.losses.FocalLoss("multiclass")
        return lambda o, y: dice(o, y) + foc(o, y)
    if kind == "ce":
        ce = nn.CrossEntropyLoss()
        return lambda o, y: ce(o.float(), y)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(ENC))
    ap.add_argument("--budget-min", type=float, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="dice_ce", choices=["dice_ce", "dice_focal", "ce"])
    ap.add_argument("--aug", default="affine", choices=list(AUG))
    ap.add_argument("--encoder", default=None, help="mặc định theo lib2d.ENC")
    ap.add_argument("--arch", default="Unet", choices=["Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN"])
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="", help="đặt tên một cấu hình ablation; kết quả vào results/<ds>/ablation/<tag>/")
    a = ap.parse_args()
    start()
    seed_all(a.seed)
    name, dev = a.name, "cuda" if torch.cuda.is_available() else "cpu"
    names = meta(name)["names"]
    K = len(names)
    out = RESULTS / RDIR[name] / ("ablation/" + a.tag if a.tag else "")
    ckname = f"{name}_{a.tag}" if a.tag else name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ck = CACHE / "ckpt"
    ck.mkdir(exist_ok=True)

    Xtr, Ytr = load_cache(name, "train")
    Xva, Yva = load_cache(name, "val")
    Xte, Yte = load_cache(name, "test")
    Xg, Yg = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
    model = build(name, not a.no_pretrain, a.encoder, a.arch).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    loss_fn = make_loss(a.loss)

    budget, t0, ep, best, best_ep, log = a.budget_min * 60, time.time(), 0, -1.0, 0, []
    while time.time() - t0 < budget:
        ep += 1
        perm, losses = torch.randperm(len(Xg), device=dev), []
        model.train()
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            frac = (time.time() - t0) / budget
            if frac >= 1:
                break
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * frac))  # cosine theo thời gian đã dùng
            for g in opt.param_groups:
                g["lr"] = lr
            idx = perm[i:i + a.bs]
            x, y = augment(prep(Xg[idx]), Yg[idx].long(), a.aug)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                o = model(x)
                loss = loss_fn(o, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        vd = val_dice(predict(model, Xva), Yva, K)
        log.append(dict(epoch=ep, elapsed_s=round(time.time() - t0), lr=lr, train_loss=np.mean(losses), val_dice=vd))
        print(log[-1], flush=True)
        if vd > best:
            best, best_ep = vd, ep
            torch.save(model.state_dict(), ck / f"{ckname}_best.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)  # lưu dần, phòng bị ngắt

    L = pd.DataFrame(log)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(L.epoch, L.train_loss)
    ax[0].set(title=f"train loss ({a.loss})", xlabel="epoch")
    ax[1].plot(L.epoch, L.val_dice)
    ax[1].set(title="val Dice (trung bình nhãn)", xlabel="epoch")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)

    model.load_state_dict(torch.load(ck / f"{ckname}_best.pt"))
    pred = predict(model, Xte)
    df = evaluate(pred, Yte, names)
    df.to_csv(out / "test_per_image.csv", index=False)
    t = metrics_table(df)
    t.to_csv(out / "metrics.csv", index=False)
    print(t)

    per = df.groupby("case")["dice"].mean().dropna().sort_values()
    pick = list(per.index[-2:][::-1]) + list(per.index[:2])
    tags = ["tốt", "tốt", "tệ", "tệ"]
    rows = [[Xte[i], overlay(Xte[i], Yte[i], K), overlay(Xte[i], pred[i], K)] for i in pick]
    show_rows(out / "figures" / "pred_vs_gt.png", rows, ["ảnh", "ground truth", "dự đoán"],
              "Hàng 1-2: ca tốt nhất; hàng 3-4: ca tệ nhất (Dice TB: " + ", ".join(f"{per[i]:.2f}" for i in pick) + ")")

    json.dump(dict(name=name, tag=a.tag, arch=a.arch, encoder=a.encoder or ENC[name],
                   pretrained=None if a.no_pretrain else "imagenet", params=n_params, input=f"{Xtr.shape[1]}x{Xtr.shape[2]}",
                   loss=a.loss, optimizer="AdamW", lr=a.lr, lr_schedule="cosine theo thời gian", batch=a.bs, seed=a.seed,
                   budget_min=a.budget_min, epochs_done=ep, best_epoch=best_ep, best_val_dice=best,
                   n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte), device=dev,
                   gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                   torch=torch.__version__, smp=smp.__version__, augment=f"{a.aug} + brightness, không lật"),
              open(out / "config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
