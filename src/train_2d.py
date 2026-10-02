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

from common import CACHE, RESULTS, SEED, metrics_table, overlay, show_rows, start
from lib2d import ENC, RDIR, build, evaluate, load_cache, meta, predict, prep, val_dice

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def augment(x, y):
    """Affine ngẫu nhiên trên GPU (xoay ±10°, scale ±10%, dịch ±5%) + đổi độ sáng/tương phản."""
    B, d = x.shape[0], x.device
    a = (torch.rand(B, device=d) - .5) * 2 * math.radians(10)
    s = 1 + (torch.rand(B, device=d) - .5) * .2
    th = torch.zeros(B, 2, 3, device=d)
    th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1] = torch.cos(a) / s, -torch.sin(a) / s, torch.sin(a) / s, torch.cos(a) / s
    th[:, :, 2] = (torch.rand(B, 2, device=d) - .5) * .1
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, align_corners=False)
    y = F.grid_sample(y[:, None].float(), g, mode="nearest", align_corners=False)[:, 0].long()
    x = x * (1 + (torch.rand(B, 1, 1, 1, device=d) - .5) * .4) + (torch.rand(B, 1, 1, 1, device=d) - .5) * .4
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(ENC))
    ap.add_argument("--budget-min", type=float, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    start()
    name, dev = a.name, "cuda" if torch.cuda.is_available() else "cpu"
    names = meta(name)["names"]
    K = len(names)
    out = RESULTS / RDIR[name]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ck = CACHE / "ckpt"
    ck.mkdir(exist_ok=True)

    Xtr, Ytr = load_cache(name, "train")
    Xva, Yva = load_cache(name, "val")
    Xte, Yte = load_cache(name, "test")
    Xg, Yg = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
    model = build(name).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    dice_l, ce_l = smp.losses.DiceLoss("multiclass", from_logits=True), nn.CrossEntropyLoss()

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
            x, y = augment(prep(Xg[idx]), Yg[idx].long())
            with torch.autocast("cuda", enabled=dev == "cuda"):
                o = model(x)
                loss = dice_l(o, y) + ce_l(o.float(), y)
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
            torch.save(model.state_dict(), ck / f"{name}_best.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)  # lưu dần, phòng bị ngắt

    L = pd.DataFrame(log)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(L.epoch, L.train_loss)
    ax[0].set(title="train loss (Dice+CE)", xlabel="epoch")
    ax[1].plot(L.epoch, L.val_dice)
    ax[1].set(title="val Dice (trung bình nhãn)", xlabel="epoch")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)

    model.load_state_dict(torch.load(ck / f"{name}_best.pt"))
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

    json.dump(dict(name=name, encoder=ENC[name], pretrained="imagenet", params=n_params, input=f"{Xtr.shape[1]}x{Xtr.shape[2]}",
                   loss="Dice+CE", optimizer="AdamW", lr=a.lr, lr_schedule="cosine theo thời gian", batch=a.bs, seed=SEED,
                   budget_min=a.budget_min, epochs_done=ep, best_epoch=best_ep, best_val_dice=best,
                   n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte), device=dev,
                   gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                   torch=torch.__version__, smp=smp.__version__, augment="affine+brightness, không lật"),
              open(out / "config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
