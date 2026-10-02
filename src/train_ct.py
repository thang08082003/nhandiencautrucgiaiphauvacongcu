"""Bước 3 (CT): 3D U-Net nhỏ (MONAI), patch 3D, loss Dice+CE, dừng theo ngân sách thời gian.
Đánh giá trên tập test ở KHÔNG GIAN ẢNH GỐC (resample dự đoán về lưới gốc) để so được với TotalSegmentator.
python train_ct.py --budget-min 30
"""
import argparse
import json
import math
import time

import matplotlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

from common import CACHE, RESULTS, SEED, eval_case, metrics_table, overlay, rows_df, show_rows, start
from data_ct import HU_HI, HU_LO, NAMES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CT = CACHE / "ct"
PATCH = (96, 96, 96)
NCLS = len(NAMES) + 1
EPOCH_ITERS = 50  # "epoch giả" = 50 bước, chỉ để vẽ đường cong


def build_model():
    return UNet(3, 1, NCLS, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="instance")


def split_ids(s):
    d = pd.read_csv(RESULTS / "splits" / "ct_split.csv")
    return d[d.split == s].case.tolist()


def load(cid):
    return np.load(CT / f"{cid}_img.npy"), np.load(CT / f"{cid}_lab.npy")


class Sampler:
    """Cắt patch: 2/3 số lần tâm tại một voxel của nhãn ngẫu nhiên, 1/3 ngẫu nhiên hoàn toàn."""

    def __init__(self, ids, rng):
        self.rng, self.cases = rng, []
        for c in ids:
            x, y = load(c)
            fg = {k: np.argwhere(y == k)[::50] for k in range(1, NCLS)}
            self.cases.append((x, y, {k: v for k, v in fg.items() if len(v)}))

    def one(self):
        x, y, fg = self.cases[self.rng.integers(len(self.cases))]
        ps, shp = np.array(PATCH), np.array(x.shape)
        if fg and self.rng.random() < 2 / 3:
            k = self.rng.choice(list(fg))
            ctr = fg[k][self.rng.integers(len(fg[k]))]
        else:
            ctr = self.rng.integers(0, shp)
        st = np.clip(ctr - ps // 2, 0, np.maximum(shp - ps, 0))
        sl = tuple(slice(s, s + p) for s, p in zip(st, ps))
        xi, yi = x[sl].astype(np.float32), y[sl]
        pad = [(0, p - s) for p, s in zip(ps, xi.shape)]
        xi, yi = np.pad(xi, pad), np.pad(yi, pad)
        xi = xi * self.rng.uniform(.9, 1.1) + self.rng.uniform(-.05, .05)
        return xi[None], yi[None]

    def batch(self, n, dev):
        b = [self.one() for _ in range(n)]
        return (torch.from_numpy(np.stack([i[0] for i in b])).to(dev),
                torch.from_numpy(np.stack([i[1] for i in b]).astype(np.int64)).to(dev))


@torch.no_grad()
def infer(model, x, dev, amp=False):  # fp16 tràn số -> logits NaN -> argmax 0 -> Dice 0 âm thầm
    model.eval()
    t = torch.from_numpy(x.astype(np.float32))[None, None].to(dev)
    with torch.autocast("cuda", enabled=amp and dev == "cuda"):
        o = sliding_window_inference(t, PATCH, 4, model, overlap=0.25)
    return o.argmax(1)[0].byte().cpu().numpy()


def val_dice(model, ids, dev):
    inter, ps, gs = np.zeros(NCLS), np.zeros(NCLS), np.zeros(NCLS)
    for c in ids:
        x, y = load(c)
        p = infer(model, x, dev)
        for k in range(1, NCLS):
            inter[k] += ((p == k) & (y == k)).sum()
            ps[k] += (p == k).sum()
            gs[k] += (y == k).sum()
    m = gs[1:] > 0
    return float(np.mean(2 * inter[1:][m] / (ps[1:][m] + gs[1:][m])))


def to_orig(pred, cid):
    g = json.load(open(CT / f"{cid}_geom.json"))
    im = sitk.GetImageFromArray(pred)
    im.SetSpacing(g["spacing"]), im.SetOrigin(g["origin"]), im.SetDirection(g["direction"])
    ref = sitk.ReadImage(str(CT / "test_raw" / f"{cid}_lab.nii.gz"))
    return sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8), ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=30)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-min", type=float, default=4, help="chu kỳ validation (phút)")
    a = ap.parse_args()
    start()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = RESULTS / "ct"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (CT / "ckpt").mkdir(exist_ok=True)
    (CT / "pred_custom").mkdir(exist_ok=True)
    tr, va, te = split_ids("train"), split_ids("val"), split_ids("test")
    sampler = Sampler(tr, np.random.default_rng(SEED))
    model = build_model().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # AMP tắt: fp16 forward tràn số -> loss NaN (đo trên T4, iter ~500)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    budget, t0, last_val, it, best, best_ep, losses, log = a.budget_min * 60, time.time(), time.time(), 0, -1.0, 0, [], []
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        x, y = sampler.batch(a.bs, dev)
        with torch.autocast("cuda", enabled=False):
            loss = loss_fn(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        it += 1
        if it % EPOCH_ITERS == 0:
            row = dict(epoch=it // EPOCH_ITERS, iters=it, elapsed_s=round(time.time() - t0), lr=lr,
                       train_loss=np.mean(losses), val_dice=np.nan)
            losses = []
            if time.time() - last_val > a.val_min * 60:
                row["val_dice"] = val_dice(model, va, dev)
                last_val = time.time()
                if row["val_dice"] > best:
                    best, best_ep = row["val_dice"], row["epoch"]
                    torch.save(model.state_dict(), CT / "ckpt" / "ct_best.pt")
            log.append(row)
            print(row, flush=True)
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    vd = val_dice(model, va, dev)  # validation cuối
    if vd > best:
        best, best_ep = vd, (it // EPOCH_ITERS) + 1
        torch.save(model.state_dict(), CT / "ckpt" / "ct_best.pt")
    log.append(dict(epoch=it // EPOCH_ITERS + 1, iters=it, elapsed_s=round(time.time() - t0), lr=0, train_loss=np.mean(losses) if losses else np.nan, val_dice=vd))
    L = pd.DataFrame(log)
    L.to_csv(out / "train_log.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(L.epoch, L.train_loss)
    ax[0].set(title=f"train loss (Dice+CE), epoch = {EPOCH_ITERS} bước", xlabel="epoch giả")
    v = L.dropna(subset=["val_dice"])
    ax[1].plot(v.epoch, v.val_dice, "o-")
    ax[1].set(title="val Dice (1.5mm, trung bình nhãn)", xlabel="epoch giả")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)

    # ---- test: không gian ảnh gốc
    model.load_state_dict(torch.load(CT / "ckpt" / "ct_best.pt"))
    per, secs = {}, {}
    for c in te:
        x, _ = load(c)
        t = time.time()
        p = infer(model, x, dev)
        sitk_p, ref = to_orig(p, c)
        secs[c] = time.time() - t
        gt = sitk.GetArrayFromImage(ref)
        pr = sitk.GetArrayFromImage(sitk_p)
        sitk.WriteImage(sitk_p, str(CT / "pred_custom" / f"{c}.nii.gz"))
        per[c] = eval_case(pr, gt, NAMES, ref.GetSpacing()[::-1])
        print(c, f"{secs[c]:.1f}s", flush=True)
    df = rows_df(per)
    df["seconds"] = df.case.map(secs)
    df.to_csv(out / "test_per_case.csv", index=False)
    t = metrics_table(df)
    t.to_csv(out / "metrics.csv", index=False)
    print(t)

    sc = df.groupby("case")["dice"].mean().sort_values()
    rows = []
    for c in [sc.index[-1], sc.index[0]]:
        im = sitk.GetArrayFromImage(sitk.ReadImage(str(CT / "test_raw" / f"{c}_img.nii.gz"))).astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(CT / "test_raw" / f"{c}_lab.nii.gz")))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(str(CT / "pred_custom" / f"{c}.nii.gz")))
        z = int(np.argmax((gt > 0).sum((1, 2))))
        w = np.clip((im[z] - HU_LO) / (HU_HI - HU_LO), 0, 1)
        rows.append([w, overlay(w, gt[z], NCLS), overlay(w, pr[z], NCLS)])
    show_rows(out / "figures" / "custom_pred_vs_gt.png", rows, ["CT", "ground truth", "3D U-Net tự huấn luyện"],
              f"Hàng 1: ca tốt nhất ({sc.index[-1]}, Dice {sc.iloc[-1]:.2f}); hàng 2: ca tệ nhất ({sc.index[0]}, Dice {sc.iloc[0]:.2f})")

    json.dump(dict(model="MONAI UNet 3D", channels=(16, 32, 64, 128, 256), res_units=2, norm="instance", params=n_params,
                   patch=PATCH, spacing_mm=1.5, hu_window=(HU_LO, HU_HI), loss="DiceCE", lr=a.lr, batch=a.bs, seed=SEED,
                   budget_min=a.budget_min, iterations=it, best_epoch=best_ep, best_val_dice_1p5mm=best,
                   n_train=len(tr), n_val=len(va), n_test=len(te), device=dev,
                   gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", torch=torch.__version__,
                   augment="chỉ nhiễu cường độ (không lật vì trái/phải liên quan tới nhãn thận)",
                   inference_seconds_note="gồm sliding window + resample về lưới gốc, chưa gồm tiền xử lý"),
              open(out / "config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
