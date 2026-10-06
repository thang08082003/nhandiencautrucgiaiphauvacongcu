"""Tuần 3-4 (CAMUS / CholecSeg8k): phân tích lỗi, bảng ablation, ensemble.
python analyze_2d.py errors   --name camus [--tag <tag>]
python analyze_2d.py ablation --name camus
python analyze_2d.py ensemble --name camus --tags a b c
Tag rỗng = mô hình gốc của tuần 3 (train_2d.py không --tag).
"""
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
import torch

from common import CACHE, RESULTS, metrics_table, start
from lib2d import RDIR, build, evaluate, load_cache, meta, prep

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rdir(name, tag):
    return RESULTS / RDIR[name] / ("ablation/" + tag if tag else "")


def load_model(name, tag, dev):
    cfg = json.load(open(rdir(name, tag) / "config.json"))
    m = build(name, False, cfg["encoder"], cfg.get("arch", "Unet")).to(dev)
    ck = f"{name}_{tag}" if tag else name
    m.load_state_dict(torch.load(CACHE / "ckpt" / f"{ck}_best.pt", map_location=dev))
    return m.eval()


@torch.no_grad()
def probs(models, X, dev, bs=32):
    """Softmax trung bình của các mô hình (1 mô hình = dự đoán thường)."""
    out = []
    for i in range(0, len(X), bs):
        x = prep(torch.from_numpy(X[i:i + bs]).to(dev))
        out.append(torch.stack([m(x).float().softmax(1) for m in models]).mean(0).argmax(1).byte().cpu().numpy())
    return np.concatenate(out)


def errors(a, dev):
    names, (X, Y) = meta(a.name)["names"], load_cache(a.name, "test")
    out = rdir(a.name, a.tag) / "errors"
    out.mkdir(parents=True, exist_ok=True)
    P = probs([load_model(a.name, a.tag, dev)], X, dev)
    K = len(names)
    # ma trận nhầm lẫn pixel (hàng = GT, cột = dự đoán), chuẩn hoá theo hàng
    cm = np.bincount(Y.astype(np.int64).ravel() * K + P.astype(np.int64).ravel(), minlength=K * K).reshape(K, K)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    pd.DataFrame(cmn, index=names, columns=names).round(4).to_csv(out / "confusion_norm.csv")
    fig, ax = plt.subplots(figsize=(0.6 * K + 3, 0.6 * K + 2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(K), yticks=range(K), xticklabels=names, yticklabels=names, xlabel="dự đoán", ylabel="ground truth")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=110)
    plt.close(fig)
    # lỗi lớn nhất: với mỗi nhãn GT, nhãn nào nó hay bị nhầm thành (trừ chính nó)
    rows = []
    df = evaluate(P, Y, names)
    df.to_csv(out / "test_per_image_full.csv", index=False)
    for k in range(1, K):
        off = cmn[k].copy()
        off[k] = 0
        j = int(off.argmax())
        d = df[df.label == names[k]]
        area = np.array([(y == k).sum() for y in Y])
        ok = ~np.isnan(d.dice.values)
        rows.append(dict(label=names[k], dice=d.dice.mean(), nsd=d.nsd.mean(), recall=cmn[k, k], most_confused_with=names[j],
                         confused_frac=off[j], frac_images_dice_lt_0p5=float((d.dice < .5).mean()),
                         frac_missed_entirely=float((d.dice == 0).mean())))
    E = pd.DataFrame(rows)
    # tương quan diện tích nhãn - Dice (nhãn nhỏ có kém hơn không?)
    corr = []
    for k in range(1, K):
        g = (Y == k).reshape(len(Y), -1).sum(1)
        d = df[df.label == names[k]].set_index("case").dice.reindex(range(len(Y)))
        m = ~np.isnan(d.values)
        corr.append(float(np.corrcoef(g[m], d.values[m])[0, 1]) if m.sum() > 2 else np.nan)
    E["corr_area_dice"] = corr
    E.round(3).to_csv(out / "error_summary.csv", index=False)
    print(E.round(3).to_string(index=False))
    # 8 ảnh tệ nhất
    per = df.groupby("case").dice.mean().dropna().sort_values()
    per.head(8).to_csv(out / "worst8.csv")
    print("Ảnh tệ nhất:", per.head(8).round(3).to_dict())


def ablation(a, dev):
    base = RESULTS / RDIR[a.name]
    cands = [("baseline (tuần 3)", base)] + [(p.name, p) for p in sorted((base / "ablation").glob("*")) if p.is_dir()]
    rows = []
    for n, p in cands:
        f, c = p / "metrics.csv", p / "config.json"
        if not (f.exists() and c.exists()):
            continue
        t, cfg = pd.read_csv(f).set_index("label"), json.load(open(c))
        rows.append(dict(config=n, arch=cfg.get("arch", "Unet"), encoder=cfg["encoder"], pretrained=cfg["pretrained"], loss=cfg["loss"],
                         augment=cfg["augment"], seed=cfg["seed"], params_M=round(cfg["params"] / 1e6, 2),
                         dice=t.loc["MEAN", "dice"], iou=t.loc["MEAN", "iou"], hd95=t.loc["MEAN", "hd95"],
                         nsd=t.loc["MEAN", "nsd"] if "nsd" in t else np.nan, best_val=cfg["best_val_dice"]))
    R = pd.DataFrame(rows)
    R["d_dice_vs_base"] = R.dice - R.dice.iloc[0]
    R.round(4).to_csv(base / "ablation_table.csv", index=False)
    print(R.round(4).to_markdown(index=False))


def ensemble(a, dev):
    names, (X, Y) = meta(a.name)["names"], load_cache(a.name, "test")
    P = probs([load_model(a.name, t, dev) for t in a.tags], X, dev)
    out = RESULTS / RDIR[a.name] / "ablation" / ("ens_" + "+".join(a.tags))
    out.mkdir(parents=True, exist_ok=True)
    t = metrics_table(evaluate(P, Y, names))
    t.to_csv(out / "metrics.csv", index=False)
    json.dump(dict(name=a.name, arch="ensemble", encoder="+".join(a.tags), pretrained="-", params=0, loss="-", augment="-",
                   seed=-1, best_val_dice=float("nan")), open(out / "config.json", "w"))
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["errors", "ablation", "ensemble"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tags", nargs="*", default=[])
    a = ap.parse_args()
    start()
    {"errors": errors, "ablation": ablation, "ensemble": ensemble}[a.cmd](a, "cuda" if torch.cuda.is_available() else "cpu")
