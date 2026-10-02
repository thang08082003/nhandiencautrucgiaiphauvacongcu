"""Tiện ích dùng chung: seed, log lệnh, metrics (Dice/IoU/HD95), đo tốc độ, vẽ overlay."""
import datetime
import os
import subprocess
import sys
import time
import random
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "cache"))


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def log_cmd(cmd=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    cmd = cmd or " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    with open(RESULTS / "commands.log", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{cmd}\n")


def start():
    """Gọi đầu mỗi script: seed 42 + ghi lệnh vào results/commands.log."""
    seed_all()
    log_cmd()


def sh(cmd):
    log_cmd(cmd)
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def write_kv(path, d):
    import pandas as pd
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"key": list(d), "value": [str(v) for v in d.values()]}).to_csv(path, index=False)


# ---------------------------------------------------------------- metrics
def _bbox(m, pad=2):
    return tuple(slice(max(i.min() - pad, 0), i.max() + pad + 1) for i in np.nonzero(m))


def hd95(p, g, spacing):
    """HD95 giữa hai mặt nạ nhị phân (đơn vị theo spacing). NaN nếu một trong hai rỗng."""
    from scipy import ndimage as ndi
    if not p.any() or not g.any():
        return np.nan
    sl = _bbox(p | g)
    p, g = p[sl], g[sl]
    sp, sg = p & ~ndi.binary_erosion(p), g & ~ndi.binary_erosion(g)
    dg = ndi.distance_transform_edt(~sg, sampling=spacing)
    dp = ndi.distance_transform_edt(~sp, sampling=spacing)
    return float(np.percentile(np.concatenate([dg[sp], dp[sg]]), 95))


def eval_case(pred, gt, names, spacing, hd=True):
    """names: {id: tên}. Nhãn vắng trong GT -> NaN (bỏ qua). GT có mà pred rỗng -> Dice=0, HD95=NaN."""
    rows = {}
    for k, n in names.items():
        p, g = pred == k, gt == k
        if not g.any():
            rows[n] = (np.nan, np.nan, np.nan)
            continue
        inter = (p & g).sum()
        rows[n] = (2 * inter / (p.sum() + g.sum()), inter / (p.sum() + g.sum() - inter),
                   hd95(p, g, spacing) if hd else np.nan)
    return rows


def rows_df(per_case):
    """per_case: {case: eval_case(...)} -> DataFrame(case,label,dice,iou,hd95)."""
    import pandas as pd
    return pd.DataFrame([dict(case=c, label=n, dice=v[0], iou=v[1], hd95=v[2])
                         for c, r in per_case.items() for n, v in r.items()])


def metrics_table(df):
    t = df.groupby("label", sort=False)[["dice", "iou", "hd95"]].mean()
    t["dice_std"] = df.groupby("label", sort=False)["dice"].std()
    t["n_cases"] = df.dropna(subset=["dice"]).groupby("label", sort=False).size()
    t.loc["MEAN"] = [t["dice"].mean(), t["iou"].mean(), t["hd95"].mean(), np.nan, np.nan]
    return t.reset_index()


def mean_dice(df):
    return float(df.groupby("label")["dice"].mean().mean())


# ---------------------------------------------------------------- tốc độ
def bench(fn, warm=10, n=100):
    import torch
    cuda = torch.cuda.is_available()
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    return dict(warmup=warm, runs=n, mean_ms=ts.mean(), p95_ms=np.percentile(ts, 95), fps=1000 / ts.mean(),
                vram_peak_mb=torch.cuda.max_memory_allocated() / 2**20 if cuda else np.nan)


# ---------------------------------------------------------------- hình
def overlay(img, mask, n_cls, alpha=0.5):
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    rgb = np.stack([img] * 3, -1) if img.ndim == 2 else img
    rgb = rgb.astype(float)
    if rgb.max() > 1:
        rgb = rgb / 255
    out = rgb.copy()
    for c in range(1, n_cls):
        m = mask == c
        out[m] = (1 - alpha) * rgb[m] + alpha * np.array(cmap(c % 20)[:3])
    return out


def show_rows(path, rows, titles, suptitle=None):
    """rows: list các hàng, mỗi hàng là list ảnh (cùng số cột với titles)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(len(rows), len(titles), figsize=(3.2 * len(titles), 3.2 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            ax[i, j].imshow(im, cmap="gray")
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(titles[j])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
