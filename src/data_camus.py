"""Bước 1B: CAMUS. Bạn tự đăng ký + tải + giải nén (xem notebook), rồi chạy:
python data_camus.py --src /content/CAMUS_public
"""
import argparse
import re
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

from common import CACHE, RESULTS, SEED, overlay, show_rows, start, write_kv

NAMES = ["background", "LV_endo", "myocardium", "LA"]
SIZE = 256


def official_split(src):
    d = {}
    for key, fn in (("train", "subgroup_training"), ("val", "subgroup_validation"), ("test", "subgroup_testing")):
        f = next(Path(src).rglob(fn + ".txt"), None)
        if f is None:
            return None
        d[key] = set(re.findall(r"patient\d+", f.read_text()))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    start()
    root = next(Path(a.src).rglob("database_nifti"))
    items = []
    for pd_ in sorted(d for d in root.iterdir() if d.is_dir() and re.fullmatch(r"patient\d+", d.name)):
        p = pd_.name
        for v in ("2CH", "4CH"):
            cfg = pd_ / f"Info_{v}.cfg"
            m = re.search(r"ImageQuality\s*:\s*(\w+)", cfg.read_text()) if cfg.exists() else None
            for ph in ("ED", "ES"):
                im, gt = pd_ / f"{p}_{v}_{ph}.nii.gz", pd_ / f"{p}_{v}_{ph}_gt.nii.gz"
                if im.exists() and gt.exists():
                    items.append((p, v, ph, m.group(1) if m else "NA", im, gt))
    patients = sorted({i[0] for i in items})
    assert items, "Không thấy cặp ảnh/nhãn"

    sp = official_split(a.src)
    if sp:
        split_src = "chia chính thức của CAMUS (subgroup_*.txt)"
    else:
        split_src = "KHÔNG thấy file chia chính thức -> tự chia theo bệnh nhân, seed 42, 70/15/15"
        perm = np.random.default_rng(SEED).permutation(patients)
        ntr, nva = round(0.7 * len(perm)), round(0.15 * len(perm))
        sp = {"train": set(perm[:ntr]), "val": set(perm[ntr:ntr + nva]), "test": set(perm[ntr + nva:])}
    p2s = {p: s for s, ps in sp.items() for p in ps}
    print("Cách chia:", split_src)

    data = {s: ([], [], []) for s in sp}
    stats, tot = [], np.zeros(4)
    for p, v, ph, q, im, gt in items:
        if p not in p2s:
            continue
        ii, gi = nib.load(str(im)), nib.load(str(gt))
        x = np.asarray(ii.dataobj).astype(np.uint8).T
        y = np.asarray(gi.dataobj).astype(np.uint8).T
        cnt = np.bincount(y.ravel(), minlength=4)[:4]
        tot += cnt
        stats.append(dict(patient=p, view=v, phase=ph, quality=q, split=p2s[p], height=x.shape[0], width=x.shape[1],
                          spacing_mm=tuple(round(float(z), 3) for z in ii.header.get_zooms()[:2]),
                          **{f"pct_{n}": round(100 * c / y.size, 2) for n, c in zip(NAMES, cnt)}))
        X, Y, idx = data[p2s[p]]
        X.append(cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        Y.append(cv2.resize(y, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))
        idx.append((p, v, ph, q))

    out = CACHE / "camus"
    out.mkdir(parents=True, exist_ok=True)
    for s, (X, Y, idx) in data.items():
        if not X:
            print(f"CẢNH BÁO: tập '{s}' rỗng (chưa tải đủ bệnh nhân?), bỏ qua")
            continue
        np.savez_compressed(out / f"{s}.npz", X=np.stack(X), Y=np.stack(Y))
        pd.DataFrame(idx, columns=["patient", "view", "phase", "quality"]).to_csv(out / f"{s}_index.csv", index=False)
    import json
    json.dump(dict(names=NAMES, in_ch=1, size=SIZE), open(out / "meta.json", "w"))

    df = pd.DataFrame(stats)
    r = RESULTS / "camus"
    r.mkdir(parents=True, exist_ok=True)
    df.to_csv(r / "dataset_stats.csv", index=False)
    (RESULTS / "splits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient": sorted(p2s), "split": [p2s[p] for p in sorted(p2s)]}).to_csv(
        RESULTS / "splits" / "camus_split.csv", index=False)
    ov = dict(split_source=split_src, n_patients_total=len(patients), n_images=len(df),
              patients_per_split={s: len(sp[s] & set(patients)) for s in sp},
              images_per_split=df.split.value_counts().to_dict(),
              images_per_view_phase=df.groupby(["view", "phase"]).size().to_dict(),
              quality_counts=df.quality.value_counts().to_dict(), orig_sizes_unique=len(set(zip(df.height, df.width))),
              orig_size_example=f"{df.height.iloc[0]}x{df.width.iloc[0]}", resized_to=f"{SIZE}x{SIZE}",
              **{f"pixel_pct_{n}": round(100 * c / tot.sum(), 2) for n, c in zip(NAMES, tot)})
    write_kv(r / "dataset_overview.csv", ov)

    X, Y, _ = data["train"]
    rows = [[X[i], overlay(X[i], Y[i], 4)] for i in (0, 1, 2)]
    show_rows(r / "figures" / "data_examples.png", rows, ["siêu âm (256x256)", "+ nhãn"])
    print("Xong bước dữ liệu CAMUS.")


if __name__ == "__main__":
    main()
