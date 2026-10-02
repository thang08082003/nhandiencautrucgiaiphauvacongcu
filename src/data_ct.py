"""Bước 1A: AMOS (CT). Tải tập con bằng HTTP range (không tải cả zip), thống kê, chia theo ca, tiền xử lý.

Chạy: python data_ct.py --url <link amos22.zip> --n 40
Script in dung lượng tập con và DỪNG chờ bạn gõ 'yes' trước khi tải.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from common import CACHE, DATA, RESULTS, SEED, show_rows, overlay, start, write_kv

AMOS = {1: "spleen", 2: "right kidney", 3: "left kidney", 4: "gallbladder", 5: "esophagus", 6: "liver",
        7: "stomach", 8: "aorta", 9: "postcava", 10: "pancreas", 11: "right adrenal gland",
        12: "left adrenal gland", 13: "duodenum", 14: "bladder", 15: "prostate/uterus"}
# (id mới, tên, id AMOS, tên TotalSegmentator)
KEEP = [(1, "spleen", 1, "spleen"), (2, "kidney_right", 2, "kidney_right"), (3, "kidney_left", 3, "kidney_left"),
        (4, "gallbladder", 4, "gallbladder"), (5, "liver", 6, "liver"), (6, "pancreas", 10, "pancreas")]
NAMES = {k: n for k, n, _, _ in KEEP}
SPACING = (1.5, 1.5, 1.5)
HU_LO, HU_HI = -175.0, 250.0
OUT = DATA / "amos"
CT = CACHE / "ct"


def download(url, n):
    from remotezip import RemoteZip
    with RemoteZip(url) as z:
        infos = {i.filename: i for i in z.infolist()}
        img = {}
        for f in infos:
            m = re.search(r"imagesTr/amos_(\d{4})\.nii\.gz$", f)
            lab = f.replace("imagesTr", "labelsTr")
            if m and int(m.group(1)) < 500 and lab in infos:  # id < 500 = CT (MRI bắt đầu từ ~0507)
                img[int(m.group(1))] = (f, lab)
        ids = sorted(img)
        chosen = sorted(np.random.default_rng(SEED).permutation(ids)[:n].tolist())
        todo = [x for i in chosen for x in img[i]]
        mb = sum(infos[f].compress_size for f in todo) / 2**20
        print(f"AMOS CT có nhãn: {len(ids)} ca. Chọn {len(chosen)} ca (seed {SEED}): {chosen}")
        print(f"Dung lượng cần tải: {mb:.0f} MB. Nguồn: {url}. Giấy phép: kiểm tra trên trang AMOS/Zenodo (dự kiến CC BY 4.0).")
        if input("Gõ 'yes' để tải: ").strip().lower() != "yes":
            sys.exit("Đã huỷ theo yêu cầu.")
        for f in todo:
            if not (OUT / f).exists():
                z.extract(f, OUT)
                print("  tải xong", f, flush=True)


def resample(im, interp, default):
    size = [int(round(s * sp / t)) for s, sp, t in zip(im.GetSize(), im.GetSpacing(), SPACING)]
    return sitk.Resample(im, size, sitk.Transform(), interp, im.GetOrigin(), SPACING, im.GetDirection(),
                         default, sitk.sitkFloat32 if interp == sitk.sitkLinear else sitk.sitkUInt8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://zenodo.org/records/7262581/files/amos22.zip")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    start()
    if not a.skip_download:
        download(a.url, a.n)

    imgs = sorted(OUT.rglob("imagesTr/amos_*.nii.gz"))
    ids = [p.name[:-7] for p in imgs]
    assert ids, f"Không thấy ảnh trong {OUT}"
    order = np.random.default_rng(SEED).permutation(len(ids))
    ntr, nva = round(0.7 * len(ids)), round(0.15 * len(ids))
    split = {}
    for r, i in enumerate(order):
        split[ids[i]] = "train" if r < ntr else "val" if r < ntr + nva else "test"
    res = RESULTS / "splits"
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case": list(split), "split": list(split.values())}).sort_values("case").to_csv(
        res / "ct_split.csv", index=False)

    CT.mkdir(parents=True, exist_ok=True)
    (CT / "test_raw").mkdir(exist_ok=True)
    lut = np.zeros(256, np.uint8)
    for new, _, amos, _ in KEEP:
        lut[amos] = new
    stats, vox_total = [], np.zeros(16)
    for p in imgs:
        cid = p.name[:-7]
        lp = p.parent.parent / "labelsTr" / p.name
        img, lab = sitk.ReadImage(str(p)), sitk.ReadImage(str(lp))
        a_, l_ = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(lab)
        assert a_.shape == l_.shape, cid
        cnt = np.bincount(l_.ravel(), minlength=16)[:16]
        vox_total += cnt
        vol_ml = np.prod(img.GetSpacing()) / 1000
        row = dict(case=cid, split=split[cid], size_xyz=img.GetSize(), spacing_xyz=tuple(round(s, 3) for s in img.GetSpacing()),
                   hu_min=a_.min(), hu_max=a_.max(), hu_mean=round(float(a_.mean()), 1),
                   hu_p005=np.percentile(a_[::4, ::4, ::4], 0.5), hu_p995=np.percentile(a_[::4, ::4, ::4], 99.5),
                   looks_like_ct=bool(a_.min() < -500))
        row.update({f"ml_{AMOS[k]}": round(cnt[k] * vol_ml, 1) for k in AMOS})
        stats.append(row)

        lab2 = sitk.GetImageFromArray(lut[l_])
        lab2.CopyInformation(lab)
        im_r = resample(sitk.Cast(img, sitk.sitkFloat32), sitk.sitkLinear, -1024.0)
        lb_r = resample(lab2, sitk.sitkNearestNeighbor, 0)
        x = np.clip((sitk.GetArrayFromImage(im_r) - HU_LO) / (HU_HI - HU_LO), 0, 1).astype(np.float16)
        np.save(CT / f"{cid}_img.npy", x)
        np.save(CT / f"{cid}_lab.npy", sitk.GetArrayFromImage(lb_r))
        json.dump(dict(spacing=im_r.GetSpacing(), origin=im_r.GetOrigin(), direction=im_r.GetDirection()),
                  open(CT / f"{cid}_geom.json", "w"))
        if split[cid] == "test":  # bản gốc để chạy TotalSegmentator và đánh giá ở không gian gốc
            shutil.copy(p, CT / "test_raw" / f"{cid}_img.nii.gz")
            sitk.WriteImage(lab2, str(CT / "test_raw" / f"{cid}_lab.nii.gz"))
        print("tiền xử lý", cid, x.shape, flush=True)

    out = RESULTS / "ct"
    df = pd.DataFrame(stats)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dataset_stats.csv", index=False)
    ls = pd.DataFrame([dict(amos_id=k, name=AMOS[k], kept=k in {m[2] for m in KEEP},
                            mean_ml_when_present=df[f"ml_{AMOS[k]}"].replace(0, np.nan).mean(),
                            n_cases_present=int((df[f"ml_{AMOS[k]}"] > 0).sum()),
                            pct_of_labeled_voxels=100 * vox_total[k] / vox_total[1:].sum()) for k in AMOS])
    ls.to_csv(out / "label_stats.csv", index=False)
    write_kv(out / "dataset_overview.csv", dict(
        n_cases=len(df), n_train=ntr, n_val=nva, n_test=len(df) - ntr - nva, all_look_like_ct=df.looks_like_ct.all(),
        hu_min=df.hu_min.min(), hu_max=df.hu_max.max(), preprocess=f"spacing {SPACING} mm, HU [{HU_LO},{HU_HI}] -> [0,1]",
        sizes_xyz_unique=len(set(df.size_xyz)), spacing_xyz_unique=len(set(df.spacing_xyz))))
    pd.DataFrame([dict(our_id=n, our_name=nm, amos_id=am, amos_name=AMOS[am], totalseg_name=ts)
                  for n, nm, am, ts in KEEP]).to_csv(out / "label_mapping.csv", index=False)

    rows = []
    for cid in [c for c in ids if split[c] == "train"][:3]:
        x = np.load(CT / f"{cid}_img.npy").astype(np.float32)
        y = np.load(CT / f"{cid}_lab.npy")
        z = int(np.argmax((y > 0).sum((1, 2))))
        rows.append([x[z], overlay(x[z], y[z], 7)])
    show_rows(out / "figures" / "data_examples.png", rows, ["CT (cửa sổ HU, 1.5mm)", "+ nhãn"])
    print("Xong bước dữ liệu CT.")


if __name__ == "__main__":
    main()
