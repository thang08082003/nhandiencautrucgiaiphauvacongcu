"""Bước 1C: CholecSeg8k. Chia theo VIDEO.
Cách 1: bạn tự tải/giải nén rồi: python data_cholec.py --src /content/cholecseg8k
Cách 2: python data_cholec.py --download   (cần ~/.kaggle/kaggle.json do BẠN tạo; script in dung lượng rồi chờ 'yes')
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import CACHE, DATA, RESULTS, SEED, overlay, show_rows, sh, start, write_kv

# Giá trị trong *_endo_watershed_mask.png (đo trên dữ liệu thật: {0,5,11,12,13,21,22,23,24,25,31,32,33,50,255}).
# Nền đen = 50 (mã màu #505050), phải đứng ĐẦU để thành chỉ số 0 (evaluate() bỏ lớp 0).
# Tên của 9 lớp ngoài {50, 11, 21, 31} theo bảng bài báo CholecSeg8k (theo trí nhớ, chưa đối chiếu): kiểm tra lại trước khi viết báo cáo.
IDS = {50: "Black Background", 11: "Abdominal Wall", 21: "Liver", 13: "Gastrointestinal Tract", 12: "Fat", 31: "Grasper",
       23: "Connective Tissue", 24: "Blood", 25: "Cystic Duct", 32: "L-hook Electrocautery", 22: "Gallbladder",
       33: "Hepatic Vein", 5: "Liver Ligament"}
KEYS = list(IDS)
NAMES = [IDS[k] for k in KEYS]
SIZE = 256
LUT = np.zeros(256, np.uint8)  # ponytail: 255 (biên watershed, ~0.7% pixel) và 0 gộp vào nền; thêm ignore_index nếu cần chính xác hơn
for i, k in enumerate(KEYS):
    LUT[k] = i


def read(args):
    vid, f = args
    im = cv2.imread(str(f))
    m = cv2.imread(str(f).replace("_endo.png", "_endo_watershed_mask.png"), cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[:, :, 0]
    mm = LUT[m]
    if set(np.unique(m)) - set(IDS) - {0, 255}:
        raise ValueError(f"{f}: giá trị mask lạ {sorted(set(np.unique(m)) - set(IDS))}; kiểm tra lại cách mã hoá nhãn")
    h, w = mm.shape
    return (vid, h, w, cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA),
            cv2.resize(mm, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST), np.bincount(mm.ravel(), minlength=len(KEYS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--slug", default="newbie/cholecseg8k")
    a = ap.parse_args()
    start()
    src = Path(a.src) if a.src else DATA / "cholecseg8k"
    if a.download:
        sh(f"kaggle datasets list -s cholecseg8k")
        print(f"Sẽ tải '{a.slug}' từ Kaggle vào {src}. Xem dung lượng/giấy phép ở danh sách trên.")
        if input("Gõ 'yes' để tải: ").strip().lower() != "yes":
            raise SystemExit("Đã huỷ.")
        sh(f"kaggle datasets download -d {a.slug} -p {src} --unzip")

    files = sorted(src.rglob("*_endo.png"))
    jobs = [(re.search(r"video(\d+)", str(f)).group(1), f) for f in files]
    assert jobs, f"Không thấy frame trong {src}"
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(read, jobs))
    vids = np.array([r[0] for r in res])
    X, Y = np.stack([r[3] for r in res]), np.stack([r[4] for r in res])
    cnt = np.stack([r[5] for r in res])  # (N, 13) pixel đếm ở độ phân giải gốc

    uv = sorted(set(vids))
    perm = np.random.default_rng(SEED).permutation(uv)
    ntr, nva = round(0.7 * len(uv)), round(0.15 * len(uv))
    sp = {"train": sorted(perm[:ntr]), "val": sorted(perm[ntr:ntr + nva]), "test": sorted(perm[ntr + nva:])}
    v2s = {v: s for s, vs in sp.items() for v in vs}

    out = CACHE / "cholec"
    out.mkdir(parents=True, exist_ok=True)
    s_arr = np.array([v2s[v] for v in vids])
    for s in sp:
        np.savez_compressed(out / f"{s}.npz", X=X[s_arr == s], Y=Y[s_arr == s])
        pd.Series(vids[s_arr == s], name="video").to_csv(out / f"{s}_index.csv", index=False)
    json.dump(dict(names=NAMES, in_ch=3, size=SIZE, ids=KEYS), open(out / "meta.json", "w"))

    r = RESULTS / "cholecseg8k"
    (r / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS / "splits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"video": uv, "split": [v2s[v] for v in uv],
                  "n_frames": [int((vids == v).sum()) for v in uv]}).to_csv(RESULTS / "splits" / "cholecseg8k_split.csv", index=False)
    pd.DataFrame([dict(class_id=k, name=IDS[k], pixel_pct=100 * cnt[:, i].sum() / cnt.sum(),
                       frames_with_class=int((cnt[:, i] > 0).sum()), videos_with_class=len(set(vids[cnt[:, i] > 0])),
                       **{f"present_in_{s}": bool((cnt[s_arr == s][:, i] > 0).any()) for s in sp})
                  for i, k in enumerate(KEYS)]).to_csv(r / "dataset_stats.csv", index=False)
    sizes = {(x[1], x[2]) for x in res}
    write_kv(r / "dataset_overview.csv", dict(
        n_videos=len(uv), n_frames=len(res), n_classes=len(KEYS), orig_size_hxw=sizes, resized_to=f"{SIZE}x{SIZE}",
        videos_train=sp["train"], videos_val=sp["val"], videos_test=sp["test"],
        frames_per_split={s: int((s_arr == s).sum()) for s in sp}))

    i0 = np.flatnonzero(s_arr == "train")[:: max(1, (s_arr == "train").sum() // 3)][:3]
    show_rows(r / "figures" / "data_examples.png", [[X[i], overlay(X[i], Y[i], 13)] for i in i0], ["khung hình", "+ nhãn"])
    print("Xong bước dữ liệu CholecSeg8k.")


if __name__ == "__main__":
    main()
