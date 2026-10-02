"""Bước 2: TotalSegmentator (baseline có sẵn) trên 3-5 ca test CT. Đo thời gian từng ca, Dice/HD95 theo bảng ánh xạ.
python totalseg_baseline.py --n 5 [--fast]
"""
import argparse
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from common import CACHE, RESULTS, eval_case, rows_df, sh, start
from data_ct import KEEP, NAMES

CT = CACHE / "ct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="dùng khi thiếu VRAM")
    a = ap.parse_args()
    start()
    split = pd.read_csv(RESULTS / "splits" / "ct_split.csv")
    te = split[split.split == "test"].case.tolist()[: a.n]
    (CT / "pred_ts").mkdir(exist_ok=True)
    roi = " ".join(ts for *_, ts in KEEP)
    dev = "gpu" if torch.cuda.is_available() else "cpu"
    per, secs = {}, {}
    for c in te:
        o = CT / "ts_out" / c
        t = time.time()
        sh(f"TotalSegmentator -i {CT / 'test_raw' / (c + '_img.nii.gz')} -o {o} --roi_subset {roi} --device {dev}"
           + (" --fast" if a.fast else ""))
        secs[c] = time.time() - t  # gồm nạp mô hình + tiền xử lý + suy luận + ghi file
        ref = sitk.ReadImage(str(CT / "test_raw" / f"{c}_lab.nii.gz"))
        gt = sitk.GetArrayFromImage(ref)
        pred = np.zeros_like(gt)
        for k, _, _, ts in KEEP:
            m = sitk.ReadImage(str(o / f"{ts}.nii.gz"))
            if m.GetSize() != ref.GetSize():
                m = sitk.Resample(m, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
            pred[sitk.GetArrayFromImage(m) > 0] = k
        p = sitk.GetImageFromArray(pred)
        p.CopyInformation(ref)
        sitk.WriteImage(p, str(CT / "pred_ts" / f"{c}.nii.gz"))
        per[c] = eval_case(pred, gt, NAMES, ref.GetSpacing()[::-1])
        print(c, f"{secs[c]:.1f}s", flush=True)
    df = rows_df(per)
    df["seconds"] = df.case.map(secs)
    df["mode"] = "fast" if a.fast else "full"
    df["device"] = dev
    (RESULTS / "ct").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "ct" / "totalseg_metrics.csv", index=False)
    print(df.groupby("label", sort=False)[["dice", "hd95"]].mean())


if __name__ == "__main__":
    main()
