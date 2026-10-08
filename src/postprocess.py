"""Hậu xử lý CT: giữ thành phần liên thông lớn nhất của mỗi nhãn, rồi đánh giá lại (so trước/sau).
python postprocess.py ct --src pred_custom      # hoặc pred_nnunet -> results/ct/<src>_pp_metrics.csv
CAMUS bỏ qua: dự đoán 2D không được lưu ra đĩa."""
import argparse

import numpy as np
import SimpleITK as sitk

from common import CACHE, RESULTS, eval_case, metrics_table, rows_df, start
from data_ct import NAMES
from nnunet_ct import split

CT = CACHE / "ct"


def largest_cc(m):
    cc = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(m.astype(np.uint8))))
    return cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1) if cc.any() else m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["ct"])
    ap.add_argument("--src", default="pred_custom")
    a = ap.parse_args()
    start()
    per = {}
    for c in split()["test"]:
        ref = sitk.ReadImage(str(CT / "test_raw" / f"{c}_lab.nii.gz"))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(str(CT / a.src / f"{c}.nii.gz")))
        out = np.zeros_like(pr)
        for k in NAMES:
            out[largest_cc(pr == k) & (pr == k)] = k
        per[c] = eval_case(out, sitk.GetArrayFromImage(ref), NAMES, ref.GetSpacing()[::-1], tol=2.0)
        print(c, flush=True)
    t = metrics_table(rows_df(per))
    t.to_csv(RESULTS / "ct" / f"{a.src}_pp_metrics.csv", index=False)
    print(t.round(3).to_string(index=False))
