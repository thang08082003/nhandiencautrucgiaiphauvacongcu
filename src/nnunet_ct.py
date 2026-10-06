"""Tuần 3 (CT): nnU-Net v2 3d_fullres trên đúng chia tập train/val/test của data_ct.py (6 nhãn), đánh giá ở không gian gốc.
python nnunet_ct.py prepare                       # dựng Dataset501, splits_final.json (fold 0 = đúng train/val), plan+preprocess
python nnunet_ct.py train --trainer nnUNetTrainer_50epochs   # đổi số epoch cho vừa ngân sách Colab (mặc định nnU-Net là 1000)
python nnunet_ct.py predict                       # dự đoán các ca test
python nnunet_ct.py evaluate                      # Dice/IoU/HD95/NSD theo nhãn -> results/ct/nnunet_metrics.csv
nnU-Net dùng val trong splits_final.json làm tập theo dõi; không dùng test lúc huấn luyện.
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk

from common import CACHE, DATA, RESULTS, eval_case, metrics_table, rows_df, sh, start
from data_ct import KEEP, NAMES, OUT

DS = "Dataset501_AMOS6"
BASE = CACHE / "nnunet"
RAW, PRE, RES = BASE / "nnUNet_raw", BASE / "nnUNet_preprocessed", BASE / "nnUNet_results"
ENV = f'nnUNet_raw="{RAW}" nnUNet_preprocessed="{PRE}" nnUNet_results="{RES}"'
CT = CACHE / "ct"


def split():
    d = pd.read_csv(RESULTS / "splits" / "ct_split.csv")
    return {s: d[d.split == s].case.tolist() for s in ("train", "val", "test")}


def prepare():
    sp = split()
    for d in ("imagesTr", "labelsTr", "imagesTs"):
        (RAW / DS / d).mkdir(parents=True, exist_ok=True)
    lut = np.zeros(256, np.uint8)
    for new, _, amos, _ in KEEP:
        lut[amos] = new
    for c in sp["train"] + sp["val"]:
        img = next(OUT.rglob(f"imagesTr/{c}.nii.gz"))
        lab = sitk.ReadImage(str(img.parent.parent / "labelsTr" / img.name))
        l2 = sitk.GetImageFromArray(lut[sitk.GetArrayFromImage(lab)])
        l2.CopyInformation(lab)
        sitk.WriteImage(l2, str(RAW / DS / "labelsTr" / f"{c}.nii.gz"))
        dst = RAW / DS / "imagesTr" / f"{c}_0000.nii.gz"
        if not dst.exists():
            shutil.copy(img, dst)  # Drive không hỗ trợ symlink
    for c in sp["test"]:
        dst = RAW / DS / "imagesTs" / f"{c}_0000.nii.gz"
        if not dst.exists():
            shutil.copy(CT / "test_raw" / f"{c}_img.nii.gz", dst)
    json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, **{n: k for k, n in NAMES.items()}},
               "numTraining": len(sp["train"]) + len(sp["val"]), "file_ending": ".nii.gz"},
              open(RAW / DS / "dataset.json", "w"), indent=1)
    sh(f"{ENV} nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity")
    json.dump([{"train": sp["train"], "val": sp["val"]}], open(PRE / DS / "splits_final.json", "w"), indent=1)


def train(a):
    sh(f"{ENV} nnUNetv2_train 501 3d_fullres 0 -tr {a.trainer} --npz" + (" --c" if a.resume else ""))


def predict(a):
    sh(f"{ENV} nnUNetv2_predict -i {RAW / DS / 'imagesTs'} -o {CT / 'pred_nnunet'} -d 501 -c 3d_fullres "
       f"-tr {a.trainer} -f 0 -chk checkpoint_best.pth")


def evaluate():
    per = {}
    for c in split()["test"]:
        ref = sitk.ReadImage(str(CT / "test_raw" / f"{c}_lab.nii.gz"))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(str(CT / "pred_nnunet" / f"{c}.nii.gz")))
        per[c] = eval_case(pr, sitk.GetArrayFromImage(ref), NAMES, ref.GetSpacing()[::-1], tol=2.0)
        print(c, flush=True)
    df = rows_df(per)
    df.to_csv(RESULTS / "ct" / "nnunet_per_case.csv", index=False)
    t = metrics_table(df)
    t.to_csv(RESULTS / "ct" / "nnunet_metrics.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train", "predict", "evaluate"])
    ap.add_argument("--trainer", default="nnUNetTrainer_50epochs")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    start()
    {"prepare": prepare, "train": lambda: train(a), "predict": lambda: predict(a), "evaluate": evaluate}[a.cmd]()
