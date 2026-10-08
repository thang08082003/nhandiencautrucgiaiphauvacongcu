"""Tuần 3 (CT): nnU-Net v2 3d_fullres trên đúng chia tập train/val/test của data_ct.py (6 nhãn), đánh giá ở không gian gốc.
python nnunet_ct.py prepare                       # dựng Dataset501, splits_final.json (fold 0 = đúng train/val), plan+preprocess
python nnunet_ct.py train --trainer nnUNetTrainer_50epochs   # đổi số epoch cho vừa ngân sách Colab (mặc định nnU-Net là 1000)
python nnunet_ct.py predict                       # dự đoán các ca test
python nnunet_ct.py evaluate                      # Dice/IoU/HD95/NSD theo nhãn -> results/ct/nnunet_metrics.csv
nnU-Net dùng val trong splits_final.json làm tập theo dõi; không dùng test lúc huấn luyện.
"""
import argparse
import json

import numpy as np
import pandas as pd
import SimpleITK as sitk

from common import CACHE, RESULTS, eval_case, metrics_table, rows_df, sh, start
from data_ct import NAMES
from train_ct import to_orig

DS = "Dataset501_AMOS6"
BASE = CACHE / "nnunet"
RAW, PRE, RES = BASE / "nnUNet_raw", BASE / "nnUNet_preprocessed", BASE / "nnUNet_results"
ENV = f'nnUNet_raw="{RAW}" nnUNet_preprocessed="{PRE}" nnUNet_results="{RES}"'
CT = CACHE / "ct"


def split():
    d = pd.read_csv(RESULTS / "splits" / "ct_split.csv")
    return {s: d[d.split == s].case.tolist() for s in ("train", "val", "test")}


def nii(arr, cid, dst):
    g = json.load(open(CT / f"{cid}_geom.json"))
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing(g["spacing"]), im.SetOrigin(g["origin"]), im.SetDirection(g["direction"])
    sitk.WriteImage(im, str(dst))


def prepare():
    # dựng từ cache/ct/*.npy (đã resample 1.5 mm, HU->[0,1]): không cần tải lại AMOS, cùng đầu vào với 3D U-Net tự huấn luyện
    sp = split()
    for d in ("imagesTr", "labelsTr", "imagesTs"):
        (RAW / DS / d).mkdir(parents=True, exist_ok=True)
    for c in sp["train"] + sp["val"]:
        nii(np.load(CT / f"{c}_img.npy").astype(np.float32), c, RAW / DS / "imagesTr" / f"{c}_0000.nii.gz")
        nii(np.load(CT / f"{c}_lab.npy"), c, RAW / DS / "labelsTr" / f"{c}.nii.gz")
    for c in sp["test"]:
        nii(np.load(CT / f"{c}_img.npy").astype(np.float32), c, RAW / DS / "imagesTs" / f"{c}_0000.nii.gz")
    json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, **{n: k for k, n in NAMES.items()}},
               "numTraining": len(sp["train"]) + len(sp["val"]), "file_ending": ".nii.gz"},
              open(RAW / DS / "dataset.json", "w"), indent=1)
    sh(f"{ENV} nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity")
    json.dump([{"train": sp["train"], "val": sp["val"]}], open(PRE / DS / "splits_final.json", "w"), indent=1)


def train(a):
    sh(f"{ENV} nnUNetv2_train 501 3d_fullres 0 -tr {a.trainer} --npz" + (" --c" if a.resume else ""))


def predict(a):
    tmp = CT / "pred_nnunet_1p5mm"
    if not all((tmp / f"{c}.nii.gz").exists() for c in split()["test"]):  # đã có (vd. kéo từ Kaggle) thì bỏ qua suy luận
        sh(f"{ENV} nnUNetv2_predict -i {RAW / DS / 'imagesTs'} -o {tmp} -d 501 -c 3d_fullres "
           f"-tr {a.trainer} -f 0 -chk checkpoint_best.pth")
    (CT / "pred_nnunet").mkdir(exist_ok=True)
    for c in split()["test"]:  # resample về lưới gốc như train_ct.to_orig để so công bằng
        pr = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp / f"{c}.nii.gz")))
        o, _ = to_orig(pr, c)
        sitk.WriteImage(o, str(CT / "pred_nnunet" / f"{c}.nii.gz"))


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
