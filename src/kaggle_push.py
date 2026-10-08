"""Khi hết quota GPU Colab: chạy phần còn lại trên Kaggle GPU rồi kéo kết quả về Drive. Chạy trên Colab (CPU được).
Cần KAGGLE_API_TOKEN (dán qua getpass như Cell 5) và tài khoản Kaggle đã xác minh số điện thoại (mới dùng được GPU).
Sau mỗi `datasets create`, đợi trang dataset trên Kaggle xử lý xong rồi mới đẩy kernel.
nnU-Net CT:
  python kaggle_push.py push     # dataset riêng tư (nnunet.tar)
  python kaggle_push.py final    # sau pull lần 1: đẩy results/ (checkpoint_final) thành dataset thứ hai
  python kaggle_push.py kernel   # kernel GPU T4 chạy kaggle_nnunet.py
  python kaggle_push.py pull     # kéo output về Drive, rồi:
  python nnunet_ct.py predict --trainer nnUNetTrainer_20epochs && python nnunet_ct.py evaluate   # predict bỏ qua suy luận
CholecSeg8k (4 lượt ablation/seed):
  python kaggle_push.py cholec         # dataset: cache/cholec + src
  python kaggle_push.py cholec_kernel  # kernel GPU T4 chạy kaggle_cholec.py (~2 giờ)
  python kaggle_push.py cholec_pull    # kéo results/ablation + ckpt về Drive
python kaggle_push.py status [kernel]  # trạng thái kernel (mặc định nnU-Net)
"""
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from common import CACHE, RESULTS, sh

USER = "thangnguyenv"
DS, KN = f"{USER}/anatomy-nnunet-ct", f"{USER}/anatomy-nnunet-train"
DS2 = f"{USER}/anatomy-nnunet-final"  # checkpoint đã train xong, để kernel chỉ dự đoán
DS3, KN2 = f"{USER}/anatomy-cholec", f"{USER}/anatomy-cholec-train"
T = "nnUNetTrainer_20epochs"
BASE = CACHE / "nnunet"
RUN = f"nnUNet_results/Dataset501_AMOS6/{T}__nnUNetPlans__3d_fullres"
STAGE = Path("/content/kaggle_stage")


def dataset(sub, ds, title, files, lic="CC-BY-4.0"):
    """files: {arcname: đường dẫn}. Đóng một file tar riêng tư rồi tạo dataset."""
    d = STAGE / sub
    d.mkdir(parents=True, exist_ok=True)
    name = next(iter(files.values()))[1]
    with tarfile.open(d / name, "w") as t:
        for arc, (p, _) in files.items():
            t.add(p, arcname=arc, filter=lambda i: None if "__pycache__" in i.name else i)
    print(f"{name}: {(d / name).stat().st_size / 2**20:.0f} MB", flush=True)
    json.dump({"title": title, "id": ds, "licenses": [{"name": lic}]}, open(d / "dataset-metadata.json", "w"))
    sh(f"kaggle datasets create -p {d}")  # mặc định riêng tư


def push():  # npz đã nén sẵn, tar không nén cho nhanh
    dataset("ds", DS, "anatomy-nnunet-ct", {s: (BASE / s, "nnunet.tar") for s in (
        "nnUNet_preprocessed", "nnUNet_raw/Dataset501_AMOS6/imagesTs", "nnUNet_raw/Dataset501_AMOS6/dataset.json", RUN)})


def final():
    dataset("ds2", DS2, "anatomy-nnunet-final", {"results": (STAGE / "out" / "results", "final.tar")})


def cholec():  # đuôi .bin để Kaggle không tự giải nén (nó giải cả .tar lẫn .nii.gz bên trong)
    dataset("ds3", DS3, "anatomy-cholec", {"cache/cholec": (CACHE / "cholec", "cholec.bin"),
                                           "src": (Path(__file__).parent, "cholec.bin")}, "CC-BY-NC-SA-4.0")


def kernel(kn=KN, title="anatomy-nnunet-train", code="kaggle_nnunet.py", sources=None):
    k = STAGE / f"kernel_{title}"
    k.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name(code), k / code)
    sources = sources or [DS] + ([DS2] if (STAGE / "ds2").exists() else [])
    json.dump({"id": kn, "title": title, "code_file": code, "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": sources,
               "machine_shape": "NvidiaTeslaT4"}, open(k / "kernel-metadata.json", "w"))
    sh(f"kaggle kernels push -p {k}")


def sh_out(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip(), flush=True)
    return r.stdout.lower()


def pull():
    o = STAGE / "out"
    sh(f"kaggle kernels output {KN} -p {o}")
    shutil.copytree(o / "pred_1p5mm", CACHE / "ct" / "pred_nnunet_1p5mm", dirs_exist_ok=True)
    shutil.copytree(o / "results", BASE / RUN, dirs_exist_ok=True)
    print("Đã kéo về Drive:", sorted(p.name for p in (o / "pred_1p5mm").iterdir()))


def cholec_pull():
    o = STAGE / "out_cholec"
    sh(f"kaggle kernels output {KN2} -p {o}")
    abl = o / "results" / "cholecseg8k" / "ablation"
    shutil.copytree(abl, RESULTS / "cholecseg8k" / "ablation", dirs_exist_ok=True)
    shutil.copytree(o / "ckpt", CACHE / "ckpt", dirs_exist_ok=True)
    print("Đã kéo về Drive:", sorted(p.name for p in abl.iterdir()))


if __name__ == "__main__":
    {"push": push, "final": final, "kernel": kernel, "pull": pull, "cholec": cholec,
     "cholec_kernel": lambda: kernel(KN2, "anatomy-cholec-train", "kaggle_cholec.py", [DS3]),
     "cholec_pull": cholec_pull,
     "status": lambda: sh_out(f"kaggle kernels status {sys.argv[2] if len(sys.argv) > 2 else KN}")}[sys.argv[1]]()
