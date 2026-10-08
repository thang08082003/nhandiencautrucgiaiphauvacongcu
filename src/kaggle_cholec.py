"""Chạy TRÊN Kaggle GPU (đẩy bằng `kaggle_push.py cholec_kernel`): 4 lượt CholecSeg8k còn thiếu, 30 phút mỗi lượt (T4 như Colab).
Output: /kaggle/working/results/cholecseg8k/ablation/<tag>/ và /kaggle/working/ckpt/cholec_<tag>_best.pt."""
import glob
import os
import shutil
import subprocess
import tarfile

W, B = "/kaggle/working", "/tmp/proj"
RUNS = {"aug_focal": "--aug strong --loss dice_focal", "base_s1": "--seed 1",
        "aug_strong_s1": "--aug strong --seed 1", "aug_focal_s1": "--aug strong --loss dice_focal --seed 1"}

tarfile.open(glob.glob("/kaggle/input/**/cholec.bin", recursive=True)[0]).extractall(B)
env = dict(os.environ, CACHE_DIR=f"{B}/cache", RESULTS_DIR=f"{W}/results")
subprocess.run("pip -q install segmentation-models-pytorch", shell=True, check=True)
for tag, args in RUNS.items():
    cmd = f"python train_2d.py --name cholec --budget-min 30 --tag {tag} {args}"
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=f"{B}/src", env=env)  # lượt lỗi không chặn các lượt sau
    print(f"{tag}: exit {r.returncode}", flush=True)
shutil.copytree(f"{B}/cache/ckpt", f"{W}/ckpt")
