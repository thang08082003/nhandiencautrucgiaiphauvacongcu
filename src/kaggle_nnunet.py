"""Chạy TRÊN Kaggle GPU (được đẩy lên bằng kaggle_push.py): train tiếp nnU-Net 20 epoch từ checkpoint của Colab,
rồi dự đoán 6 ca test ở lưới 1.5 mm. Output: /kaggle/working/pred_1p5mm/*.nii.gz và /kaggle/working/results/ (checkpoint, log).
Nếu input có checkpoint_final.pth (dataset anatomy-nnunet-final) thì bỏ qua train, chỉ dự đoán."""
import glob
import os
import shutil
import subprocess
import tarfile

T = "nnUNetTrainer_20epochs"
W = "/kaggle/working"
B = "/tmp/nnunet"  # input chỉ đọc, phải chép ra chỗ ghi được
RUN = f"{B}/nnUNet_results/Dataset501_AMOS6/{T}__nnUNetPlans__3d_fullres"


def sh(c):
    print("$", c, flush=True)
    subprocess.run(c, shell=True, check=True)


tars = glob.glob("/kaggle/input/**/nnunet.tar", recursive=True)
if tars:
    tarfile.open(tars[0]).extractall(B)
else:  # Kaggle đã tự giải nén
    shutil.copytree(glob.glob("/kaggle/input/**/nnUNet_preprocessed", recursive=True)[0].rsplit("/", 1)[0], B)
# Kaggle tự giải nén cả .nii.gz thành .nii -> nnU-Net (file_ending .nii.gz) thấy 0 ca; nén lại
for f in glob.glob(f"{B}/nnUNet_raw/Dataset501_AMOS6/imagesTs/*.nii"):
    sh(f"gzip {f}")
env = f"nnUNet_raw={B}/nnUNet_raw nnUNet_preprocessed={B}/nnUNet_preprocessed nnUNet_results={B}/nnUNet_results"
sh("pip -q install nnunetv2")
final = glob.glob("/kaggle/input/**/fold_0/checkpoint_final.pth", recursive=True)
if final:
    shutil.copytree(os.path.dirname(os.path.dirname(final[0])), RUN, dirs_exist_ok=True)
else:
    sh(f"{env} nnUNetv2_train 501 3d_fullres 0 -tr {T} --npz --c")
sh(f"{env} nnUNetv2_predict -i {B}/nnUNet_raw/Dataset501_AMOS6/imagesTs -o {W}/pred_1p5mm -d 501 -c 3d_fullres "
   f"-tr {T} -f 0 -chk checkpoint_best.pth")
assert glob.glob(f"{W}/pred_1p5mm/*.nii.gz"), "không có dự đoán nào"
shutil.copytree(RUN, f"{W}/results", dirs_exist_ok=True)
