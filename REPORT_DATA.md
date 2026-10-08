# REPORT_DATA (số liệu thật từ các lần chạy; sinh bởi src/report_tables.py)

## a) Môi trường
# Môi trường
- OS: Linux-6.6.122+-x86_64-with-glibc2.39
- Python: 3.13.15
- RAM: total        used        free      shared  buff/cache   available
Mem:            12Gi       1.7Gi       2.0Gi        18Mi       9.3Gi        10Gi
- Đĩa trống (/content/drive/MyDrive/anatomy_project): 5 GB / 15 GB
- PyTorch: 2.11.0+cu130, CUDA build: 13.0, GPU thấy được: True
- Kiểm tra nhanh GPU (matmul 1024x1024): OK

```
Wed Oct  7 08:39:37 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   41C    P0             25W /   70W |     175MiB /  15360MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5897      C   python3                                 172MiB |
+-----------------------------------------------------------------------------------------+
```


## b) Dữ liệu
### CT (AMOS, tập con)
- n_cases: 40
- n_train: 28
- n_val: 6
- n_test: 6
- all_look_like_ct: True
- hu_min: -3024.0
- hu_max: 6894.0
- preprocess: spacing (1.5, 1.5, 1.5) mm, HU [-175.0,250.0] -> [0,1]
- sizes_xyz_unique: 36
- spacing_xyz_unique: 31

|   amos_id | name                | kept   |   mean_ml_when_present |   n_cases_present |   pct_of_labeled_voxels |
|----------:|:--------------------|:-------|-----------------------:|------------------:|------------------------:|
|         1 | spleen              | True   |                194.132 |                40 |                   7.572 |
|         2 | right kidney        | True   |                139.41  |                40 |                   5.214 |
|         3 | left kidney         | True   |                145.862 |                40 |                   5.437 |
|         4 | gallbladder         | True   |                 27.087 |                38 |                   1.01  |
|         5 | esophagus           | False  |                 15.697 |                39 |                   0.622 |
|         6 | liver               | True   |               1335.44  |                40 |                  49.716 |
|         7 | stomach             | False  |                340.518 |                39 |                  12.196 |
|         8 | aorta               | False  |                110.005 |                40 |                   4.331 |
|         9 | postcava            | False  |                 75.188 |                40 |                   2.869 |
|        10 | pancreas            | True   |                 73.508 |                40 |                   2.736 |
|        11 | right adrenal gland | False  |                  3.215 |                40 |                   0.126 |
|        12 | left adrenal gland  | False  |                  3.748 |                40 |                   0.151 |
|        13 | duodenum            | False  |                 54.565 |                40 |                   2.228 |
|        14 | bladder             | False  |                122.503 |                38 |                   4.119 |
|        15 | prostate/uterus     | False  |                 50.711 |                37 |                   1.673 |

Ánh xạ nhãn:

|   our_id | our_name     |   amos_id | amos_name    | totalseg_name   |
|---------:|:-------------|----------:|:-------------|:----------------|
|        1 | spleen       |         1 | spleen       | spleen          |
|        2 | kidney_right |         2 | right kidney | kidney_right    |
|        3 | kidney_left  |         3 | left kidney  | kidney_left     |
|        4 | gallbladder  |         4 | gallbladder  | gallbladder     |
|        5 | liver        |         6 | liver        | liver           |
|        6 | pancreas     |        10 | pancreas     | pancreas        |

### CAMUS
- split_source: chia chính thức của CAMUS (subgroup_*.txt)
- n_patients_total: 500
- n_images: 2000
- patients_per_split: {'train': 400, 'val': 50, 'test': 50}
- images_per_split: {'train': 1600, 'test': 200, 'val': 200}
- images_per_view_phase: {('2CH', 'ED'): 500, ('2CH', 'ES'): 500, ('4CH', 'ED'): 500, ('4CH', 'ES'): 500}
- quality_counts: {'Good': 1010, 'Medium': 758, 'Poor': 232}
- orig_sizes_unique: 74
- orig_size_example: 389x549
- resized_to: 256x256
- pixel_pct_background: 77.07
- pixel_pct_LV_endo: 8.54
- pixel_pct_myocardium: 9.29
- pixel_pct_LA: 5.1
### CholecSeg8k
- n_videos: 17
- n_frames: 8080
- n_classes: 13
- orig_size_hxw: {(480, 854)}
- resized_to: 256x256
- videos_train: [np.str_('01'), np.str_('12'), np.str_('17'), np.str_('20'), np.str_('24'), np.str_('25'), np.str_('27'), np.str_('28'), np.str_('35'), np.str_('37'), np.str_('52'), np.str_('55')]
- videos_val: [np.str_('09'), np.str_('18'), np.str_('48')]
- videos_test: [np.str_('26'), np.str_('43')]
- frames_per_split: {'train': 6400, 'val': 640, 'test': 1040}

|   class_id | name                   |   pixel_pct |   frames_with_class |   videos_with_class | present_in_train   | present_in_val   | present_in_test   |
|-----------:|:-----------------------|------------:|--------------------:|--------------------:|:-------------------|:-----------------|:------------------|
|         50 | Black Background       |      27.241 |                8080 |                  17 | True               | True             | True              |
|         11 | Abdominal Wall         |      21.569 |                7255 |                  16 | True               | True             | True              |
|         21 | Liver                  |      21.271 |                8080 |                  17 | True               | True             | True              |
|         13 | Gastrointestinal Tract |       1.881 |                4558 |                  13 | True               | True             | True              |
|         12 | Fat                    |      14.72  |                7510 |                  17 | True               | True             | True              |
|         31 | Grasper                |       2.422 |                6020 |                  16 | True               | True             | True              |
|         23 | Connective Tissue      |       2.267 |                1600 |                   2 | True               | False            | False             |
|         24 | Blood                  |       0.422 |                 692 |                   1 | True               | False            | False             |
|         25 | Cystic Duct            |       0.037 |                 248 |                   3 | True               | False            | False             |
|         32 | L-hook Electrocautery  |       1.311 |                2254 |                   9 | True               | True             | False             |
|         22 | Gallbladder            |       6.43  |                6861 |                  16 | True               | True             | True              |
|         33 | Hepatic Vein           |       0.009 |                 317 |                   1 | True               | False            | False             |
|          5 | Liver Ligament         |       0.419 |                 240 |                   1 | False              | True             | False             |

Danh sách chia tập: results/splits/*.csv

## c) CT: TotalSegmentator và mô hình tự huấn luyện (cùng các ca test, không gian ảnh gốc)
> TotalSegmentator có thể đã thấy dữ liệu tương tự (kể cả AMOS) lúc huấn luyện, nên đây chỉ là baseline tham khảo.

| label        |   dice_totalseg |   hd95_totalseg |   dice_custom |   hd95_custom |
|:-------------|----------------:|----------------:|--------------:|--------------:|
| spleen       |           0.94  |           2.06  |         0.871 |        83.025 |
| kidney_right |           0.883 |           6.12  |         0.819 |         7.19  |
| kidney_left  |           0.882 |           3.998 |         0.766 |        43.908 |
| gallbladder  |           0.712 |          11.505 |         0.505 |        33.593 |
| liver        |           0.962 |           3.422 |         0.944 |         6.939 |
| pancreas     |           0.777 |           9.674 |         0.6   |        60.445 |
| MEAN         |           0.859 |           6.13  |         0.751 |        39.183 |

| mô hình                |   giây/ca (TB) | ghi chú                                        |
|:-----------------------|---------------:|:-----------------------------------------------|
| TotalSegmentator       |           74.3 | gồm nạp mô hình+tiền xử lý+ghi file, full, gpu |
| 3D U-Net tự huấn luyện |            2.6 | sliding window + resample về lưới gốc          |

Mô hình tự huấn luyện, toàn bộ ca test:

| label        |   dice |   iou |   hd95 |   dice_std |   n_cases |
|:-------------|-------:|------:|-------:|-----------:|----------:|
| spleen       |  0.885 | 0.798 | 69.482 |      0.065 |         6 |
| kidney_right |  0.833 | 0.735 |  6.588 |      0.151 |         6 |
| kidney_left  |  0.754 | 0.63  | 39.491 |      0.179 |         6 |
| gallbladder  |  0.566 | 0.464 | 28.583 |      0.358 |         6 |
| liver        |  0.947 | 0.901 |  6.616 |      0.019 |         6 |
| pancreas     |  0.632 | 0.475 | 52.053 |      0.141 |         6 |
| MEAN         |  0.77  | 0.667 | 33.802 |    nan     |       nan |

Cấu hình: {
 "model": "MONAI UNet 3D",
 "channels": [
  16,
  32,
  64,
  128,
  256
 ],
 "res_units": 2,
 "norm": "instance",
 "params": 4812026,
 "patch": [
  96,
  96,
  96
 ],
 "spacing_mm": 1.5,
 "hu_window": [
  -175.0,
  250.0
 ],
 "loss": "DiceCE",
 "lr": 0.002,
 "batch": 2,
 "seed": 42,
 "budget_min": 30.0,
 "iterations": 6585,
 "best_epoch": 132,
 "best_val_dice_1p5mm": 0.8607383942442722,
 "n_train": 28,
 "n_val": 6,
 "n_test": 6,
 "device": "cuda",
 "gpu": "Tesla T4",
 "torch": "2.11.0+cu130",
 "augment": "ch\u1ec9 nhi\u1ec5u c\u01b0\u1eddng \u0111\u1ed9 (kh\u00f4ng l\u1eadt v\u00ec tr\u00e1i/ph\u1ea3i li\u00ean quan t\u1edbi nh\u00e3n th\u1eadn)",
 "inference_seconds_note": "g\u1ed3m sliding window + resample v\u1ec1 l\u01b0\u1edbi g\u1ed1c, ch\u01b0a g\u1ed3m ti\u1ec1n x\u1eed l\u00fd"
}

## d) CAMUS và CholecSeg8k (test, trung bình theo ảnh; HD95 đơn vị pixel ở 256x256; MEAN = trung bình các nhãn trừ nền)
### CAMUS
| label      |   dice |   iou |   hd95 |   dice_std |   n_cases |
|:-----------|-------:|------:|-------:|-----------:|----------:|
| LV_endo    |  0.941 | 0.89  |  5.116 |      0.033 |       200 |
| myocardium |  0.883 | 0.792 |  5.76  |      0.043 |       200 |
| LA         |  0.919 | 0.854 |  6.123 |      0.063 |       200 |
| MEAN       |  0.914 | 0.846 |  5.666 |    nan     |       nan |
### CholecSeg8k
| label                  |    dice |     iou |    hd95 |   dice_std |   n_cases |
|:-----------------------|--------:|--------:|--------:|-----------:|----------:|
| Abdominal Wall         |   0.922 |   0.867 |  22.547 |      0.092 |      1023 |
| Liver                  |   0.912 |   0.844 |  27.057 |      0.069 |      1040 |
| Gastrointestinal Tract |   0.881 |   0.8   |  10.655 |      0.097 |       317 |
| Fat                    |   0.901 |   0.828 |  19.898 |      0.083 |      1039 |
| Grasper                |   0.779 |   0.658 |  29.29  |      0.153 |       684 |
| Connective Tissue      | nan     | nan     | nan     |    nan     |       nan |
| Blood                  | nan     | nan     | nan     |    nan     |       nan |
| Cystic Duct            | nan     | nan     | nan     |    nan     |       nan |
| L-hook Electrocautery  | nan     | nan     | nan     |    nan     |       nan |
| Gallbladder            |   0.642 |   0.524 |  23.156 |      0.277 |      1038 |
| Hepatic Vein           | nan     | nan     | nan     |    nan     |       nan |
| Liver Ligament         | nan     | nan     | nan     |    nan     |       nan |
| MEAN                   |   0.839 |   0.753 |  22.101 |    nan     |       nan |

## e) Tốc độ
### CT
|   warmup |   runs |   mean_ms |   p95_ms |    fps |   vram_peak_mb | backend                           | precision   | input                     | scope                                 | gpu      |
|---------:|-------:|----------:|---------:|-------:|---------------:|:----------------------------------|:------------|:--------------------------|:--------------------------------------|:---------|
|       10 |    100 |    24.312 |   24.636 | 41.131 |        107.247 | pytorch                           | fp32        | patch (96, 96, 96)        | 1 patch                               | Tesla T4 |
|       10 |    100 |    13.026 |   14.822 | 76.77  |         90.458 | pytorch                           | fp16        | patch (96, 96, 96)        | 1 patch                               | Tesla T4 |
|        1 |      5 |  1420.57  | 1428.14  |  0.704 |       1404.62  | pytorch                           | fp32        | amos_0405 (300, 301, 301) | cả khối (sliding window, overlap .25) | Tesla T4 |
|        1 |      5 |   668.257 |  670.019 |  1.496 |       1260.16  | pytorch                           | fp16        | amos_0405 (300, 301, 301) | cả khối (sliding window, overlap .25) | Tesla T4 |
|       10 |    100 |    28.211 |   29.468 | 35.447 |        nan     | onnxruntime/CUDAExecutionProvider | fp32        | patch (96, 96, 96)        | 1 patch                               | Tesla T4 |
### CAMUS
|   warmup |   runs |   mean_ms |   p95_ms |     fps |   vram_peak_mb | backend                           | precision   | input   |   scale |   dice_mean_fg | gpu      |
|---------:|-------:|----------:|---------:|--------:|---------------:|:----------------------------------|:------------|:--------|--------:|---------------:|:---------|
|       10 |    100 |     9.679 |   12.243 | 103.319 |         80.771 | pytorch                           | fp32        | 256x256 |    1    |          0.914 | Tesla T4 |
|       10 |    100 |     7.06  |   11.261 | 141.648 |        108.973 | pytorch                           | fp16        | 256x256 |    1    |          0.914 | Tesla T4 |
|       10 |    100 |     6.989 |    8.988 | 143.084 |         70.341 | pytorch                           | fp32        | 192x192 |    0.75 |          0.905 | Tesla T4 |
|       10 |    100 |     9.318 |   12.612 | 107.324 |         98.551 | pytorch                           | fp16        | 192x192 |    0.75 |          0.905 | Tesla T4 |
|       10 |    100 |     4.89  |    5.998 | 204.485 |         61.927 | pytorch                           | fp32        | 128x128 |    0.5  |          0.694 | Tesla T4 |
|       10 |    100 |     5.941 |    8.158 | 168.325 |         89.243 | pytorch                           | fp16        | 128x128 |    0.5  |          0.694 | Tesla T4 |
|       10 |    100 |     9.965 |   10.935 | 100.352 |        nan     | onnxruntime/CUDAExecutionProvider | fp32        | 256x256 |    1    |          0.914 | Tesla T4 |
### CholecSeg8k
|   warmup |   runs |   mean_ms |   p95_ms |     fps |   vram_peak_mb | backend                           | precision   | input   |   scale |   dice_mean_fg | gpu      |
|---------:|-------:|----------:|---------:|--------:|---------------:|:----------------------------------|:------------|:--------|--------:|---------------:|:---------|
|       10 |    100 |    10.677 |   13.216 |  93.656 |        135.773 | pytorch                           | fp32        | 256x256 |    1    |          0.839 | Tesla T4 |
|       10 |    100 |     9.931 |   11.554 | 100.698 |         61.03  | pytorch                           | fp16        | 256x256 |    1    |          0.839 | Tesla T4 |
|       10 |    100 |     7.834 |    8.423 | 127.65  |         37.142 | pytorch                           | fp32        | 192x192 |    0.75 |          0.826 | Tesla T4 |
|       10 |    100 |    12.645 |   17.063 |  79.08  |         51.41  | pytorch                           | fp16        | 192x192 |    0.75 |          0.826 | Tesla T4 |
|       10 |    100 |     7.424 |    8.581 | 134.707 |         30.687 | pytorch                           | fp32        | 128x128 |    0.5  |          0.685 | Tesla T4 |
|       10 |    100 |    10.056 |   12.065 |  99.444 |         44.315 | pytorch                           | fp16        | 128x128 |    0.5  |          0.684 | Tesla T4 |
|       10 |    100 |     8.537 |   12.796 | 117.139 |        nan     | onnxruntime/CUDAExecutionProvider | fp32        | 256x256 |    1    |          0.839 | Tesla T4 |

## f) Tuần 3-4
### CT: nnU-Net 3D (cùng ca test, không gian ảnh gốc)
| label        |   dice |   iou |    hd95 |   nsd |   dice_std |   n_cases |
|:-------------|-------:|------:|--------:|------:|-----------:|----------:|
| spleen       |  0.933 | 0.876 |  24.626 | 0.918 |      0.015 |         6 |
| kidney_right |  0.84  | 0.749 |  24.281 | 0.758 |      0.165 |         6 |
| kidney_left  |  0.83  | 0.741 |  22.904 | 0.765 |      0.192 |         6 |
| gallbladder  |  0.536 | 0.411 |  89.091 | 0.486 |      0.297 |         6 |
| liver        |  0.948 | 0.902 |  54.334 | 0.839 |      0.013 |         6 |
| pancreas     |  0.683 | 0.519 | 149.591 | 0.616 |      0.043 |         6 |
| MEAN         |  0.795 | 0.7   |  60.804 | 0.73  |    nan     |       nan |
### Phân tích lỗi 2D
CAMUS:

| label      |   dice |   nsd |   recall | most_confused_with   |   confused_frac |   frac_images_dice_lt_0p5 |   frac_missed_entirely |   corr_area_dice |
|:-----------|-------:|------:|---------:|:---------------------|----------------:|--------------------------:|-----------------------:|-----------------:|
| LV_endo    |  0.941 | 0.693 |    0.945 | myocardium           |           0.048 |                     0     |                      0 |            0.226 |
| myocardium |  0.883 | 0.642 |    0.873 | background           |           0.08  |                     0     |                      0 |           -0.371 |
| LA         |  0.919 | 0.648 |    0.896 | background           |           0.093 |                     0.005 |                      0 |            0.206 |

CholecSeg8k:

| label                  |    dice |     nsd |   recall | most_confused_with   |   confused_frac |   frac_images_dice_lt_0p5 |   frac_missed_entirely |   corr_area_dice |
|:-----------------------|--------:|--------:|---------:|:---------------------|----------------:|--------------------------:|-----------------------:|-----------------:|
| Abdominal Wall         |   0.922 |   0.752 |    0.95  | Fat                  |           0.024 |                     0.006 |                  0     |            0.74  |
| Liver                  |   0.912 |   0.723 |    0.932 | Abdominal Wall       |           0.039 |                     0     |                  0     |            0.349 |
| Gastrointestinal Tract |   0.881 |   0.683 |    0.881 | Gallbladder          |           0.074 |                     0.001 |                  0     |            0.59  |
| Fat                    |   0.901 |   0.751 |    0.938 | Gallbladder          |           0.023 |                     0.001 |                  0     |            0.629 |
| Grasper                |   0.779 |   0.742 |    0.738 | Abdominal Wall       |           0.142 |                     0.032 |                  0.007 |            0.456 |
| Connective Tissue      | nan     | nan     |    0     | Black Background     |           0     |                     0     |                  0     |          nan     |
| Blood                  | nan     | nan     |    0     | Black Background     |           0     |                     0     |                  0     |          nan     |
| Cystic Duct            | nan     | nan     |    0     | Black Background     |           0     |                     0     |                  0     |          nan     |
| L-hook Electrocautery  | nan     | nan     |    0     | Black Background     |           0     |                     0     |                  0     |          nan     |
| Gallbladder            |   0.642 |   0.525 |    0.82  | Fat                  |           0.108 |                     0.238 |                  0.075 |            0.402 |
| Hepatic Vein           | nan     | nan     |    0     | Black Background     |           0     |                     0     |                  0     |          nan     |
| Liver Ligament         | nan     | nan     |    0     | Black Background     |           0     |                     0     |                  0     |          nan     |
### Ablation 2D (test; mỗi dòng đổi một thứ so với baseline)
CAMUS:

| config                            | arch     | encoder                       | pretrained   | loss       | augment                        |   seed |   params_M |   dice |   iou |   hd95 |     nsd |   best_val |   d_dice_vs_base |
|:----------------------------------|:---------|:------------------------------|:-------------|:-----------|:-------------------------------|-------:|-----------:|-------:|------:|-------:|--------:|-----------:|-----------------:|
| baseline (tuần 3)                 | Unet     | resnet18                      | imagenet     | Dice+CE    | affine+brightness, không lật   |     42 |      14.32 |  0.914 | 0.846 |  5.666 | nan     |      0.92  |            0     |
| aug_strong                        | Unet     | resnet18                      | imagenet     | dice_ce    | strong + brightness, không lật |     42 |      14.32 |  0.912 | 0.842 |  5.755 |   0.651 |      0.92  |           -0.002 |
| enc_r34                           | Unet     | resnet34                      | imagenet     | dice_ce    | affine + brightness, không lật |     42 |      24.43 |  0.912 | 0.841 |  5.869 |   0.647 |      0.919 |           -0.002 |
| ens_loss_focal+aug_strong+enc_r34 | ensemble | loss_focal+aug_strong+enc_r34 | -            | -          | -                              |     -1 |       0    |  0.918 | 0.852 |  5.435 |   0.677 |    nan     |            0.004 |
| loss_focal                        | Unet     | resnet18                      | imagenet     | dice_focal | affine + brightness, không lật |     42 |      14.32 |  0.915 | 0.847 |  5.627 |   0.664 |      0.919 |            0.001 |
| no_pretrain                       | Unet     | resnet18                      | nan          | dice_ce    | affine + brightness, không lật |     42 |      14.32 |  0.913 | 0.844 |  5.836 |   0.657 |      0.917 |           -0.001 |

CholecSeg8k:

| config                            | arch     | encoder                       | pretrained   | loss       | augment                        |   seed |   params_M |   dice |   iou |   hd95 |     nsd |   best_val |   d_dice_vs_base |
|:----------------------------------|:---------|:------------------------------|:-------------|:-----------|:-------------------------------|-------:|-----------:|-------:|------:|-------:|--------:|-----------:|-----------------:|
| baseline (tuần 3)                 | Unet     | mobilenet_v2                  | imagenet     | Dice+CE    | affine+brightness, không lật   |     42 |       6.63 |  0.839 | 0.753 | 22.101 | nan     |      0.775 |            0     |
| aug_focal                         | Unet     | mobilenet_v2                  | imagenet     | dice_focal | strong + brightness, không lật |     42 |       6.63 |  0.856 | 0.774 | 25.132 |   0.735 |      0.781 |            0.017 |
| aug_focal_s1                      | Unet     | mobilenet_v2                  | imagenet     | dice_focal | strong + brightness, không lật |      1 |       6.63 |  0.852 | 0.764 | 22.253 |   0.713 |      0.78  |            0.013 |
| aug_strong                        | Unet     | mobilenet_v2                  | imagenet     | dice_ce    | strong + brightness, không lật |     42 |       6.63 |  0.848 | 0.764 | 18.081 |   0.73  |      0.776 |            0.009 |
| aug_strong_s1                     | Unet     | mobilenet_v2                  | imagenet     | dice_ce    | strong + brightness, không lật |      1 |       6.63 |  0.854 | 0.765 | 26.064 |   0.723 |      0.777 |            0.015 |
| base_s1                           | Unet     | mobilenet_v2                  | imagenet     | dice_ce    | affine + brightness, không lật |      1 |       6.63 |  0.846 | 0.761 | 20.67  |   0.703 |      0.776 |            0.006 |
| enc_r34                           | Unet     | resnet34                      | imagenet     | dice_ce    | affine + brightness, không lật |     42 |      24.44 |  0.827 | 0.739 | 29.54  |   0.696 |      0.777 |           -0.012 |
| ens_loss_focal+aug_strong+enc_r34 | ensemble | loss_focal+aug_strong+enc_r34 | -            | -          | -                              |     -1 |       0    |  0.872 | 0.794 | 16.648 |   0.753 |    nan     |            0.032 |
| loss_focal                        | Unet     | mobilenet_v2                  | imagenet     | dice_focal | affine + brightness, không lật |     42 |       6.63 |  0.856 | 0.773 | 23.642 |   0.72  |      0.771 |            0.016 |
| no_pretrain                       | Unet     | mobilenet_v2                  | nan          | dice_ce    | affine + brightness, không lật |     42 |       6.63 |  0.74  | 0.652 | 32.464 |   0.56  |      0.743 |           -0.099 |

## g) Danh sách hình
- `camus/ablation/aug_strong/curves.png`: (Claude ghi ý nghĩa sau)
- `camus/ablation/aug_strong/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `camus/ablation/enc_r34/curves.png`: (Claude ghi ý nghĩa sau)
- `camus/ablation/enc_r34/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `camus/ablation/loss_focal/curves.png`: (Claude ghi ý nghĩa sau)
- `camus/ablation/loss_focal/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `camus/ablation/no_pretrain/curves.png`: (Claude ghi ý nghĩa sau)
- `camus/ablation/no_pretrain/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `camus/curves.png`: (Claude ghi ý nghĩa sau)
- `camus/dice_vs_fps.png`: (Claude ghi ý nghĩa sau)
- `camus/errors/confusion.png`: (Claude ghi ý nghĩa sau)
- `camus/figures/data_examples.png`: (Claude ghi ý nghĩa sau)
- `camus/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/aug_focal/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/aug_focal/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/aug_focal_s1/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/aug_focal_s1/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/aug_strong/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/aug_strong/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/aug_strong_s1/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/aug_strong_s1/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/base_s1/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/base_s1/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/enc_r34/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/enc_r34/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/loss_focal/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/loss_focal/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/no_pretrain/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/ablation/no_pretrain/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/curves.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/dice_vs_fps.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/errors/confusion.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/figures/data_examples.png`: (Claude ghi ý nghĩa sau)
- `cholecseg8k/figures/pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `ct/curves.png`: (Claude ghi ý nghĩa sau)
- `ct/figures/compare_amos_0015.png`: (Claude ghi ý nghĩa sau)
- `ct/figures/compare_amos_0035.png`: (Claude ghi ý nghĩa sau)
- `ct/figures/compare_amos_0078.png`: (Claude ghi ý nghĩa sau)
- `ct/figures/compare_amos_0127.png`: (Claude ghi ý nghĩa sau)
- `ct/figures/compare_amos_0376.png`: (Claude ghi ý nghĩa sau)
- `ct/figures/custom_pred_vs_gt.png`: (Claude ghi ý nghĩa sau)
- `ct/figures/data_examples.png`: (Claude ghi ý nghĩa sau)
