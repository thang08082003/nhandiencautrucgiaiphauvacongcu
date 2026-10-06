"""Bước 5: gom các CSV thật thành results/REPORT_DATA.md (phần số liệu) + hình so sánh CT.
Các mục g (lỗi gặp phải), h (hạn chế), i (đề xuất) và nhận xét nhãn kém được viết tay sau khi có số liệu thật.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from common import CACHE, RESULTS, overlay, show_rows, start
from data_ct import HU_HI, HU_LO

R = RESULTS


def md(p, **kw):
    p = Path(p)
    return pd.read_csv(p).round(3).to_markdown(index=False, **kw) if p.exists() else f"_(chưa có {p.relative_to(R)})_"


def kv(p):
    p = Path(p)
    return "\n".join(f"- {k}: {v}" for k, v in pd.read_csv(p).values) if p.exists() else f"_(chưa có {p.relative_to(R)})_"


def ct_compare():
    ts, cu = R / "ct" / "totalseg_metrics.csv", R / "ct" / "test_per_case.csv"
    if not (ts.exists() and cu.exists()):
        return "_(thiếu totalseg_metrics.csv hoặc test_per_case.csv)_"
    a, b = pd.read_csv(ts), pd.read_csv(cu)
    b = b[b.case.isin(a.case)]
    cmp = a.groupby("label", sort=False)[["dice", "hd95"]].mean().join(
        b.groupby("label", sort=False)[["dice", "hd95"]].mean(), lsuffix="_totalseg", rsuffix="_custom")
    cmp.loc["MEAN"] = cmp.mean()
    cmp.reset_index().to_csv(R / "ct" / "compare_totalseg_vs_custom.csv", index=False)
    t = pd.DataFrame({"mô hình": ["TotalSegmentator", "3D U-Net tự huấn luyện"],
                      "giây/ca (TB)": [a.groupby("case").seconds.first().mean(), b.groupby("case").seconds.first().mean()],
                      "ghi chú": [f"gồm nạp mô hình+tiền xử lý+ghi file, {a['mode'].iloc[0]}, {a.device.iloc[0]}",
                                  "sliding window + resample về lưới gốc"]})
    for c in sorted(a.case.unique()):  # hình so sánh
        im = sitk.GetArrayFromImage(sitk.ReadImage(str(CACHE / "ct" / "test_raw" / f"{c}_img.nii.gz"))).astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(CACHE / "ct" / "test_raw" / f"{c}_lab.nii.gz")))
        pt = sitk.GetArrayFromImage(sitk.ReadImage(str(CACHE / "ct" / "pred_ts" / f"{c}.nii.gz")))
        pc = sitk.GetArrayFromImage(sitk.ReadImage(str(CACHE / "ct" / "pred_custom" / f"{c}.nii.gz")))
        z = int(np.argmax((gt > 0).sum((1, 2))))
        w = np.clip((im[z] - HU_LO) / (HU_HI - HU_LO), 0, 1)
        show_rows(R / "ct" / "figures" / f"compare_{c}.png", [[overlay(w, m[z], 7) for m in (gt, pt, pc)]],
                  ["ground truth", "TotalSegmentator", "tự huấn luyện"], c)
    return cmp.reset_index().round(3).to_markdown(index=False) + "\n\n" + t.round(1).to_markdown(index=False)


def main():
    start()
    figs = sorted(p.relative_to(R) for p in R.rglob("*.png"))
    txt = f"""# REPORT_DATA (số liệu thật từ các lần chạy; sinh bởi src/report_tables.py)

## a) Môi trường
{(R / 'env_report.md').read_text() if (R / 'env_report.md').exists() else '_(chưa có env_report.md)_'}

## b) Dữ liệu
### CT (AMOS, tập con)
{kv(R / 'ct' / 'dataset_overview.csv')}

{md(R / 'ct' / 'label_stats.csv')}

Ánh xạ nhãn:

{md(R / 'ct' / 'label_mapping.csv')}

### CAMUS
{kv(R / 'camus' / 'dataset_overview.csv')}
### CholecSeg8k
{kv(R / 'cholecseg8k' / 'dataset_overview.csv')}

{md(R / 'cholecseg8k' / 'dataset_stats.csv')}

Danh sách chia tập: results/splits/*.csv

## c) CT: TotalSegmentator và mô hình tự huấn luyện (cùng các ca test, không gian ảnh gốc)
> TotalSegmentator có thể đã thấy dữ liệu tương tự (kể cả AMOS) lúc huấn luyện, nên đây chỉ là baseline tham khảo.

{ct_compare()}

Mô hình tự huấn luyện, toàn bộ ca test:

{md(R / 'ct' / 'metrics.csv')}

Cấu hình: {(R / 'ct' / 'config.json').read_text() if (R / 'ct' / 'config.json').exists() else '_(chưa có)_'}

## d) CAMUS và CholecSeg8k (test, trung bình theo ảnh; HD95 đơn vị pixel ở 256x256; MEAN = trung bình các nhãn trừ nền)
### CAMUS
{md(R / 'camus' / 'metrics.csv')}
### CholecSeg8k
{md(R / 'cholecseg8k' / 'metrics.csv')}

## e) Tốc độ
### CT
{md(R / 'ct' / 'speed_benchmark.csv')}
### CAMUS
{md(R / 'camus' / 'speed_benchmark.csv')}
### CholecSeg8k
{md(R / 'cholecseg8k' / 'speed_benchmark.csv')}

## f) Tuần 3-4
### CT: nnU-Net 3D (cùng ca test, không gian ảnh gốc)
{md(R / 'ct' / 'nnunet_metrics.csv')}
### Phân tích lỗi 2D
CAMUS:

{md(R / 'camus' / 'errors' / 'error_summary.csv')}

CholecSeg8k:

{md(R / 'cholecseg8k' / 'errors' / 'error_summary.csv')}
### Ablation 2D (test; mỗi dòng đổi một thứ so với baseline)
CAMUS:

{md(R / 'camus' / 'ablation_table.csv')}

CholecSeg8k:

{md(R / 'cholecseg8k' / 'ablation_table.csv')}

## g) Danh sách hình
""" + "\n".join(f"- `{f}`: (Claude ghi ý nghĩa sau)" for f in figs) + "\n"
    (R / "REPORT_DATA.md").write_text(txt)
    print("Đã ghi", R / "REPORT_DATA.md")


if __name__ == "__main__":
    main()
