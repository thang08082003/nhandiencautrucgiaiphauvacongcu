# Tuần 3-4: mô hình chính xác, phân tích lỗi, cải tiến, ablation

Theo `Ke_hoach_6_tuan.docx`. Tuần 3 lập mốc tham chiếu và đánh giá chuẩn; tuần 4 thử 2-3 cải tiến, đo đóng góp từng cải tiến, chốt mô hình.

## Đã thêm vào code
| Việc | Nơi làm |
|---|---|
| Thêm **NSD** (ngưỡng 2 mm cho CT, 2 px cho 2D) cạnh Dice/IoU/HD95; mọi `metrics.csv` mới có cột `nsd` | `src/common.py` |
| `train_2d.py` cấu hình được: `--loss {dice_ce,dice_focal,ce}`, `--aug {none,affine,strong}`, `--encoder`, `--arch`, `--no-pretrain`, `--seed`, `--tag` (kết quả vào `results/<ds>/ablation/<tag>/`) | `src/train_2d.py`, `src/lib2d.py` |
| Phân tích lỗi (ma trận nhầm lẫn, nhãn hay bị nhầm với nhãn nào, tỉ lệ ảnh Dice<0.5, tương quan diện tích nhãn - Dice, ảnh tệ nhất), bảng ablation, ensemble | `src/analyze_2d.py` |
| nnU-Net v2 3d_fullres trên đúng chia tập train/val/test hiện có, đánh giá ở không gian gốc | `src/nnunet_ct.py` |
| Cell 10-13 trong notebook, mục "Tuần 3-4" trong `REPORT_DATA.md` | `colab_run.ipynb`, `src/report_tables.py` |

Giá trị mặc định của `train_2d.py` giữ nguyên cấu hình tuần 2, nên baseline cũ vẫn tái lập được. Baseline tuần 2 không có cột NSD: chạy lại `train_2d.py` (hoặc `analyze_2d.py errors`, nó tự đánh giá lại và ghi `errors/test_per_image_full.csv` có NSD) để có đủ bốn chỉ số.

## Trình tự chạy (Colab, sau Cell 7)
1. **Cell 10** phân tích lỗi CAMUS và CholecSeg8k.
2. **Cell 11** nnU-Net CT. Dùng `nnUNetTrainer_50epochs` cho vừa Colab; mặc định 1000 epoch là quá lâu. Nếu có thể, tăng lên 100-250 epoch rồi ghi rõ số epoch trong báo cáo.
3. **Cell 12** bốn cấu hình ablation mỗi bộ dữ liệu, mỗi cấu hình đổi đúng một thứ: `loss_focal`, `aug_strong`, `enc_r34`, `no_pretrain`. Ước tính 8 lượt x 30 phút = 4 giờ GPU.
4. **Cell 13** ensemble các cấu hình tốt, bảng ablation, gom `REPORT_DATA.md`.

## Cách đọc kết quả (cho báo cáo)
- Cải tiến chỉ đáng tin nếu chênh Dice lớn hơn dao động giữa các lần chạy. Hiện mỗi cấu hình chạy 1 seed; với chênh nhỏ (dưới ~0.01) hãy chạy lại với `--seed` khác trước khi kết luận, hoặc ghi rõ là chưa đủ bằng chứng.
- Bảng ablation có cột `d_dice_vs_base` so với baseline; `no_pretrain` cho biết pretrain ImageNet đóng góp bao nhiêu.
- `error_summary.csv`: `most_confused_with` + `confused_frac` trả lời "nhãn nào kém và vì sao"; `corr_area_dice` dương cao nghĩa là nhãn nhỏ kém hơn.
- Kết quả không cải thiện vẫn ghi vào báo cáo như kết quả âm tính (theo kế hoạch mục 5).

## Hạn chế cần nêu
- Test CT chỉ khoảng 6 ca (40 ca x 15%), độ không chắc chắn lớn; nên báo cáo thêm độ lệch chuẩn theo ca (`dice_std`).
- nnU-Net dùng val trong `splits_final.json` để chọn checkpoint, không đụng test.
- Ablation 2D và nnU-Net khác nhau về ngân sách huấn luyện; so sánh chéo chỉ mang tính tham khảo.
- Mã mới chỉ kiểm tra cú pháp (`py_compile`) vì môi trường này không có numpy/GPU/dữ liệu; chưa chạy thật trên Colab. Nếu gặp lỗi, dán nguyên văn output để sửa.
