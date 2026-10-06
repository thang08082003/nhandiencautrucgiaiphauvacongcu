# STATUS (cập nhật 2026-10-06, buổi tối)

Tệp này để trợ lý khác (chỉ đọc được GitHub repo này và Drive `MyDrive/anatomy_project/`) nắm trạng thái chạy thực nghiệm.

## Đã xong (kết quả nằm ở Drive `results/`, bảng tổng ở `results/REPORT_DATA.md`)
- Tuần 1-2: 3 mô hình nền (CT 3D U-Net, CAMUS U-Net ResNet-18, CholecSeg8k U-Net MobileNetV2).
- Tuần 3-4 phần 2D: Cell 10 (phân tích lỗi), Cell 12 (8 lượt ablation, 0 lỗi, không NaN), Cell 13 (ensemble + bảng ablation + `REPORT_DATA.md`).
- Tất cả chạy 1 seed (42). Cột NSD của dòng `baseline (tuần 3)` là `nan` (baseline huấn luyện trước khi có NSD).

## Chưa xong: nnU-Net 3D CT (Cell 11)
Chưa có `results/ct/nnunet_metrics.csv`; mục f của `REPORT_DATA.md` ghi "(chưa có ct/nnunet_metrics.csv)". Lý do, theo thứ tự:
1. `src/nnunet_ct.py` dòng 39 (`next(OUT.rglob(...))`) báo `StopIteration`: `OUT = /content/data/amos` bị mất khi runtime Colab reset. Cách xử lý: chạy lại Cell 3 (`python data_ct.py --n 40`, cần gõ `yes` xác nhận tải ~1525 MB).
2. `os.symlink` trên ổ Drive báo `OSError: [Errno 95] Operation not supported`. Đã sửa bằng `shutil.copy` (commit `561e753` trên `main`, bản trên Drive cũng đã sửa).
3. Sau khi sửa, `prepare` chạy qua, nnU-Net đang lập kế hoạch/tiền xử lý thì Colab báo hết hạn mức GPU. Chưa tới bước `train`. Thư mục `cache/nnunet/` trên Drive có trạng thái `prepare` dở.

## Việc tiếp theo (khi có lại GPU Colab)
Chạy tuần tự: Cell 1 -> Cell 3 (người dùng đồng ý `yes`) -> Cell 11 -> (nếu có) Cell 14 -> `python report_tables.py` để cập nhật `REPORT_DATA.md`.

## Quy ước làm việc
- Code thay đổi phải lên GitHub `main`; người dùng chép `src/` và `colab_run.ipynb` lên Drive (Colab đọc từ Drive, không đọc GitHub).
- Đừng gửi patch qua tệp cục bộ; hãy sửa thẳng trên `main` hoặc mô tả thay đổi để người dùng áp dụng.
- Không đưa token (Kaggle, API) vào repo hay chat. Repo giữ private (dữ liệu CAMUS/CholecSeg8k là CC BY-NC-SA 4.0).
- HD95: 2D tính bằng pixel (256x256), CT tính bằng mm, không so sánh trực tiếp.
- Đừng bật lại AMP trong `src/train_ct.py` (fp16 gây NaN).
- Cell có lệnh `rm -r` trong notebook Colab trên Drive (cell tạm) chỉ dùng một lần; không chạy lại, sẽ xóa kết quả ablation thật.
