# Đưa mã nguồn lên GitHub lần đầu

Hướng dẫn này dành cho chủ tài khoản `hieudc-vub`, repository `dl-landslide-vietnam`. Gói mã nguồn đã được chuẩn bị trên máy; hướng dẫn không có nghĩa repository đã được cập nhật tự động.

## 1. Chọn đúng thư mục

Giải nén gói source mới. Mở thư mục `dl-landslide-vietnam` chứa `README.md`, `docs/translations/README.vi.md`, `src`, `configs`, `.github`. Chỉ đăng nội dung thư mục này. Không đăng thư mục tổng `Reproducibility_DATA`, dataset, `.venv`, các file `.pth`, kết quả chạy hoặc notebook gốc. Bật **View → Show → Hidden items** trong File Explorer để nhìn thấy `.github` và `.gitignore`.

## 2. Mở repository

Đăng nhập GitHub, mở https://github.com/hieudc-vub/dl-landslide-vietnam. Nếu chưa có repository, bấm dấu **+ → New repository**, chọn owner `hieudc-vub`, nhập `dl-landslide-vietnam`. Nên giữ **Private** trong lúc kiểm tra; có thể chuyển Public khi sẵn sàng. Không tạo thêm README hoặc license mặc định nếu sẽ tải bộ file này lên.

Nếu repository đã có nội dung, trước tiên bấm **Code → Download ZIP** để lưu bản hiện tại. Upload có thể cập nhật file trùng tên; kiểm tra danh sách thay đổi trước khi commit. Các file cũ không có trong gói mới không tự biến mất khi upload. Nếu có nhiều nội dung cũ khác cấu trúc, gửi ảnh/danh sách cây thư mục để được hướng dẫn đối chiếu trước khi xóa bất cứ thứ gì.

## 3. Tải file lên

1. Trong repository, chọn **Add file → Upload files**. Repository trống có liên kết **uploading an existing file**.
2. Từ trong thư mục `dl-landslide-vietnam`, chọn các file/thư mục con và kéo vào vùng upload. Không kéo cả thư mục cha, tránh tạo thêm một tầng `dl-landslide-vietnam` trong repository.
3. Đợi upload xong. Kiểm tra có `README.md`, `CITATION.cff`, `LICENSE`, `.github/workflows/tests.yml`, `src/landslide/cli.py`.
4. Nhập commit message: `Add reproducible Windows benchmark source v0.4.0`.
5. Đọc lại danh sách file, sau đó bấm **Commit changes**.

Không upload file ZIP làm nội dung duy nhất của repository. GitHub cần nhìn thấy các file nguồn để hiển thị README, trích dẫn và chạy kiểm thử. Upload trình duyệt có giới hạn số lượng/kích thước; gói nguồn nhỏ này không chứa raster hoặc checkpoint lớn. Nếu giao diện từ chối một nhóm file, chia thành các lượt nhỏ và giữ nguyên đường dẫn.

## 4. Điền phần About

Ở trang chính repository, bấm biểu tượng bánh răng cạnh **About**. Description:

```text
PyTorch workflows for training and evaluating binary landslide segmentation models using prepared Sentinel-2 and ALOS AW3D30 patches.
```

Topics, nhập từng mục:

```text
deep-learning
landslide-detection
sentinel-2
alos-aw3d30
remote-sensing
benchmark
pytorch
semantic-segmentation
reproducibility
```

Bấm **Save changes**. `CITATION.cff` đã đặt `repository-code` đúng URL. Khi GitHub nhận file hợp lệ, mục **Cite this repository** sẽ xuất hiện.

## 5. Kiểm tra tự động

Bấm tab **Actions**, mở lần chạy mới nhất của workflow **Windows CPU tests**. Dấu tích xanh nghĩa là kiểm thử CI đó đã qua; không có nghĩa 13 mô hình đã được huấn luyện đủ trên dataset. Nếu đỏ, mở job, chụp hoặc sao chép thông báo lỗi để được hỗ trợ. Không bỏ qua lỗi bằng cách xóa kiểm thử.

Tải lại mã từ GitHub vào một thư mục khác, làm theo `docs/translations/README.vi.md` đến bước kiểm thử để kiểm tra cách độc giả nhận gói. Không thử trong notebook/môi trường gốc.

## 6. Hoàn thiện dữ liệu rồi công bố

Dataset đã có DOI 10.5281/zenodo.23838723. Dùng hướng dẫn docs/DATASET_RELEASE.md: tải ba ZIP raster từ Zenodo, sử dụng schema và manifest trong GitHub. Không yêu cầu archive metadata thứ tư. DOI này dành cho dataset, không phải bài báo hay mã nguồn.

Sau khi kiểm tra xong và muốn công khai: **Settings → General → Danger Zone → Change repository visibility → Public**, đọc xác nhận của GitHub trước khi thực hiện. Nếu repository đã Public thì bỏ qua bước này.

Tạo phiên bản: trang chính → **Releases → Draft a new release**, tạo tag `v0.4.0`, title `v0.4.0 — Windows reproducibility source`, mô tả theo `CHANGELOG.md`. Nêu rõ chưa có checkpoint và trạng thái dataset. Chỉ bấm **Publish release** sau khi kiểm tra thông tin. Không điền DOI bài báo vì bài còn chuẩn bị submission.

Bạn không cần cung cấp mật khẩu GitHub hoặc token cho người hỗ trợ. Nếu cần hỗ trợ bước tiếp theo, gửi URL repository và ảnh màn hình bước đang gặp khó khăn.
