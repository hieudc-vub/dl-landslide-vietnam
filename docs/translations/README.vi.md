[English](../../README.md) · [Français](README.fr.md) · [Nederlands](README.nl.md) · **Tiếng Việt**

# dl-landslide-vietnam

Quy trình PyTorch cho phân đoạn nhị phân trượt lở đất

Mã nguồn đi kèm nghiên cứu lập bản đồ trượt lở đất từ Sentinel-2 hai thời điểm và ALOS AW3D30 tại miền Bắc Việt Nam. Bộ mã cung cấp 13 cấu hình kiến trúc cho patch đã chuẩn bị theo schema dự án. Độc giả tự huấn luyện mô hình; checkpoint đã huấn luyện của tác giả không được phân phối.

[Cài đặt và kiểm tra môi trường](#installation) · [Dataset](#dataset) · [Bắt đầu chạy](#quick-start) · [Trích dẫn và ghi công](#citation)

| Nhu cầu | Xem mục |
|---|---|
| Hiểu quy trình | [Hiểu quy trình](#overview) |
| Kiểm tra cài đặt | [Kiểm tra cài đặt](#installation) |
| Chuẩn bị đầu vào | [Chuẩn bị đầu vào](#dataset) |
| Chạy một mô hình | [Chạy một mô hình](#quick-start) |
| Tìm file được tạo | [Tìm file được tạo](#outputs) |
| Báo lỗi mã nguồn | [Báo lỗi mã nguồn](#support) |

<a id="overview"></a>

## Tổng quan

Đầu vào: patch 256 × 256 với đúng 20 band theo thứ tự dự án và nhãn nhị phân (0 nền, 1 trượt lở). Quy trình: huấn luyện, chọn checkpoint và hiệu chỉnh ngưỡng trên VAL, đánh giá TEST, dự đoán. Đầu ra được tạo cục bộ khi người dùng chạy. Repository không chứa kết quả trong bài báo.

<a id="before"></a>

## Trước khi bắt đầu

Quy trình được hướng dẫn dùng Windows, Python 3.12 (tham chiếu 3.12.8), PyTorch 2.9.0 với CUDA 12.8. Luồng huấn luyện FP16 được mô tả dùng GPU NVIDIA CUDA. CPU chạy được kiểm thử cài đặt giả lập. Nhu cầu bộ nhớ khác nhau giữa kiến trúc; chưa xác định cấu hình GPU tối thiểu ở đây. Internet cần cho cài đặt, dataset và trọng số ImageNet khởi tạo. Các phụ thuộc trực tiếp được cố định phiên bản, chưa phải khóa toàn bộ phụ thuộc gián tiếp.

| Yêu cầu | Cấu hình được mô tả | Kiểm tra / hướng dẫn |
|---|---|---|
| OS | Windows local | Mở PowerShell tại thư mục dự án |
| Python | Python 3.12; tham chiếu 3.12.8 | Lệnh kiểm tra phiên bản bên dưới |
| GPU | CUDA cho luồng huấn luyện FP16 được mô tả | Chẩn đoán môi trường sau cài đặt |
| Data | Dataset dự án đã chuẩn bị | Schema và kiểm tra dataset bên dưới |

<a id="installation"></a>

## Cài đặt và kiểm tra môi trường

Tải ZIP repository và giải nén. Mở thư mục chứa README.md và pyproject.toml. Bấm thanh địa chỉ File Explorer, gõ powershell rồi Enter. Chạy các lệnh dưới đây tại thư mục này; chúng tạo môi trường riêng cho dự án. Đợi dòng SETUP COMPLETE. Tùy chọn execution policy chỉ áp dụng cho tiến trình PowerShell đó.

Nếu chưa cài Python, mở [trang Python 3.12.8 chính thức](https://www.python.org/downloads/release/python-3128/) để lấy bản tham chiếu. Trong trình cài Windows, bật Python launcher và tùy chọn PATH rồi mở lại PowerShell. Tại [trang repository](https://github.com/hieudc-vub/dl-landslide-vietnam), chọn **Code → Download ZIP**, sau đó **Extract All** trong Windows. Luôn chạy lệnh từ thư mục gốc đã giải nén, kể cả khi đang đọc bản dịch.

```powershell
py -3.12 --version
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Compute cuda
.\.venv\Scripts\python.exe -m landslide check-env
```

Sau đó chạy kiểm thử cài đặt giả lập. Kiểm thử tạo dữ liệu tạm, không tải dataset nghiên cứu hoặc trọng số pretrained. Trạng thái cuối cần có: OK. Đây là kiểm tra hành vi phần mềm, không phải hiệu năng nghiên cứu. Đầu ra tạm được dọn sau kiểm thử.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```


<a id="dataset"></a>

## Dataset

Dataset v1.0.0: [Zenodo](https://doi.org/10.5281/zenodo.23838723). Tải ba ZIP raster; schema và manifest được cung cấp trong repository này. Xem [hướng dẫn tải và xác minh](../DATASET_RELEASE.md). Chạy đầy đủ cần SAMPLE_DATASET_256x256. Đặt dưới data/ như cây thư mục hoặc truyền đường dẫn hiện có trong dấu ngoặc kép. Không thay thứ tự band hoặc chuẩn hóa thêm. Giá trị phải hữu hạn trong [-1,1], với dung sai kiểm tra đầu vào 1e-5; pixel không hợp lệ/NoData bị từ chối. Tên file và metadata không gian của STACK/LABEL phải khớp.

```text
data/SAMPLE_DATASET_256x256/
├── TRAIN_256/   STACK/ + LABEL/
├── VAL_256/     STACK/ + LABEL/
└── TEST_256/    STACK/ + LABEL/
```

Cấu hình dự án khai báo rõ độ phân giải danh nghĩa 10 m/pixel. Diện tích bằng số pixel × 100 m²; đây không phải độ phân giải suy ra từ CRS của patch. Thiếu hoặc khai báo pixel_size_m không hợp lệ sẽ bị từ chối. Chi tiết schema và nguồn dữ liệu nằm trong tài liệu liên kết. Dữ liệu và TRAIN/VAL/TEST phải được chuẩn bị trước khi dùng bộ mã.

<a id="quick-start"></a>

## Bắt đầu chạy

Kiểm tra dataset rồi chạy một cấu hình ví dụ. Kiểm tra dữ liệu đọc và tính mã băm từng raster. Lệnh run bắt đầu huấn luyện thật, hiệu chỉnh trên VAL, đánh giá TEST và ghi đầu ra cục bộ; quá trình có thể kéo dài. Mỗi lần chạy dùng thư mục đầu ra mới. Kiến trúc ví dụ chỉ minh họa thao tác, không hàm ý ưu thế hiệu năng.

```powershell
.\.venv\Scripts\python.exe -m landslide validate-data --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\dataset_validation
.\.venv\Scripts\python.exe -m landslide run --config .\configs\models\04_unetplusplus_resnet50.json --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\04_unetplusplus_resnet50 --device cuda
```

Dự đoán là lệnh riêng, dùng checkpoint và ngưỡng do lần chạy của bạn tạo:

```powershell
.\.venv\Scripts\python.exe -m landslide predict --checkpoint .\runs\04_unetplusplus_resnet50\best_model.pth --threshold .\runs\04_unetplusplus_resnet50\calibration\threshold.json --input .\data\SAMPLE_DATASET_256x256\TEST_256\STACK --out .\runs\04_predictions --device cuda
```


<a id="outputs"></a>

## Các file được tạo

Cây dưới đây chỉ minh họa vị trí file. Checkpoint, bảng, hình và raster được tạo cục bộ, mặc định bị loại khỏi Git. Không có số đo hoặc ảnh kết quả ở đây. JSON null hoặc ô CSV trống nghĩa là chỉ số không xác định, không phải bằng 0.

```text
runs/04_unetplusplus_resnet50/
├── config.json, environment.json, split_inventory.json
├── best_model.pth, last_model.pth
├── training_log.csv, training_summary.json, training_curves.png
├── calibration/threshold.json, val_threshold_sweep.csv
├── evaluation/
│   ├── test_summary.json, test_per_image.csv
│   ├── Table_MainResults.csv, Table_SizeRecall.csv, Table_Boundary.csv
│   ├── size_stratified_object_recall.csv
│   └── confusion_matrix.png
└── benchmark.json
runs/04_predictions/
├── <input-name>_probability.tif
├── <input-name>_mask.tif
└── prediction_meta.json
```


<a id="method"></a>

## Mô hình và phương pháp

Cả 13 cấu hình dùng batch 16, learning rate 1e-4, weight decay 0.02, seed 42, tối đa 150 epoch, early-stopping patience 20 và num_workers=0. Tắt augmentation trực tuyến và weighted sampler. Huấn luyện CUDA dùng FP16 autocast; chọn ngưỡng, đánh giá và export dùng FP32. Tài liệu protocol định nghĩa loss, thống kê overlap có điều kiện, đối tượng 8-connectivity và các chỉ số biên.

```text
TRAIN → Training
VAL → Checkpoint selection → Threshold calibration
Best checkpoint + frozen threshold + TEST → Evaluation
Best checkpoint + frozen threshold + STACK → Prediction
```

| ID | Decoder | Encoder |
|---|---|---|
| 01 | unet | efficientnet-b4 |
| 02 | unet | resnet50 |
| 03 | unetplusplus | efficientnet-b4 |
| 04 | unetplusplus | resnet50 |
| 05 | unet | tu-hrnet_w18 |
| 06 | unet | tu-hrnet_w32 |
| 07 | deeplabv3plus | resnet50 |
| 08 | deeplabv3plus | mobilenet_v2 |
| 09 | upernet | resnet50 |
| 10 | segformer | mit_b0 |
| 11 | segformer | mit_b2 |
| 12 | segformer | mit_b5 |
| 13 | unet | tu-swin_tiny_patch4_window7_224 |


<a id="documentation"></a>

## Tài liệu

Các liên kết dưới đây dẫn tới schema, định nghĩa chỉ số, cách đọc đầu ra, lệnh chạy 13 mô hình và xử lý lỗi. Tài liệu kỹ thuật bổ trợ hiện bằng tiếng Anh; hướng dẫn đăng GitHub dành cho chủ dự án bằng tiếng Việt.

[Dataset](../../docs/DATASET.md) · [Protocol](../../docs/PROTOCOL.md) · [Outputs](../../docs/interpreting_outputs.md) · [All models](../../docs/RUNNING.md) · [Troubleshooting](../../docs/troubleshooting.md) · [Tests](../../docs/VALIDATION.md) · [Changelog](../../CHANGELOG.md)


<a id="support"></a>

## Phạm vi, kiểm thử và hỗ trợ

Hỗ trợ áp dụng cho mã đã phát hành, schema dataset dự án và quy trình được hướng dẫn. Người dùng tự chịu trách nhiệm điều chỉnh và kiểm tra khi dùng dataset hoặc phương pháp khác; việc điều chỉnh nằm ngoài phạm vi hỗ trợ. Báo lỗi mã đã phát hành qua GitHub Issues sau khi xóa đường dẫn riêng và thông tin bí mật. Không cam kết thời gian phản hồi. Trạng thái kiểm thử chỉ áp dụng cho các kiểm thử và môi trường đã dùng; không xác lập tính đúng đắn khoa học, độc lập không gian giữa các tập hoặc kết quả giống nhau trên mọi máy.

[GitHub Issues](https://github.com/hieudc-vub/dl-landslide-vietnam/issues) · [Contributing](../../CONTRIBUTING.md)

<a id="citation"></a>

## Trích dẫn và ghi công

Bản thảo liên quan: *A Systematic Benchmark of Deep Learning Architectures for Landslide Inventory Mapping with Bi-Temporal Sentinel-2 and ALOS AW3D30 in Northern Vietnam*. Tác giả: Cong Hieu Duong, Jonathan Cheung-Wai Chan, Chinh Luu, Matthieu Kervyn. Bài đang chuẩn bị submission; không gán DOI bài báo ở đây. Dùng CITATION.cff và trích dẫn dataset riêng bằng DOI [10.5281/zenodo.23838723](https://doi.org/10.5281/zenodo.23838723). Thông tin nguồn dữ liệu được ghi riêng; không hàm ý tổ chức bảo trợ hay chứng nhận.

[CITATION.cff](../../CITATION.cff) · [Data sources](../../docs/THIRD_PARTY_DATA.md)

<a id="licenses"></a>

## Giấy phép và bản dịch

Mã dự án: MIT. Dataset đã chuẩn bị: CC BY-NC 4.0 trong phạm vi quyền của chủ dữ liệu, giữ thông báo nguồn. Thư viện bên thứ ba và trọng số khởi tạo có điều khoản riêng. Polygon gốc không nằm trong gói phát hành.

[MIT](../../LICENSE) · [Dataset license](../../LICENSE-DATA.md)

README có tiếng Anh, Pháp, Hà Lan và Việt. Bản tiếng Anh ở thư mục gốc là nội dung đối chiếu; bản dịch nằm trong docs/translations/. Có thể đề xuất sửa bản dịch bằng pull request vào file tương ứng. Bốn bản được đồng bộ với phiên bản 0.4.0; thông tin này không phải chứng nhận dịch thuật độc lập.
