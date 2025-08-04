# YOLOv8 Person Detection System

Hệ thống phát hiện người sử dụng YOLOv8 với khả năng xử lý ảnh, video và thời gian thực.

## 📋 Mô tả

Dự án này là một hệ thống phát hiện người sử dụng mô hình YOLOv8, được thiết kế để phát hiện và đếm số lượng người trong ảnh, video hoặc từ camera thời gian thực. Hệ thống cung cấp giao diện dòng lệnh dễ sử dụng với nhiều tùy chọn xử lý khác nhau.

## ✨ Tính năng

- **Phát hiện người thời gian thực**: Sử dụng webcam để phát hiện người theo thời gian thực
- **Xử lý video**: Phát hiện người trong file video với khả năng lưu kết quả
- **Xử lý ảnh**: Phát hiện người trong ảnh tĩnh
- **Hiển thị thông tin chi tiết**: Bounding box, độ tin cậy, số lượng người phát hiện được
- **Tùy chỉnh tham số**: Điều chỉnh ngưỡng tin cậy và IoU
- **Lưu kết quả**: Khả năng lưu frame và video đã xử lý
- **Hiệu suất cao**: Sử dụng YOLOv8 tối ưu cho tốc độ và độ chính xác

## 🏗️ Cấu trúc dự án

```
YOLO/
├── config/
│   └── settings.py          # Cấu hình mô hình và hiển thị
├── data/                    # Thư mục chứa dữ liệu mẫu
│   ├── cat.jpg
│   ├── human.jpg
│   ├── istockphoto-2193050456-640_adpp_is.mp4
│   └── nguoi.mp4
├── models/
│   └── yolov8n.pt          # Mô hình YOLOv8 nano
└── src/
    ├── detector.py          # Lớp phát hiện người chính
    ├── main.py              # File chính với giao diện CLI
    ├── real_time.py         # Xử lý thời gian thực
    ├── utils.py             # Các hàm tiện ích
    └── video_processor.py   # Xử lý video
```

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Webcam (cho chức năng thời gian thực)

### Cài đặt dependencies

```bash
# Cài đặt OpenCV
pip install opencv-python

# Cài đặt Ultralytics YOLO
pip install ultralytics

# Hoặc cài đặt tất cả dependencies
pip install -r requirements.txt
```

### Tải mô hình

Mô hình YOLOv8n sẽ được tải tự động khi chạy lần đầu. Hoặc bạn có thể tải thủ công:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```

## 📖 Cách sử dụng

### Chạy chương trình chính

```bash
python src/main.py
```

Sau khi chạy, bạn sẽ thấy menu với 3 tùy chọn:

1. **Real-time detection with webcam** - Phát hiện thời gian thực
2. **Process video file** - Xử lý file video
3. **Process image file** - Xử lý file ảnh