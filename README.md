# Phân Cụm Khách Sạn Dựa Trên Đặc Tính Và Cảm Xúc Của Khách Hàng

Đồ án cơ sở ngành Khoa học Dữ liệu – Đại học Công nghệ TP.HCM (HUTECH).  
Hệ thống phân tích dữ liệu đánh giá (reviews) từ Booking.com, trích xuất đặc trưng ngữ nghĩa bằng mô hình ngôn ngữ lớn (Multilingual-E5-Large), gán khía cạnh dịch vụ (Aspect) và áp dụng thuật toán phân cụm (KMeans) để phân nhóm khách sạn theo trải nghiệm khách hàng.

---

## 📌 Tổng Quan Dữ Liệu & Thực Nghiệm
- **Dữ liệu đầu vào:** 63 tỉnh/thành, 2.325 khách sạn và 18.779 reviews thu thập từ Booking.com.
- **Biểu diễn văn bản:** Sử dụng mô hình `intfloat/multilingual-e5-large` để tạo vector ngữ nghĩa 1024 chiều cho review đa ngôn ngữ.
- **Aspect trích xuất:** 10 khía cạnh chính (`room`, `cleanliness`, `service`, `location`, `price`, `food`, `facilities`, `wifi`, `noise`, `checkin`).
- **Xử lý đặc trưng:** Xây dựng 54 đặc trưng cấp khách sạn, chuẩn hóa bằng `RobustScaler` và giảm chiều dữ liệu bằng `PCA` xuống 8 thành phần chính (tỷ lệ giữ lại phương sai 74,05%).
- **Mô hình tối ưu:** KMeans ($K=3$) đạt Silhouette Score = 0.3701, Davies-Bouldin Index = 1.2159, Calinski-Harabasz Index = 791.19.

---

## 📂 Cấu Trúc Thư Mục

```text
├── data/               # Dữ liệu khách sạn và đánh giá thô
├── docs/               # Báo cáo đồ án chi tiết (PDF)
├── src/                # Pipeline mã nguồn xử lý dự án
│   ├── config.py       # Cấu hình đường dẫn và MongoDB
│   ├── mongo_utils.py  # Hàm tiện ích kết nối cơ sở dữ liệu MongoDB
│   ├── 01_scan_raw_data.py
│   ├── 02_import_raw_reviews_to_mongodb.py
│   ├── 03_setup_and_check_mongodb.py
│   ├── 04_eda_mongodb_data.py
│   ├── 05_preprocess_reviews.py
│   ├── 06_export_reviews_for_colab.py
│   ├── 07_colab_e5_large_embedding_topic_aspect.ipynb
│   ├── 08_import_colab_e5_large_results_to_mongodb.py
│   ├── 09_build_hotel_features.py
│   ├── 10_compare_and_train_clustering.py
│   └── 11_visualize_cluster_praise_complaints.ipynb
├── .gitignore          # File cấu hình bỏ qua cache, venv, mô hình nặng
├── README.md           # Tài liệu hướng dẫn dự án
└── requirements.txt    # Danh sách thư viện Python cần thiết

## 💾 Dữ liệu dự án (Dataset)
Do kích thước dữ liệu lớn, toàn bộ raw reviews và dữ liệu khách sạn được lưu trữ tại:
- **Link tải dữ liệu:** [Tải tại Google Drive](https://drive.google.com/file/d/1spjRqGWrX0vy3vWISffNICgSEATC1brJ/view?usp=sharing)
- Sau khi tải về, giải nén và đặt thư mục `data/` vào thư mục gốc của dự án.