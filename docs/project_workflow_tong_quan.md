# QUY TRÌNH & HƯỚNG DẪN TỔNG QUAN DỰ ÁN BÀI TẬP LỚN KHOA HỌC DỮ LIỆU
## Đề tài: Phân tích giá thuê nhà trọ/căn hộ theo khu vực
### Môn học: Nhập môn Khoa học Dữ liệu (IT4142)

> 📖 Đây là **bản tổng quan** — tóm tắt quy trình và yêu cầu.
> Xem thêm [bản chi tiết](project_workflow_chi_tiet.md) để có code mẫu và hướng dẫn từng bước.

---

## PHẦN I: YÊU CẦU CHUNG CỦA ĐỒ ÁN MÔN HỌC (Slide L0)

Đồ án môn học (chiếm **40%** tổng điểm học phần) đánh giá năng lực thực hành toàn diện của sinh viên từ thu thập, xử lý đến xây dựng mô hình dự báo trên dữ liệu thực tế.

### 1. Thành phần sản phẩm bàn giao (Deliverables)
Cuối kỳ, nhóm phải đóng gói sản phẩm bao gồm:
*   **Mã nguồn (Source codes):** Lưu trong một file nén (.zip/.rar) chứa toàn bộ code thu thập, tiền xử lý, phân tích EDA và huấn luyện mô hình.
*   **Tài liệu hướng dẫn (`README.md`):** Mô tả chi tiết cách thức cài đặt môi trường, biên dịch, chạy chương trình và danh sách các gói phần mềm (packages/libraries) kèm phiên bản được sử dụng.
*   **Tài liệu báo cáo kết quả đồ án (.pdf):** Trình bày học thuật theo cấu trúc bắt buộc:
    1. Giới thiệu và mô tả cụ thể về bài toán thực tế được giải quyết.
    2. Chi tiết của các phương pháp học máy (ML) và các tập dữ liệu (datasets) được sử dụng.
    3. Các kết quả thí nghiệm đánh giá hiệu năng của hệ thống học máy đối với các tập dữ liệu.
    4. Các chức năng chính của hệ thống thử nghiệm và hướng dẫn sử dụng.
    5. Cấu trúc mã nguồn chương trình, vai trò của các lớp (classes) và phương thức (methods) chính.
    6. Các vấn đề/khó khăn gặp phải trong quá trình thực hiện đồ án và phương án giải quyết.
    7. Các khám phá mới (insights) hoặc kết luận khoa học rút ra.

### 2. Tiêu chí đánh giá đồ án
Giảng viên sẽ đánh giá điểm đồ án dựa trên các tiêu chí sau:
*   **Độ phức tạp / khó khăn** của bài toán thực tế được giải quyết.
*   **Chất lượng và sự phù hợp** của phương pháp học máy được chọn.
*   **Đánh giá và lựa chọn kỹ lưỡng mô hình** (hyperparameter tuning).
*   **Cài đặt hệ thống thử nghiệm** (chạy thực tế, có giao diện trực quan, dễ sử dụng).
*   **Chất lượng tài liệu báo cáo** (.pdf) khoa học, rõ ràng.
*   **Chất lượng bài thuyết trình (Presentation):** Tối đa **15 phút**, tất cả thành viên trong nhóm bắt buộc phải thuyết trình và trả lời câu hỏi phản biện.
*   *Lưu ý đạo văn:* Nếu sử dụng lại, kế thừa hoặc khai thác mã nguồn/gói phần mềm có sẵn, nhóm phải **nêu rõ ràng và chính xác** trong tài liệu báo cáo và slide thuyết trình.

---

## PHẦN II: QUY TRÌNH THỰC HIỆN TỔNG QUAN (Áp dụng lý thuyết từ Slide L1 - L12)

Quy trình tuân theo vòng đời **Khai phá Tri thức** (Slide L1):

```
Thu thập → Xử lý → Trực quan hóa & Hiểu dữ liệu → Phân tích / ML → Ra quyết định
```

> 💡 Bước "Thu thập + Xử lý dữ liệu" chiếm **70-90%** toàn bộ quá trình (John Dickerson, UMD).

---

### BƯỚC 1: THU THẬP DỮ LIỆU (WEB SCRAPING) — Slide L2

**Lý thuyết cốt lõi:**
*   Web Scraping trích xuất dữ liệu HTML thành dạng có cấu trúc khi nguồn không cung cấp API.
*   Sử dụng **Scrapy** (Python) — hỗ trợ throttling, concurrency, lọc trùng URL, retry on error.
*   Bóc tách bằng **XPath / CSS Selectors** trên cấu trúc cây DOM.
*   Tuân thủ chính sách: **Politeness policy** (tránh quá tải), **robots.txt** (giới hạn cào).

**Việc cần làm:**
*   Mỗi thành viên viết một spider cào riêng cho từng website:
    *   *Hồng Dương:* `batdongsan.com.vn` → `data/raw/batdongsan/`
    *   *Thành An:* `alonhadat.com.vn` → `data/raw/alonhadat/`
    *   *Gia Phát:* `nhatot.com` → `data/raw/nhatot/`
    *   *Thành Đạt:* `phongtro123.com` → `data/raw/phongtro123/`
    *   *Danh Dũng:* `lozido.com` → `data/raw/lozido/`
    *   *Mạnh Quyền:* `chothuephongtro.me` → `data/raw/chothuephongtro/`
*   Cấu hình `DOWNLOAD_DELAY = 1.5` hoặc `AutoThrottle` trong Scrapy.
*   Bóc tách các trường: tiêu đề, giá thuê, diện tích, địa chỉ, số phòng ngủ/tắm, ngày đăng.
*   Xuất dữ liệu thô ra `.csv` vào `data/raw/`.

---

### BƯỚC 2: TÍCH HỢP & LÀM SẠCH DỮ LIỆU — Slide L3

**Lý thuyết cốt lõi:**
*   **Chất lượng dữ liệu:** Xử lý giá trị thiếu, sai cú pháp, trùng lặp, không nhất quán đơn vị.
*   **ETL (Extract-Transform-Load)** chiếm **70-80%** công việc dự án.
*   Xử lý missing: bỏ qua, điền median/mean, hoặc dùng mô hình dự đoán.
*   **Chuẩn hóa:** Min-Max, Z-score, Decimal scaling.

**Việc cần làm:**
*   **Làm sạch `Hanoi_housing_dataset.csv` (82,497 dòng):**
    *   Tách bỏ ký tự chữ bằng Regex (`"100 m2"` → `100.0`).
    *   Loại bỏ cột `Dài` (thiếu 75.9%) và `Rộng` (thiếu 57.0%).
    *   Cột `Số tầng` thiếu 55.9%: điền mặc định = 1.
*   **Làm sạch `hn.csv`, `hcm.csv`, `dn.csv`:**
    *   Loại bỏ dòng `price = -1` ("Giá thỏa thuận").
    *   Lọc outlier diện tích: chỉ giữ `area ≤ 150 m²`.
*   **Tích hợp dữ liệu:**
    *   Xây dựng lược đồ trung gian chung: `title, price, area, bedrooms, bathrooms, district, city, source`.
    *   Khử trùng lặp bản ghi bằng so khớp tiêu đề + địa chỉ.
*   **Feature Engineering:**
    *   Chuẩn hóa địa chỉ → trích xuất Quận/Huyện, Thành phố.
    *   Tính khoảng cách trung tâm (`dtcenter`).
    *   Mã hóa One-Hot cho biến danh mục và tiện ích.

---

### BƯỚC 3: PHÂN TÍCH KHÁM PHÁ (EDA) & TRỰC QUAN HÓA — Slide L4, L5, L6

**Lý thuyết cốt lõi:**
*   **EDA** quan tâm tới cấu trúc, ngoại lệ và mô hình dữ liệu (John Tukey).
*   Thống kê mô tả: Mean, Median, Mode, Variance, Std Dev, Skewness, Kurtosis.
*   Trực quan: Histogram (phân phối), Box plot (so sánh nhóm), Scatter plot (tương quan), Heatmap (ma trận tương quan), Violin plot (phân bố + density).
*   Phân cụm: K-Means, AHC. Giảm chiều: PCA.

**Việc cần làm:**
*   **Thống kê mô tả:** Tính Mean, Median, Min, Max, Std, Skewness cho `price` và `area`.
*   **Trực quan hóa:**
    *   Histogram: phân phối giá thuê và diện tích.
    *   Box plot / Violin plot: so sánh giá theo quận/huyện.
    *   Scatter plot: `area` vs `price`, `dtcenter` vs `price`.
    *   Heatmap: ma trận tương quan Pearson giữa các biến số.
    *   Bar chart: giá thuê trung bình theo thành phố.
*   **Phân cụm (tùy chọn):** K-Means gom cụm dữ liệu theo `price` + `area` → phân khúc thị trường.
*   **PCA (tùy chọn):** Giảm chiều dữ liệu đa biến về 2D để trực quan hóa.

---

### BƯỚC 4: XÂY DỰNG MÔ HÌNH HỌC MÁY — Slide L7

**Lý thuyết cốt lõi:**
*   Bài toán: **Hồi quy** (đầu ra `y = price` là số thực liên tục).
*   **OLS Linear Regression:** Tìm `w*` cực tiểu hóa RSS. Nhược: dễ overfitting, ma trận suy biến.
*   **Ridge (L2):** Thêm phạt `λ‖w‖²`. Ưu: luôn tồn tại nghiệm, giảm overfitting.
*   **LASSO (L1):** Thêm phạt `λ‖w‖₁`. Ưu: tạo nghiệm thưa → lựa chọn đặc trưng tự động.
*   **Random Forest:** Kết hợp nhiều cây quyết định (Bagging + ngẫu nhiên hóa thuộc tính). Mạnh nhất, ít overfitting.
*   **XGBoost:** Sử dụng thư viện xgboost để huấn luyện mô hình xgboost.
*   **No-Free-Lunch:** Phải thử nhiều thuật toán để so sánh.

**Việc cần làm:**
*   Chuẩn bị `X` (features) và `y` (price). Chia Train/Validation/Test (70/15/15).
*   Huấn luyện 4 mô hình: **OLS**, **Ridge**, **LASSO**, **Random Forest**.
*   Tối ưu siêu tham số: `λ` cho Ridge/LASSO, `n_estimators` cho RF.
*   Trích xuất Feature Importance từ Random Forest.
*   Lưu mô hình tốt nhất (`.pkl`).

---

### BƯỚC 5: ĐÁNH GIÁ & LỰA CHỌN MÔ HÌNH — Slide L12

**Lý thuyết cốt lõi:**
*   **Hold-out:** Chia Train + Test. **k-fold CV:** Chia k phần, lặp k lần.
*   **Bootstrap:** Lấy mẫu có trùng lặp (63.2% train).
*   Tiêu chí hồi quy: **MAE**, **RMSE**, **R² Score**.
*   Quy trình: Model Selection (tối ưu trên Validation) → Model Assessment (đánh giá trên Test).

**Việc cần làm:**
*   Sử dụng **10-fold Cross-Validation** trên tập Train để đánh giá ổn định.
*   Tối ưu siêu tham số trên tập Validation.
*   Đánh giá cuối cùng trên tập **Test** (chỉ chạy **1 lần duy nhất**).
*   Lập bảng so sánh MAE/RMSE/R² cho tất cả mô hình.
*   Vẽ biểu đồ Actual vs Predicted.

---

### BƯỚC 6: HỆ THỐNG THỬ NGHIỆM & BÁO CÁO — Slide L0

**Việc cần làm:**
*   **Xây dựng ứng dụng web demo (Streamlit):**
    *   Cho người dùng nhập thông tin phòng trọ (diện tích, quận, tiện ích...).
    *   Dự báo giá thuê bằng mô hình ML tốt nhất.
    *   Hiển thị biểu đồ EDA hỗ trợ ra quyết định.
*   **Đóng gói:** `requirements.txt`, `README.md` hướng dẫn chạy.
*   **Báo cáo PDF** theo cấu trúc 7 phần bắt buộc.
*   **Slide thuyết trình:** Tối đa 15 phút, tất cả thành viên thuyết trình.

---

## PHẦN III: KẾ HOẠCH TIẾN ĐỘ & PHÂN CÔNG

| Giai đoạn  | Nội dung                                 | Phụ trách                 | Sản phẩm                | Hạn chót |
| :--------- | :--------------------------------------- | :------------------------ | :---------------------- | :------- |
| **Bước 1** | Cào dữ liệu thô từ 6 website             | Cả nhóm (mỗi người 1 web) | `data/raw/*.csv`        | Tuần 5   |
| **Bước 2** | Làm sạch, tích hợp, feature engineering  | Cả nhóm                   | `data/cleaned_data.csv` | Tuần 8   |
| **Bước 3** | EDA, trực quan hóa                       | Thành Đạt, Mạnh Quyền     | Biểu đồ + notebook      | Tuần 10  |
| **Bước 4** | Huấn luyện ML, tinh chỉnh hyperparameter | Hồng Dương, Gia Phát      | Mô hình `.pkl`          | Tuần 12  |
| **Bước 5** | k-fold CV, bảng so sánh mô hình          | Danh Dũng, Thành An       | Bảng kết quả            | Tuần 13  |
| **Bước 6** | Web Streamlit, báo cáo PDF, slide        | Cả nhóm                   | App + Báo cáo + Slide   | Tuần 15  |
