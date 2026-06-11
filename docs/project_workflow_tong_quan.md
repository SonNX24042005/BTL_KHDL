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
*   **Random Forest:** Kết hợp nhiều cây quyết định (Bagging + ngẫu nhiên hóa thuộc tính). Ít overfitting, mạnh với dữ liệu phi tuyến.
*   **XGBoost (Extreme Gradient Boosting):** Thuật toán Gradient Boosting cải tiến — mỗi cây mới học để sửa lỗi (residual) của cây trước theo hướng gradient descent. Hàm mục tiêu gồm **loss function** (giảm sai số) và **regularization** (L1 + L2 kiểm soát độ phức tạp cây, tránh overfitting). Ưu điểm nổi bật: xử lý missing value tự động, hỗ trợ `early_stopping_rounds`, song song hóa tính toán, cho độ chính xác cao hơn Random Forest trên nhiều bài toán thực tế.
*   **No-Free-Lunch:** Phải thử nhiều thuật toán để so sánh.

**Việc cần làm:**
*   Chuẩn bị `X` (features) và `y` (price). Chia Train/Validation/Test (70/15/15).
*   Huấn luyện 5 mô hình: **OLS**, **Ridge**, **LASSO**, **Random Forest**, **XGBoost**.
*   Tối ưu siêu tham số:
    *   Ridge/LASSO: `alpha` — thử grid `[0.01, 0.1, 1, 10, 100]`.
    *   Random Forest: `n_estimators` ∈ {100, 200, 500}, `max_depth` ∈ {5, 10, 20, None}, `min_samples_split` ∈ {2, 5, 10}.
    *   **XGBoost** (dùng scikit-learn API `XGBRegressor`):
        *   `n_estimators` ∈ {100, 300, 500} — số cây boosting.
        *   `learning_rate` (eta) ∈ {0.01, 0.05, 0.1, 0.3} — bước học; nhỏ hơn → cần nhiều cây hơn.
        *   `max_depth` ∈ {3, 5, 7} — độ sâu mỗi cây; càng sâu càng dễ overfit.
        *   `subsample` ∈ {0.6, 0.8, 1.0} — tỉ lệ lấy mẫu hàng mỗi vòng.
        *   `colsample_bytree` ∈ {0.6, 0.8, 1.0} — tỉ lệ lấy mẫu cột mỗi cây.
        *   `reg_alpha` (L1) ∈ {0, 0.1, 1} và `reg_lambda` (L2) ∈ {1, 5, 10}.
        *   Truyền `eval_set=[(X_val, y_val)]` và `early_stopping_rounds=50` vào `fit()` để dừng sớm khi validation loss không cải thiện — tránh overfitting và tiết kiệm thời gian huấn luyện.
*   **Feature Importance:**
    *   Random Forest: dùng `model.feature_importances_` (Mean Decrease Impurity).
    *   XGBoost: dùng `model.feature_importances_` hoặc `xgb.plot_importance(model, importance_type='gain')` (gain = mức giảm loss trung bình khi dùng feature đó).
    *   Vẽ bar chart top-10 feature quan trọng nhất của cả hai mô hình để so sánh.
*   Lưu mô hình tốt nhất bằng `joblib.dump(model, 'model_best.pkl')`.

---

### BƯỚC 5: ĐÁNH GIÁ & LỰA CHỌN MÔ HÌNH — Slide L12

**Lý thuyết cốt lõi:**
*   **Hold-out:** Chia Train + Validation + Test — nhanh nhưng phụ thuộc cách chia.
*   **k-fold Cross-Validation:** Chia k phần, lặp k lần, lấy trung bình → đánh giá ổn định hơn, ít phụ thuộc random seed.
*   **Bootstrap:** Lấy mẫu có trùng lặp (~63.2% mẫu train duy nhất), phần còn lại làm OOB test.
*   **Tiêu chí hồi quy:**
    *   **MAE** (Mean Absolute Error) = trung bình |y − ŷ| — dễ diễn giải theo đơn vị giá thuê.
    *   **RMSE** (Root Mean Squared Error) = √(trung bình (y − ŷ)²) — phạt nặng lỗi lớn hơn MAE.
    *   **R² Score** = 1 − SS_res/SS_tot ∈ (−∞, 1] — tỉ lệ phương sai được giải thích; R² = 1 là hoàn hảo.
*   **Quy trình chuẩn:** Model Selection (so sánh + chọn mô hình trên Validation/CV) → Model Assessment (đánh giá lần cuối trên Test, **chỉ chạy 1 lần duy nhất**).

**Việc cần làm:**
*   **Cross-Validation trên tập Train:**
    *   Dùng `cross_val_score(model, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')` cho từng mô hình.
    *   Báo cáo mean ± std của RMSE và R² qua 10 fold để đánh giá độ ổn định.
*   **Tối ưu siêu tham số trên tập Validation:**
    *   Dùng `GridSearchCV` hoặc `RandomizedSearchCV` với `cv=5` trên tập Train.
    *   Chọn bộ tham số tốt nhất theo RMSE trên Validation.
*   **Đánh giá cuối cùng trên tập Test** (chỉ gọi `predict` **1 lần duy nhất** sau khi đã chốt mô hình):
    *   Tính MAE, RMSE, R² bằng `mean_absolute_error`, `mean_squared_error(squared=False)`, `r2_score` từ `sklearn.metrics`.
*   **Bảng so sánh tất cả mô hình** (ghi rõ kết quả trên cả CV-Train và Test):

    | Mô hình       | CV RMSE (mean±std) | Test MAE | Test RMSE | Test R² |
    | :------------ | :----------------- | :------- | :-------- | :------ |
    | OLS           |                    |          |           |         |
    | Ridge         |                    |          |           |         |
    | LASSO         |                    |          |           |         |
    | Random Forest |                    |          |           |         |
    | XGBoost       |                    |          |           |         |

*   **Biểu đồ đánh giá:**
    *   **Actual vs Predicted** (scatter plot): điểm nằm gần đường y = x → mô hình tốt.
    *   **Residual plot** (residual = y − ŷ vs ŷ): phân bố ngẫu nhiên quanh 0 → không có bias hệ thống.
    *   **Learning curve** (tùy chọn): vẽ train/val score theo kích thước tập train → chẩn đoán underfitting/overfitting.
*   Chọn mô hình cuối dựa trên **Test RMSE thấp nhất** và **R² cao nhất**, lưu lại bằng `joblib.dump()`.

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
