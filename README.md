# BTL_KHDL — Phân tích & Dự báo Giá thuê Nhà trọ/Căn hộ theo Khu vực

> Đồ án môn **Nhập môn Khoa học Dữ liệu (IT4142)** — HUST.
> Thu thập dữ liệu cho thuê BĐS từ nhiều nguồn web Việt Nam → làm sạch → EDA → xây dựng &
> đánh giá mô hình hồi quy dự báo **giá thuê (triệu VNĐ/tháng)** → app web demo.

---

## 📊 Kết quả chính

| | Mô hình triển khai | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| **`models/model_best.pkl`** | **XGBoost** | **1.10 triệu** | **1.75 triệu** | **0.632** |

Mô hình dự báo giá dựa trên **diện tích, khu vực (thành phố/quận), số phòng, 12 tiện ích**. Mô
hình cây (XGBoost/Random Forest) vượt trội mô hình tuyến tính (Test R² 0.63 so với 0.18). *Cố ý
KHÔNG dùng `source` (nguồn web)* làm đặc trưng vì đó là artifact thu thập, không phải yếu tố định giá.

---

## 📁 Cấu trúc thư mục

```
btl_khdl/
├── app.py                       # ★ App Streamlit dự báo giá (Bước 6)
├── requirements.txt             # Danh sách thư viện + phiên bản
├── README.md
├── data/
│   ├── raw/                     # Dữ liệu thô từ các nguồn web (Bước 1)
│   └── processed/               # Dữ liệu đã làm sạch (Bước 2)
│       └── rental_data_clean.csv   # 133k bản ghi — đầu vào cho EDA & mô hình
├── notebooks/
│   ├── 03_eda.ipynb             # Bước 3 — EDA & trực quan hóa
│   ├── 04_modeling.ipynb        # Bước 4 — huấn luyện + tinh chỉnh 5 mô hình
│   └── 05_evaluation.ipynb      # Bước 5 — 10-fold CV + đánh giá Test
├── script/
│   ├── <nguồn>/                 # Spider cào dữ liệu từng web (Bước 1)
│   ├── eda/build_eda_notebook.py
│   └── model/                   # ★ Mã nguồn mô hình
│       ├── features.py             # Feature engineering DÙNG CHUNG train/serve
│       ├── build_model_notebook.py # dựng notebook Bước 4
│       ├── build_eval_notebook.py  # dựng notebook Bước 5
│       └── build_app_assets.py     # sinh dữ liệu phụ trợ cho app
├── models/                      # Mô hình & metadata (sinh ở Bước 4–5)
│   ├── model_best.pkl              # Mô hình triển khai (XGBoost)
│   ├── model_{ols,ridge,lasso,xgboost}.pkl
│   ├── metadata.json               # cột, siêu tham số, chỉ số CV/Test
│   └── app_assets.json             # danh mục khu vực + thống kê cho app
└── docs/
    ├── bao_cao/eda/             # biểu đồ EDA (PNG + HTML tương tác)
    ├── bao_cao/model/           # biểu đồ Bước 4 (so sánh mô hình, importance)
    ├── bao_cao/eval/            # biểu đồ Bước 5 (Test, learning curve)
    └── project_workflow*.md     # tài liệu quy trình 6 bước
```
> `models/model_random_forest.pkl` (~95 MB) bị `.gitignore` vì vượt giới hạn 100 MB của GitHub;
> tái tạo được từ `metadata.json` (`best_params`) bằng 1 lần `fit`.

---

## ⚙️ Cài đặt môi trường

Yêu cầu **Python 3.12**. Khuyến nghị dùng môi trường ảo (conda hoặc venv):

```bash
# Cách 1 — conda
conda create -n btl_khdl python=3.12 -y
conda activate btl_khdl

# Cách 2 — venv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

---

## ▶️ Cách chạy

### Chạy app demo (Bước 6) — *xem ngay không cần huấn luyện lại*
Mô hình đã được huấn luyện sẵn (`models/model_best.pkl`), chỉ cần:
```bash
streamlit run app.py
# Mở trình duyệt tại http://localhost:8501
```
App gồm 3 tab: **🔮 Dự báo giá** (nhập diện tích/khu vực/tiện ích → giá + đối chiếu mặt bằng quận),
**📊 Phân tích thị trường** (biểu đồ EDA), **🤖 Về mô hình** (chỉ số & biểu đồ đánh giá).

### Tái lập toàn bộ pipeline (Bước 3 → 5)
```bash
# Bước 3 — EDA (dựng + chạy notebook, nhúng sẵn biểu đồ)
python script/eda/build_eda_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb

# Bước 4 — Huấn luyện & tinh chỉnh 5 mô hình  (~10 phút)
python script/model/build_model_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/04_modeling.ipynb

# Bước 5 — 10-fold CV + đánh giá Test  (~10 phút)
python script/model/build_eval_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation.ipynb

# Sinh lại dữ liệu phụ trợ cho app
python script/model/build_app_assets.py
```
> Hoặc mở trực tiếp các notebook trong Jupyter/VS Code và Run All.

---

## 🔬 Quy trình 6 bước

| Bước | Nội dung | Sản phẩm |
|---|---|---|
| 1 | Thu thập dữ liệu (web scraping) | `data/raw/*` |
| 2 | Tích hợp & làm sạch | `data/processed/rental_data_clean.csv` |
| 3 | EDA & trực quan hóa | `notebooks/03_eda.ipynb`, `docs/bao_cao/eda/` |
| 4 | Huấn luyện 5 mô hình (OLS, Ridge, LASSO, RF, XGBoost) + tinh chỉnh + **thực nghiệm đối sánh** (thô↔FE↔FE+PCA, `price`↔`log(price)`, có↔không lọc outlier) | `notebooks/04_modeling.ipynb`, `models/` |
| 5 | 10-fold CV + đánh giá Test (1 lần) | `notebooks/05_evaluation.ipynb` |
| 6 | App Streamlit + đóng gói | `app.py`, `requirements.txt`, README |

**Phương pháp ML:** bài toán hồi quy; chia Train/Val/Test = 70/15/15; chuẩn hóa số (Z-score),
One-Hot khu vực (gộp hạng mục hiếm), mục tiêu giữ nguyên `price` gốc (log-transform bị thực
nghiệm 11.2 bác bỏ — làm RMSE tệ hơn và dự đoán hụt phân khúc giá cao); tinh chỉnh
GridSearchCV (tuyến tính) + RandomizedSearchCV (cây) + early stopping (XGBoost); chọn mô hình theo
RMSE/R² trên Validation rồi đánh giá cuối trên Test. Mọi quyết định tiền xử lý đều được **kiểm
chứng bằng thực nghiệm đối sánh** (notebook 04, mục 11): Feature Engineering ↔ dữ liệu thô ↔ PCA,
log-transform mục tiêu, và lọc outlier. Chi tiết: `docs/project_workflow_tong_quan.md`.

---

## 🌐 Nguồn dữ liệu thô (Bước 1)

| Thư mục | Website nguồn |
|---|---|
| `data/raw/batdongsan/` | batdongsan.com.vn |
| `data/raw/alonhadat/` | alonhadat.com.vn |
| `data/raw/nhatot/` | nhatot.com |
| `data/raw/phongtro123/` | phongtro123.com |
| `data/raw/lozido/` | lozido.com |
| `data/raw/chothuephongtro/` | chothuephongtro.me |

> Dữ liệu thô là dữ liệu gốc, **không chỉnh sửa trực tiếp**.

---

## 📦 Thư viện sử dụng
Xem `requirements.txt` (đã ghim phiên bản). Chính: pandas, numpy, scikit-learn, xgboost, joblib,
matplotlib, seaborn, plotly, streamlit, nbformat/nbconvert.

> *Lưu ý đạo văn:* dự án sử dụng các thư viện mã nguồn mở nêu trên; mọi thuật toán/được kế thừa
> đều được trích dẫn trong báo cáo theo yêu cầu môn học.
