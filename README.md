# BTL_KHDL — Phân tích & Dự báo Giá thuê Nhà trọ/Căn hộ theo Khu vực

> Đồ án môn **Nhập môn Khoa học Dữ liệu (IT4930)** — HUST.
> Thu thập dữ liệu cho thuê BĐS từ nhiều nguồn web Việt Nam → làm sạch → EDA → xây dựng &
> đánh giá mô hình hồi quy dự báo **giá thuê (triệu VNĐ/tháng)** → app web demo.

---

## 📊 Kết quả chính

| Mô hình | Test MAE | Test RMSE | Test R² | Ghi chú |
|---|---|---|---|---|
| **XGBoost** | **1.26 triệu** | **1.90 triệu** | **0.565** | ★ Mô hình triển khai (`model_best.pkl`) |
| Random Forest | 1.00 triệu | 1.67 triệu | 0.664 | Chính xác hơn nhưng nặng hơn (~5.6 MB) |
| OLS / Ridge / LASSO | ~1.96 triệu | ~2.59 triệu | ~0.19 | Mô hình tuyến tính — baseline |

Mô hình dự báo giá dựa trên **diện tích, khu vực (thành phố/quận), số phòng ngủ, số phòng tắm, và
12 tiện ích**. Mô hình cây (XGBoost/RF) vượt trội mô hình tuyến tính rõ rệt (R² 0.56+ so với 0.19).
*Cố ý KHÔNG dùng `source` (nguồn web) làm đặc trưng* vì đó là artifact thu thập, không phải yếu tố
định giá.

Tập dữ liệu: **133,586 bản ghi** sau lọc (chia 70/15/15 = 93 563 / 19 985 / 20 038).

---

## 📁 Cấu trúc thư mục

```
btl_khdl/
├── app.py                          # ★ App Streamlit dự báo giá (Bước 6)
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                        # Dữ liệu thô từ các nguồn (không track trong git)
│   └── processed/
│       ├── rental_data_clean.csv   # ~133k bản ghi sạch — đầu vào EDA & mô hình
│       └── rental_data.csv         # Dữ liệu thuê đã xử lý sơ bộ (trước lọc outlier)
│
├── notebooks/
│   ├── 02_data_cleaning.ipynb      # Bước 2 — ETL & làm sạch
│   ├── 03_eda.ipynb                # Bước 3 — EDA & trực quan hóa
│   ├── 04_modeling.ipynb           # Bước 4 — huấn luyện + tinh chỉnh 5 mô hình
│   └── 05_evaluation.ipynb         # Bước 5 — 10-fold CV + đánh giá Test
│
├── script/
│   ├── cleaning/
│   │   └── build_clean_notebook.py # Dựng notebook Bước 2
│   ├── eda/
│   │   └── build_eda_notebook.py   # Dựng notebook Bước 3
│   ├── model/
│   │   ├── features.py             # ★ Feature engineering dùng chung train/serve
│   │   ├── build_model_notebook.py # Dựng notebook Bước 4
│   │   └── build_eval_notebook.py  # Dựng notebook Bước 5
│   └── report/
│       ├── extract_report_numbers.py
│       └── report_numbers.json     # Các chỉ số tổng hợp cho báo cáo
│
├── models/
│   ├── model_best.pkl              # Mô hình triển khai — XGBoost (~1.1 MB)
│   ├── model_xgboost.pkl
│   ├── model_ols.pkl
│   ├── model_ridge.pkl
│   ├── model_lasso.pkl
│   ├── model_random_forest.pkl     # Gitignored (~5.6 MB); tái tạo từ metadata.json
│   └── metadata.json               # Cột đặc trưng, siêu tham số, chỉ số CV/Test
│
└── docs/
    ├── cleaning/                   # Biểu đồ Bước 2 (5 PNG + 1 HTML)
    ├── eda/                        # Biểu đồ Bước 3 (15 PNG + 5 HTML tương tác)
    ├── model/                      # Biểu đồ Bước 4 (so sánh mô hình, feature importance, ablation)
    ├── eval/                       # Biểu đồ Bước 5 (CV so sánh, Test, learning curve)
    └── Bao_Cao_PDF/                # Báo cáo LaTeX (gitignored — dùng Overleaf)
```

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

### Chạy app demo — *xem ngay không cần huấn luyện lại*

Mô hình đã được huấn luyện sẵn (`models/model_best.pkl`), chỉ cần:

```bash
streamlit run app.py
# Mở trình duyệt tại http://localhost:8501
```

App gồm 3 tab: **🔮 Dự báo giá** (nhập diện tích/khu vực/tiện ích → giá ước tính + đối chiếu mặt
bằng quận), **📊 Phân tích thị trường** (biểu đồ EDA), **🤖 Về mô hình** (chỉ số & biểu đồ đánh giá).

### Tái lập toàn bộ pipeline (Bước 2 → 5)

```bash
# Bước 2 — ETL & làm sạch
python script/cleaning/build_clean_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_cleaning.ipynb

# Bước 3 — EDA
python script/eda/build_eda_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb

# Bước 4 — Huấn luyện & tinh chỉnh 5 mô hình  (~10 phút)
python script/model/build_model_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/04_modeling.ipynb

# Bước 5 — 10-fold CV + đánh giá Test  (~10 phút)
python script/model/build_eval_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation.ipynb
```

> Hoặc mở trực tiếp các notebook trong Jupyter/VS Code và Run All.

---

## 🔬 Quy trình 6 bước

| Bước | Nội dung | Sản phẩm |
|---|---|---|
| 1 | Thu thập dữ liệu (web scraping 7 nguồn) | `data/raw/` |
| 2 | Tích hợp, làm sạch & kiểm định tiện ích | `data/processed/rental_data_clean.csv`, `notebooks/02_data_cleaning.ipynb` |
| 3 | EDA & trực quan hóa (15 biểu đồ, K-Means, PCA, Simpson's paradox) | `notebooks/03_eda.ipynb`, `docs/eda/` |
| 4 | Huấn luyện 5 mô hình (OLS, Ridge, LASSO, RF, XGBoost) + tinh chỉnh + 3 thực nghiệm đối sánh | `notebooks/04_modeling.ipynb`, `models/`, `docs/model/` |
| 5 | 10-fold CV + đánh giá Test (1 lần duy nhất) | `notebooks/05_evaluation.ipynb`, `docs/eval/` |
| 6 | App Streamlit + đóng gói | `app.py`, `requirements.txt` |

**Phương pháp ML:** bài toán hồi quy; chia Train/Val/Test = 70/15/15; chuẩn hóa số (Z-score),
One-Hot khu vực (gộp hạng mục hiếm `< 10 mẫu`), mục tiêu giữ nguyên `price` gốc (log-transform
được thực nghiệm 05-Bước4 bác bỏ — làm RMSE tệ hơn); tinh chỉnh GridSearchCV (tuyến tính) +
RandomizedSearchCV (RF) + early stopping (XGBoost); chọn mô hình theo RMSE/R² trên Validation rồi
đánh giá cuối trên Test. Ba thực nghiệm đối sánh (notebook 04): Feature Engineering ↔ dữ liệu thô
↔ PCA; log-transform mục tiêu; lọc outlier.

---

## 🌐 Nguồn dữ liệu thô (Bước 1)

Dữ liệu thô lưu ở `data/raw/` (không track trong git). Các nguồn:

| Thư mục | Nguồn |
|---|---|
| `data/raw/nhatot/` | nhatot.com |
| `data/raw/phongtro123/` | phongtro123.com |
| `data/raw/chothuephongtro/` | chothuephongtro.me |
| `data/raw/lozido/` | lozido.com |
| `data/raw/ChothuenhadattoanVN/` | chothuenhadattoavn.com |
| `data/raw/facebook/` | Facebook Marketplace |
| `data/raw/github/` | Dataset công khai (GitHub) |

---

## 📦 Thư viện sử dụng

Xem `requirements.txt` (đã ghim phiên bản). Chính: `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`joblib`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `nbformat`, `nbconvert`.
