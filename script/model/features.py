"""
Đặc trưng dùng chung cho Bước 4 (xây dựng mô hình) — *single source of truth*.

Đặt riêng ở đây để **train (notebook 04) và serve (app Streamlit ở Bước 6) dùng CHUNG
một logic feature engineering**, tránh lệch train/serve (train/serve skew). Logic trích
xuất `city`/`district` từ `full_address` kế thừa nguyên vẹn từ Bước 3 (notebook EDA).

Quan trọng — KHÔNG dùng làm đặc trưng:
  • `source`                     → **nguồn web cào** chỉ là artifact thu thập dữ liệu, KHÔNG phải
                                   yếu tố định giá thực tế; app demo (Bước 6) cũng không có input này.
                                   Mô hình phải dự báo từ đặc điểm BĐS, không từ "tin này lấy ở web nào".
  • `n_features`                 → ĐÃ BỎ (2026-06-11): là tổng của 12 cờ tiện ích, trong đó 7 cờ bị
                                   CHẾ TÁC với phần lớn nguồn (xem MNAR bên dưới) ⇒ phép đếm vô nghĩa.
  • `price_per_m2`, `log_price`  → suy ra từ `price`  ⇒ rò rỉ nhãn (data leakage).
  • `posted_date`                → ~46% là giá trị sentinel ngày tương lai (xem EDA 3.2).
  • `description`, `detail_link` → thiếu phần lớn / không mang tín hiệu giá.
  • `full_address`              → đã thay bằng `city` + `district` đã tách.

MNAR — cờ tiện ích bị chế tác khi gộp dữ liệu (phát hiện 2026-06-11, đối chiếu raw ↔ combined):
  Chỉ 5 cờ có cột gốc thật ở đa số nguồn (máy lạnh=`ac`, máy giặt=`washingmachine`,
  nước nóng=`hotwater`, bảo vệ 24/24=`security`, hầm để xe=`parking`). 7 cờ còn lại chỉ có thật
  ở phongtro123 (11 cờ) và lozido (10 cờ); với các nguồn khác chúng được điền ngẫu nhiên lúc gộp
  (corr với giá trong từng nguồn ≈ 0). `chothuenhadattoanvn` + `github_hcm/hn/dn` không có cờ
  thật nào. ⇒ `mask_fabricated()` chuyển các ô chế tác thành NaN ("không khai báo" ≠ "không có"):
  XGBoost/RF xử lý NaN tự nhiên; còn lúc serve người dùng khai báo trực tiếp nên luôn là 0/1 thật.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Danh sách tỉnh/thành để tách `city` từ địa chỉ (giống Bước 3 — notebook EDA).
PROVINCES = [
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Bình Dương", "Hưng Yên",
    "Hải Phòng", "Quảng Ninh", "Khánh Hòa", "Đồng Nai", "Cần Thơ",
    "Bắc Ninh", "Hải Dương", "Nghệ An", "Thừa Thiên Huế", "Huế",
    "Bà Rịa", "Vũng Tàu", "Lâm Đồng", "Thanh Hóa", "Quảng Nam",
]
_DIST_RE = re.compile(r"^(Quận|Huyện|Thị xã|Thành phố|TP)\s", re.IGNORECASE)

# Nhãn dùng khi không tách được khu vực (để OneHotEncoder coi là 1 hạng mục hợp lệ).
UNKNOWN = "(không rõ)"

# ---- Nhóm cột đầu vào của mô hình -------------------------------------------
TARGET = "price"
NUM_COLS = ["area", "bedrooms", "bathrooms"]  # biến số (Z-score) — `n_features` ĐÃ BỎ (xem docstring)
CAT_COLS = ["city", "district"]               # biến danh mục (One-Hot) — CỐ Ý bỏ `source`
# BIN_COLS = 12 cờ tiện ích `feature_*` (0/1 hoặc NaN sau mask) — lấy động bằng get_feature_cols().

# ---- Cờ tiện ích THẬT theo nguồn (đối chiếu dữ liệu raw, 2026-06-11) ---------
# Cờ không nằm trong danh sách của nguồn = bị chế tác khi gộp → mask_fabricated() đặt NaN.
STD5_FLAGS = ["feature_Có máy lạnh", "feature_Có máy giặt", "feature_Có nước nóng",
              "feature_Có bảo vệ 24/24", "feature_Có hầm để xe"]
OBSERVED_FLAGS = {
    "chothuephongtro": STD5_FLAGS,
    "nhatot": STD5_FLAGS,
    "facebook": STD5_FLAGS,
    "github_housing": STD5_FLAGS,
    "lozido": STD5_FLAGS + ["feature_Có gác", "feature_Có kệ bếp", "feature_Có thang máy",
                            "feature_Không chung chủ", "feature_Giờ giấc tự do"],
    # phongtro123 có sẵn 11 cờ feature_* trong raw (thiếu duy nhất nước nóng)
    "phongtro123": ["feature_Đầy đủ nội thất", "feature_Có gác", "feature_Có kệ bếp",
                    "feature_Có máy lạnh", "feature_Có máy giặt", "feature_Có tủ lạnh",
                    "feature_Có thang máy", "feature_Không chung chủ", "feature_Giờ giấc tự do",
                    "feature_Có bảo vệ 24/24", "feature_Có hầm để xe"],
    # Các nguồn dưới đây KHÔNG có cột tiện ích thật nào trong raw:
    "chothuenhadattoanvn": [],
    "github_hcm": [],
    "github_hn": [],
    "github_dn": [],
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Danh sách 12 cột cờ tiện ích nhị phân `feature_*` có trong DataFrame."""
    return [c for c in df.columns if c.startswith("feature_")]


def parse_address(addr) -> tuple[str | None, str | None]:
    """Tách (city, district) từ chuỗi `full_address`. Kế thừa nguyên vẹn từ EDA.

    - city: tỉnh/thành khớp PROVINCES (duyệt từ phải sang); nếu không khớp lấy cụm cuối.
    - district: cụm bắt đầu bằng Quận/Huyện/Thị xã/Thành phố/TP.
    """
    parts = [p.strip() for p in str(addr).split(",") if p.strip()]
    city = None
    for p in reversed(parts):
        for prov in PROVINCES:
            if prov.lower() in p.lower():
                city = prov
                break
        if city:
            break
    if city is None and parts:
        city = parts[-1]
    district = next((p for p in parts if _DIST_RE.match(p)), None)
    return city, district


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm `city`, `district`, `n_features` (không đụng các cột gốc). Trả về bản sao.

    `n_features` chỉ còn dùng cho mô tả/EDA — KHÔNG còn là đặc trưng mô hình (xem docstring).
    """
    df = df.copy()
    df.columns = [c.lstrip("﻿") for c in df.columns]  # bỏ BOM nếu có
    fcols = get_feature_cols(df)
    parsed = df["full_address"].apply(parse_address)
    df["city"] = parsed.map(lambda x: x[0]).fillna(UNKNOWN)
    df["district"] = parsed.map(lambda x: x[1]).fillna(UNKNOWN)
    df["n_features"] = df[fcols].sum(axis=1)
    return df


def mask_fabricated(df: pd.DataFrame) -> pd.DataFrame:
    """NaN-hóa các ô (source, cờ tiện ích) bị CHẾ TÁC khi gộp Bước 2. Trả về bản sao.

    NaN = "nguồn này không thu thập tiện ích đó" (không khai báo ≠ không có). Nguồn lạ
    ngoài OBSERVED_FLAGS được coi thận trọng là không quan sát toàn bộ cờ.
    """
    df = df.copy()
    fcols = get_feature_cols(df)
    for src, ok in OBSERVED_FLAGS.items():
        bad = [c for c in fcols if c not in ok]
        if bad:
            df.loc[df["source"] == src, bad] = np.nan
    unknown = ~df["source"].isin(OBSERVED_FLAGS)
    if unknown.any():
        df.loc[unknown, fcols] = np.nan
    return df


def build_Xy(df: pd.DataFrame, mask_mnar: bool = True):
    """Sinh (X, y, col_groups) sẵn sàng cho ColumnTransformer.

    mask_mnar=True (mặc định): áp dụng mask_fabricated() nếu có cột `source` —
    dùng cho TRAIN. Lúc serve (app) không đi qua hàm này: input người dùng luôn 0/1 thật.
    col_groups = {"num": [...], "cat": [...], "bin": [...]} để dựng preprocessor.
    """
    df = add_engineered(df)
    if mask_mnar and "source" in df.columns:
        df = mask_fabricated(df)
    fcols = get_feature_cols(df)
    col_groups = {"num": NUM_COLS, "cat": CAT_COLS, "bin": fcols}
    feature_columns = NUM_COLS + CAT_COLS + fcols
    X = df[feature_columns].copy()
    y = df[TARGET].copy()
    return X, y, col_groups


def build_preprocessor(col_groups: dict, min_frequency: int = 100, impute_bin: bool = False):
    """ColumnTransformer dùng chung cho cả 5 mô hình.

    - num: StandardScaler (cần cho mô hình tuyến tính; vô hại với cây).
    - cat: OneHotEncoder gộp hạng mục hiếm (< `min_frequency`) để chặn bùng nổ chiều
      do `district` (360) / `city` (204) cardinality cao; `infrequent_if_exist` còn xử lý
      được hạng mục lạ lúc dự đoán (app demo).
    - bin: 12 cờ tiện ích (0/1 hoặc NaN sau mask_fabricated):
        impute_bin=False → passthrough (XGBoost/RandomForest xử lý NaN native);
        impute_bin=True  → SimpleImputer(mean) cho mô hình tuyến tính (OLS/Ridge/LASSO
        không nhận NaN; mean = tỷ lệ quan sát được, trung hòa sau khi chuẩn hóa).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    bin_step = SimpleImputer(strategy="mean") if impute_bin else "passthrough"
    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), col_groups["num"]),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=min_frequency,
                                  sparse_output=False), col_groups["cat"]),
            ("bin", bin_step, col_groups["bin"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Xuất DataFrame (giữ TÊN cột) để monotone_constraints dạng dict của XGBoost
    # khớp theo tên — bền với thay đổi số cột One-Hot giữa các fold CV.
    return ct.set_output(transform="pandas")


def monotone_constraints(col_groups: dict, constrain_area: bool = False) -> dict:
    """Dict ràng buộc đơn điệu cho XGBoost: {tên cột sau preprocessor: +1}.

    +1 cho 12 cờ tiện ích — tick thêm tiện ích KHÔNG được làm giảm giá dự báo (chặn
    confound "phòng rẻ quảng cáo nhiều tiện ích, căn hộ đắt không liệt kê"); tùy chọn
    +1 cho `area`. Cột không nêu tên mặc định không ràng buộc. Yêu cầu preprocessor
    xuất pandas (build_preprocessor đã set_output) để XGBoost thấy tên cột.
    """
    cons = {c: 1 for c in col_groups["bin"]}
    if constrain_area:
        cons["area"] = 1
    return cons


class MonotoneRandomForestRegressor(RandomForestRegressor):
    """RandomForest áp **ràng buộc đơn điệu +1** cho các cờ tiện ích — bản RF
    "công bằng" để so trực tiếp với XGBoost (cùng yêu cầu: tick thêm tiện ích
    KHÔNG được làm giảm giá dự báo).

    `bin_cols` = tên 12 cờ tiện ích. monotonic_cst được TÍNH LẠI theo TÊN cột
    mỗi lần `fit`, nên bền với việc số chiều One-Hot thay đổi giữa các fold CV
    (vd. 64 chiều trong fold so với 72 chiều trên toàn bộ Train). Yêu cầu
    preprocessor xuất pandas (`build_preprocessor` đã `set_output`) để thấy tên cột.

    Khai báo đầy đủ chữ ký `__init__` của RandomForestRegressor để `get_params`/
    `clone` hoạt động đúng trong RandomizedSearchCV (nếu chỉ dùng **kwargs thì
    sklearn không thấy `n_estimators`, `max_depth`, … khi set_params).
    """

    def __init__(self, bin_cols=None, n_estimators=100, *, criterion="squared_error",
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                 n_jobs=None, random_state=None, verbose=0, warm_start=False,
                 ccp_alpha=0.0, max_samples=None):
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs,
            random_state=random_state, verbose=verbose, warm_start=warm_start,
            ccp_alpha=ccp_alpha, max_samples=max_samples)
        self.bin_cols = bin_cols

    def fit(self, X, y, **fit_kwargs):
        if self.bin_cols is not None and hasattr(X, "columns"):
            bs = set(self.bin_cols)
            self.monotonic_cst = np.array([1 if c in bs else 0 for c in X.columns])
        return super().fit(X, y, **fit_kwargs)
