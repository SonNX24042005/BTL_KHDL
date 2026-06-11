"""
Đặc trưng dùng chung cho Bước 4 (xây dựng mô hình) — *single source of truth*.

Đặt riêng ở đây để **train (notebook 04) và serve (app Streamlit ở Bước 6) dùng CHUNG
một logic feature engineering**, tránh lệch train/serve (train/serve skew). Logic trích
xuất `city`/`district` từ `full_address` và `n_features` kế thừa nguyên vẹn từ Bước 3
(notebook EDA) để giữ nhất quán xuyên suốt dự án.

Quan trọng — KHÔNG dùng làm đặc trưng:
  • `source`                     → **nguồn web cào** chỉ là artifact thu thập dữ liệu, KHÔNG phải
                                   yếu tố định giá thực tế; app demo (Bước 6) cũng không có input này.
                                   Mô hình phải dự báo từ đặc điểm BĐS, không từ "tin này lấy ở web nào".
  • `price_per_m2`, `log_price`  → suy ra từ `price`  ⇒ rò rỉ nhãn (data leakage).
  • `posted_date`                → ~46% là giá trị sentinel ngày tương lai (xem EDA 3.2).
  • `description`, `detail_link` → thiếu phần lớn / không mang tín hiệu giá.
  • `full_address`              → đã thay bằng `city` + `district` đã tách.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

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
NUM_COLS = ["area", "bedrooms", "bathrooms", "n_features"]  # biến số (chuẩn hóa Z-score)
CAT_COLS = ["city", "district"]                             # biến danh mục (One-Hot) — CỐ Ý bỏ `source`
# BIN_COLS = 12 cờ tiện ích `feature_*` (đã 0/1) — lấy động bằng get_feature_cols().


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
    """Thêm `city`, `district`, `n_features` (không đụng các cột gốc). Trả về bản sao."""
    df = df.copy()
    df.columns = [c.lstrip("﻿") for c in df.columns]  # bỏ BOM nếu có
    fcols = get_feature_cols(df)
    parsed = df["full_address"].apply(parse_address)
    df["city"] = parsed.map(lambda x: x[0]).fillna(UNKNOWN)
    df["district"] = parsed.map(lambda x: x[1]).fillna(UNKNOWN)
    df["n_features"] = df[fcols].sum(axis=1)
    return df


def build_Xy(df: pd.DataFrame):
    """Sinh (X, y, col_groups) sẵn sàng cho ColumnTransformer.

    col_groups = {"num": [...], "cat": [...], "bin": [...]} để dựng preprocessor.
    """
    df = add_engineered(df)
    fcols = get_feature_cols(df)
    col_groups = {"num": NUM_COLS, "cat": CAT_COLS, "bin": fcols}
    feature_columns = NUM_COLS + CAT_COLS + fcols
    X = df[feature_columns].copy()
    y = df[TARGET].copy()
    return X, y, col_groups


def build_preprocessor(col_groups: dict, min_frequency: int = 100):
    """ColumnTransformer dùng chung cho cả 5 mô hình.

    - num: StandardScaler (cần cho mô hình tuyến tính; vô hại với cây).
    - cat: OneHotEncoder gộp hạng mục hiếm (< `min_frequency`) để chặn bùng nổ chiều
      do `district` (360) / `city` (204) cardinality cao; `infrequent_if_exist` còn xử lý
      được hạng mục lạ lúc dự đoán (app demo).
    - bin: passthrough (12 cờ tiện ích đã 0/1).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), col_groups["num"]),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=min_frequency,
                                  sparse_output=False), col_groups["cat"]),
            ("bin", "passthrough", col_groups["bin"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
