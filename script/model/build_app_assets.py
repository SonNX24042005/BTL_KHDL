"""
Sinh `models/app_assets.json` — dữ liệu phụ trợ GỌN cho app Streamlit (Bước 6).

Mục tiêu: app **không phải nạp lại CSV 24 MB** mỗi lần chạy → chỉ cần model_best.pkl +
metadata.json + app_assets.json (nhỏ) + ảnh EDA. Asset gồm:
  • Danh sách thành phố (lọc về TỈNH/THÀNH THẬT trong F.PROVINCES, đủ mẫu) và quận/huyện theo
    từng thành phố — tránh nhiễu do parse_address fallback (vd '11 tổ dân phố số 6').
  • Trung vị giá & đơn giá/m² theo (thành phố, quận) và theo thành phố → ngữ cảnh thị trường.
  • Nhãn đẹp cho 12 cờ tiện ích; miền giá trị area/price.

Chạy:  python script/model/build_app_assets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "script" / "model"))
import features as F

DATA = ROOT / "data" / "processed" / "rental_data_clean.csv"
MODELS = ROOT / "models"
MIN_COUNT = 100          # khớp min_frequency của OneHotEncoder → vùng mô hình "phân biệt" được

meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
BIN_COLS = meta["feature_columns"]["bin"]

df = F.add_engineered(pd.read_csv(DATA, low_memory=False))
df["price_per_m2"] = df["price"] / df["area"]

# ---- Thành phố: chỉ giữ tỉnh/thành THẬT, đủ mẫu ----------------------------
prov_set = set(F.PROVINCES)
city_counts = df["city"].value_counts()
cities = [c for c in city_counts.index if c in prov_set and city_counts[c] >= MIN_COUNT]

# ---- Quận/huyện theo từng thành phố (đủ mẫu) -------------------------------
districts_by_city, region_stats = {}, {}
for city in cities:
    sub = df[df["city"] == city]
    dc = sub["district"].value_counts()
    dists = [d for d in dc.index if dc[d] >= MIN_COUNT and d != F.UNKNOWN]
    # sắp theo tên cho dễ tra; đưa lựa chọn "(không rõ/khác)" lên đầu để có default an toàn
    dists = sorted(dists)
    districts_by_city[city] = dists
    for d in dists:
        s = sub[sub["district"] == d]
        region_stats[f"{city}|{d}"] = {
            "median_price": round(float(s["price"].median()), 2),
            "median_ppm": round(float(s["price_per_m2"].median()), 4),
            "n": int(len(s)),
        }

# Thống kê theo thành phố (khi không chọn quận cụ thể)
city_stats = {
    c: {"median_price": round(float(df.loc[df.city == c, "price"].median()), 2),
        "median_ppm": round(float((df.loc[df.city == c, "price"] /
                                   df.loc[df.city == c, "area"]).median()), 4),
        "n": int((df.city == c).sum())}
    for c in cities
}

assets = {
    "cities": cities,                                   # đã sắp theo số lượng giảm dần
    "districts_by_city": districts_by_city,
    "region_stats": region_stats,
    "city_stats": city_stats,
    "amenities": [{"col": c, "label": c.replace("feature_", "")} for c in BIN_COLS],
    "ranges": {
        "area":  [float(df.area.min()), float(df.area.max())],
        "price": [float(df.price.min()), float(df.price.max()), float(df.price.median())],
        "bedrooms":  [int(df.bedrooms.min()), int(df.bedrooms.max())],
        "bathrooms": [int(df.bathrooms.min()), int(df.bathrooms.max())],
    },
    "overall_median_price": round(float(df.price.median()), 2),
    "unknown_label": F.UNKNOWN,
}

(MODELS / "app_assets.json").write_text(
    json.dumps(assets, ensure_ascii=False, indent=2), encoding="utf-8")

print("Đã ghi", MODELS / "app_assets.json")
print(f"  • {len(cities)} thành phố: {cities}")
print(f"  • Tổng quận/huyện (đủ mẫu): {sum(len(v) for v in districts_by_city.values())}")
print(f"  • {len(region_stats)} vùng có thống kê giá; {len(BIN_COLS)} tiện ích")
print(f"  • Kích thước file: {(MODELS/'app_assets.json').stat().st_size/1024:.1f} KB")
