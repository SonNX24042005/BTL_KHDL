"""
Bước 6 — Hệ thống thử nghiệm: App web dự báo giá thuê nhà trọ/căn hộ (Streamlit).

Đề tài: "Phân tích & dự báo giá thuê nhà trọ/căn hộ theo khu vực" — IT4142.

App nạp **mô hình triển khai** đã chốt ở Bước 4–5 (`models/model_best.pkl` = XGBoost, refit trên
Train+Val) và **dùng chung feature engineering** với lúc huấn luyện (`script/model/features.py`)
để đảm bảo nhất quán train/serve. Người dùng nhập đặc điểm phòng → app dự báo giá + đối chiếu mặt
bằng khu vực, kèm các biểu đồ EDA (Bước 3) và thông tin đánh giá mô hình (Bước 4–5) hỗ trợ ra quyết định.

Chạy:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "script" / "model"))
import features as F  # feature engineering DÙNG CHUNG với lúc train

MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed" / "rental_data_clean.csv"
EDA_FIG = ROOT / "docs" / "eda"
MODEL_FIG = ROOT / "docs" / "model"
EVAL_FIG = ROOT / "docs" / "eval"
MIN_COUNT = 100          # khớp min_frequency của OneHotEncoder → vùng mô hình "phân biệt" được

st.set_page_config(page_title="Dự báo giá thuê nhà trọ", page_icon="🏠", layout="wide")


# ============================ NẠP TÀI NGUYÊN ================================
@st.cache_resource
def load_model():
    return joblib.load(MODELS / "model_best.pkl")


@st.cache_data
def load_json(name):
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


@st.cache_data
def build_assets(bin_cols):
    """Dựng dữ liệu phụ trợ cho app TRỰC TIẾP từ CSV đã làm sạch (cache 1 lần).

    Thay cho `models/app_assets.json` đã bỏ — giữ app tự chứa, luôn khớp dữ liệu/mô hình
    hiện tại. Logic giống hệt builder cũ: chỉ giữ TỈNH/THÀNH thật & quận/huyện đủ mẫu
    (≥ MIN_COUNT), trung vị giá + đơn giá/m² theo (thành phố, quận) và theo thành phố.
    """
    df = F.add_engineered(pd.read_csv(DATA, low_memory=False))
    df["price_per_m2"] = df["price"] / df["area"]

    prov_set = set(F.PROVINCES)
    city_counts = df["city"].value_counts()
    cities = [c for c in city_counts.index if c in prov_set and city_counts[c] >= MIN_COUNT]

    districts_by_city, region_stats = {}, {}
    for city in cities:
        sub = df[df["city"] == city]
        dc = sub["district"].value_counts()
        dists = sorted(d for d in dc.index if dc[d] >= MIN_COUNT and d != F.UNKNOWN)
        districts_by_city[city] = dists
        for d in dists:
            s = sub[sub["district"] == d]
            region_stats[f"{city}|{d}"] = {
                "median_price": round(float(s["price"].median()), 2),
                "median_ppm": round(float(s["price_per_m2"].median()), 4),
                "n": int(len(s)),
            }

    city_stats = {
        c: {"median_price": round(float(df.loc[df.city == c, "price"].median()), 2),
            "median_ppm": round(float((df.loc[df.city == c, "price"] /
                                       df.loc[df.city == c, "area"]).median()), 4),
            "n": int((df.city == c).sum())}
        for c in cities
    }

    return {
        "cities": cities,
        "districts_by_city": districts_by_city,
        "region_stats": region_stats,
        "city_stats": city_stats,
        "amenities": [{"col": c, "label": c.replace("feature_", "")} for c in bin_cols],
        "ranges": {
            "area": [float(df.area.min()), float(df.area.max())],
            "price": [float(df.price.min()), float(df.price.max()), float(df.price.median())],
            "bedrooms": [int(df.bedrooms.min()), int(df.bedrooms.max())],
            "bathrooms": [int(df.bathrooms.min()), int(df.bathrooms.max())],
        },
        "overall_median_price": round(float(df.price.median()), 2),
        "unknown_label": F.UNKNOWN,
    }


model = load_model()
meta = load_json("metadata.json")

BIN_COLS = meta["feature_columns"]["bin"]
assets = build_assets(BIN_COLS)
TEST = meta.get("test_metrics", {}).get(meta["best_model"], {})


def predict_price(area, bedrooms, bathrooms, city, district, amenities):
    """Dựng đúng 1 dòng input theo schema của mô hình rồi dự báo (triệu VNĐ/tháng)."""
    row = {"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
           "city": city, "district": district}
    for c in BIN_COLS:
        row[c] = int(amenities.get(c, 0))                    # người dùng khai báo → luôn 0/1 thật
    X = pd.DataFrame([row])[F.NUM_COLS + F.CAT_COLS + BIN_COLS]
    return float(model.predict(X)[0])


def show_img(path, caption=None):
    if Path(path).exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"(Chưa có biểu đồ: {Path(path).name})")


# ================================ HEADER ===================================
st.title("🏠 Dự báo Giá thuê Nhà trọ / Căn hộ theo Khu vực")
st.caption("Đồ án Nhập môn Khoa học Dữ liệu (IT4142) — HUST · Mô hình: "
           f"**{meta['best_model']}** · "
           f"Test R² = {TEST.get('Test_R2', float('nan')):.3f}, "
           f"MAE = {TEST.get('Test_MAE', float('nan')):.2f} triệu/tháng")

tab_predict, tab_eda, tab_model = st.tabs(
    ["🔮 Dự báo giá", "📊 Phân tích thị trường (EDA)", "🤖 Về mô hình"])

# ============================== TAB 1: DỰ BÁO ==============================
with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Nhập thông tin phòng")
        amin, amax = assets["ranges"]["area"]
        area = st.slider("📐 Diện tích (m²)", float(amin), float(amax), 25.0, 1.0)

        c1, c2 = st.columns(2)
        bmin, bmax = assets["ranges"]["bedrooms"]
        bedrooms = c1.number_input("🛏️ Số phòng ngủ", bmin, bmax, 1)
        wmin, wmax = assets["ranges"]["bathrooms"]
        bathrooms = c2.number_input("🚿 Số phòng tắm", wmin, wmax, 1)

        c3, c4 = st.columns(2)
        city = c3.selectbox("🏙️ Thành phố", assets["cities"])
        UNK = assets["unknown_label"]
        dist_opts = [f"— Khác / {UNK} —"] + assets["districts_by_city"].get(city, [])
        district_sel = c4.selectbox("📍 Quận / Huyện", dist_opts)
        district = UNK if district_sel.startswith("— Khác") else district_sel

        st.markdown("**🛋️ Tiện ích** (tick những gì có):")
        amenities, acols = {}, st.columns(3)
        for i, am in enumerate(assets["amenities"]):
            amenities[am["col"]] = acols[i % 3].checkbox(am["label"], value=False)

    with right:
        st.subheader("Kết quả dự báo")
        price = predict_price(area, bedrooms, bathrooms, city, district, amenities)
        mae = TEST.get("Test_MAE", 1.0)
        lo, hi = max(0.3, price - mae), price + mae

        st.metric("💰 Giá thuê dự báo", f"{price:.1f} triệu/tháng",
                  help="Ước lượng điểm từ mô hình XGBoost.")
        st.caption(f"Khoảng tham khảo (± MAE trên Test): **{lo:.1f} – {hi:.1f}** triệu/tháng")
        if area > 0:
            st.caption(f"Tương đương đơn giá ~ **{price/area*1000:,.0f} nghìn/m²/tháng**")

        # Đối chiếu mặt bằng khu vực
        key = f"{city}|{district}"
        region = assets["region_stats"].get(key)
        base = region or assets["city_stats"].get(city)
        if base:
            med = base["median_price"]
            scope = (f"quận **{district}**" if region else f"thành phố **{city}**")
            delta = price - med
            st.metric(f"So với trung vị {scope}", f"{med:.1f} triệu",
                      delta=f"{delta:+.1f} triệu so với mức của bạn", delta_color="inverse")
            comp = pd.DataFrame({
                "Mức giá": ["Dự báo của bạn", f"Trung vị {('quận' if region else 'TP')}"],
                "Triệu/tháng": [round(price, 2), med]}).set_index("Mức giá")
            st.bar_chart(comp, horizontal=True, color="#2D5BFF")
            if region:
                st.caption(f"*(Dựa trên {region['n']:,} tin đăng tại khu vực này.)*")

        st.info("ℹ️ Mô hình **không dùng** 'nguồn web' làm đặc trưng — dự báo chỉ dựa trên "
                "đặc điểm thật của phòng (diện tích, khu vực, tiện ích). Tiện ích được huấn luyện "
                "với **ràng buộc đơn điệu**: tick thêm tiện ích không bao giờ làm giảm giá dự báo. "
                "Kết quả mang tính tham khảo.")

# ============================== TAB 2: EDA =================================
with tab_eda:
    st.subheader("Bức tranh thị trường thuê (từ phân tích EDA — Bước 3)")
    st.markdown("Các biểu đồ dưới đây giúp đặt giá dự báo vào **bối cảnh thị trường**.")
    g1, g2 = st.columns(2)
    with g1:
        show_img(EDA_FIG / "02_histogram_price_area.png", "Phân phối giá thuê & diện tích")
        show_img(EDA_FIG / "04_boxplot_price_by_district.png", "Giá thuê theo Quận/Huyện (Top nhiều tin)")
        show_img(EDA_FIG / "08_amenity_price_uplift.png", "Tiện ích nào gắn với giá cao hơn")
    with g2:
        show_img(EDA_FIG / "05_price_per_m2_ranking.png", "Khu vực đắt nhất theo đơn giá/m²")
        show_img(EDA_FIG / "11_heatmap_correlation.png", "Tương quan giữa các biến số")
        show_img(EDA_FIG / "07_amenity_prevalence.png", "Mức độ phổ biến của tiện ích")
    st.caption("Biểu đồ tương tác (HTML) và đầy đủ phân tích xem trong `notebooks/03_eda.ipynb`.")

# ============================ TAB 3: VỀ MÔ HÌNH ============================
with tab_model:
    st.subheader("Mô hình & cách đánh giá (Bước 4–5)")
    st.markdown(f"""
- **Bài toán:** hồi quy — dự báo `price` (triệu VNĐ/tháng).
- **Mô hình triển khai:** **{meta['best_model']}** (gradient boosting + early stopping), huấn luyện
  lại trên Train+Val. Random Forest chính xác tương đương nhưng nặng ~40× nên chọn XGBoost.
- **Đặc trưng:** diện tích, số phòng ngủ/tắm, **thành phố/quận**, 12 cờ tiện ích.
  *Cố ý loại `source` (nguồn web)* — là artifact thu thập, không phải yếu tố định giá thực.
- **Làm sạch tiện ích (MNAR):** đối chiếu dữ liệu gốc cho thấy nhiều cờ tiện ích bị điền
  ngẫu nhiên ở bước gộp nguồn → các ô đó được coi là **"không khai báo" (NaN)** thay vì "không có"
  khi huấn luyện; đồng thời áp **ràng buộc đơn điệu** để tiện ích không làm giảm giá dự báo.
- **Quy trình đánh giá:** chia 70/15/15 → tinh chỉnh siêu tham số → 10-fold CV → đánh giá **1 lần
  trên Test**.
""")
    if TEST:
        st.markdown("**Hiệu năng trên tập Test (đánh giá 1 lần):**")
        st.dataframe(pd.DataFrame({
            "Chỉ số": ["MAE", "RMSE", "R²"],
            "Giá trị": [f"{TEST['Test_MAE']:.3f} triệu", f"{TEST['Test_RMSE']:.3f} triệu",
                        f"{TEST['Test_R2']:.3f}"]}), hide_index=True, width="stretch")

    m1, m2 = st.columns(2)
    with m1:
        show_img(MODEL_FIG / "01_model_comparison_val.png", "So sánh 5 mô hình (Validation)")
        show_img(EVAL_FIG / "02_test_actual_vs_pred.png", "Thực tế vs Dự đoán + Phần dư (Test)")
    with m2:
        show_img(MODEL_FIG / "02_feature_importance.png", "Đặc trưng quan trọng nhất")
        show_img(EVAL_FIG / "03_residual_learning_curve.png", "Phân phối phần dư & Learning curve")
    st.caption("Chi tiết: `notebooks/04_modeling.ipynb` (huấn luyện) và "
               "`notebooks/05_evaluation.ipynb` (đánh giá).")
