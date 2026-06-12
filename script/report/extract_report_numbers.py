"""
Trích xuất TOÀN BỘ số liệu v2 cho báo cáo PDF (v2/docs/Bao_Cao_PDF).
- Phần lớn chỉ số mô hình/ablation đọc từ v2/models/metadata.json (sản phẩm Bước 4/5).
- Thống kê EDA (mục Dữ liệu) tính lại từ v2/data/processed/rental_data_clean.csv.
- Thực nghiệm D (kiểm toán): huấn luyện 3 biến thể XGBoost + đo "cờ âm"/đơn điệu + sinh
  hình audit_amenity_effects.png.
Kết quả: in JSON ra stdout + ghi v2/script/report/report_numbers.json; hình → v2/docs/Bao_Cao_PDF/images/.
Chạy: python v2/script/report/extract_report_numbers.py
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RS = 42
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "v2" / "script" / "model"))
import features as F

DATA = ROOT / "v2" / "data" / "processed"
MODELS = ROOT / "v2" / "models"
IMG = ROOT / "v2" / "docs" / "Bao_Cao_PDF" / "images"
IMG.mkdir(parents=True, exist_ok=True)
PALETTE = ["#2D5BFF", "#00C2A8", "#FF8C42", "#E5446D", "#7B61FF", "#FFC857"]
OK, BAD = "#00C2A8", "#E5446D"

R = {}   # gom mọi số liệu

# ============================ DỮ LIỆU + EDA ============================
df = pd.read_csv(DATA / "rental_data_clean.csv", low_memory=False)
df.columns = [c.lstrip("﻿") for c in df.columns]
de = F.add_engineered(df)                       # thêm city/district/n_features
fcols = F.get_feature_cols(df)
N = len(df)
R["n_total"] = N

# Bảng nguồn
src = df["source"].value_counts()
R["sources"] = [{"name": s, "count": int(c), "pct": round(c / N * 100, 1)} for s, c in src.items()]
R["imbalance_top"] = {"name": src.index[0], "pct": round(src.iloc[0] / N * 100, 1)}

# Thiếu
def miss_pct(col):
    s = df[col]
    return round(float((s.isna() | (s.astype(str).str.strip() == "")).mean()) * 100, 1)
R["missing"] = {c: miss_pct(c) for c in ["detail_link", "description"]}
dist_unknown = int((de["district"] == F.UNKNOWN).sum())
R["missing"]["district_pct"] = round(dist_unknown / N * 100, 1)
R["missing"]["district_count"] = dist_unknown

# posted_date sentinel
d = pd.to_datetime(df["posted_date"], errors="coerce")
R["posted_future_pct"] = int(round((d > pd.Timestamp("2026-06-06")).mean() * 100))

# Thống kê mô tả
de["n_features"] = de[fcols].sum(axis=1)
desc = {}
for col in ["price", "area", "bedrooms", "bathrooms", "n_features"]:
    s = de[col]
    desc[col] = {"mean": round(float(s.mean()), 2), "std": round(float(s.std()), 2),
                 "min": round(float(s.min()), 1), "median": round(float(s.median()), 2),
                 "max": round(float(s.max()), 1), "skew": round(float(s.skew()), 2),
                 "kurt": round(float(s.kurt()), 2)}
R["descriptive"] = desc

# Địa lý
city = de["city"].value_counts()
R["geo"] = [{"city": c, "count": int(n), "pct": round(n / N * 100, 1)} for c, n in city.head(3).items()]
R["geo_provinces"] = int(sum(1 for p in F.PROVINCES if de["city"].eq(p).any()))
R["district_cardinality"] = int(de["district"].nunique())

# Spearman với price
de["price_per_m2"] = de["price"] / de["area"]
sp = de[["price", "area", "bedrooms", "bathrooms", "n_features"]].corr("spearman")["price"]
R["spearman_price"] = {k: round(float(sp[k]), 3) for k in ["area", "bedrooms", "bathrooms", "n_features"]}

# Uplift tiện ích (mean có - mean không)
uplift = {}
for c in fcols:
    has = de.loc[de[c] == 1, "price"].mean(); no = de.loc[de[c] == 0, "price"].mean()
    uplift[c.replace("feature_Có ", "").replace("feature_", "")] = round(float(has - no), 2)
R["uplift"] = uplift

# Simpson: Spearman(n_features, price) theo nguồn (>=2000) + toàn cục
simp = {}
for s, g in de.groupby("source"):
    if len(g) >= 2000:
        simp[s] = round(float(g["n_features"].corr(g["price"], method="spearman")), 3)
R["simpson_by_source"] = dict(sorted(simp.items(), key=lambda kv: kv[1], reverse=True))
R["simpson_global"] = round(float(de["n_features"].corr(de["price"], method="spearman")), 3)

# PCA
from sklearn.decomposition import PCA
Xs = StandardScaler().fit_transform(de[["price", "area", "bedrooms", "bathrooms", "n_features"]].dropna())
evr = PCA(n_components=2).fit(Xs).explained_variance_ratio_
R["pca"] = {"pc1": round(float(evr[0]) * 100, 1), "pc2": round(float(evr[1]) * 100, 1),
            "total": round(float(evr[:2].sum()) * 100, 1)}

# ============================ METADATA (Bước 4/5) ============================
meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
R["split"] = meta["split"]
R["best_params"] = meta["best_params"]
R["cv_metrics"] = meta["cv_metrics"]
R["test_metrics"] = meta["test_metrics"]
R["most_accurate_test"] = meta.get("most_accurate_test")
R["ablation"] = meta["ablation"]
# FE dims (số chiều sau preprocessor)
X, y, CG = F.build_Xy(df)
prep_demo = F.build_preprocessor(CG, min_frequency=100).fit(X)
R["fe_dims"] = int(len(prep_demo.get_feature_names_out()))
R["cat_dims"] = int(sum(1 for c in prep_demo.get_feature_names_out()
                        if c.startswith("city") or c.startswith("district")))
# Kích thước file mô hình
R["size_xgb_kb"] = round((MODELS / "model_best.pkl").stat().st_size / 1024)
rf = MODELS / "model_random_forest.pkl"
R["size_rf_kb"] = round(rf.stat().st_size / 1024) if rf.exists() else None

# ============================ THỰC NGHIỆM D — KIỂM TOÁN ============================
print(">> Thực nghiệm D: huấn luyện 3 biến thể + behavioral...", file=sys.stderr)
hp = {k.replace("xgb__", ""): v for k, v in meta["best_params"]["XGBoost"].items()
      if k.replace("xgb__", "") in ("learning_rate", "max_depth", "subsample",
                                     "colsample_bytree", "reg_alpha", "reg_lambda")}

def split_tv_test(Xf, yf):
    Xtv, Xte, ytv, yte = train_test_split(Xf, yf, test_size=0.15, random_state=RS)
    Xtr, Xv, ytr, yv = train_test_split(Xtv, ytv, test_size=0.176, random_state=RS)
    return Xtr, Xv, Xte, ytr, yv, yte, Xtv, ytv

# --- chuẩn bị 3 bộ dữ liệu ---
NUM = F.NUM_COLS; CAT = F.CAT_COLS
def prep_std(impute=False):
    return F.build_preprocessor(CG, min_frequency=100, impute_bin=impute)
# Biến thể "Ban đầu": cờ THÔ (không mask) + n_features là biến số
de_raw = F.add_engineered(df)   # chưa mask
de_raw["n_features"] = de_raw[fcols].sum(axis=1)
X_v1 = de_raw[NUM + ["n_features"] + CAT + fcols].copy()
def prep_v1():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM + ["n_features"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                              sparse_output=False), CAT),
        ("bin", "passthrough", fcols)],
        remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
# Biến thể "mask only" và "mask+monotone": dùng X (đã mask qua build_Xy)
X_m = X.copy()
mono = F.monotone_constraints(CG)

def run_variant(Xf, prepfn, mono):
    Xtr, Xv, Xte, ytr, yv, yte, Xtv, ytv = split_tv_test(Xf, y)
    p0 = prepfn(); Xtr_t = p0.fit_transform(Xtr)
    m = XGBRegressor(tree_method="hist", random_state=RS, n_jobs=-1,
                     early_stopping_rounds=50, n_estimators=2000,
                     monotone_constraints=mono, **hp)
    m.fit(Xtr_t, ytr, eval_set=[(p0.transform(Xv), yv)], verbose=False)
    nbest = int(m.best_iteration)
    p1 = prepfn(); Xtv_t = p1.fit_transform(Xtv)
    mf = XGBRegressor(tree_method="hist", random_state=RS, n_jobs=-1,
                      n_estimators=nbest, monotone_constraints=mono, **hp)
    mf.fit(Xtv_t, ytv)
    pipe = Pipeline([("prep", p1), ("xgb", mf)])
    pred = pipe.predict(Xte)
    return pipe, {"MAE": round(float(mean_absolute_error(yte, pred)), 3),
                  "RMSE": round(float(root_mean_squared_error(yte, pred)), 3),
                  "R2": round(float(r2_score(yte, pred)), 3)}, nbest

m1, met1, n1 = run_variant(X_v1, prep_v1, None)            # Ban đầu (thô + n_features)
m2, met2, n2 = run_variant(X_m, lambda: prep_std(False), None)   # chỉ mask
m3, met3, n3 = run_variant(X_m, lambda: prep_std(False), mono)   # mask + monotone (deployed)

# --- behavioral: input app điển hình ---
top_dist = de[de.city == "Hà Nội"]["district"].value_counts().index[0]
base = {"area": 25.0, "bedrooms": 1, "bathrooms": 1, "city": "Hà Nội", "district": top_dist}
base.update({c: 0 for c in fcols})

def eff_per_flag(pipe, cols, with_nf):
    def pred(over):
        row = {**base, **over}
        if with_nf: row["n_features"] = sum(row[c] for c in fcols)
        return float(pipe.predict(pd.DataFrame([row])[cols])[0])
    base_pred = pred({})
    effs = {c.replace("feature_Có ", "").replace("feature_", ""): round(pred({c: 1}) - base_pred, 3)
            for c in fcols}
    # đơn điệu: bật dồn 1..12
    seq = [pred({c: 1 for c in fcols[:k]}) for k in range(len(fcols) + 1)]
    viol = sum(1 for k in range(1, len(seq)) if seq[k] < seq[k - 1] - 1e-9)
    neg = sum(1 for v in effs.values() if v < -1e-6)
    return effs, neg, viol

cols_v1 = NUM + ["n_features"] + CAT + fcols
cols_std = NUM + CAT + fcols
eff1, neg1, vio1 = eff_per_flag(m1, cols_v1, with_nf=True)
eff3, neg3, vio3 = eff_per_flag(m3, cols_std, with_nf=False)
_, neg2, vio2 = eff_per_flag(m2, cols_std, with_nf=False)

R["audit"] = {
    "typical_input": {"area": 25, "bedrooms": 1, "bathrooms": 1, "city": "Hà Nội", "district": top_dist},
    "variants": {
        "initial":      {**met1, "neg_flags": neg1, "mono_viol": vio1, "trees": n1},
        "mask_only":    {**met2, "neg_flags": neg2, "mono_viol": vio2, "trees": n2},
        "mask_monotone":{**met3, "neg_flags": neg3, "mono_viol": vio3, "trees": n3},
    },
    "ac_effect_initial": eff1.get("máy lạnh"),
    "neg_effects_initial": {k: v for k, v in sorted(eff1.items(), key=lambda kv: kv[1]) if v < 0},
}

# nhatot dấu vân tị nguồn
nha = de[de.source == "nhatot"]; oth = de[de.source != "nhatot"]
R["audit"]["nhatot"] = {
    "median_price": round(float(nha["price"].median()), 1),
    "others_median": round(float(oth["price"].median()), 1),
    "ac_rate": round(float(nha["feature_Có máy lạnh"].mean()) * 100),
    "wash_rate": round(float(nha["feature_Có máy giặt"].mean()) * 100),
    "hotwater_rate": round(float(nha["feature_Có nước nóng"].mean()) * 100),
    "share_in_ac0": round(float((de.loc[de["feature_Có máy lạnh"] == 0, "source"] == "nhatot").mean()) * 100, 1),
    "share_in_ac1": round(float((de.loc[de["feature_Có máy lạnh"] == 1, "source"] == "nhatot").mean()) * 100, 1),
}
# Tỷ lệ dữ liệu theo nhóm nguồn (bảng kiểm toán cờ)
zero_src = [s for s, v in F.OBSERVED_FLAGS.items() if len(v) == 0]
R["audit"]["share_zero_flag_src"] = round(float(df["source"].isin(zero_src).mean()) * 100, 1)
R["audit"]["share_7fab_real_src"] = round(float(df["source"].isin(["phongtro123", "lozido"]).mean()) * 100, 1)

# ============================ HÌNH audit_amenity_effects ============================
labels = list(eff1.keys())
v_init = [eff1[k] for k in labels]; v_dep = [eff3[k] for k in labels]
order = np.argsort(v_init)
labels = [labels[i] for i in order]; v_init = [v_init[i] for i in order]; v_dep = [v_dep[i] for i in order]
yp = np.arange(len(labels)); h = 0.4
fig, ax = plt.subplots(figsize=(11, 6.5))
b1 = ax.barh(yp + h/2, v_init, height=h, color=[BAD if v < 0 else "#F2A6B6" for v in v_init],
             label="Mô hình ban đầu (cờ thô)")
b2 = ax.barh(yp - h/2, v_dep, height=h, color=OK, label="Mô hình triển khai (mask + ràng buộc)")
ax.axvline(0, color="#333", lw=1)
ax.set_yticks(yp); ax.set_yticklabels(labels)
ax.set_xlabel("Hiệu ứng lên giá dự báo khi tick tiện ích (triệu/tháng) — input app điển hình")
ax.set_title("Kiểm toán hành vi: hiệu ứng từng tiện ích — ban đầu (đỏ, có hiệu ứng âm phi lý)\n"
             "vs. mô hình triển khai sau kiểm toán (xanh, không còn âm)")
ax.legend(loc="lower right"); sns.despine(fig)
plt.tight_layout(); fig.savefig(IMG / "audit_amenity_effects.png", dpi=150, bbox_inches="tight",
                                facecolor="white"); plt.close(fig)

# ============================ XUẤT ============================
out = ROOT / "v2" / "script" / "report" / "report_numbers.json"
out.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=2))
print("\n>> Đã ghi", out, "| hình audit →", IMG / "audit_amenity_effects.png", file=sys.stderr)
