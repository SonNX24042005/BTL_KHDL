# BTL_KHDL — Bài Tập Lớn Khoa Học Dữ Liệu

Dự án thu thập và phân tích dữ liệu bất động sản từ nhiều nguồn website khác nhau tại Việt Nam.

---

## 📁 Cấu trúc thư mục

```
btl_khdl/
├── README.md
├── data/
│   └── raw/                    # Dữ liệu thô đã được craw
│       ├── alonhadat/
│       ├── batdongsan/
│       ├── chothuephongtro/
│       ├── lozido/
│       ├── nhatot/
│       └── phongtro123/
├── docs/                       # Tài liệu dự án
└── script/                     # Script craw dữ liệu
    ├── alonhadat/
    ├── batdongsan/
    ├── chothuephongtro/
    ├── lozido/
    ├── nhatot/
    └── phongtro123/
```

---

## 📂 `data/raw/` — Dữ liệu thô

Thư mục `data/raw/` chứa toàn bộ dữ liệu thô được thu thập (craw) từ các website bất động sản. Mỗi thư mục con tương ứng với một nguồn website riêng biệt.

| Thư mục con | Website nguồn |
|---|---|
| `data/raw/alonhadat/` | [alonhadat.com.vn](https://alonhadat.com.vn) |
| `data/raw/batdongsan/` | [batdongsan.com.vn](https://batdongsan.com.vn) |
| `data/raw/chothuephongtro/` | [chothuephongtro.com](https://chothuephongtro.com) |
| `data/raw/lozido/` | [lozido.com](https://lozido.com) |
| `data/raw/nhatot/` | [nhatot.com](https://nhatot.com) |
| `data/raw/phongtro123/` | [phongtro123.com](https://phongtro123.com) |

> **Lưu ý:** Dữ liệu trong thư mục này là dữ liệu gốc, chưa qua xử lý. Không chỉnh sửa trực tiếp các file trong thư mục này.

---

## 📂 `script/` — Script craw dữ liệu

Thư mục `script/` chứa các script dùng để craw dữ liệu từ từng website. Mỗi thư mục con chứa script tương ứng với nguồn dữ liệu cùng tên trong `data/raw/`.

| Thư mục con | Script craw cho |
|---|---|
| `script/alonhadat/` | Craw dữ liệu từ alonhadat.com.vn |
| `script/batdongsan/` | Craw dữ liệu từ batdongsan.com.vn |
| `script/chothuephongtro/` | Craw dữ liệu từ chothuephongtro.com |
| `script/lozido/` | Craw dữ liệu từ lozido.com |
| `script/nhatot/` | Craw dữ liệu từ nhatot.com |
| `script/phongtro123/` | Craw dữ liệu từ phongtro123.com |

---

## 🔗 Mapping Script ↔ Dữ liệu thô

| Script | → | Dữ liệu đầu ra |
|---|:---:|---|
| `script/alonhadat/` | → | `data/raw/alonhadat/` |
| `script/batdongsan/` | → | `data/raw/batdongsan/` |
| `script/chothuephongtro/` | → | `data/raw/chothuephongtro/` |
| `script/lozido/` | → | `data/raw/lozido/` |
| `script/nhatot/` | → | `data/raw/nhatot/` |
| `script/phongtro123/` | → | `data/raw/phongtro123/` |
