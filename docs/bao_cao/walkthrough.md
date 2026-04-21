# Báo cáo Phân tích Thống kê Dữ liệu Nhà ở (Housing Data)

Dưới đây là kết quả phân tích thống kê mô tả thô quét trực tiếp từ toàn bộ các tệp `.csv` hiện có trong thư mục của bạn. Quá trình quét giúp đánh giá sơ bộ chất lượng dữ liệu để chuẩn bị cho các bước tiền xử lý (Preprocessing) tiếp theo.

## 1. Tệp dữ liệu `Housing_data_final.csv` (Dữ liệu đã qua làm sạch)
Đây là tập dữ liệu có chất lượng tốt nhất trong số các file, tất cả các biến đã được chuẩn hoá dưới dạng số học (numeric) và không có bất kỳ dữ liệu nào bị khuyết thiếu.

* **Kích thước:** `1635 dòng × 10 cột`
* **Dữ liệu thiếu (Null/NA):** Hoàn toàn sạch (0 giá trị null).

**Các biến định lượng:**
| Thành phần (Cột) | Min | Max | Trung bình (Mean) |
| :--- | :--- | :--- | :--- |
| `price` (Giá) | 921,053 | 7,335,780 | 2,572,058.52 |
| `area` (Diện tích) | 17 | 177 | 52.9 |
| `bedrooms` (Phòng ngủ) | 1 | 3 | 1.2 |
| `bathrooms` (Phòng tắm) | 0 | 2 | 1.04 |
| `dtcenter` (K/C trung tâm) | 0.27 km | 21.44 km | 4.53 km |

*(Ghi chú: Các biến tiện ích như `washingmachine`, `hotwater`, `ac`, `parking`, `security` đều mang giá trị kiểu số [0, 1] hoặc [0, 1, 2] đại diện đại lượng Categorical phân loại).*

---

## 2. Các tệp dữ liệu vùng miền (Hà Nội, TP.HCM, Đà Nẵng)
Ba tập dữ liệu `hn.csv`, `hcm.csv` và `dn.csv` có cùng một cấu trúc gồm `5 trường thông tin`: `title`, `price`, `published`, `acreage`, `address`.

> [!WARNING]
> Cả 3 tệp dữ liệu này đều có chứa những quan sát với **`price` là -1**. Rất có thể đây là cách ghi nhận cho mục `"Giá thoả thuận"` trên các website bất động sản. Cần lọc hoặc thay thế các giá trị này (Imputation) trước khi huấn luyện mô hình.

### Kích thước
- **Hà Nội (`hn.csv`):** 3,284 dòng
- **TP.HCM (`hcm.csv`):** 9,733 dòng
- **Đà Nẵng (`dn.csv`):** 768 dòng
- *Tình trạng Null:* Cả ba tệp hiện tại không phát hiện ô cứng trống (Empty/NA).

### Thống kê các cột định lượng (Price & Acreage)
**Giá trị `price`:**
- **Đà Nẵng:** Trung bình 2.12 (Min: -1, Max: 15)
- **Hà Nội:** Trung bình 2.66 (Min: -1, Max: 100)
- **TP.HCM:** Trung bình 3.39 (Min: -1, Max: 150)

**Giá trị `acreage` (Diện tích):**
- Diễn biến diện tích trung bình trên các khu vực khá đồng đều: ~26 m².
- Tuy nhiên tại Đà Nẵng và HCM xuất hiện các giá trị Maximum diện tích trần rất lớn (lên tới 600m² - 1000m²), có khả năng là nhiễu (outliers) hoặc dữ liệu dành cho mặt bằng nguyên toà nhà/đất nền.

---

## 3. Tệp dữ liệu thô `Hanoi_housing_dataset.csv`
Đây là tệp dữ liệu mang mức độ "thô" (raw) cao nhất, được scrape chủ yếu với nhiều khuyết thiếu cần được chú ý.

* **Kích thước:** `82,497 dòng × 13 cột`
* **Cấu trúc trường:** Các cột ghi nhận trực tiếp bằng tiếng Việt có dấu (`Ngày`, `Địa chỉ`, `Quận`, `Huyện`, `Loại hình nhà ở`, `Giấy tờ pháp lý`, `Số tầng`, `Số phòng ngủ`, `Dài`, `Rộng`, `Giá/m2`).

> [!CAUTION]
> **Cảnh báo Dữ liệu Khuyết Thiếu (Missing values) rất lớn:**
> - `Dài`: 62,670 dòng bị thiếu.
> - `Rộng`: 47,052 dòng bị thiếu.
> - `Số tầng`: 46,098 dòng bị thiếu.
> - `Giấy tờ pháp lý`: 28,887 dòng bị thiếu.
> - *Không nên sử dụng trực tiếp các cột Dài/Rộng/Số tầng cho mô hình khi lượng hao hụt quá 50%.*

**Ghi chú đối với kiểu dữ liệu của tệp này:** Do các con số (giá trị giá, diện tích) được cào về bị dính chung với ký tự chữ hoặc dấu phẩy (Ví dụ: "100 m2", "1,5 Tỷ"), toàn bộ các cột này đang bị phần mềm lưu trữ dưới dạng Chuỗi (Text/Categorical) thay vì dạng định lượng Numeric chuẩn. Bước tiền xử lý (Tách chữ, xoá dấu) là **việc bắt buộc phải làm làm đầu tiên** đối với tệp này.
