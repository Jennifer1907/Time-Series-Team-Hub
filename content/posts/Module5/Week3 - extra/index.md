---
title: "Module 5 - Tuần 3+4: Dự đoán giá nhà với Machine Learning – Từ tiền xử lý đến mô hình hóa"
date: 2025-11-01T10:00:00+07:00
description: "Hành trình xây dựng mô hình dự đoán giá nhà: xử lý dữ liệu thiếu, kỹ thuật đặc trưng, và tối ưu mô hình hồi quy."
image: images/house_price_prediction.png
categories:
  - minutes
tags:
  - feature
math: true
draft: false
---

🏡 **Dự án Tuần 3+4 của Module 5** là một bài học thực chiến đầy đủ quy trình của một dự án **Machine Learning**: từ thu thập, xử lý dữ liệu, đến huấn luyện và đánh giá mô hình.
Chủ đề trọng tâm là **Dự đoán giá nhà (House Price Prediction)** — một bài toán kinh điển nhưng chứa đựng nhiều thách thức trong xử lý dữ liệu thực tế.

---

## 🎯 Mục tiêu dự án

- Làm quen với quy trình xử lý dữ liệu thực tế: nhận diện giá trị khuyết, chuẩn hóa, mã hóa.
- Hiểu rõ vai trò của **Feature Engineering** và **Feature Selection**.
- Xây dựng, huấn luyện, và đánh giá các mô hình hồi quy: **Linear Regression**, **Ridge**, **Lasso**, **ElasticNet**, **Random Forest**, **XGBoost**.
- So sánh, lựa chọn mô hình tối ưu dựa trên độ chính xác và khả năng tổng quát hóa.

---

## 🧩 Quy trình thực hiện

Dự án được triển khai theo 5 giai đoạn chính:

### 1️⃣ Khám phá dữ liệu (EDA)

Phân tích phân phối của các đặc trưng, kiểm tra giá trị ngoại lai, và đánh giá mối tương quan giữa các biến độc lập với biến mục tiêu (`SalePrice`).
Sử dụng biểu đồ phân tán, histogram và ma trận heatmap để phát hiện đặc trưng ảnh hưởng mạnh nhất đến giá nhà, chẳng hạn như:
- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `TotalBsmtSF`

---

### 2️⃣ Xử lý giá trị khuyết (Missing Values)

Một phần quan trọng trong dự án là xử lý các giá trị `N/A`.
Nhóm đã áp dụng chiến lược sau:

- Loại bỏ các cột có tỷ lệ khuyết trên 50\%.
- Với biến số: thay thế bằng giá trị trung bình (mean imputation).
- Với biến phân loại: thay thế bằng nhãn `"None"`.

---

### 3️⃣ Kỹ thuật đặc trưng (Feature Engineering)

Nhóm đã:
- Sinh thêm các đặc trưng có ý nghĩa (như tổng diện tích sàn, tuổi nhà, diện tích gara).
- Áp dụng **One-Hot Encoding** để biến đổi dữ liệu phân loại.
- Chuẩn hóa dữ liệu bằng **StandardScaler** để giảm ảnh hưởng của đơn vị đo lường.
- Giảm chiều dữ liệu bằng **PCA (Principal Component Analysis)** nhằm loại bỏ nhiễu và tăng tốc độ huấn luyện.

---

### 4️⃣ Huấn luyện mô hình và đánh giá

Nhóm thử nghiệm nhiều mô hình hồi quy khác nhau:

| Mô hình | Đặc điểm chính | Điểm R² (Validation) |
|----------|----------------|----------------------|
| Linear Regression | Cơ bản, không regularization | 0.84 |
| Ridge Regression | Giảm overfitting bằng L2 penalty | 0.86 |
| Lasso Regression | Loại bỏ feature không quan trọng (L1 penalty) | 0.87 |
| ElasticNet | Kết hợp L1 và L2 | 0.88 |
| Random Forest | Ensemble, robust với nhiễu | 0.91 |
| XGBoost | Tối ưu gradient, hiệu năng cao | **0.93** |

Mô hình **XGBoost** đạt kết quả tốt nhất với sai số thấp nhất trên tập kiểm định.

---

### 5️⃣ Đánh giá và cải tiến

Các kỹ thuật đánh giá bao gồm:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **Cross-validation (k=5)**

Nhóm cũng quan sát ảnh hưởng của siêu tham số như `max_depth`, `n_estimators`, `learning_rate` đến hiệu quả mô hình, và tinh chỉnh chúng bằng **GridSearchCV**.

---

## 📊 Kết quả nổi bật

- RMSE trung bình trên tập test: **0.1265**
- Độ chính xác R2 đạt **93%** với mô hình XGBoost.
- Thời gian huấn luyện giảm đáng kể sau khi áp dụng PCA và chuẩn hóa dữ liệu.

Biểu đồ dưới đây minh họa sự khác biệt về hiệu suất giữa các mô hình:

{{< figure src="/Time-Series-Team-Hub/images/house_model_comparison.png" title="So sánh hiệu suất giữa các mô hình hồi quy" >}}

---

## 💡 Bài học rút ra

- Dữ liệu sạch và đặc trưng phù hợp ảnh hưởng lớn hơn cả mô hình.
- Feature Engineering tốt giúp mô hình tuyến tính đạt kết quả ngang bằng mô hình phi tuyến.
- XGBoost và Random Forest cho kết quả ổn định hơn khi dữ liệu có nhiễu.

---

## 📄 Tài liệu đính kèm

👉 [Tải bản PDF chi tiết tại đây]
{{< pdf src="/Time-Series-Team-Hub/pdf/M5W3D5_House_Price_Prediction.pdf" title="House Price Prediction" height="700px" >}}

---

## 🧠 Dành cho ai?

- Người học muốn luyện kỹ năng **xử lý dữ liệu thực tế**.
- Người muốn **nâng cấp kỹ năng mô hình hóa** với bài toán hồi quy.
- Người yêu thích việc **tối ưu mô hình và hiểu sâu về ảnh hưởng của đặc trưng**.

---

✍️ *Bài viết được thực hiện bởi nhóm Machine Learning - Time Series Team Hub, với mong muốn chia sẻ quy trình làm dự án ML thực tế từ góc nhìn sinh viên đến cộng đồng.*

🧩 _Repository:_ [Time Series Team Hub](https://github.com/Jennifer1907/Time-Series-Team-Hub)
