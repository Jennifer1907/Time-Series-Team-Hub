---
title: "Module 5 - Tuần 4: Advanced Regression & AI Agent for Housing Appraisal"
date: 2025-10-28T10:00:00+07:00
description: "Tuần 4 của Module 5 tập trung vào ứng dụng nâng cao của Regression, kết hợp chọn đặc trưng thông minh (Correlation & F-statistics), tối ưu mô hình qua Ensemble, và mở rộng thành Agent AI hỗ trợ thẩm định giá nhà thực tế."
image: images/house_prediction.jpg
caption: Illustration by AI Vietnam Team
categories:
  - minutes
tags:
  - feature
draft: false
---
🎓 **All-in-One Course 2025 – aivietnam.edu.vn**  
📘 **Project: Module 5 – Week 4**  
🏠 **Chủ đề:** Advanced Regression Techniques & AI Agent for Housing Appraisal

> 💡 *Dự án này mở rộng từ mô hình dự đoán giá nhà cổ điển (House Prices - Kaggle), không chỉ dừng ở dự đoán giá mà còn tiến tới xây dựng Agent hỗ trợ phân tích & thẩm định giá bất động sản dựa trên mô hình ML và LLM.*

---

## 🎯 **Mục tiêu dự án**

- Ứng dụng các kỹ thuật **Advanced Regression** để dự đoán giá nhà với độ chính xác cao.  
- Thực hành **chọn đặc trưng (Feature Selection)** bằng tương quan và F-statistics.  
- Xây dựng và so sánh nhiều **pipeline mô hình** (Scaled / Raw / PCA / GA).  
- Kết hợp **Ensemble Learning** để cải thiện hiệu năng dự đoán.  
- Mở rộng ứng dụng mô hình thành **AI Agent for Housing Appraisal** — trợ lý thẩm định giá tự động dựa trên Machine Learning + LLM.

---

## ⚙️ **Pipeline tổng quan**

![Project Pipeline](Project_Module5_Pipeline.png)

**Các giai đoạn chính:**
1. Tiền xử lý dữ liệu (xử lý giá trị thiếu, mã hóa biến phân loại, chuẩn hóa số liệu).  
2. Chọn đặc trưng dựa trên **tương quan (Correlation)** và **F-statistics (SelectKBest)**.  
3. Tạo nhiều pipeline với các chiến lược khác nhau:
   - `GA_base_scaled`  
   - `GA_base_raw`  
   - `GA_pca`  
   - `Stats_base_scaled`  
   - `Stats_base_raw`  
   - `Stats_pca`
4. Huấn luyện mô hình (Linear, Ridge, Lasso, ElasticNet, RandomForest, Gradient Boosting, Ensemble).  
5. So sánh hiệu năng qua RMSE và $R^2$.  
6. Tạo **AI Appraisal Agent** giúp phân tích, so sánh, và sinh báo cáo giá trị căn nhà bằng ngôn ngữ tự nhiên.

---

## 🧩 **Feature Selection**

### 🔹 1. Correlation-based Selection

```python
correlation = numeric_df.corr()["SalePrice"].abs().sort_values(ascending=False)
selected_numeric_stats = correlation[correlation >= 0.1].index.tolist()
selected_numeric_stats.remove("SalePrice")
```

- Chỉ giữ các biến có hệ số tương quan ≥ 0.1 so với `SalePrice`.  
- Giúp giảm nhiễu và tăng khả năng giải thích mô hình.

---

### 🔹 2. F-statistics Selection (ANOVA)

```python
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=min(15, len(categorical_cols)))
X_cat_selected = selector.fit_transform(X_cat, y)
selected_cat_mask = selector.get_support()
selected_categorical_stats = [
    categorical_cols[i] for i in range(len(categorical_cols)) if selected_cat_mask[i]
]
```

- Dựa trên mức độ ảnh hưởng của từng biến phân loại đến biến mục tiêu.  
- Giúp chọn top-k đặc trưng có ý nghĩa thống kê cao nhất.  
- Kết quả được kết hợp vào danh sách `selected_features_stats`.

---

## 🤖 **Huấn luyện và so sánh mô hình**

### 🔸 Kết quả tổng hợp

| Pipeline | Model | RMSE | R² |
|-----------|--------|------|------|
| Stats_base_raw | **Ensemble** | **0.135** | **0.902** |
| Stats_base_raw | Gradient Boosting | 0.136 | 0.901 |
| Stats_base_scaled | Linear Regression | 0.139 | 0.896 |
| Stats_base_scaled | Ridge | 0.144 | 0.888 |
| Stats_base_raw | Random Forest | 0.145 | 0.888 |
| Stats_base_scaled | ElasticNet | 0.145 | 0.887 |
| GA_base_raw | Ensemble | 0.145 | 0.887 |
| GA_pca | Gradient Boosting | 0.248 | 0.670 |
| Stats_pca | Random Forest | 0.259 | 0.640 |

🔥 **Best Model:**  
📊 Pipeline: `Stats_base_raw`  
🤖 Model: `Ensemble`  
📈 R²: 0.9018  

> PCA tỏ ra không hiệu quả trong bài toán này do mất thông tin định danh (categorical encoding).  
> Ensemble kết hợp nhiều mô hình nền (Linear, Ridge, Lasso, RandomForest, GBM) giúp ổn định kết quả và giảm phương sai.

---

## 🧠 **Agent for Housing Appraisal**

Sau khi huấn luyện mô hình tốt nhất, nhóm mở rộng ứng dụng thành **AI Agent** có khả năng:

- Trích xuất đặc trưng của căn nhà (`extract_property_features`).  
- Tìm bất động sản tương đồng theo trọng số diện tích, khu phố, chất lượng, tuổi nhà (`find_comparable_properties_advanced`).  
- Sinh prompt cho LLM để tạo báo cáo thẩm định (`generate_comprehensive_analysis_prompt`, `create_property_report_prompt`).  
- Kết hợp dự đoán của mô hình với khả năng diễn giải của LLM → báo cáo “thị trường hóa”.

---

### 📄 **Kết quả Prompt mẫu**

```
You are a real estate valuation assistant.
The target property (Index 50) is located in Gilbert.
It has 3 bedrooms, 2 garage(s), 1470 square feet of living area, built in 1997, with an overall quality rating of 6.

Actual sale price in dataset: $177,000.
Predicted price from ML model: $175,451.
The prediction differs by -0.9% from the actual sale price.

The dataset's mean sale price is $180,921, ranging from $34,900 to $755,000.

Here are the top 5 comparable properties:
- The first comparable: Located in Gilbert, built in 1997, 1511 sqft, 3 bedrooms, quality 6/10, 2 garage(s), selling for $185,000 (similarity score: 0.992).
- The second comparable: Located in Gilbert, built in 1994, 1481 sqft, 3 bedrooms, quality 6/10, 2 garage(s), selling for $174,000 (similarity score: 0.989).
- The third comparable: Located in Gilbert, built in 1995, 1498 sqft, 3 bedrooms, quality 6/10, 2 garage(s), selling for $187,500 (similarity score: 0.988).
- The fourth comparable: Located in Gilbert, built in 1993, 1470 sqft, 3 bedrooms, quality 6/10, 2 garage(s), selling for $185,000 (similarity score: 0.988).
- The fifth comparable: Located in Gilbert, built in 1993, 1501 sqft, 3 bedrooms, quality 6/10, 2 garage(s), selling for $165,600 (similarity score: 0.982).

Based on these comparables and model estimates, analyze whether the predicted value is reasonable. Explain which features likely contributed most to the price difference.
```

---

## 🧩 **Kiến thức chính**

### ✅ Advanced Regression
- Ứng dụng Ridge, Lasso, ElasticNet, Gradient Boosting, và Ensemble.  
- So sánh ảnh hưởng của scaling, PCA và feature selection.  

### ✅ Feature Engineering & Selection
- Kết hợp Correlation và F-statistics để chọn biến hiệu quả.  
- Tránh overfitting và tăng khả năng giải thích mô hình.

### ✅ Model Evaluation
- RMSE đo sai số tuyệt đối trung bình.  
- $R^2$ đánh giá mức độ giải thích biến mục tiêu.  
- Ensemble giúp mô hình ổn định hơn và tổng quát tốt hơn.

### ✅ AI Integration
- Sử dụng kết quả ML làm đầu vào cho LLM để sinh báo cáo tự động.  
- Mô hình ML + LLM có thể tái sử dụng trong các hệ thống định giá thực tế (real-estate valuation assistant).

---

## 📚 **Tài liệu kèm theo**

{{< pdf src="/Time-Series-Team-Hub/pdf/M5W4_Housing_Price.pdf" title="M5W4_Housing_Price.pdf" height="700px" >}}  

---

🧠 _Repository managed by [AI Vietnam Team Hub](https://github.com/AI-Vietnam-Institution/All-in-One-Course)_  
📍 _Blog thuộc series **All-in-One Course 2025** – chương trình đào tạo toàn diện AI, Data Science, và MLOps tại [aivietnam.edu.vn](https://aivietnam.edu.vn)_