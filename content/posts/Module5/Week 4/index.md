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

> 💡 *Dự án này mở rộng từ bài toán kinh điển “House Prices Prediction (Kaggle)” và hướng đến một pipeline hiện đại hơn – nơi Machine Learning, Explainable AI, và Large Language Model kết hợp để xây dựng một hệ thống thẩm định giá nhà tự động (AI Appraisal Agent).*

---
## 🧪 Trải nghiệm Agent tại đây: 
**Dùng thử ngay**: **[Streamlit](https://housingpriceai.streamlit.app/)** 

## 🧪 File Source Code: 
[Google_Colab] (https://colab.research.google.com/drive/1a1ap0Th2R9K8CzVXttu4E6JTsnPjYx-g?usp=sharing)

## 🎯 **Mục tiêu dự án**

- Ứng dụng các kỹ thuật **Advanced Regression** để dự đoán giá nhà với độ chính xác cao.  
- Kết hợp **phân tích thống kê truyền thống** (Correlation, F-statistics) và **tối ưu hóa tiến hóa** (Genetic Algorithm).  
- Thiết kế và huấn luyện các **pipeline đa dạng** (Raw, Scaled, PCA, GA).  
- Áp dụng **Ensemble Learning** để nâng cao độ ổn định và khả năng tổng quát hóa.  
- Mở rộng mô hình thành **AI Agent for Housing Appraisal** có khả năng sinh báo cáo thẩm định giá bằng ngôn ngữ tự nhiên.

---

## ⚙️ **Cải tiến trong pipeline mới**

Pipeline mới được nhóm đề xuất theo định hướng **kết hợp giữa khả năng tự động hóa, phân tích đa chiều và giải thích mô hình**.  
Các cải tiến chính bao gồm:

### 🔹 (a) Chọn đặc trưng đa hướng  
- Kết hợp **Genetic Algorithm (GA)** để tìm tập biến tối ưu với **Correlation / F-test** nhằm đánh giá thống kê truyền thống.  
- Mục tiêu là tạo ra tập đặc trưng vừa có ý nghĩa thống kê, vừa tối ưu theo góc nhìn tìm kiếm toàn cục (global search).  

### 🔹 (b) Phân nhánh huấn luyện có và không có PCA  
- Tạo hai pipeline song song để **so sánh tác động của giảm chiều dữ liệu** (dimensionality reduction) đến hiệu quả mô hình và khả năng diễn giải.  
- Kết quả cho thấy PCA giúp giảm nhiễu nhưng làm mất tính giải thích đối với các biến định danh (categorical).

### 🔹 (c) Mở rộng phạm vi mô hình  
- Bổ sung và thử nghiệm các thuật toán nâng cao: **ElasticNet, Random Forest, Gradient Boosting, Ensemble Learning.**  
- Ensemble được chọn là mô hình mạnh nhất, giúp giảm phương sai và tận dụng sức mạnh của nhiều bộ dự đoán cơ sở.

### 🔹 (d) Giải thích bằng SHAP  
- Áp dụng **SHAP (SHapley Additive exPlanations)** để lượng hóa mức đóng góp của từng đặc trưng vào dự đoán cuối cùng.  
- Phương pháp này tăng tính minh bạch và hỗ trợ phân tích nguyên nhân định lượng của từng quyết định mô hình.

### 🔹 (e) Hướng đến ứng dụng thực tế  
- Tích hợp bước **Prompting – AI Agent for Housing Appraisal**, cho phép mô hình tạo báo cáo định giá và giải thích chi tiết từng yếu tố ảnh hưởng.  
- Đây là bước đệm để triển khai hệ thống **AI hỗ trợ thẩm định giá bất động sản tự động** trong tương lai.

---

## 📊 **Pipeline tổng quan**

![Project Pipeline](Project_Module5_Pipeline.png)

**Các giai đoạn chính:**
1. Tiền xử lý dữ liệu (xử lý giá trị thiếu, mã hóa, chuẩn hóa).  
2. Chọn đặc trưng bằng **Correlation / F-statistics** và **GA optimization**.  
3. Phân nhánh pipeline (Raw / Scaled / PCA / GA).  
4. Huấn luyện mô hình (Linear, Ridge, Lasso, ElasticNet, RF, GBM, Ensemble).  
5. Đánh giá hiệu năng qua RMSE và R².  
6. Tạo **AI Agent** để phân tích và sinh báo cáo định giá nhà tự động.

---

## 🧩 **Feature Selection**

### 🔹 Correlation-based Selection
```python
correlation = numeric_df.corr()["SalePrice"].abs().sort_values(ascending=False)
selected_numeric_stats = correlation[correlation >= 0.1].index.tolist()
selected_numeric_stats.remove("SalePrice")
```
- Chỉ giữ các biến có hệ số tương quan ≥ 0.1 so với `SalePrice`.  
- Giúp loại bỏ nhiễu và tăng khả năng giải thích mô hình dựa trên thống kê truyền thống.

### 🔹 F-statistics Selection (ANOVA)
```python
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=min(15, len(categorical_cols)))
X_cat_selected = selector.fit_transform(X_cat, y)
selected_cat_mask = selector.get_support()
selected_categorical_stats = [
    categorical_cols[i] for i in range(len(categorical_cols)) if selected_cat_mask[i]
]
```
- Lựa chọn các đặc trưng có ý nghĩa thống kê cao nhất thông qua kiểm định F-test.  
- Đây là hướng **chọn đặc trưng truyền thống** giúp mô hình dễ giải thích, ổn định và có cơ sở thống kê rõ ràng.

---
### 🔹 Genertic Algorithm (GA) Feature Selection

Ngoài các phương pháp thống kê, nhóm còn phát triển một pipeline riêng sử dụng **Genetic Algorithm (GA)** để tự động hóa việc lựa chọn tập đặc trưng tối ưu.

GA hoạt động dựa trên cơ chế tiến hóa tự nhiên — chọn lọc, lai ghép, và đột biến — nhằm tìm ra tập biến mang lại hiệu quả dự đoán cao nhất cho mô hình.

**Quy trình chính:**
1. Mỗi “cá thể” trong quần thể đại diện cho một tập biến.  
2. Tính “độ thích nghi” (fitness) dựa trên hiệu năng mô hình (RMSE hoặc R²).  
3. Chọn lọc các cá thể tốt nhất, thực hiện lai ghép và đột biến để tạo thế hệ mới.  
4. Lặp lại đến khi hội tụ hoặc đạt ngưỡng cải thiện tối đa.

**Ưu điểm của GA:**
- Tự động tìm ra tập biến tối ưu mà không cần giả định tuyến tính.  
- Phù hợp với các bài toán có tương tác phức tạp giữa biến độc lập.  
- Bổ sung cho các phương pháp truyền thống, mở ra pipeline riêng (`GA_base_raw`, `GA_base_scaled`, `GA_pca`) giúp so sánh tính hiệu quả.

---

## 🤖 **Kết quả huấn luyện và so sánh nhóm 10 mô hình có thành tích tốt nhất**

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

> PCA không phù hợp cho bài toán này do làm mất thông tin định danh (categorical).  
> Ensemble giúp mô hình ổn định và khái quát hóa tốt hơn so với từng mô hình riêng lẻ.

---

## 🧠 **AI Agent for Housing Appraisal**

Sau khi xác định mô hình tốt nhất, nhóm mở rộng ứng dụng thành **AI Agent** có khả năng:

- Trích xuất đặc trưng chi tiết của từng căn nhà (`extract_property_features`).  
- Tìm kiếm bất động sản tương đồng bằng hàm tương tự có trọng số (`find_comparable_properties_advanced`).  
- Sinh báo cáo định giá chi tiết qua LLM (`generate_comprehensive_analysis_prompt`, `create_property_report_prompt`).  
- Kết hợp ML + LLM để tạo **báo cáo thẩm định tự động** như chuyên viên bất động sản.

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
## 📚 **Tài liệu kèm theo**

{{< pdf src="/Time-Series-Team-Hub/pdf/M5W4_Housing_Price.pdf" title="M5W4_Housing_Price.pdf" height="700px" >}}  

---

🧠 _Repository managed by [AI Vietnam Team Hub](https://github.com/AI-Vietnam-Institution/All-in-One-Course)_  
📍 _Blog thuộc series **All-in-One Course 2025** – chương trình đào tạo toàn diện AI, Data Science, và MLOps tại [aivietnam.edu.vn](https://aivietnam.edu.vn)_