---
title: "Module 4 - Tuần 3: Tree Review & LightGBM – Cây cối nở rộ và nở hoa với Time-series"
date: 2025-09-23T10:00:00+07:00
description: "Tuần 3 là tuần 'cây cối nở rộ': chúng ta ôn tập toàn bộ họ nhà Tree từ Random Forest đến XGBoost, rồi học LightGBM và SHAP, kèm MLOps với Docker và project dự báo doanh số!"
image: images/tree_tree.png
categories:  
  - minutes  
tags:  
  
draft: false
---

🎉 **Chào mừng đến với blog Tuần 3 – Module 4 của AIO Course 2025!**

---

🌟 **Giới thiệu**

Nếu tuần trước chúng ta dừng lại ở Gradient Boosting & XGBoost, thì tuần này là một **hành trình tổng hợp**:  
- Ôn tập toàn bộ dòng họ **Tree-based models**.  
- Khám phá **LightGBM** – “người kế vị tốc độ” của Gradient Boosting.  
- Thực hành **MLOps cơ bản với Docker**.  
- Project thực tế: **Sales Forecasting với Explainable AI** (LightGBM + SHAP + Optuna + Streamlit).  

---

## 📅 Lịch trình tuần học

🗓️ **Thứ Ba – 16/09/2025**  
🔥 **Warm-up: Cơ bản về XGBoost** *(Extra class)*  
**Instructor:** TA Đình Thắng  
- Thảo luận tổng quát về giải thuật XGBoost.  
- Làm ví dụ tính tay đơn giản, sau đó cài đặt lại bằng Python.  

---

🗓️ **Thứ Tư – 17/09/2025**  
🌲 **Tree Review – Tổng hợp lý thuyết** *(Main session)*  
**Instructor:** Dr. Đình Vinh  
- Ôn tập toàn bộ các giải thuật dựa trên cây: Decision Tree, Random Forest, Gradient Boosting, XGBoost.  
- Thảo luận ứng dụng vào **bài toán time-series**.  
- So sánh ưu/nhược điểm từng mô hình khi dữ liệu thay đổi theo thời gian.  

---

🗓️ **Thứ Năm – 18/09/2025**  
⚙️ **Basic MLOps: Docker for ML**  
**Instructor:** TA Dương Thuận  
- Giới thiệu về Docker: container, image, registry.  
- Các bước đóng gói & triển khai mô hình ML trong Docker.  
- Thảo luận về Docker Compose, Docker Swarm và ứng dụng trong ML pipeline.  

---

🗓️ **Thứ Sáu – 19/09/2025**  
⚡ **LightGBM & SHAP** *(Main session)*  
**Instructor:** Dr. Quang Vinh  
- Nhược điểm của Gradient Boosting và cách LightGBM giải quyết bằng histogram-based, leaf-wise growth, GOSS, EFB.  
- LightGBM trong classification và regression.  
- Giới thiệu nhanh về **SHAP** để giải thích mô hình (model interpretability).  

---

🗓️ **Thứ Bảy – 20/09/2025**  
📊 **Project: Sales Forecasting with Explainable AI**  
**Instructor:** Dr. Thái Hà  
- Đề bài: Dự báo doanh số bán hàng (time-series).  
- Giải bài toán bằng LightGBM + SHAP.  
- Tối ưu tham số với **Optuna**.  
- Triển khai giao diện nhanh bằng **Streamlit**.  

---

🗓️ **Chủ Nhật – 21/09/2025**  
💪 **Exercise Session: Tree Review & LightGBM**  
**Instructor:** TA Quốc Thái  
- Ôn nhanh nội dung chính của buổi Tree Review (Thứ 4) và LightGBM (Thứ 6).  
- Giải các bài tập tổng hợp: từ tính tay entropy/gini → chạy code → đọc kết quả SHAP.  

---

## 🎯 Mục tiêu học tập

📌 **Tree Review**  
- Hệ thống lại các mô hình cây: Decision Tree, Random Forest, Gradient Boosting, XGBoost.  
- So sánh khả năng áp dụng trong time-series.  

📌 **LightGBM**  
- Hiểu rõ cải tiến của LightGBM: histogram-based, leaf-wise, GOSS, EFB.  
- Biết cách tính toán tại lá: \( w^* = -G / (H+\lambda) \).  
- Dùng SHAP để giải thích mô hình.  

📌 **MLOps với Docker**  
- Biết cách container hoá và triển khai ML models.  
- Hiểu vai trò của Compose, Swarm trong hệ thống ML.  

📌 **Project thực tế**  
- Tích hợp LightGBM, Optuna, SHAP và Streamlit.  
- Giải quyết bài toán dự báo doanh số – một ứng dụng phổ biến của AI trong kinh doanh.  

---

## 📂 Tài liệu kèm theo

({{< pdf src="/Time-Series-Team-Hub/pdf/M4W4D2D4_All_about_tree.pdf" title="M4W4D2D4_All_about_tree" height="700px" >}}) 
{{< pdf src="/Time-Series-Team-Hub/pdf/M4W3D3_DockerForML.pdf" title="M4W3D3_DockerForML" height="700px" >}}  
---

🧠 **_Repository được quản lý bởi [Time Series Team Hub](https://github.com/Jennifer1907/Time-Series-Team-Hub)_**
