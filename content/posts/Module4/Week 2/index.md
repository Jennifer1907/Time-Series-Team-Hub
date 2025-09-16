---
title: "Module 4 - Tuần 2: Tree Series Tiếp Tục – Gradient Boosting, XGBoost & FastAPI Deployment"
date: 2025-09-14T10:00:00+07:00
description: "Tuần này chúng ta tiếp nối series Tree-based models với Gradient Boosting và XGBoost, học từ ví dụ tính tay đến code thật, kèm bonus 1 ngày FastAPI để đưa mô hình lên web!"
image: images/Boosting.png
categories:
  - minutes
tags:
  - gradient-boosting
  - xgboost
  - fastapi
  - timeseries
draft: false

---

🎉 **Chào mừng đến với blog Tuần 2 của team Time Series – Module 4!**

🌟 **Giới thiệu**

Sau khi “làm bạn” với **Random Forest** và **AdaBoost** ở tuần trước, tuần này series Tree-based models tiếp tục với hai cao thủ đình đám: **Gradient Boosting** và **XGBoost**.  

Điểm đặc biệt của tuần này:  
- **Gradient Boosting**: đi từ ví dụ tính tay từng bước để hiểu rõ cơ chế “fit residuals”, rồi code lại với scikit-learn, kèm phần **Behind the Scenes** (giải thích nội bộ: residual = negative gradient, leaf value từ minimization).  
- **XGBoost**: đào sâu **Behind the Scenes** hơn nữa — công thức Taylor bậc 2, similarity score, regularization — và áp dụng cho regression, classification, time series.  
- **FastAPI**: một ngày “MLOps nhẹ nhàng” để học cách biến model thành web API với CRUD, response model, và mini case study “Fashion Detection App”.  

---

📅 **Lịch trình tuần học**

🗓️ **Thứ Ba - 09/09/2025**  
🔍 **Warm-up: Gradient Boosting cơ bản** *(Extra Class)*  
**Giảng viên:** TA Đình Thắng  
- Ôn lại decision tree, supervised learning.  
- Làm **ví dụ tính tay step-by-step** cho regression (MSE) và classification (logistic loss).  
- Thấy rõ từng bước: khởi tạo F0 (mean/log-odds), tính pseudo-residuals, chọn split, tính leaf value, update model.  

🗓️ **Thứ Tư - 10/09/2025**  
🧠 **Gradient Boosting – Lý thuyết & ứng dụng** *(Main Session)*  
**Giảng viên:** Dr. Đình Vinh  
- Gradient Boosting cho regression & classification: loss, gradient, leaf output.  
- Phần **Behind the Scenes**: tại sao residual = negative gradient, tại sao dùng Taylor approximation để xấp xỉ loss.  
- Ứng dụng vào **time series forecasting**: walk-forward CV, rolling vs expanding window, đánh giá bằng MAE/MSE/MAPE.  

🗓️ **Thứ Năm - 11/09/2025**  
⚡ **Triển khai ML model với FastAPI** *(Basic MLOps)*  
**Giảng viên:** TA Đăng Nhã  
- Từ **khái niệm API** (request/response, client-server) → **ASGI & Uvicorn**.  
- Hiểu async/await để code concurrent.  
- **CRUD với FastAPI**: routing, Pydantic models, status codes, `/docs` & `/redoc`.  
- **Case study**: Fashion Detection App – dựng endpoint cho mô hình object detection, gửi ảnh → trả bounding boxes.  

🗓️ **Thứ Sáu - 12/09/2025**  
🚀 **XGBoost (1)** *(Main Session)*  
**Giảng viên:** Dr. Đình Vinh  
- XGBoost cho regression & classification.  
- **Behind the Scenes**: công thức Taylor bậc 2 (gradient + Hessian), similarity score để chọn split, leaf output.  
- Regularization: λ (L2), γ (min split loss), shrinkage (learning rate).  
- Handling missing values, ứng dụng cho time series.  

🗓️ **Thứ Bảy - 13/09/2025**  
📊 **Project Time-series: PG&E Energy Analytics Challenge**  
**Giảng viên:** PhD-c Võ Nguyên  
- Tìm hiểu data năng lượng, đặc trưng seasonal/winter-summer.  
- Thử nghiệm Decision Tree, Random Forest, XGBoost.  
- So sánh ưu/nhược điểm từng cách, chọn chiến lược tốt cho dự báo.  

🗓️ **Chủ Nhật - 14/09/2025**  
💪 **Exercise Session: Gradient Boosting & XGBoost**  
**Giảng viên:** TA Quốc Thái  
- Ôn tập trọng tâm tuần.  
- Làm bài tập tổng hợp regression, classification, time series.  
- Củng cố hiểu biết qua hands-on coding.  

---

🎯 **Mục tiêu học tập**

📌 **Gradient Boosting**  
- Hiểu cơ chế boosting stage-wise: mỗi cây fit pseudo-residuals.  
- Làm ví dụ tay cho regression (MSE) & classification (logistic).  
- Biết cách kiểm soát overfit bằng learning rate, depth nhỏ, subsample.  
- Hiểu **Behind the Scenes**: residual = negative gradient, leaf value từ minimization/Taylor.  
- Áp dụng cho time series với walk-forward CV.  

📌 **XGBoost**  
- Hiểu công thức Taylor bậc 2 (gradient + Hessian).  
- Biết cách tính **Similarity Score**, leaf value.  
- Nắm vai trò regularization (λ, γ).  
- Xử lý missing values và ứng dụng cho dữ liệu time series.  

📌 **FastAPI Deployment**  
- Hiểu khái niệm API, ASGI, async/await.  
- Viết CRUD endpoints, dùng Pydantic để validate input/output.  
- Biết dựng response model, status code chuẩn.  
- Triển khai mini-app cho mô hình ML.  

---

📂 **_Tài liệu đi kèm:_**  
{{< pdf src="/Time-Series-Team-Hub/pdf/M4W2D1+2_GradientBoosting.pdf" title="M4W2D1+2_GradientBoosting" height="700px" >}}  
{{< pdf src="/Time-Series-Team-Hub/pdf/M4W2D3_WebDeploymentUsingFastAPI.pdf" title="M4W2D3_WebDeploymentUsingFastAPI" height="700px" >}}  
{{< pdf src="/Time-Series-Team-Hub/pdf/M4W2D4_XGBoost.pdf" title="M4W2D4_XGBoost" height="700px" >}} 

---

🧠 **_Repository managed by [Time Series Team Hub](https://github.com/Jennifer1907/Time-Series-Team-Hub)_**
