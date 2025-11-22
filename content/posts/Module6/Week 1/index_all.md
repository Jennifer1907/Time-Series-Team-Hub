---
title: "Module 6 - Tuần 1 - Bước chân đầu tiên vào thế giới Deep Learning "
date: 2025-06-28T13:03:07+07:00
description: Các Thước Đo Đánh Giá Mô Hình Hồi Quy 
image: images/M6W1.png
caption:
categories:
  - minutes
tags:
  
draft: false
---

# 📘 Study Guide – Module 6, Week 1  
**Chủ đề chính:** Logistic Regression & Loss / Metrics cho Regression  

Tuần này tập trung vào ba trục kiến thức chính:

1. Từ **Linear Regression → Logistic Regression** (mô hình hoá xác suất và phân loại).  
2. **Loss function** cho Logistic/Regression (MSE, BCE) và cách cài đặt bằng vector/matrix + NumPy.  
3. **Metrics cho Regression** và hướng nghiên cứu mở rộng (buổi thứ 7 – liên kết sang bài blog chi tiết).

---

## 🎯 Mục tiêu học tập

Sau tuần 1 của Module 6, học viên sẽ:

- Hiểu quy trình xây dựng mô hình **Logistic Regression**:
  - Từ trực giác của Linear Regression.
  - Cách chuyển sang bài toán phân loại khi label là category (0/1).
- Nắm được sự khác nhau giữa:
  - **Loss function** dùng để *train* mô hình (MSE, BCE).  
  - **Evaluation metrics** dùng để *đánh giá* sau khi train (MAE, RMSE, \(R^2\), v.v. – với regression).
- Biết sử dụng **vector/matrix** và **NumPy** để cài đặt Logistic Regression cho nhiều sample.
- Có cái nhìn ban đầu về **MLOps** với Airflow: lập lịch và quản lý các tác vụ cho pipeline AI.
- Hiểu bức tranh lớn về **metrics cho regression** và những thách thức nghiên cứu hiện tại.

---

## 📅 Lịch học chi tiết

### 1️⃣ Thứ 3 (04/11/2025) – Warm-up Logistic cơ bản  

**Buổi:** Warm-up  
**Instructor:** TA Quốc Thái  

**Nội dung:**

- Ôn/nhắc lại các bước trong quy trình **Logistic Regression**:
  - Chuẩn bị dữ liệu, chia train/test.
  - Xây dựng mô hình logistic cho bài toán nhị phân.
  - Tối ưu tham số bằng gradient descent.
- Làm **ví dụ tính tay đơn giản**:
  - Tính logit, sigmoid, và cập nhật tham số trong 1–2 bước gradient.
  - Giúp học viên “cảm” được mô hình, không chỉ nhìn code.

---

### 2️⃣ Thứ 4 (05/11/2025) – From Linear Regression to Logistic Regression  

**Buổi:** Học chính  
**Instructor:** Dr. Quang Vinh  

**Nội dung:**

- Thảo luận câu hỏi:  
  > “Khi nhãn là **category (0/1)** thì linear regression gặp vấn đề gì?”
- So sánh:
  - Linear Regression vs Logistic Regression về:
    - Miền giá trị dự đoán (ℝ vs (0,1)).
    - Ý nghĩa dự đoán (giá trị liên tục vs xác suất).
- Giải bài toán logistic với:
  - **Hàm loss MSE** (Mean Squared Error) – trực giác dễ hiểu nhưng không phù hợp.  
  - **Hàm loss BCE** (Binary Cross Entropy) – chuẩn hơn cho classification:
    - Liên quan đến maximum likelihood.
    - Gradient ổn định hơn, hội tụ tốt hơn.

---

### 3️⃣ Thứ 5 (06/11/2025) – MLOps với Airflow  

**Buổi:** MLOps  
**Instructor:** TA Dương Thuận  

**Nội dung:**

- Giới thiệu khái niệm **MLOps**:
  - Tự động hoá pipeline AI: ETL → Train → Evaluate → Deploy → Monitor.
- Làm quen với **Apache Airflow**:
  - DAG, task, scheduling.
- Thảo luận:
  - Cách lập lịch các tác vụ cho một chương trình AI đang chạy (training định kỳ, re-training, batch inference, báo cáo).
- Demo minh hoạ:
  - Xây một DAG nhỏ cho pipeline AI (ví dụ: tải dữ liệu → train logistic → log kết quả).

---

### 4️⃣ Thứ 6 (07/11/2025) – Advanced Logistic Regression  

**Buổi:** Học chính  
**Instructor:** Dr. Quang Vinh  

**Nội dung:**

- Mở rộng Logistic Regression lên **nhiều sample**:
  - Viết lại mô hình bằng **vector/matrix**, không còn tính từng điểm riêng lẻ.
- Cài đặt bằng **NumPy**:
  - Vector hoá forward pass: \( \hat{y} = \sigma(Xw + b) \).  
  - Vector hoá gradient, cập nhật tham số.
- Kết nối với thực hành:
  - Từ công thức tay → code Python → kiểm tra lại bằng ví dụ nhỏ.

---

### 5️⃣ Thứ 7 (08/11/2025) – Metric for Regression  

**Buổi:** Nâng cao  
**Instructor:** Dr. Đình Vinh  

**Nội dung:**

- Thảo luận các cách đo lường độ chính xác cho mô hình **regression**:
  - Sai số tuyệt đối (MAE), sai số bình phương (MSE, RMSE).
  - Hệ số tương quan \(R\), \(R^2\), Adjusted \(R^2\), VAF.
  - Các metric chuẩn hoá (MAPE, MASE, RMSSE, v.v.).
- Làm ví dụ:
  - Tính tay MAE, RMSE cho bộ dữ liệu nhỏ.  
  - Phân tích trường hợp có **outlier**: tại sao RMSE nhảy rất mạnh, MAE thì “hiền” hơn.
- Thảo luận các thách thức & hướng nghiên cứu:
  - Khi metric đánh lừa ta (benchmark quá tốt, dữ liệu nhiều số 0, dữ liệu đa chuỗi,…).  
  - Chọn metric nào cho đúng với mục tiêu business.

🔗 **Tài liệu đọc thêm / Pre-reading cho buổi thứ 7:**  
👉 [Bài viết: Các Thước Đo Đánh Giá Mô Hình Hồi Quy (Evaluation Metrics for Regression)](./evaluation-metrics-regression.md)  
*(thay đường dẫn trên bằng link thực tế trên website của bạn)*

---

### 6️⃣ Chủ nhật (09/11/2025) – Logistic Regression Exercise  

**Buổi:** Học chính (luyện tập)  
**Instructor:** TA Đình Thắng  

**Nội dung:**

- Ôn nhanh:
  - Nội dung buổi thứ 4: From Linear → Logistic, MSE vs BCE.  
  - Nội dung buổi thứ 6: Vector hoá Logistic Regression + NumPy.
- Giải bài tập:
  - Bài tập tính tay nhỏ để củng cố lý thuyết.  
  - Bài tập code (Python/NumPy) cho logistic nhiều sample.  
  - Thảo luận cách debug khi mô hình không hội tụ / accuracy thấp.

---

## 📚 Gợi ý học trước & sau mỗi buổi

- **Trước Thứ 4 & Thứ 6:**  
  - Ôn lại kiến thức linear regression, đạo hàm cơ bản, vector/matrix.
- **Trước Thứ 7:**  
  - Đọc lướt bài blog về metric cho regression:  
    [Các Thước Đo Đánh Giá Mô Hình Hồi Quy](./evaluation-metrics-regression.md)  
    (đặc biệt phần MAE vs RMSE, MAPE, MASE/RMSSE).
- **Sau Chủ nhật:**  
  - Tự cài lại logistic regression từ 0 (không dùng thư viện cao cấp),  
  - Thử thay loss function và metric đánh giá, xem mô hình thay đổi như thế nào.

---

## 🧠 Lời nhắn cuối tuần

Tuần này là cầu nối giữa:

- **Toán & trực giác** (logistic, loss, vector/matrix),  
- **Kỹ thuật triển khai** (MLOps với Airflow),  
- **Cách đánh giá mô hình** (metrics cho regression).

Đừng chỉ “chạy code cho ra kết quả”, hãy tự hỏi:

> *“Mô hình của mình đang tối ưu cái gì?  
> Và mình đang đánh giá nó bằng thước đo nào?”*

Nếu hai thứ này lệch nhau, mô hình có thể “đẹp trên giấy, xấu trong thực tế”.  
Tuần 1 – Module 6 là lúc chúng ta học cách tránh bẫy đó. 🚀

📂 _Tài liệu đi kèm:_
{{< pdf src="/Time-Series-Team-Hub/pdf/M6W1D5_Evaluation_Metrics.pdf" title="M6W1D5_Evaluation_Metrics" height="700px" >}}