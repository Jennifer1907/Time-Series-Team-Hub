---
title: "Module 5 - Tuần 1 + 2: Chinh phục Linear Regression từ cơ bản tới nâng cao & Hủy Diệt Data Conflicts cùng MLOps Versioning và Feast"
date: 2025-10-13T10:00:00+07:00
description: "Tuần 2 của Module 5 đào sâu vào Advanced Linear Regression — từ vector hóa công thức đến triển khai dự án ML thực chiến với Feast. Blog này tập trung vào Linear Regression và MLOps, trong khi XAI (LIME–ANCHOR–SHAP) sẽ được tổng hợp trong một bài blog đặc biệt sắp tới!"
image: images/Advanced_Linear_FeatureStore.jpeg
caption: Illustration by AI Vietnam Team
categories:
  - minutes
tags:
  - feature
draft: false
---

🎓 **All-in-One Course 2025 – aivietnam.edu.vn**
📘 **Study Guide: Module 5 – Week 1 + 2**
🧩 **Chủ đề:** Chinh phục Linear Regression từ Cơ Bản tới Nâng Cao & Hủy Diệt Data Conflicts cùng MLOps Versioning và Feast

> 🧠 *Lưu ý:* Blog này chỉ bao gồm **Linear Regression từ Cơ Bản tới Nâng Cao** và **Triển khai, quản lý dự án ML với Data Versioning và Feast**.
> Toàn bộ phần **XAI (LIME – ANCHOR – SHAP)** sẽ được **tổng hợp, mở rộng và xuất bản trong bài blog đặc biệt sắp tới:
> “All-in-One XAI Series 2025 – Giải thích mô hình Machine Learning từ cơ bản đến nâng cao.”*

---

## 📅 **Lịch trình học và nội dung chính**

### 🧑‍🏫 **Thứ 3 – Ngày 07/10/2025**

_(Buổi warm-up – TA Đình Thắng)_

**Chủ đề:** Linear Regression dùng Vector/Matrix và Numpy
- Mô tả công thức hồi quy tuyến tính bằng vector và ma trận.
- Cài đặt bài toán Linear Regression cơ bản với **Numpy**.
- Trực quan hóa mối quan hệ giữa dữ liệu và mô hình tuyến tính.

---

### 👨‍🏫 **Thứ 4 – Ngày 08/10/2025**

_(Buổi học chính – Dr. Quang Vinh)_

**Chủ đề:** Loss Functions cho Linear Regression
- So sánh **MSE (Mean Squared Error)**, **MAE (Mean Absolute Error)** và **Huber Loss**.
- Thảo luận về **tính lồi (convexity)** trong tối ưu hóa và gradient descent.
- Ứng dụng đạo hàm và cập nhật tham số mô hình với learning rate.

---

### ⚙️ **Thứ 5 Tuần 1 – Ngày 02/10/2025**

_(MLOps Session – TA Dương Thuận)_

**Chủ đề:** Kiểm soát phiên bản dữ liệu cho dự án ML/AI
- Tổng quan về AI, MLOps và Data Versioning
- 3 Thách thức chính trong Quản lý Dữ liệu và Code
- Giới thiệu về Data Version Control (DVC) qua so sánh DvC và Git
- Case Study: Triển khai DVC cho Dataset MNIST
- Tự động hóa Pipelines và Các khái niệm Versioning

### ⚙️ **Thứ 5 Tuần 2 – Ngày 09/10/2025**

_(MLOps Session – TA Dương Thuận)_

**Chủ đề:** Triển khai và Quản lý Dự án ML với Feast
- Giới thiệu **Feature Store** trong MLOps – nơi quản lý các feature phục vụ mô hình học máy.
- Nguyên tắc đảm bảo **nhất quán giữa training và serving (online–offline consistency)**.
- Phiên bản hóa feature, truy cập thời gian thực và tích hợp pipeline với Feast.
- Thực hành mini demo: từ dữ liệu thô → feature → model → deployment pipeline.

---

### 👨‍🏫 **Thứ 6 – Ngày 10/10/2025**

_(Buổi học chính – Dr. Quang Vinh)_

**Chủ đề:** Vectorized Linear Regression
- Biểu diễn công thức hồi quy cho 1, m và N samples bằng vector/matrix.
- Tối ưu hóa phép tính bằng vectorization để tăng tốc độ huấn luyện.
- Cài đặt hoàn chỉnh bằng **Numpy** và phân tích hiệu năng.

---

### 🔬 **Thứ 7 – Ngày 11/10/2025**

_(XAI Session – Dr. Đình Vinh)_

**Chủ đề:** Giải thuật ANCHOR trong XAI *(preview – sẽ có trong blog riêng)*
- Giới thiệu khái niệm **Explai­nable AI (XAI)** và ý tưởng của **ANCHOR** trong việc giải thích mô hình theo điều kiện “if–then”.
- Các ví dụ cài đặt minh họa sẽ được **tổng hợp trong blog “All-in-One XAI Series 2025”**.

---

### 👨‍🎓 **Chủ nhật – Ngày 12/10/2025**

_(Buổi ôn tập – TA Quốc Thái)_

**Chủ đề:** Advanced Linear Regression – Exercise
- Ôn tập nội dung của buổi thứ 4 và thứ 6.
- Giải bài tập về vectorization, loss function và gradient descent.
- Thảo luận cách mở rộng Linear Regression thành Ridge/Lasso Regression.

---

## 📌 **Điểm nhấn và kiến thức chính**

### ✅ **Linear Regression – Nền tảng mở rộng**

- Hiểu rõ bản chất ma trận trong hồi quy tuyến tính:
$$
\hat{y} = X\beta + \varepsilon
$$
- So sánh và ứng dụng MSE, MAE, Huber trong thực tế.
- Áp dụng vectorization để tăng hiệu quả huấn luyện.
- Xử lý Colinearity trong Hồi quy tuyến tính bội (Multiple Linear Regression)
- Áp dụng regulization để tránh tình trạng overfitting

---

### ✅ **MLOps với Feast – Quản lý Feature trong ML**

- Làm quen với khái niệm **Feature Store** và cách Feast hỗ trợ version control, lineage và serving.
- Triển khai pipeline từ dữ liệu thô đến mô hình.
- Đảm bảo tính nhất quán dữ liệu và khả năng mở rộng khi deploy ML model.

---

### 🧩 **XAI (LIME – ANCHOR – SHAP) – Giới thiệu**

> 🔜 *Phần XAI sẽ được giới thiệu chi tiết trong blog riêng sắp tới:
> “All-in-One XAI Series 2025: LIME – ANCHOR – SHAP – Giải thích mô hình ML toàn diện.”*
>
> Bài viết đó sẽ bao gồm:
> - Giải thích trực quan từng phương pháp.
> - Cài đặt minh họa bằng Python (Sklearn, SHAP, LIME).
> - So sánh độ ổn định, độ tin cậy, và trade-off giữa các phương pháp.

---

## 📚 **Tài liệu đi kèm**

{{< pdf src="/Time-Series-Team-Hub/pdf/M5W1W2_LinearRegression.pdf" title="Advanced Linear Regression" height="700px" >}}

{{< pdf src="/Time-Series-Team-Hub/pdf/M5W1W2_MLOps_Data_Versioning.pdf" title="MLOps with Feast" height="700px" >}}

{{< pdf src="/Time-Series-Team-Hub/pdf/M5W1W2_MLOps_with_Feast.pdf" title="MLOps with Feast" height="700px" >}}
---

🧠 _Repository managed by [AI Vietnam Team Hub](https://github.com/AI-Vietnam-Institution/All-in-One-Course)_
📍 _Blog thuộc series **All-in-One Course 2025** – chương trình đào tạo toàn diện AI, Data Science, và MLOps tại [aivietnam.edu.vn](https://aivietnam.edu.vn)_
