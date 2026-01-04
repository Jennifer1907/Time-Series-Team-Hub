---
title: "Module 6 - Tuần 2: Softmax Regression & MLFlow for Model Tracking"
date: 2025-11-21T10:00:00+07:00
description: "Tuần 2 của Module 6 đi sâu vào Softmax Regression cho bài toán phân loại đa lớp, cùng với ứng dụng thực tế trong MLOps bằng MLFlow. Sinh viên được thực hành tính toán, lập trình với Numpy và PyTorch, và tìm hiểu cách quản lý mô hình bằng MLFlow."
image: images/Softmax_MLFlow.png
caption: Illustration by AI Vietnam Team
categories:
  - minutes
tags:
 
draft: false
---

🎓 **All-in-One Course 2025 – aivietnam.edu.vn**  
📘 **Study Guide: Module 6 – Week 2**  
🧩 **Chủ đề:** Softmax Regression & MLFlow for Model Tracking and Versioning

> 💡 *Tuần này làm quen với Softmax Regression – một mô hình then chốt trong phân loại đa lớp – đồng thời tìm hiểu về cách theo dõi và quản lý mô hình Machine Learning qua MLFlow.*

---

## 📅 **Lịch trình học và nội dung chính**

### 🧑‍🏫 **Thứ 3 – Ngày 11/11/2025**
_(Buổi warm-up – TA Đăng Nhã)_

**Chủ đề:** Softmax cơ bản  
**Nội dung:**
- Thảo luận các bước trong **Softmax Regression**.  
- Làm ví dụ tính tay từng bước để hiểu cơ chế tính xác suất và logit.  
- Chuẩn bị cho phần lập trình bằng Numpy trong buổi kế tiếp.

---

### 👨‍🏫 **Thứ 4 – Ngày 12/11/2025**
_(Buổi học chính – Dr. Quang Vinh)_

**Chủ đề:** Softmax Regression cho Phân loại đa lớp  
**Nội dung:**
- Giới thiệu bài toán **multi-class classification**.  
- Phân tích công thức **forward** và **backward propagation** trong Softmax.  
- Cài đặt hoàn chỉnh bằng **Numpy** để hiểu rõ cơ chế học và gradient.  

---

### ⚙️ **Thứ 5 – Ngày 13/11/2025**
_(Buổi MLOps – TA Nguyễn Thuận)_

**Chủ đề:** MLFlow cho Model Tracking và Versioning  
**Nội dung:**
- Giới thiệu tổng quan về **MLFlow** trong hệ thống MLOps.  
- Cách theo dõi thí nghiệm, log siêu tham số và quản lý phiên bản mô hình.  
- Demo thực hành nhỏ về **Model Tracking & Versioning**.  

---

### 🧠 **Thứ 6 – Ngày 14/11/2025**
_(Buổi học chính – Dr. Quang Vinh)_

**Chủ đề:** PyTorch Framework – Cài đặt Regression và Classification  
**Nội dung:**
- Giới thiệu cơ bản về **PyTorch Framework**.  
- Cài đặt các mô hình **Linear Regression**, **Logistic Regression** và **Softmax Regression** bằng PyTorch.  
- Thảo luận về cách huấn luyện, tính loss và cập nhật tham số tự động.  

---

### 🔬 **Thứ 7 – Ngày 15/11/2025**
_(Buổi nâng cao – Dr. Đình Vinh)_

**Chủ đề:** Loss Functions cho Bài toán Phân loại  
**Nội dung:**
- Thảo luận chi tiết các loại **Loss Function**: Cross-Entropy, Negative Log-Likelihood, Focal Loss, v.v.  
- So sánh các hàm mất mát trong bối cảnh **class imbalance** và **multi-label**.  
- Thảo luận hướng nghiên cứu mở rộng và những thách thức thực tế.

---

### 👨‍🎓 **Chủ nhật – Ngày 16/11/2025**
_(Buổi ôn tập – MSc. Quốc Thái)_

**Chủ đề:** Softmax Regression – Exercise  
**Nội dung:**
- Ôn tập nhanh nội dung trọng tâm của buổi thứ 4 và thứ 6.  
- Giải bài tập tính toán gradient, loss, và dự đoán xác suất.  
- Củng cố kỹ năng lập trình Softmax Regression.  

---

## 📌 **Điểm nhấn và kiến thức chính**

### ✅ **Softmax Regression – Phân loại đa lớp**
- Biểu diễn xác suất cho từng lớp:
  $$
  P(y_i = k \mid \mathbf{x}_i) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}_i}}{\sum_j e^{\mathbf{w}_j^\top \mathbf{x}_i}}
  $$
- Mô hình tổng quát cho **multi-class classification**.  
- Kết nối trực tiếp với **Logistic Regression** và **Cross-Entropy Loss**.  
- Phân tích **gradient descent** trong không gian nhiều chiều.

---

### ✅ **PyTorch – Triển khai hiện đại**
- Sử dụng `torch.nn.Module` để tạo mô hình Regression.  
- Sử dụng `torch.optim` cho cập nhật tham số tự động.  
- Hiểu cơ chế **autograd** trong việc lan truyền gradient.

---

### ✅ **MLOps với MLFlow**
- Theo dõi và ghi lại toàn bộ quá trình huấn luyện (metrics, parameters, model files).  
- So sánh các phiên bản mô hình qua UI hoặc CLI.  
- Lưu trữ mô hình và phục hồi để **deploy hoặc fine-tune**.

---

### ✅ **Loss Functions nâng cao**
- Phân biệt các loại loss trong phân loại:
  - `CrossEntropyLoss`
  - `KLDivLoss`
  - `Focal Loss`
- Ứng dụng từng loại loss cho bài toán có dữ liệu mất cân bằng.

---

## 📚 **Tài liệu đi kèm**

{{< pdf src="/Time-Series-Team-Hub/pdf/M6W2D2+3_Softmax.pdf" title="M6W2D_Softmax" height="700px" >}}

{{< pdf src="/Time-Series-Team-Hub/pdf/M6W2D4+6_Prometheus_Grafana.pdf" title="M6W2D4+6_Prometheus_Grafana_MLOPs" height="700px" >}}


{{< pdf src="/Time-Series-Team-Hub/pdf/M6W2D7_LossFuncForClassify.pdf" title="M6W2D7_LossFuncForClassify" height="700px" >}}


---

🧠 _Repository managed by [AI Vietnam Team Hub](https://github.com/AI-Vietnam-Institution/All-in-One-Course)_  
📍 _Blog thuộc series **All-in-One Course 2025** – chương trình đào tạo toàn diện AI, Data Science, và MLOps tại [aivietnam.edu.vn](https://aivietnam.edu.vn)_
