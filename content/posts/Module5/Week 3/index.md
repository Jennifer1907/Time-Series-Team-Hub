---
title: "Module 5 - Tuần 3: Genetic Algorithms & XAI (SHAP)"
date: 2025-10-20T10:00:00+07:00
description: "Tuần 3 của Module 5 tập trung vào nền tảng của Giải thuật Di truyền (Genetic Algorithms), kết hợp MLOps AWS và giải thích mô hình qua SHAP trong XAI. Hướng dẫn cài đặt Python cơ bản, thảo luận randomness, và áp dụng GA trong bài toán tối ưu hóa & dự đoán thực tế."
image: images/GA.jpg
caption: Illustration by AI Vietnam Team
categories:
  - minutes
tags:
  - feature
draft: false
---

🎓 **All-in-One Course 2025 – aivietnam.edu.vn**  
📘 **Study Guide: Module 5 – Week 3**  
🧬 **Chủ đề:** Genetic Algorithms & Explainable AI (SHAP)

> 💡 *Tuần này, ta chính thức bước vào thế giới của các giải thuật tiến hóa — nơi quá trình “chọn lọc tự nhiên” được mô phỏng trong lập trình.  
> Bên cạnh đó, phần mở rộng XAI tuần này sẽ đi sâu vào SHAP — phương pháp giải thích mô hình bằng giá trị đóng góp từng feature.*

---

## 📅 **Lịch trình học và nội dung chính**

### 🧩 **Thứ 3 – Ngày 14/10/2025**
_(Buổi warm-up – TA Dương Thuận)_

**Chủ đề:** Genetic Algorithm cơ bản  
**Nội dung:**
- Thảo luận các bước trong giải thuật di truyền (khởi tạo quần thể, chọn lọc, lai ghép, đột biến, đánh giá).  
- Cài đặt bằng **Python cơ bản** (dùng List).  
- Thực hành minh họa tiến trình tiến hóa quần thể qua các thế hệ.

---

### ⚙️ **Thứ 4 – Ngày 15/10/2025**
_(Buổi học chính – Dr. Quang Vinh)_

**Chủ đề:** Randomness và Ứng dụng trong Tối ưu hóa  
**Nội dung:**
- Phân tích **quy luật và sự kiểm soát hệ thống ngẫu nhiên** trong tối ưu hóa.  
- Ứng dụng randomness trong bài toán tìm cực trị và tránh local minima.  
- Cài đặt bằng **List và Numpy**, so sánh các hàm ngẫu nhiên `random()`, `randint()`, `choice()`, `shuffle()` trong GA.

---

### ☁️ **Thứ 5 – Ngày 16/10/2025**
_(Buổi MLOps – TA Quang Tuấn)_

**Chủ đề:** AWS trong MLOps  
**Nội dung:**
- Tổng quan **MLOps dùng Cloud**.  
- Thảo luận ví dụ thực tế với **AWS EC2, S3, ECR và ECS**.  
- Giới thiệu cách container hóa quy trình huấn luyện GA và lưu trữ mô hình.

---

### 🧠 **Thứ 6 – Ngày 17/10/2025**
_(Buổi học chính – Dr. Đình Vinh)_

**Chủ đề:** Genetic Algorithms và Ứng dụng  
**Nội dung:**
- Phân tích sâu **các thành phần GA**: biểu diễn gene, fitness function, selection, crossover, mutation.  
- Ứng dụng GA cho các bài toán **tối ưu hóa** (tối đa hóa hàm, tìm tham số tốt nhất) và **dự đoán** (feature selection, hyperparameter tuning).  
- Minh họa quy trình “tiến hóa hội tụ” qua đồ thị Fitness vs Generation.

---

### 🔍 **Thứ 7 – Ngày 18/10/2025**
_(Buổi XAI – Dr. Đình Vinh)_

**Chủ đề:** Giải thuật SHAP trong Explainable AI  
**Nội dung:**
- Giới thiệu phương pháp **SHAP (SHapley Additive exPlanations)** trong XAI.  
- Phân tích cách SHAP giúp **giải thích dự đoán của mô hình** dựa trên giá trị đóng góp từng feature.  
- Làm ví dụ minh họa: SHAP cho mô hình Regression & Tree-based.  
- Hướng dẫn cài đặt `shap` library và hiển thị biểu đồ đóng góp.

---

### 🎯 **Chủ nhật – Ngày 19/10/2025**
_(Buổi ôn tập – TA Đình Thắng)_

**Chủ đề:** Genetic Algorithms – Exercise  
**Nội dung:**
- Ôn nhanh nội dung trọng tâm buổi thứ 4 và thứ 6.  
- Giải bài tập lập trình GA (các phép chọn lọc, lai ghép, đột biến).  
- Thảo luận chiến lược hội tụ và cải tiến hiệu năng GA.  

---

## 📌 **Điểm nhấn và kiến thức chính**

### ✅ **Genetic Algorithms – Tối ưu hóa qua tiến hóa**
- Cấu trúc cơ bản của GA:
  **Population → Selection → Crossover → Mutation → Next Generation**
- Hiểu rõ cách xây dựng **fitness function** và các cơ chế chọn lọc (roulette wheel, tournament, rank).  
- Ứng dụng trong tối ưu hóa tham số mô hình, bài toán tìm kiếm, và học máy.

### ✅ **Randomness & Control**
- Sự cân bằng giữa ngẫu nhiên và kiểm soát là chìa khóa trong tối ưu hóa tiến hóa.  
- Randomness giúp khám phá không gian nghiệm; deterministic giúp hội tụ ổn định.

### ✅ **MLOps với AWS**
- Tích hợp MLOps trên Cloud: EC2 (Compute), S3 (Storage), ECR/ECS (Container & Deployment).  
- Quản lý pipeline huấn luyện và deploy mô hình GA ở quy mô lớn.

### ✅ **Explainable AI – SHAP**
- Phân tích và giải thích kết quả mô hình bằng **giá trị Shapley**.  
- So sánh SHAP với LIME và ANCHOR: SHAP mang tính nhất quán và có nền tảng lý thuyết từ game theory.  
- Minh họa cách trực quan hóa SHAP summary plot và dependence plot.

---

## 📚 **Tài liệu đi kèm**

{{< pdf src="/Time-Series-Team-Hub/pdf/M5W3_GA.pdf" title="Genetic Algorithm - Giải thuật di truyền" height="700px" >}}
{{< pdf src="/Time-Series-Team-Hub/pdf/M5W3D3_MLOps_with_AWS.pdf" title="MLOPs với AWS" height="700px" >}}


---

🧠 _Repository managed by [AI Vietnam Team Hub](https://github.com/AI-Vietnam-Institution/All-in-One-Course)_  
📍 _Blog thuộc series **All-in-One Course 2025** – chương trình đào tạo toàn diện AI, Data Science, và MLOps tại [aivietnam.edu.vn](https://aivietnam.edu.vn)_
