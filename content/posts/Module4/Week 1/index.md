---
title: "Module 4 - Tuần 1: Random Forest • AdaBoost • Gradio demo • Tổng quan Time-series"
date: 2025-09-02T09:00:00+07:00
description: "Tuần mở màn của Module 4 tập trung vào hai thuật toán ensemble chủ lực (Random Forest, AdaBoost), thực hành dựng demo bằng Gradio (Basic MLOps), và mở rộng nền tảng về dữ liệu chuỗi thời gian – một tuần học thiên về thực hành, ứng dụng và kết nối giữa mô hình & triển khai."
image: assets/images/RF_AdaBoost.png
caption: "Minh họa dòng chảy học tập tuần: Random Forest → Gradio → AdaBoost → Time-series → Exercise."
categories:
  - minutes
tags:
  - random-forest
  - adaboost
  - boosting
  - ensemble-learning
  - gradio
  - mlops
  - time-series
  - module-4
  - week-1
draft: false
---

🎉 **Chào mừng đến với blog Tuần 1 của team Time Series – Module 4!**

🌟 **Giới thiệu**

Tuần này, chúng ta bước vào các kỹ thuật **ensemble learning** trọng yếu: **Random Forest** và **AdaBoost**. Bên cạnh phần thuật toán, bạn sẽ học cách **triển khai demo mô hình bằng Gradio** để trình bày và kiểm thử nhanh, đồng thời **củng cố nền tảng về dữ liệu time-series** nhằm định hướng mô hình và đánh giá đúng bản chất chuỗi thời gian. Kết thúc tuần là một phiên **bài tập tổng hợp** để bạn liên kết toàn bộ kiến thức đã học vào thực hành end-to-end. (Lịch & nội dung chi tiết dựa trên Study Guide tuần này. :contentReference[oaicite:0]{index=0})

---

📅 **Lịch trình tuần học**

🗓️ **Thứ Ba – 02/09/2025**  
🔥 **Cơ bản về Random Forest** *(Warm-up Session)*  
**Giảng viên:** Dr. Đinh Quang Vinh  
**Nội dung:** Ôn tập **Entropy** & **Gini**, thảo luận nguyên lý **Random Forest**, làm tay ví dụ đơn giản & cài đặt cơ bản.

---

🗓️ **Thứ Tư – 03/09/2025**  
🌲 **Random Forest (Main Session)**  
**Giảng viên:** Dr. Đình Vinh  
**Nội dung:** Thảo luận mở rộng về **Random Forest**; xử lý **dữ liệu thiếu/mất**, các chủ đề nâng cao, và các **ứng dụng trong data science**.

---

🗓️ **Thứ Năm – 04/09/2025**  
⚙️ **Basic MLOps: Gradio for ML model demos**  
**Giảng viên:** TA Dương Thuận  
**Nội dung:** Giới thiệu **Gradio** và cách dựng **demo mô hình ML** nhanh; thực hành qua **các ví dụ minh họa**.

---

🗓️ **Thứ Sáu – 05/09/2025**  
⚡ **AdaBoost (Main Session)**  
**Giảng viên:** Dr. Đình Vinh  
**Nội dung:** Tổng quan **boosting** & giải thuật **AdaBoost**; minh họa **ứng dụng cho dữ liệu time-series**.

---

🗓️ **Thứ Bảy – 06/09/2025**  
📈 **Time-series: Tổng quát & Ứng dụng**  
**Giảng viên:** PhD-c Nguyễn Khoa  
**Nội dung:** Đặc trưng **dữ liệu time-series**, **mô hình phù hợp**, **đánh giá hiệu quả** và các **hướng phát triển** trong mảng này.

---

🗓️ **Chủ Nhật – 07/09/2025**  
💪 **Exercise: Random Forest & AdaBoost**  
**Giảng viên:** TA Quốc Thái  
**Nội dung:** Ôn nhanh trọng tâm buổi **Thứ Tư** & **Thứ Sáu**; giải **bài tập tổng hợp**.

---

🎯 **Mục tiêu học tập**

🔍 **Random Forest**
- Nắm cách kết hợp **bagging** + **tập con thuộc tính** để giảm phương sai.  
- Xử lý **dữ liệu thiếu/mất** và hiểu tác động tới hiệu năng.  
- Thực hành cài đặt, phân tích **feature importance**, và đánh giá mô hình.

⚡ **AdaBoost**
- Hiểu cơ chế **trọng số mẫu** & **cập nhật weak learner**.  
- Áp dụng AdaBoost cho **bài toán thực tế** (bao gồm minh họa trên **time-series**).  
- So sánh điểm mạnh/yếu giữa **bagging** vs **boosting**.

🛠️ **Gradio (Basic MLOps)**
- Dựng **UI demo** cho mô hình ML để thử nghiệm & chia sẻ nhanh.  
- Tổ chức **inputs/outputs**, xử lý batch, và xuất bản demo nội bộ.

📊 **Time-series**
- Nhận diện **cấu trúc chuỗi thời gian** (trend/seasonality/autocorrelation).  
- Chọn **mô hình** & **chỉ số đánh giá** phù hợp.  
- Định hướng **pipeline** cho các bài toán dự báo/phân loại theo thời gian.

---

📂 **_Tài liệu đi kèm:_**
{{< pdf src="/Time-Series-Team-Hub/pdf/M4W1.pdf" title="M4W1 height="700px" >}}


---

🧠 **_Repository managed by [Time Series Team Hub](https://github.com/Jennifer1907/Time-Series-Team-Hub)_**

