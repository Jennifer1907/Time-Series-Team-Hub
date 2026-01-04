---
title: "Module 7: Hiểu và Xây dựng CNN từ Gốc"
date: 2025-11-21T10:00:00+07:00
description: "Module 7 đi sâu vào Mạng Nơ-ron Tích chập (CNN). Chúng ta sẽ đi từ những hạn chế của MLP truyền thống trong xử lý ảnh đến 'Góc nhìn Bộ lọc' của CNN, khám phá các cơ chế chính như Stride, Padding, Pooling, và các khái niệm nâng cao như Backpropagation và Tích chập 1x1."
image: images/CNN_Visual.png
caption: Minh họa bởi AI Vietnam Team
categories:
  - minutes
tags:
  - Deep Learning
  - Computer Vision
  - CNN
draft: false
---

🎓 **All-in-One Course 2025 – aivietnam.edu.vn**
📘 **Study Guide: Module 7**
🧩 **Chủ đề:** Hiểu và Xây dựng CNN từ Gốc và Trực giác

> 💡 *Tuần này, chúng ta sẽ giải mã "Ma thuật" của Thị giác Máy tính (Computer Vision). Chúng ta sẽ "mổ xẻ" Mạng Nơ-ron Tích chập (CNN) để hiểu cách nó "nhìn" thế giới thông qua các Bộ lọc (Filters), Bước nhảy (Strides), và Lề (Padding), vượt ra khỏi cách tiếp cận hộp đen của các thư viện Deep Learning thông thường.*

---

## 📅 **Nội dung chính**

Module này được thiết kế để xây dựng trực giác của bạn từ con số 0, bao gồm 5 phần chính:

1.  **Nút thắt cổ chai của MLP:** Tại sao các mạng Nơ-ron truyền thống thất bại khi xử lý hình ảnh (Vấn đề bùng nổ tham số).
2.  **Góc nhìn Bộ lọc (The Filter Perspective):** Hiểu về Kernels, Cơ chế khớp đặc trưng (Feature Matching), và toán học đằng sau phép Tích chập.
3.  **Kiến trúc CNN:** Cách các lớp (Conv, ReLU, Pooling) xếp chồng lên nhau để tạo thành một hệ thống phân cấp thị giác.
4.  **Backpropagation Chuyên sâu:** Toán học ẩn giấu đằng sau việc huấn luyện CNN, bao gồm yêu cầu bắt buộc về "Xoay Kernel" (Kernel Rotation).
5.  **Các khái niệm nâng cao:** Phân biệt CNN 2D vs 3D và mở khóa sức mạnh của Tích chập 1x1 trong việc giảm chiều dữ liệu.

---

## 📚 **Tài liệu đi kèm**

Dưới đây là tài liệu bài giảng chi tiết cho module này. Nó bao gồm nền tảng lý thuyết, các dẫn xuất toán học, và các giải thích trực quan về cơ chế hoạt động của CNN.

{{< pdf src="/Time-Series-Team-Hub/pdf/M7_Understand_and_BuildCNN_from_theGrounthUp.pdf" title="M7_Understand_and_BuildCNN_from_theGrounthUp" height="700px" >}}

---

🧠 *Repository managed by [AI Vietnam Team Hub](https://github.com/AI-Vietnam-Institution/All-in-One-Course)*
📍 *Blog thuộc series **All-in-One Course 2025** – Chương trình đào tạo toàn diện AI, Data Science, và MLOps tại [aivietnam.edu.vn](https://aivietnam.edu.vn)*