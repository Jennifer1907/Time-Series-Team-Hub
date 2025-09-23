---
title: "Module 4 - Tuần 3: 🌳 Hành trình khám phá **Tree-Based Models**: Từ lý thuyết đến thực chiến"
date: 2025-09-23T10:00:00+07:00
description: "Tuần 3 là tuần 'cây cối nở rộ': chúng ta ôn tập toàn bộ họ nhà Tree từ Random Forest đến XGBoost, rồi học LightGBM!"
image: images/tree_tree.png
categories:  
  - minutes  
tags:  
  - feature
  
draft: false

---

✨ Đây không chỉ là một bài blog, mà là toàn bộ **tinh túy quá trình học tập và tổng hợp** của nhóm với các mô hình cây. Từ những bước tính tay đơn giản nhất cho một cây quyết định, đến việc so sánh benchmark hàng loạt mô hình hiện đại như **XGBoost** và **LightGBM**.  

---

## 🚀 Vì sao viết blog này?

Trong hành trình học Machine Learning, nhóm nhận ra:  
- Các mô hình cây (**Decision Tree, Random Forest, Gradient Boosting…**) là nền tảng cho rất nhiều hệ thống thực tế.  
- Nhưng khi học qua sách vở/tài liệu, thường bị rời rạc: công thức thì khô khan, code thì chạy “mặc định”, ít khi kết nối với **ý nghĩa thực sự**.  

👉 Blog này là nỗ lực của nhóm để **ghép nối tất cả**:  
- **Giải thích nguyên lý cặn kẽ** (Entropy, Gini, residual, similarity score…).  
- **Chứng minh công thức** bằng toán học (Taylor expansion, tối ưu convex).  
- **Ví dụ tính tay** từng bước trên tập dữ liệu toy nhỏ.  
- **Code Python** đối chiếu để kiểm chứng.  
- **Benchmark toàn diện** trên tập dữ liệu chuẩn (`breast_cancer`) để thấy hiệu quả thực tế.

---

## 📚 Nội dung chính

- 🌱 **Decision Tree**: nguyên lý, entropy vs gini, ví dụ tính tay, cắt tỉa cây.  
- 🌲 **Random Forest**: bootstrap, OOB error, minh họa bỏ phiếu.  
- 🔥 **AdaBoost**: trọng số mẫu, công thức alpha, bảng cập nhật từng vòng.  
- 📈 **Gradient Boosting**: residual, Taylor expansion, cập nhật từng bước.  
- ⚡ **XGBoost**: công thức lá tối ưu, similarity score, chứng minh vì sao xuất hiện.  
- 💡 **LightGBM**: histogram-based, leaf-wise growth, GOSS, EFB – và vì sao nó nhanh.  

Mỗi phần đều có:  
✔ Công thức toán học rõ ràng  
✔ Ví dụ tính tay minh họa  
✔ Code Python để chạy lại  
✔ Kết quả benchmark thực tế

---

## 🏆 Benchmark & Kết quả

Nhóm đã chạy thử trên **tập breast_cancer** của sklearn, với cross-validation nhiều lần.  
Kết quả (ROC-AUC, Accuracy, thời gian fit & predict) được so sánh giữa:  

- DecisionTree  
- RandomForest  
- ExtraTrees  
- GradientBoosting  
- HistGradientBoosting  
- AdaBoost (stump)  
- XGBoost  
- LightGBM  

📊 Biểu đồ ROC-AUC, Accuracy, thời gian training/predict đều có sẵn trong blog.  

---

## 🔑 Điểm đặc biệt của blog này

- **Không phải chỉ là lý thuyết** – mà là sự kết nối giữa toán học, trực giác, và code.  
- **Không chỉ copy từ sách vở** – mà là sản phẩm của quá trình học, tính tay, debug, thử sai rồi rút ra kết luận.  
- **Không chỉ một mô hình** – mà là toàn bộ hệ sinh thái tree-based, từ “cây thô sơ” đến “rừng tối tân”.

---

## 📄 Tài liệu chi tiết

Blog này đi kèm với một bản PDF chi tiết, trong đó:  
- Trình bày công thức đầy đủ, chứng minh chi tiết.  
- Có bảng “Tính tay vs Python” để đối chiếu từng bước.  
- Có code benchmark để bạn tự chạy lại.  

👉 [Tải bản PDF chi tiết tại đây]
({{< pdf src="/Time-Series-Team-Hub/pdf/M4W4D2D4_All_about_tree.pdf" title="M4W4D2D4_All_about_tree" height="700px" >}})

---

## 🎯 Dành cho ai?

- Người mới học ML muốn có cái nhìn **hệ thống, dễ hiểu**.  
- Người đã biết cơ bản nhưng muốn **đào sâu công thức**.  
- Người làm thực tế muốn biết **mô hình nào nhanh & chính xác nhất**.  

---

💡 Nếu bạn cũng từng chật vật với mớ công thức khô khan, hoặc hoang mang trước hàng chục thuật toán boosting khác nhau, nhóm tin blog này sẽ giúp bạn tiết kiệm rất nhiều thời gian – vì nó là tất cả những gì tụi mình đã gói gọn lại sau nhiều tháng học tập và thử nghiệm.

---

✍️ *Viết bởi nhóm gồm các thành viên đam mê dữ liệu, mong muốn biến hành trình học tập thành tài liệu hữu ích cho cộng đồng.*
