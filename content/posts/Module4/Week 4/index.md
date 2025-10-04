---
title: "Module 4 - Tuần 4 - Tabular Data Project: Phân loại khả năng mắc bệnh tim dựa vào các triệu chứng"
date: 2025-09-06T10:00:00+07:00
description: Mô hình dự đoán bệnh tim mở rộng hướng kết hợp ensemble learning và tăng cường dataset bằng dữ liệu ảnh.
image: images/heart.jpg
caption:
categories:  
  - minutes  
tags:  
  - feature

draft: false
---

## 🧠 Module 4 - Tuần 4 — ❤️ Heart Disease Classifier – Time-Series Team

---

Chào mừng bạn đến với Heart Disease Classifier – dự án mở rộng từ nhóm Time-Series Team.
Hệ thống này tập trung vào chẩn đoán và phân loại bệnh tim dựa trên nhiều hướng tiếp cận tiên tiến trong Machine Learning & Deep Learning, đồng thời tích hợp Explainable AI (XAI) giúp giải thích quyết định mô hình.
---

## 🧪 Trải nghiệm Heart Disease Classifier tại đây
- **Slide giới thiệu sản phẩm**: [Canvas](https://www.canva.com/design/DAG0zM148Qg/hn0w-MEamlx4noLQBs0JtA/view?utm_content=DAG0zM148Qg&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h532a96a668)


### 🧪 Mã nguồn / Notebook
- **Google Colab**: [Colab-Pipeline 1](https://colab.research.google.com/drive/1b4kaHX-tU743qJOBEm71biyhZE2e3Q1z?usp=sharing)
- **Google Colab**: [Colab-Pipeline 2](https://colab.research.google.com/drive/1Sy2U8LzNYSV9NmpGZJb5d9Oqpo7ssiI-?usp=sharing)


### 👉 Đối tượng người dùng
- Nhà nghiên cứu AI Y tế: thử nghiệm mô hình đa phương thức trong chẩn đoán tim mạch.
- Sinh viên/Học viên: tài liệu tham khảo khi nghiên cứu Ensemble Learning & Fusion.
- Bác sĩ & bệnh viện: định hướng triển khai hệ thống gợi ý lâm sàng minh bạch.
---

## 🎯 Hai hướng mở rộng chính
1. **So sánh Ensemble Learning**: Stacking vs TSA
- Pipeline tổng quát: Chuẩn hoá dữ liệu Cleveland, huấn luyện nhiều mô hình cơ sở.
- Stacking Model: Kết hợp dự đoán của Random Forest, XGBoost, LightGBM, CatBoost → Logistic Regression meta-learner.
- Tunicate Swarm Algorithm (TSA): Thuật toán meta-heuristic lấy cảm hứng từ sinh học, tối ưu trọng số kết hợp mô hình.
- Triển khai & Kết quả: So sánh độ chính xác, ROC–AUC, và độ ổn định khi thay đổi dữ liệu.
- XAI: Dùng beeswarm plot và feature importance để minh hoạ vai trò của từng mô hình con trong ensemble.

2. **CardioFusion**: Ghép đa phương thức EchoNet + Cleveland
- Pipeline tổng quát: Hợp nhất dữ liệu tabular (Cleveland) và video tim (EchoNet).
- Mô hình thành phần:
    MLP cho dữ liệu bảng (Cleveland).
    CNN/ResNet50 cho dữ liệu hình ảnh tim (EchoNet).
- Fusion Model: Cơ chế kết hợp đặc trưng (feature-level fusion) → mô hình dự đoán chung.
- Triển khai & Kết quả: Fusion cho thấy cải thiện rõ rệt so với dùng từng nguồn dữ liệu riêng lẻ.

## 🔍 Giải thích & Ứng dụng
Explainable AI (XAI):
- Với ensemble: đánh giá tầm ảnh hưởng toàn cục của mỗi mô hình cơ sở.
- Với fusion: trực quan hóa trọng số đặc trưng tim mạch và chỉ số lâm sàng.

Ứng dụng tiềm năng:
- Hệ thống gợi ý lâm sàng sớm → cảnh báo nguy cơ bệnh tim.
- Cơ sở cho việc xây dựng API phòng ngừa bệnh tim, kết nối với hệ thống bệnh viện.
---

## 📊 Kết quả tiêu biểu
- **Pipeline 1**: Stacking + meta-model và stacking equal weight cho thấy khả năng tối ưu trọng số dự đoán tốt hơn các mô hình riêng lẻ.

- **Pipeline 2**: Sử dụng ResNet-50 để trích đặc trưng từ EchoNet và MLP để trích đặc trưng từ Cleveland. Hợp nhất (fusion) đặc trưng của CNN + MLP để dự đoán bệnh tim

---

## 🔍 Tính năng nổi bật

- **So sánh Ensemble Learning**: triển khai Stacking Model và TSA (Tunicate Swarm Algorithm) để tối ưu trọng số dự đoán.
- **Fusion đa phương thức**: kết hợp dữ liệu tabular (Cleveland) với hình ảnh tim (EchoNet) qua ResNet50 + MLP.
- **Explainable AI (XAI)**: trực quan hóa mức ảnh hưởng của từng mô hình con trong ensemble và mức đóng góp của từng đặc trưng trong fusion.
- **Dashboard trực quan**: biểu đồ ROC–AUC, Confusion Matrix, Beeswarm Plot và Feature Importance hỗ trợ phân tích kết quả.

---

## ♻️ Ưu điểm

- **Độ chính xác cao**: CardioFusion cho kết quả vượt trội so với dùng đơn nguồn dữ liệu.
- **Minh bạch**: XAI giúp hiểu vì sao hệ thống đưa ra quyết định → tăng độ tin cậy trong ứng dụng y tế.
- **Linh hoạt**: dễ mở rộng với nhiều thuật toán ensemble hoặc mô hình fusion khác.
- **Ứng dụng thực tiễn**: có thể phát triển thành API gợi ý lâm sàng hỗ trợ ngăn ngừa bệnh tim.

---

## 🛠️ Công nghệ sử dụng

| **Thành phần**            | **Công cụ**                                                                  |
|---------------------------|------------------------------------------------------------------------------|
| Dữ liệu                   | Cleveland Heart Disease, EchoNet Dataset                                     |
| Ensemble Models           | Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression              |
| Meta-Heuristic            | Tunicate Swarm Algorithm (TSA)                                               |
| Deep Learning             | ResNet50, CNN, MLP                                                           |
| Giải thích XAI            | SHAP (Beeswarm plot), Feature Importance                                     |
| Trực quan hoá             | Matplotlib, Seaborn, Plotly                                                  |
| Triển khai                | Notebook (Colab), LaTeX report, API (prototype)                              |

---

🗂️ Tài liệu đính kèm

{{< pdf src="/Time-Series-Team-Hub/pdf/M3W4D1_Abstract_Classifier.pdf" title="M3W4D1_Abstract_Classifier" height="700px" >}}
