---
title: "Module 6 - Tuần 1 - Đừng yêu một metric duy nhất"
date: 2025-06-28T13:03:07+07:00
description: Các Thước Đo Đánh Giá Mô Hình Hồi Quy 
image: images/metric.jpg
caption:
categories:
  - minutes
tags:
  - feature
draft: false
---

# Các Thước Đo Đánh Giá Mô Hình Hồi Quy  
## (Evaluation Metrics for Regression)

Bài viết/bài blog này đi kèm với file LaTeX:

> **“Các Thước Đo Đánh Giá Mô Hình Hồi Quy (Evaluation Metrics for Regression)”**

Dựa trên hai tài liệu chính của Plevris et al. và Muraina et al., bài viết hệ thống lại các thước đo cho bài toán hồi quy và nhấn mạnh một ý quan trọng:  
**Không có một con số nào đủ để “đánh giá mô hình” – luôn cần kết hợp nhiều metric để kể một câu chuyện đầy đủ và trung thực.**

---

## 🎯 Mục tiêu bài viết

- Giải thích **trực giác** đằng sau các thước đo lỗi trong hồi quy:  
  MB, MAE, MSE, RMSE, các dạng chuẩn hoá (MAPE/MNGE, NME, FGE, UI, IOA), \(R\), VAF, \(R^2\), Adjusted \(R^2\), MASE, RMSSE, v.v.
- Minh hoạ bằng **ví dụ tính tay** và **case study** (giá nhà, doanh số, outlier, benchmark) để thấy:
  - Khi nào metric hoạt động tốt.
  - Khi nào metric có thể đánh lừa chúng ta (dữ liệu gần 0, outlier, benchmark quá tốt, v.v.).
- Làm rõ sự khác nhau giữa:
  - **Loss function** dùng để *huấn luyện* mô hình.
  - **Evaluation metric** dùng để *đánh giá* và *báo cáo* cho business.
- Kết nối với **bài toán phân loại** (accuracy, precision, recall, F1) qua ví dụ ma trận nhầm lẫn “bác sĩ chẩn đoán mang thai”.

---

## 🧩 Nội dung chính

Bài viết được trình bày dạng kể chuyện (narrative), không liệt kê khô cứng, gồm các phần:

### 1. Giới thiệu

- Vì sao đánh giá mô hình quan trọng không kém việc chọn thuật toán.  
- Câu hỏi “mô hình sai bao nhiêu là chấp nhận được” trong bối cảnh thực tế  
  (dự báo doanh số, dự báo tải trọng, y tế, tài chính,…).

### 2. Thiết lập ký hiệu & khung bài toán

- \(r_i\): giá trị thực, \(p_i\): giá trị dự đoán, \(e_i = p_i - r_i\).  
- Bài viết tập trung vào hồi quy nhưng nhiều ý có thể chuyển sang classification.

### 3. Nhóm metric dựa trên sai số tuyệt đối & bình phương

- MB, MAE, MSE, RMSE.  
- Ví dụ giá nhà với 1–2 outlier:
  - Thể hiện điểm mạnh của MAE (ít bị kéo bởi outlier)  
  - Và điểm yếu: MAE có thể “làm mờ” các lỗi cực lớn nếu gói chung vào trung bình.
- So sánh MAE và RMSE:
  - Khi RMSE >> MAE → dấu hiệu đuôi sai số dày hoặc có outlier rất lớn.

### 4. Metric chuẩn hoá & những cái bẫy thường gặp

- MAPE / MNGE, NME, FB, FGE, UI, IOA.  
- Ví dụ sMAPE cho thấy:
  - Dự đoán 10 thay vì 1000 và 1000 thay vì 10 đều cho sMAPE ≈ 196%  
    → metric đối xứng về mặt toán học nhưng không phản ánh hết ý nghĩa thực tiễn.
- MRAE, GMRAE, RelMAE, RSE:
  - Ví dụ cụ thể cho thấy benchmark quá tốt hoặc xuất hiện outlier  
    có thể khiến metric bị “phóng đại” hoặc “dìm bớt” lỗi của mô hình.

### 5. \(R\), VAF, \(R^2\), Adjusted \(R^2\)

- Khi nào \(R^2\) hữu ích (hồi quy tuyến tính).  
- Khi nào \(R^2\) dễ gây hiểu nhầm (mô hình phi tuyến, neural network, tree,…).  
- VAF như một cách nhìn khác về tỷ lệ phương sai được giải thích, có thể âm nếu mô hình còn tệ hơn cả đoán bằng trung bình.

### 6. Taylor Diagram & so sánh nhiều mô hình

- Cách biểu đồ Taylor gom 3 thông tin: độ lệch chuẩn, hệ số tương quan, CRMSD vào một hình.  
- Gợi ý cách đọc: điểm mô hình càng gần điểm “REF” thì càng tốt.

### 7. MASE, RMSSE và bài toán nhiều chuỗi thời gian / nhiều sản phẩm

- Ví dụ chi tiết 2 sản phẩm:
  - Sản phẩm A bán vài chục đơn vị.  
  - Sản phẩm B bán vài nghìn đơn vị.  
  → MAE/RMSE thô không thể so sánh trực tiếp.
- MASE & RMSSE:
  - Chuẩn hoá lỗi so với mô hình naive (“hôm nay = hôm qua”).  
  - Cho phép so sánh, xếp hạng mô hình trên nhiều chuỗi với thang đo khác nhau.

### 8. Loss vs Metric: MSE, MAE, Huber loss

- Ví dụ 4 điểm “bình thường” + 1 outlier:
  - Train bằng MSE → mô hình học về *mean*,  
  - Trong khi MAE – metric đánh giá – lại gắn với *median*.  
  → Mô hình có thể “tốt theo loss” nhưng không tốt theo metric mà business dùng.
- Huber loss:
  - Nhỏ thì giống MSE (trơn, dễ tối ưu).  
  - Lớn thì giống MAE (robust với outlier).  
  - Là lựa chọn dung hoà giữa “dễ train” và “gần với metric thực tế”.

### 9. Liên hệ classification: ví dụ chẩn đoán mang thai

- Hình minh hoạ ma trận nhầm lẫn:  
  - True Positive / True Negative  
  - False Positive (Type I error – báo động giả)  
  - False Negative (Type II error – bỏ sót ca bệnh)
- Từ đó liên hệ:
  - Tại sao không thể chỉ nhìn accuracy.  
  - Vai trò của precision, recall, F1-score khi chi phí của FP và FN rất khác nhau.

### 10. Kết luận: Đừng yêu một metric duy nhất

- Trong classification: tối thiểu cần **accuracy + precision + recall + F1**.  
- Trong regression: tối thiểu cần **MAE + RMSE + \(R^2\)/VAF**, và nếu có nhiều series/thang đo thì thêm **MASE/RMSSE** hoặc metric chuẩn hoá khác.
- Metric là “ngôn ngữ” để kể câu chuyện về mô hình;  
  nếu chọn sai ngôn ngữ, câu chuyện sẽ bị méo hoặc thiếu.

---

📂 _Tài liệu đi kèm:_
{{< pdf src="/Time-Series-Team-Hub/pdf/M6W1D5_Evaluation_Metrics.pdf" title="M6W1D5_Evaluation_Metrics" height="700px" >}}


🧠 _Repository managed by [Time Series Team Hub](https://github.com/Jennifer1907/Time-Series-Team-Hub)_