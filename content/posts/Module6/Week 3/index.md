---

title: "Module 6 - Tuần 3: Multilayer Perceptron & Metrics for Classification"
date: 2025-11-28T10:00:00+07:00
description: "Tuần 3 của Module 6 nâng cấp từ Softmax Regression lên Multilayer Perceptron (MLP), đi kèm khảo sát Activation, Initialization, Optimizer và các hệ metric dành cho bài toán phân loại. Sinh viên vừa học công thức forward/backward, vừa code PyTorch, đồng thời kết nối với thế giới MLOps qua Prometheus & Grafana."
image: images/MLP_Metrics.png
caption: Illustration by AI Vietnam Team
categories:
  - minutes
tags:

draft: false
---

🎓 **All-in-One Course 2025 – aivietnam.edu.vn**
📘 **Study Guide: Module 6 – Week 3** 
🧩 **Chủ đề:** Multilayer Perceptron & Metrics for Classification

> 💡 *Tuần này là bước chuyển từ “mô hình tuyến tính” sang “mạng nơ-ron sâu”: MLP, Activation, Initialization, Optimizer – và cách đo lường mô hình phân loại bằng các hệ metric khác nhau để chuẩn bị cho bài Loss ở các tuần sau.*

---

## 📅 **Lịch trình học và nội dung chính**

### 🧑‍🏫 **Thứ 3 – Ngày 18/11/2025**

*(Buổi warm-up – MSc. Quốc Thái)* 

**Chủ đề:** MLP cơ bản – từ Perceptron tới Multi-layer Perceptron
**Nội dung:**

* Nhắc lại Softmax/Logistic Regression, lý do phải **thêm hidden layer**.
* Thảo luận các bước trong **MLP pipeline**: chuẩn bị dữ liệu → chuẩn hóa → xây network → khởi tạo tham số.
* Làm 1 ví dụ tính tay đơn giản forward qua 1 hidden layer để hiểu:
  $$
  \mathbf{h} = \sigma(W_1 \mathbf{x} + \mathbf{b}_1), \quad
  \hat{\mathbf{y}} = \text{softmax}(W_2 \mathbf{h} + \mathbf{b}_2)
  $$

---

### 👨‍🏫 **Thứ 4 – Ngày 19/11/2025**

*(Buổi học chính – Dr. Quang Vinh)* 

**Chủ đề:** Xây dựng MLP – Forward & Backward, từ lý thuyết tới PyTorch
**Nội dung:**

* So sánh **Softmax Regression vs MLP**: khi nào tuyến tính là không đủ, vì sao cần non-linearity.
* Xây từng bước công thức **forward** và **backward** cho nhiều layer; giải thích vì sao backprop trông “rối” nhưng thực chất chỉ là chain rule áp đi áp lại.
* Thảo luận câu hỏi đau đầu khi design network:

  * Bao nhiêu **hidden layers**, mỗi layer bao nhiêu **neurons**?
  * Đặt **activation** ở đâu: `fc1 → act → fc2 → act → fc3` hay `fc1 → fc2 → act → fc3` – và vì sao “affine → non-linear → affine” mới thật sự tăng được năng lực biểu diễn.
  * Khi nào dùng `self.activation = nn.ReLU()` trong `__init__` vs dùng `F.relu()` trực tiếp trong `forward`.
* Cài đặt MLP bằng **PyTorch** cho tabular / image đơn giản; luyện thói quen đọc `model.parameters()` và xem kích thước từng layer.

---

### ⚙️ **Thứ 5 – Ngày 20/11/2025**

*(Buổi MLOps – TA Nguyễn Thuận)* 

**Chủ đề:** Prometheus & Grafana – Tracking & Logging cho hệ thống AI

**Nội dung:**

* Nhìn lại **MLFlow** tuần trước, kết nối sang hệ **monitoring/observability** bằng Prometheus & Grafana.
* Hiểu dòng chảy: model MLP huấn luyện → log metrics, loss, latency → Prometheus thu thập → Grafana vẽ dashboard.
* Demo pipeline nhỏ: track một MLP đang train (loss, accuracy, thời gian batch, GPU memory) và hiển thị trên Grafana.

---

### 🧠 **Thứ 6 – Ngày 21/11/2025**

*(Buổi học chính – Dr. Quang Vinh)* 

**Chủ đề:** Activation Functions, Initialization & Optimizer trong MLP

**Nội dung:**

* Khảo sát các **activation function** quan trọng:

  * ReLU, LeakyReLU, GELU, Sigmoid, Tanh…
  * Phân tích ưu/nhược: ReLU đơn giản, nhưng dễ bị **dying ReLU**; GELU mượt hơn, hợp với transformer-style MLP.
* Thảo luận chi tiết:

  * Khi nào nên thay ReLU bằng GELU / LeakyReLU?
  * Vì sao activation đặt **giữa các fully-connected layers** là mấu chốt để network không sụp thành một phép biến đổi tuyến tính duy nhất.
* **Initialization & Optimizer**:

  * He / Xavier initialization, và liên hệ với loại activation đang dùng.
  * Adam, SGD (with momentum): cách chúng di chuyển trong không gian tham số, ưu – nhược từng loại.
* Thực nghiệm nhanh: thay activation / initializer / optimizer trên cùng một MLP, xem ảnh hưởng tới tốc độ hội tụ và chất lượng nghiệm.

---

### 📊 **Thứ 7 – Ngày 22/11/2025**

*(Buổi chuyên đề – Dr. Đình Vinh)*

**Chủ đề:** Metrics cho Bài toán Phân loại (Binary, Multiclass, Multilabel)

**Nội dung:**

* Ôn nhanh khái niệm **Confusion Matrix** và các thành phần TP, TN, FP, FN. 
* Với **binary classification**:

  * Accuracy, Precision, Recall (TPR), Specificity (TNR), FPR, FNR, F1.
  * Kịch bản “bệnh hiếm” để thấy vì sao **accuracy** có thể “lừa mình dối người”. 
* Với **multiclass**:

  * Micro / Macro / Weighted Precision–Recall–F1, **Balanced Accuracy**, **Fβ-score** khi muốn ưu tiên Recall hơn Precision. 
* Với **multilabel**:

  * Exact Match Ratio, 0/1 Loss, **Hamming Loss**, multilabel Precision/Recall/F1. 
* Thảo luận: cách chọn **metric phù hợp với bài toán**, và vì sao nhiều metric **không differentiable**, nên ta dùng **loss function** như Cross-Entropy / Focal Loss làm “surrogate” để tối ưu.

---

### 👨‍🎓 **Chủ nhật – Ngày 23/11/2025**

*(Buổi ôn tập – TA Đình Thắng)* 

**Chủ đề:** MLP – Exercise & Mini Project

**Nội dung:**

* Ôn tập nhanh nội dung chính của buổi thứ 4 và thứ 6: từ forward/backward MLP đến chọn activation, init, optimizer.
* Giải bài tập code và giấy:

  * Tính gradient đơn giản cho một MLP 2-layer.
  * Sửa một đoạn PyTorch: chuyển từ dùng `F.relu` sang `self.activation`, thử thay ReLU bằng GELU.
* Kết nối với buổi metric: chạy một mô hình MLP nhỏ, log các metric **Accuracy, F1, Recall, Balanced Accuracy**, so sánh chúng với nhau.

---

## 📌 **Điểm nhấn và kiến thức chính**

### ✅ Từ Softmax Regression tới Multilayer Perceptron

* Softmax Regression là **mô hình tuyến tính** trên feature:\
  $$
  \hat{\mathbf{y}} = \text{softmax}(W\mathbf{x} + \mathbf{b})
  $$
  → không đủ để học các biên quyết định phi tuyến.
* MLP thêm **hidden layers + activation** để biểu diễn hàm phi tuyến phức tạp:
  $$
  \mathbf{h}^{(l)} = \sigma(W^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})
  $$
* Các câu hỏi thực tế khi design MLP:

  * Bao nhiêu layer là “vừa đẹp” cho bài toán hiện tại?
  * Layer nào nên mỏng, layer nào nên dày?
  * Dùng activation gì, đặt ở đâu để tránh dying ReLU nhưng vẫn train nhanh?

---

### ✅ Activation, Initialization & Optimizer – Bộ ba quyết định “tính cách” của MLP

* **ReLU**: nhanh, đơn giản nhưng có nguy cơ chết nơ-ron (dying ReLU).
* **LeakyReLU / GELU**: mềm hơn, giữ gradient tốt hơn ở vùng âm, phù hợp mạng sâu.
* **Initialization**:

  * He init: hợp với ReLU/variants.
  * Xavier: hợp với Tanh/Sigmoid.
* **Optimizer**:

  * SGD + Momentum: đơn giản, ổn định nhưng cần tuning cẩn thận.
  * Adam: tự điều chỉnh learning rate, rất được dùng cho MLP thực tế.

> Với cùng một kiến trúc, chỉ cần đổi activation + init + optimizer là **curve loss/metric** đã có thể khác hẳn – đây là chỗ rất đáng thử nghiệm trong tuần này.

---

### ✅ Backprop qua nhiều layer – Làm chủ “mặt tối” của MLP

* Backward MLP thực chất chỉ là **chain rule** chồng nhiều lớp:

  * Từ (\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}) → (\frac{\partial \mathcal{L}}{\partial W^{(L)}}), (\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L-1)}}) → lan ngược dần xuống input.
* Tuần này tập trung giải thích **trực giác**: mỗi layer học cách “biến đổi không gian feature” sao cho lớp cuối dễ phân tách bằng Softmax/Logistic.
* Đây cũng là chỗ dễ dẫn tới **vanishing/exploding gradient**, nên lựa chọn activation + init trở nên cực kỳ quan trọng.

---

### ✅ Metrics cho Classification & Liên hệ với Loss

* **Binary**: Accuracy, Precision, Recall, Specificity, F1, FPR, FNR.
* **Multiclass**: Micro/Macro/Weighted F1, Balanced Accuracy, Fβ, … – dùng khi dữ liệu **mất cân bằng** hoặc khi muốn coi trọng từng lớp khác nhau. 
* **Multilabel**: Exact Match Ratio (rất khó), 0/1 Loss, Hamming Loss, multilabel Precision/Recall/F1. 
* **Kết nối với Loss**:

  * Metric là “thước đo chất lượng” mà ta quan tâm (F1, Balanced Accuracy, …).
  * Loss là hàm **differentiable** để backprop (Cross-Entropy, BCEWithLogits, Focal Loss…).
  * Ta không backprop trực tiếp trên F1, nhưng sẽ chọn **loss phù hợp với metric mục tiêu** (ví dụ: class imbalance → dùng Weighted Cross-Entropy / Focal Loss thay vì MSE).

---

## 📚 **Tài liệu đi kèm**

* {{< pdf src="/Time-Series-Team-Hub/pdf/M06W03-StudyGuide.pdf" title="M06W03 – Study Guide" height="700px" >}}
* {{< pdf src="/Time-Series-Team-Hub/pdf/M06W03-MLP.pdf" title="M06W03 – Multilayer Perceptron Slides" height="700px" >}}
* {{< pdf src="/Time-Series-Team-Hub/pdf/M06W03-InsightIntoMLP.pdf" title="M06W03 – Insight into MLP" height="700px" >}}
* {{< pdf src="/Time-Series-Team-Hub/pdf/M06W03-MetricsForClassification.pdf" title="M06W03 – Metrics for Classification" height="700px" >}}

---

🧠 *Repository managed by [AI Vietnam Team Hub](https://github.com/AI-Vietnam-Institution/All-in-One-Course)*
📍 *Blog thuộc series **All-in-One Course 2025** – chương trình đào tạo toàn diện AI, Data Science, và MLOps tại [aivietnam.edu.vn](https://aivietnam.edu.vn)*
