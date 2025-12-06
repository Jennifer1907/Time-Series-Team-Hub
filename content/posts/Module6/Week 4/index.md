---
title: "Module 6 - Tuần 4: FPT Forecasting Challenge"
date: 2025-12-06T10:00:00+07:00
description: "Dự báo giá cổ phiếu FPT 100 ngày bằng mô hình Hybrid Linear + ML + Regime-aware Pricing"
image: images/FPT.png
caption: Illustration by AI Vietnam Team
categories:
  - minutes
tags:
  - feature
draft: false
---
### 🚀 Trải nghiệm ứng dụng ngay hôm nay
- **Dùng thử ngay - Bảng Chuyên nghiệp**: **[Streamlit](https://aio-timeseries-fpt-stock-prediction.streamlit.app/)** 
- **Dùng thử ngay - Bảng Beta**: **[Streamlit](https://fptstock.streamlit.app/)** 
- **Video demo - Bảng Chuyên nghiệp:**: **[Video](https://drive.google.com/file/d/13w62NgvbqopSAVusEDSrZ5RnRHh8CPUS/view)**
- **Video demo - Bảng Beta:**: **[Video](https://us05web.zoom.us/clips/share/zvX8aKz4R0i8WfVX48Z_aQ)**

Ứng dụng Streamlit này cho phép bạn:
- 🔮 Dự báo giá FPT 100 ngày tiếp theo
- 📊 Theo dõi đồ thị xu hướng, đường trung vị, biên độ bất định (uncertainty)
- 🧠 Hiểu cách mô hình học và phản ứng với từng giai đoạn thị trường
- ⚙️ Tùy chỉnh Pricing Layer theo các trạng thái Bull / Bear / Sideways

Được thiết kế đặc biệt cho thị trường chứng khoán Việt Nam, nơi biến động ngắn hạn mạnh nhưng xu hướng dài hạn của FPT luôn bền vữn

### 🚀 Slide giới AI chuyên gia tài chính
- - **Slide**: **[Canvas](https://www.canva.com/design/DAG6wb_GkSk/GGwWp5HRd_JerCEb5ScXIw/view?utm_content=DAG6wb_GkSk&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h50e134ab07)**

---
###  🚀 LTSF-Linear — FPT Forecasting Challenge
#### **Hybrid Trend + XGBoost Residual + Regime-aware Pricing**

#### 🎯 Mục tiêu dự án
Xây dựng mô hình **dự báo giá đóng cửa FPT 100 ngày tương lai (T+100)**
chỉ từ một file duy nhất: `FPT_train.csv`.

Điểm đặc biệt của thử thách này:

- Horizon dài (**100 ngày liên tục**)
- Dữ liệu ít (4.5 năm)
- Biến động regime mạnh của thị trường Việt Nam
- Baseline Linear gần như *mất toàn bộ phương sai* khi forecast nhiều bước

Dự án này đề xuất một pipeline 3 lớp để giải quyết vấn đề theo cách **ổn định, có thể giải thích và bền vững theo thị trường**:

1. **Math Backbone** – mô hình hóa *động học giá dài hạn* bằng xu hướng log-price
2. **XGBoost Residual** – “bắt” phần cấu trúc phi tuyến còn lại mà Linear không thể học
3. **Pricing Layer (Regime-aware)** – điều tiết dự báo theo Bull/Bear/Sideways + mean reversion

Toàn bộ thiết kế hướng đến mục tiêu:

> **Không cố tiên tri giá từng ngày, mà dựng nên một trajectory hợp lý, có vật lý, có kinh tế và có ràng buộc.**
---

### 🧪 Source Code
🔗 **Google Colab Notebook**
https://drive.google.com/file/d/1i1CL8pMqbykRZiGpC6qojPCeSSwLGOVA/view?usp=sharing

---

### Problem Overview — *Tại sao baseline Linear thất bại?*

Hầu hết mô hình LTSF (Linear / NLinear / DLinear) khi dự báo cuốn-chiếu 100 bước đều tạo ra:

👉 Một **đường thẳng hoàn hảo**
👉 Dao động gần như bằng 0
👉 Không phản ánh biến động thực tế của thị trường

Hiện tượng này trong time series gọi là:

> **“Death of Variance” — Cái chết của phương sai**

Nguyên nhân:

- Log-price vốn đã mượt → thêm Linear → càng mượt
- NLinear chuẩn hóa theo giá cuối → xoá luôn nhiễu
- Recursive forecasting → nhiễu nhỏ bị triệt tiêu sau mỗi bước
- Horizon dài → sự bất định hội tụ về đúng một slope

Điều này khiến mô hình **không còn tính thị trường**, không có bull/bear, không có shock, không có volatility.

---
### 🚀 Đặc trưng của cổ phiếu FPT:

- Là bluechip đầu ngành công nghệ
- Xu hướng **tăng trưởng dài hạn (strong secular uptrend)**
- Rất nhạy theo từng **regime thị trường**:
  - 2020–2021: Bull mạnh
  - Cuối 2022: Điều chỉnh sâu
  - 2023–2024: Sideways rộng + hồi phục
- Volume có tính chu kỳ theo quý và theo sóng ngành IT

### 🔎 Hạn chế của dataset:

- Chỉ có OHLCV hằng ngày (không có macro, news sentiment, ETF flow…)
- Chỉ ~1150 mẫu, nhưng yêu cầu dự báo tới 100 ngày
- Biến động thị trường Việt Nam đôi lúc phi tuyến mạnh (gap, trần/sàn)

Chính vì vậy mô hình cần:

✔ Khả năng mô hình hóa trend
✔ Khả năng nắm bắt residual phi tuyến
✔ Cơ chế “vật lý” để giữ cho đường dự báo hợp lý

---

### Pipeline giải pháp: Hybrid 3 lớp

Dưới đây là cấu trúc của pipeline dự báo:

(1) Math Backbone (Trend) --> (2) XGBoost Residual Model --> (3) Regime-aware Pricing Layer --> (4) Forecast Path (Base → Central → Uncertainty)

---

#### **3.1 Math Backbone — Chuyên gia Trend**

Backbone đơn giản nhưng cực kỳ quan trọng:

- Fit Linear Regression vào log-price
- Dự báo quỹ đạo dài hạn
- Loại bỏ nhiễu ngắn hạn
- Tạo một baseline mà ML có thể học residual

Backbone được tách thành một **Expert độc lập trong Ensemble** để:

- Neo dự báo tránh đi quá xa khỏi xu hướng vĩ mô
- Giảm variance khi ML lỡ “quá sáng tạo”
- Tạo Fail-safe khi dữ liệu out-of-sample

---


#### **3.2 XGBoost Residual — Bắt nhiễu phi tuyến**

Thay vì dự báo giá trực tiếp, XGB học: residual = future_return – math_return

Ưu điểm:
- Học tốt tương tác phi tuyến (OHLCV, volume patterns…)
- Không bị drift dài hạn vì backbone đã lo phần slope
- Giữ được volatility thực tế của thị trường

Kết quả cross-validation nhiều cutoff (2021–2024) cho thấy:
- Train MAE ~ 0.003–0.005
- Test MAE ~ 0.008–0.02
- Residual std ~ 0.007–0.013
- Tức nhiễu thị trường FPT thường nằm trong 0.7%–1.3% log-return

Residual std này được chuyển cho Monte Carlo để tạo *uncertainty band*.

---

#### **3.3 Pricing Layer — Regime-aware + Mean Reversion**

Đây là trái tim của pipeline, giúp dự báo:

- Không bay quá cao khi bull
- Không lao quá mạnh khi bear
- Giữ sự hợp lý theo vật lý thị trường Việt Nam

Các thành phần chính:

##### 1️⃣ Clipping
Giới hạn biên độ return mỗi ngày để tránh dự báo “điên rồ”

##### 2️⃣ Damping
Biến động càng xa hiện tại → càng giảm (nhiễu không lan vô hạn)

##### 3️⃣ Mean Reversion
Giá luôn có điểm cân bằng (fair level) mà nó dao động quanh

##### 4️⃣ Regime-aware
Tham số thay đổi theo 3 chế độ:

- **Bull:** cho phép upside lớn hơn
- **Bear:** tăng lực hồi khi rơi sâu
- **Sideways:** ổn định, ít điều chỉnh

Pricing layer được tối ưu bằng **Random Search + Time-based Cross-validation** trên nhiều cutoff thực tế.

---

#### Final Result — FPT 100-day Forecast

![FPT Forecast](images/FPT_forecast.png)

Dự báo cuối cùng được xây dựng từ 3 đường:

- **BASE**: Hybrid + Pricing
- **TREND**: Tuyến tính dài hạn
- **CENTRAL_DET**: 0.7 * BASE + 0.25 * TREND + 0.05 * Risk_center

Và dải bất định Monte Carlo:

- **Uncertainty upper/lower band** thể hiện rủi ro thị trường
- Bề rộng band phản ánh độ bất ổn tăng dần theo thời gian

#### Điểm nổi bật:

✔ Forecast không phẳng như Linear
✔ Có volatility hợp lý
✔ Không overshoot không cần thiết
✔ Phản ánh đúng trạng thái thị trường hiện tại (SIDEWAYS)

---

#### 5. Reliability — Độ tin cậy của mô hình

Dự án cung cấp nhiều lớp bảo vệ rủi ro:

- ✔ Backbone neo quỹ đạo dài hạn
- ✔ XGBoost Residual duy trì nhiễu có cấu trúc
- ✔ Regime-aware Pricing kiểm soát overshoot
- ✔ Monte Carlo mô phỏng sự bất định

Sự kết hợp này tạo thành một mô hình:
> **Ổn định – Có thể giải thích – Bền vững theo nhiều pha thị trường**

Band bất định rộng dần theo horizon, cho thấy:

- Thị trường càng xa hiện tại càng khó dự đoán
- Nhưng mô hình vẫn giữ một mean trajectory rất logic
- Không bị lệch trend như đa số mô hình thuần ML

---

#### 📚 Tài liệu đính kèm

{{< pdf src="/Time-Series-Team-Hub/pdf/M6W4D1+6_Project_Module.pdf" title="M6W4D1+6_Project_Module" height="700px" >}}
