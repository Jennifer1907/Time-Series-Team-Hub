---
title: "Module 6 - Tuần 4: FPT Forecasting Challenge  "
date: 2025-12-06T10:00:00+07:00
description: "Tuần 4 của Module 6 bài toán dự đoán giá cổ phiếu FPT"
image: images/FPT.png
caption: Illustration by AI Vietnam Team
categories:
  - minutes
tags:
  - feature
draft: false
---

# LTSF-Linear FPT Forecasting Challenge  
Hybrid Trend + XGBoost Residual + Regime-aware Pricing

> **Goal:**  
> Dự báo **giá đóng cửa FPT 100 ngày tiếp theo** (T+100) chỉ từ file `FPT_train.csv`,  
> với trọng tâm là **dài hạn (long horizon)** và **ổn định qua nhiều pha thị trường**. :contentReference[oaicite:0]{index=0}  

---
### 🧪 File Source Code: 
[Google_Colab] (https://drive.google.com/file/d/1i1CL8pMqbykRZiGpC6qojPCeSSwLGOVA/view?usp=sharing)  

---

## 1. Problem Overview

Trong các mô hình baseline (Linear / NLinear / DLinear), khi forecast cuốn chiếu 100 ngày,  
đường dự báo rất dễ trở thành **một đường thẳng mượt**, gần như **mất hết volatility** – hiện tượng gọi là  
**“Cái chết của phương sai” (Death of Variance)**. :contentReference[oaicite:1]{index=1}  

Nguyên nhân chính:

- Dùng **log-price** (vốn đã mượt) + chuẩn hoá NLinear làm phẳng dao động. :contentReference[oaicite:2]{index=2}  
- Mô hình chỉ có **một tầng Linear** trên cửa sổ 14→3, nên chủ yếu học được **slope trung bình** của log-price. :contentReference[oaicite:3]{index=3}  
- Forecast cuốn chiếu nhiều bước ⇒ mọi nhiễu nhỏ bị “là phẳng” dần và hội tụ thành đường thẳng. :contentReference[oaicite:4]{index=4}  

Dự án này đề xuất một **pipeline Hybrid 3 lớp** để giải bài toán:

1. **Math Backbone (Trend)** – mô hình hoá quỹ đạo dài hạn trên log-price. :contentReference[oaicite:5]{index=5}  
2. **XGBoost Residual (ML Layer)** – học phần nhiễu có cấu trúc còn lại (residual). :contentReference[oaicite:6]{index=6}  
3. **Pricing Layer (Regime-aware)** – kiểm soát biên độ, mean reversion và chế độ thị trường. :contentReference[oaicite:7]{index=7}  

Mục tiêu chính **không phải** dự đoán chính xác từng ngày,  
mà là dựng được **một trajectory giá hợp lý, bền vững** cho FPT. :contentReference[oaicite:8]{index=8}  

---

## 2. Dataset

- File: `FPT_train.csv`  
- Các cột chính: `time`, `open`, `high`, `low`, `close`, `volume`, `symbol`. :contentReference[oaicite:9]{index=9}  
- Đặc trưng FPT:
  - Cổ phiếu công nghệ đầu ngành.
  - **Xu hướng dài hạn tăng (uptrend)** rõ rệt.
  - Biến động mạnh theo **regime thị trường**: Bull / Bear / Sideways. :contentReference[oaicite:10]{index=10}  

Hạn chế:

- Chỉ có **OHLCV daily**, không có news / macro / sentiment. :contentReference[oaicite:11]{index=11}  
- Khoảng **~4.5 năm dữ liệu** nhưng phải dự báo 100 ngày – horizon khá dài. :contentReference[oaicite:12]{index=12}  

---

## 3. Project Structure (suggested)

Bạn có thể tổ chức repo như sau:

```text
.
├── README.md                 # File này
├── FPT_train.csv             # Dữ liệu gốc
├── src/
│   ├── features.py           # Feature engineering (OHLCV, STL, returns, patterns,…)
│   ├── backbone.py           # Math Backbone (linear trend on log-price)
│   ├── residual_xgb.py       # XGBoost residual model
│   ├── pricing_layer.py      # Clipping, damping, mean reversion, regime-aware pricing
│   ├── ensemble.py           # Kết hợp BASE + TREND + RISK (central_det, bull, bear)
│   └── main.py               # Pipeline end-to-end (CV + training + forecast + plot)
└── notebooks/
    └── eda_and_visualization.ipynb  # EDA, charts, sanity-checks
```
---
## 📚 **Tài liệu đi kèm**

* {{< pdf src="/Time-Series-Team-Hub/pdf/M6W4D1+6_Project_Module.pdf" title="M6W4D1+6_Project_Module" height="700px" >}}
