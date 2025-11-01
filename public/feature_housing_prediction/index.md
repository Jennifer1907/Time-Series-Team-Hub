# 🏠 House Price Prediction Agent

Ứng dụng Streamlit dự đoán giá nhà thông minh sử dụng Machine Learning và Explainable AI.

## 🌟 Tính năng chính

### 1. 📊 Data Overview & EDA
- Hiển thị thông tin tổng quan về dataset
- Phân tích missing data với visualization
- Phân phối giá nhà (normal và log-transformed)
- Correlation heatmap của các features quan trọng

### 2. 🧹 Data Processing
- **Data Cleaning**: Xử lý missing values thông minh
  - Categorical features: Fill với 'None' hoặc mode
  - Numerical features: Fill với 0 hoặc median theo neighborhood
- **Feature Engineering**: Tạo 13 features mới
  - TotalSF: Tổng diện tích
  - TotalBathrooms: Tổng số phòng tắm
  - HouseAge, YearsSinceRemod: Tuổi nhà
  - TotalPorchSF: Tổng diện tích sân
  - OverallGrade: Quality interaction
  - Binary features: HasBasement, HasGarage, HasFireplace, HasPool
  - Ratios: LivLotRatio, GarageLotRatio

### 3. 🤖 Model Training
- **Feature Selection**: Sử dụng statistical methods
  - Correlation analysis cho numerical features
  - ANOVA F-test cho categorical features
- **Multiple Models**:
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
  - **Ensemble Model** (Voting Regressor)
- Hiển thị metrics: RMSE, R², MAE
- Visualize performance comparison

### 4. 🎯 Prediction Interface
- Chọn property bằng index
- Dự đoán giá nhà với ensemble model
- Hiển thị:
  - Predicted price vs Actual price
  - Error percentage
  - Property details
  - **Top 5 Comparable Properties** với similarity scores
  - Price comparison chart

### 5. 💡 Model Explainability (SHAP)
- SHAP Summary Plot: Feature importance
- SHAP Bar Plot: Overall feature impact
- Giải thích prediction một cách trực quan

## 🚀 Cài đặt và Sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
streamlit run house_price_predictor_app.py
```

### 3. Upload data

#### Option 1: Sử dụng data của bạn
- Chuẩn bị file CSV với cấu trúc tương tự dataset House Prices
- Upload qua sidebar

#### Option 2: Download sample data
- Truy cập [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Download file `train.csv`
- Upload vào ứng dụng

## 📋 Cấu trúc Data

Dataset cần có các cột chính sau:
- `SalePrice`: Giá bán (target variable)
- `GrLivArea`: Diện tích sống
- `OverallQual`: Chất lượng tổng thể (1-10)
- `YearBuilt`: Năm xây dựng
- `BedroomAbvGr`: Số phòng ngủ
- `GarageCars`: Số xe garage
- `Neighborhood`: Khu phố
- Và nhiều features khác...

## 🎯 Workflow

1. **Upload Data** → Upload file CSV qua sidebar
2. **Data Overview** → Khám phá và hiểu dữ liệu
3. **Data Processing** → Click "Run Data Cleaning & Feature Engineering"
4. **Model Training** → Click "Train Models" để train các mô hình
5. **Prediction** → Nhập index của property và click "Predict Price"
6. **Explainability** → Xem SHAP analysis để hiểu model

## 🔧 Các Algorithm sử dụng

### Feature Selection
- **Pearson Correlation**: Cho numerical features (threshold: 0.1)
- **ANOVA F-test**: Cho categorical features (threshold: 10)

### Models
1. **Ridge Regression** (alpha=10): Linear model với L2 regularization
2. **Lasso Regression** (alpha=0.001): Linear model với L1 regularization
3. **Random Forest** (100 trees): Ensemble của decision trees
4. **Gradient Boosting** (100 estimators): Boosting algorithm
5. **Ensemble Model**: Voting regressor kết hợp Ridge, RF, và GB

### Explainability
- **SHAP (SHapley Additive exPlanations)**: Giải thích contribution của từng feature

## 📊 Metrics đánh giá

- **RMSE** (Root Mean Squared Error): Đo sai số trung bình
- **R²** (R-squared): Đo độ fit của model (0-1, cao hơn = tốt hơn)
- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình

## 🎨 Giao diện

### Sidebar
- Upload file CSV
- Menu navigation
- Hướng dẫn sử dụng

### Main Tabs
1. **Data Overview**: EDA và statistics
2. **Data Processing**: Cleaning và feature engineering
3. **Model Training**: Train và evaluate models
4. **Prediction**: Dự đoán giá nhà mới
5. **Explainability**: SHAP analysis

## 💡 Tips

- **SHAP Analysis**: Có thể mất vài phút với dataset lớn
- **Feature Engineering**: Đã được tối ưu cho House Prices dataset
- **Model Selection**: Ensemble model thường cho kết quả tốt nhất
- **Comparable Properties**: Sử dụng weighted similarity scoring

## 🐛 Troubleshooting

### Lỗi khi load data
- Kiểm tra format CSV
- Đảm bảo có cột `SalePrice`
- Kiểm tra encoding (UTF-8)

### SHAP timeout
- Giảm số samples trong code (default: 100)
- Sử dụng subset nhỏ hơn của data

### Memory issues
- Giảm n_estimators trong Random Forest và Gradient Boosting
- Sử dụng subset của data cho training

## 📚 Tài liệu tham khảo

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Hãy tạo issue hoặc pull request.

## 👨‍💻 Author

Developed with ❤️ using Streamlit, Scikit-learn, and SHAP

---

**Happy Predicting! 🏠💰**