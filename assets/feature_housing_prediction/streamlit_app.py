# -*- coding: utf-8 -*-
"""
Streamlit App: House Price Prediction Agent
Dự đoán giá nhà sử dụng Machine Learning và Explainable AI
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Any, Optional
import pickle
import io

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# XAI Libraries
import shap

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🏠 House Price Prediction Agent",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'df_engineered' not in st.session_state:
    st.session_state.df_engineered = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}


# ==================== HELPER FUNCTIONS ====================

def load_data(file):
    """Load CSV data"""
    try:
        df = pd.read_csv(file)
        st.session_state.df_original = df
        st.session_state.data_loaded = True
        return df
    except Exception as e:
        st.error(f"Lỗi khi load data: {str(e)}")
        return None


def clean_data(df):
    """Clean data - handle missing values"""
    df_clean = df.copy()
    
    # Categorical columns to fill with 'None'
    categorical_fill_none = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    
    for col in categorical_fill_none:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('None')
    
    # Fill categorical with mode
    cat_cols = df_clean.select_dtypes(include='object').columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
    
    # Numerical columns to fill with 0
    numerical_fill_zero = [
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
        'GarageYrBlt', 'GarageCars', 'GarageArea'
    ]
    
    for col in numerical_fill_zero:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # LotFrontage - fill by neighborhood median
    if 'LotFrontage' in df_clean.columns:
        df_clean['LotFrontage'] = df_clean.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Fill remaining numerical with median
    num_cols = df_clean.select_dtypes(exclude='object').columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean


def engineer_features(df):
    """Feature engineering"""
    df_eng = df.copy()
    
    # Total square footage
    df_eng['TotalSF'] = (df_eng['TotalBsmtSF'] + 
                         df_eng['1stFlrSF'] + 
                         df_eng['2ndFlrSF'])
    
    # Total bathrooms
    df_eng['TotalBathrooms'] = (df_eng['FullBath'] + 
                                (df_eng['HalfBath'] * 0.5) + 
                                df_eng['BsmtFullBath'] + 
                                (df_eng['BsmtHalfBath'] * 0.5))
    
    # Age features
    df_eng['HouseAge'] = df_eng['YrSold'] - df_eng['YearBuilt']
    df_eng['YearsSinceRemod'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
    
    # Porch features
    df_eng['TotalPorchSF'] = (df_eng['OpenPorchSF'] + 
                              df_eng['3SsnPorch'] + 
                              df_eng['EnclosedPorch'] + 
                              df_eng['ScreenPorch'] + 
                              df_eng['WoodDeckSF'])
    
    # Quality interaction
    df_eng['OverallGrade'] = df_eng['OverallQual'] * df_eng['OverallCond']
    
    # Garage age
    df_eng['GarageAge'] = df_eng['YrSold'] - df_eng['GarageYrBlt']
    df_eng['GarageAge'] = df_eng['GarageAge'].fillna(0)
    
    # Binary features
    df_eng['HasBasement'] = (df_eng['TotalBsmtSF'] > 0).astype(int)
    df_eng['HasGarage'] = (df_eng['GarageArea'] > 0).astype(int)
    df_eng['HasFireplace'] = (df_eng['Fireplaces'] > 0).astype(int)
    df_eng['HasPool'] = (df_eng['PoolArea'] > 0).astype(int)
    
    # Ratios
    df_eng['LivLotRatio'] = df_eng['GrLivArea'] / df_eng['LotArea']
    df_eng['GarageLotRatio'] = df_eng['GarageArea'] / df_eng['LotArea']
    
    return df_eng


def select_features_stats(df):
    """Feature selection using statistical methods"""
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Numerical features - correlation
    correlation = numeric_df.corr()["SalePrice"].abs().sort_values(ascending=False)
    selected_numeric = correlation[correlation >= 0.1].index.tolist()
    if "SalePrice" in selected_numeric:
        selected_numeric.remove("SalePrice")
    
    # Categorical features - ANOVA F-test
    selected_categorical = []
    X_cat = df[categorical_cols].copy()
    y = df['SalePrice'].copy()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_cat[col] = le.fit_transform(X_cat[col].astype(str))
        label_encoders[col] = le
    
    selector = SelectKBest(f_regression, k='all')
    selector.fit(X_cat, y)
    scores = pd.Series(selector.scores_, index=categorical_cols)
    selected_categorical = scores[scores > 10].index.tolist()
    
    selected_features = selected_numeric + selected_categorical
    
    return selected_features, label_encoders


def prepare_ml_data(df, selected_features, label_encoders):
    """Prepare data for ML training"""
    X = df[selected_features].copy()
    y = df['SalePrice'].copy()
    
    # Transform categorical features
    for col in X.columns:
        if col in label_encoders:
            X[col] = label_encoders[col].transform(X[col].astype(str))
    
    # Log transform target
    y_log = np.log1p(y)
    
    return X, y_log


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and create ensemble"""
    models = {
        'Ridge': Ridge(alpha=10),
        'Lasso': Lasso(alpha=0.001),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
        trained_models[name] = model
    
    # Create ensemble
    ensemble = VotingRegressor([
        ('ridge', models['Ridge']),
        ('rf', models['RandomForest']),
        ('gb', models['GradientBoosting'])
    ])
    ensemble.fit(X_train, y_train)
    
    y_pred_ensemble = ensemble.predict(X_test)
    results['Ensemble'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
        'R2': r2_score(y_test, y_pred_ensemble),
        'MAE': mean_absolute_error(y_test, y_pred_ensemble)
    }
    trained_models['Ensemble'] = ensemble
    
    return trained_models, results


def find_comparable_properties(target_house, df, n_comps=5):
    """Find comparable properties using weighted similarity"""
    weights = {
        'sqft': 0.3,
        'neighborhood': 0.2,
        'quality': 0.2,
        'age': 0.15,
        'bedrooms': 0.1,
        'garage': 0.05,
    }
    
    scores = []
    for idx, row in df.iterrows():
        if idx == target_house.name:
            continue
        score = 0
        
        if pd.notna(row['GrLivArea']) and pd.notna(target_house['GrLivArea']):
            diff = abs(row['GrLivArea'] - target_house['GrLivArea']) / target_house['GrLivArea']
            score += weights['sqft'] * (1 - min(diff, 1))
        
        if row['Neighborhood'] == target_house['Neighborhood']:
            score += weights['neighborhood']
        
        if pd.notna(row['OverallQual']) and pd.notna(target_house['OverallQual']):
            diff = abs(row['OverallQual'] - target_house['OverallQual']) / 10
            score += weights['quality'] * (1 - diff)
        
        if pd.notna(row['YearBuilt']) and pd.notna(target_house['YearBuilt']):
            diff = abs(row['YearBuilt'] - target_house['YearBuilt']) / 50
            score += weights['age'] * (1 - min(diff, 1))
        
        if pd.notna(row['BedroomAbvGr']) and pd.notna(target_house['BedroomAbvGr']):
            diff = abs(row['BedroomAbvGr'] - target_house['BedroomAbvGr']) / 5
            score += weights['bedrooms'] * (1 - min(diff, 1))
        
        if pd.notna(row['GarageCars']) and pd.notna(target_house['GarageCars']):
            diff = abs(row['GarageCars'] - target_house['GarageCars']) / 3
            score += weights['garage'] * (1 - min(diff, 1))
        
        scores.append((idx, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:n_comps]
    
    comparable_list = []
    for idx, similarity in top:
        comp = df.loc[idx]
        comparable_list.append({
            'Index': idx,
            'Similarity': f"{similarity:.3f}",
            'Price': f"${comp['SalePrice']:,.0f}",
            'Sqft': f"{comp['GrLivArea']:,.0f}",
            'Price/Sqft': f"${comp['SalePrice']/comp['GrLivArea']:.0f}",
            'Bedrooms': int(comp['BedroomAbvGr']),
            'Quality': f"{comp['OverallQual']}/10",
            'Year': int(comp['YearBuilt']),
            'Neighborhood': comp['Neighborhood']
        })
    
    return comparable_list


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown('<p class="main-header">🏠 House Price Prediction Agent</p>', unsafe_allow_html=True)
    st.markdown("### Dự đoán giá nhà thông minh với Machine Learning & Explainable AI")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/home.png", width=100)
        st.title("📋 Menu")
        
        # Data Upload
        st.subheader("📂 Upload Data")
        uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
        
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.success(f"✅ Loaded {len(df)} properties!")
        
        st.markdown("---")
        
        # Sample data option
        use_sample = st.checkbox("Hoặc sử dụng sample data")
        if use_sample:
            st.info("📝 Hướng dẫn: Tải file train.csv từ [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)")
        
        st.markdown("---")
        st.markdown("### 🎯 Các bước thực hiện:")
        st.markdown("""
        1. Upload file CSV
        2. Khám phá dữ liệu (EDA)
        3. Làm sạch & Feature Engineering
        4. Train mô hình
        5. Dự đoán giá nhà
        """)
    
    # Main content - Tabs
    if st.session_state.data_loaded:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🧹 Data Processing", 
            "🤖 Model Training",
            "🎯 Prediction",
            "💡 Explainability"
        ])
        
        # TAB 1: DATA OVERVIEW
        with tab1:
            st.header("📊 Data Overview & EDA")
            df = st.session_state.df_original
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Số dòng", f"{len(df):,}")
            with col2:
                st.metric("Số cột", f"{df.shape[1]}")
            with col3:
                st.metric("Missing values", f"{df.isnull().sum().sum()}")
            
            st.subheader("🔍 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Missing data visualization
            st.subheader("❌ Missing Data Analysis")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_data.plot.barh(color="lightcoral", edgecolor="black", ax=ax)
                ax.set_title("Missing Data Count")
                ax.set_xlabel("Count")
                ax.set_ylabel("Features")
                st.pyplot(fig)
            else:
                st.success("✅ Không có missing data!")
            
            # Target distribution
            if 'SalePrice' in df.columns:
                st.subheader("🎯 Target Variable Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df['SalePrice'], bins=50, color='skyblue', edgecolor='black')
                    ax.axvline(df['SalePrice'].mean(), color='red', linestyle='--', label='Mean')
                    ax.axvline(df['SalePrice'].median(), color='green', linestyle='--', label='Median')
                    ax.set_title("Sale Price Distribution")
                    ax.set_xlabel("Price")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    log_price = np.log1p(df['SalePrice'])
                    ax.hist(log_price, bins=50, color='lightgreen', edgecolor='black')
                    ax.axvline(log_price.mean(), color='red', linestyle='--', label='Mean')
                    ax.axvline(log_price.median(), color='green', linestyle='--', label='Median')
                    ax.set_title("Log-transformed Sale Price")
                    ax.set_xlabel("Log(Price)")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    st.pyplot(fig)
            
            # Correlation heatmap
            st.subheader("🔥 Correlation Heatmap (Top Features)")
            numeric_df = df.select_dtypes(include=[np.number])
            if 'SalePrice' in numeric_df.columns:
                top_corr = numeric_df.corr()['SalePrice'].abs().sort_values(ascending=False)[:15]
                top_features = top_corr.index.tolist()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(numeric_df[top_features].corr(), 
                           annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax, linewidths=0.5)
                ax.set_title("Correlation Matrix - Top 15 Features")
                st.pyplot(fig)
        
        # TAB 2: DATA PROCESSING
        with tab2:
            st.header("🧹 Data Processing")
            
            if st.button("🚀 Run Data Cleaning & Feature Engineering", type="primary"):
                with st.spinner("Đang xử lý dữ liệu..."):
                    # Clean data
                    df_clean = clean_data(st.session_state.df_original)
                    st.session_state.df_clean = df_clean
                    
                    # Engineer features
                    df_eng = engineer_features(df_clean)
                    st.session_state.df_engineered = df_eng
                    
                    st.success("✅ Data processing completed!")
            
            if st.session_state.df_clean is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Cleaned Data Info")
                    st.write(f"Rows: {len(st.session_state.df_clean):,}")
                    st.write(f"Columns: {st.session_state.df_clean.shape[1]}")
                    st.write(f"Missing values: {st.session_state.df_clean.isnull().sum().sum()}")
                
                with col2:
                    st.subheader("🔧 Engineered Features")
                    if st.session_state.df_engineered is not None:
                        new_features = set(st.session_state.df_engineered.columns) - set(st.session_state.df_original.columns)
                        st.write(f"New features added: {len(new_features)}")
                        with st.expander("View new features"):
                            st.write(list(new_features))
                
                st.subheader("📊 Processed Data Sample")
                st.dataframe(st.session_state.df_engineered.head(), use_container_width=True)
        
        # TAB 3: MODEL TRAINING
        with tab3:
            st.header("🤖 Model Training")
            
            if st.session_state.df_engineered is None:
                st.warning("⚠️ Vui lòng xử lý dữ liệu trước ở tab 'Data Processing'")
            else:
                if st.button("🎯 Train Models", type="primary"):
                    with st.spinner("Đang train models..."):
                        # Feature selection
                        selected_features, label_encoders = select_features_stats(st.session_state.df_engineered)
                        st.session_state.selected_features = selected_features
                        st.session_state.label_encoders = label_encoders
                        
                        st.info(f"Selected {len(selected_features)} features")
                        
                        # Prepare data
                        X, y_log = prepare_ml_data(st.session_state.df_engineered, 
                                                   selected_features, 
                                                   label_encoders)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_log, test_size=0.2, random_state=42
                        )
                        
                        st.session_state.X_train = X_train
                        
                        # Train models
                        trained_models, results = train_models(X_train, y_train, X_test, y_test)
                        
                        st.session_state.trained_model = trained_models['Ensemble']
                        st.session_state.model_trained = True
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Display results
                        st.subheader("📊 Model Performance")
                        results_df = pd.DataFrame(results).T
                        st.dataframe(results_df.style.highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen')
                                                   .highlight_max(axis=0, subset=['R2'], color='lightgreen'),
                                    use_container_width=True)
                        
                        # Visualization
                        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                        
                        metrics = ['RMSE', 'R2', 'MAE']
                        for i, metric in enumerate(metrics):
                            values = [results[model][metric] for model in results.keys()]
                            axes[i].bar(results.keys(), values, color='skyblue', edgecolor='black')
                            axes[i].set_title(f'{metric} Comparison')
                            axes[i].set_ylabel(metric)
                            axes[i].tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                if st.session_state.model_trained:
                    st.success("✅ Model đã sẵn sàng để dự đoán!")
        
        # TAB 4: PREDICTION
        with tab4:
            st.header("🎯 House Price Prediction")
            
            if not st.session_state.model_trained:
                st.warning("⚠️ Vui lòng train model trước ở tab 'Model Training'")
            else:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🏡 Chọn Property để Dự đoán")
                    
                    # Select by index
                    property_idx = st.number_input(
                        "Nhập Index của property", 
                        min_value=0, 
                        max_value=len(st.session_state.df_engineered)-1,
                        value=0
                    )
                    
                    if st.button("🔮 Predict Price"):
                        target_property = st.session_state.df_engineered.iloc[property_idx]
                        
                        # Prepare features
                        X_pred = target_property[st.session_state.selected_features].copy()
                        for col in X_pred.index:
                            if col in st.session_state.label_encoders:
                                le = st.session_state.label_encoders[col]
                                X_pred[col] = le.transform([str(X_pred[col])])[0]
                        
                        # Predict
                        X_pred_array = X_pred.values.reshape(1, -1)
                        y_pred_log = st.session_state.trained_model.predict(X_pred_array)[0]
                        predicted_price = np.expm1(y_pred_log)
                        actual_price = target_property['SalePrice']
                        
                        # Display results
                        st.subheader("💰 Prediction Results")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Predicted Price", f"${predicted_price:,.0f}")
                        with col_b:
                            st.metric("Actual Price", f"${actual_price:,.0f}")
                        
                        error_pct = abs(predicted_price - actual_price) / actual_price * 100
                        st.metric("Error", f"{error_pct:.2f}%")
                        
                        # Property details
                        st.subheader("🏠 Property Details")
                        details_df = pd.DataFrame({
                            'Feature': ['Neighborhood', 'GrLivArea', 'OverallQual', 'YearBuilt', 
                                       'BedroomAbvGr', 'GarageCars', 'TotalBathrooms'],
                            'Value': [
                                target_property['Neighborhood'],
                                f"{target_property['GrLivArea']:,.0f} sqft",
                                f"{target_property['OverallQual']}/10",
                                int(target_property['YearBuilt']),
                                int(target_property['BedroomAbvGr']),
                                int(target_property['GarageCars']),
                                f"{target_property['TotalBathrooms']:.1f}"
                            ]
                        })
                        st.dataframe(details_df, use_container_width=True, hide_index=True)
                
                with col2:
                    if 'predicted_price' in locals():
                        st.subheader("📊 Comparable Properties")
                        comparables = find_comparable_properties(
                            target_property, 
                            st.session_state.df_engineered,
                            n_comps=5
                        )
                        
                        comp_df = pd.DataFrame(comparables)
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
                        
                        # Price comparison chart
                        st.subheader("💵 Price Comparison")
                        prices = [float(c['Price'].replace('$', '').replace(',', '')) for c in comparables]
                        prices.insert(0, predicted_price)
                        labels = ['Predicted'] + [f"Comp {i+1}" for i in range(len(comparables))]
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['red'] + ['skyblue'] * len(comparables)
                        ax.bar(labels, prices, color=colors, edgecolor='black')
                        ax.axhline(actual_price, color='green', linestyle='--', 
                                  linewidth=2, label=f'Actual: ${actual_price:,.0f}')
                        ax.set_ylabel('Price ($)')
                        ax.set_title('Price Comparison with Comparable Properties')
                        ax.legend()
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
        
        # TAB 5: EXPLAINABILITY
        with tab5:
            st.header("💡 Model Explainability (SHAP)")
            
            if not st.session_state.model_trained:
                st.warning("⚠️ Vui lòng train model trước")
            else:
                st.info("⚙️ Đang tính toán SHAP values... Có thể mất vài phút")
                
                try:
                    # Select a subset for SHAP
                    n_samples = min(100, len(st.session_state.X_train))
                    X_sample = st.session_state.X_train.sample(n_samples, random_state=42)
                    
                    # SHAP explainer
                    explainer = shap.Explainer(st.session_state.trained_model, X_sample)
                    shap_values = explainer(X_sample)
                    
                    st.subheader("📊 SHAP Summary Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot(fig)
                    
                    st.subheader("🎯 Feature Importance (SHAP)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                    st.pyplot(fig)
                    
                    st.success("✅ SHAP analysis completed!")
                    
                except Exception as e:
                    st.error(f"Lỗi khi tính SHAP: {str(e)}")
                    st.info("💡 Tip: SHAP có thể mất nhiều thời gian với dataset lớn")
    
    else:
        # Welcome screen
        st.info("👆 Vui lòng upload file CSV ở sidebar để bắt đầu")
        
        st.markdown("""
        ### 🎯 Tính năng chính:
        
        - **📊 EDA tự động**: Phân tích và visualize dữ liệu
        - **🧹 Data Cleaning**: Xử lý missing values thông minh
        - **🔧 Feature Engineering**: Tạo features mới từ dữ liệu gốc
        - **🤖 Multi-Model Training**: Train nhiều models và ensemble
        - **🎯 Prediction**: Dự đoán giá nhà với độ chính xác cao
        - **💡 Explainability**: Giải thích prediction với SHAP
        - **📊 Comparable Analysis**: Tìm properties tương tự
        
        ### 📝 Hướng dẫn sử dụng:
        
        1. Upload file CSV (hoặc download từ [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques))
        2. Khám phá dữ liệu trong tab "Data Overview"
        3. Xử lý dữ liệu trong tab "Data Processing"
        4. Train model trong tab "Model Training"
        5. Dự đoán giá nhà trong tab "Prediction"
        6. Xem giải thích model trong tab "Explainability"
        """)


if __name__ == "__main__":
    main()