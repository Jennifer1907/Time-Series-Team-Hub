# 📈 FPT Stock Price Forecast - Streamlit App

A comprehensive stock price forecasting application using a **Hybrid Model** combining:
- **Math Backbone**: Linear trend on log-price
- **XGBoost Residual**: Machine learning for residual prediction
- **Pricing Layer**: Regime-aware clipping, damping, and mean-reversion

## 🚀 Features

- Interactive parameter tuning
- Real-time forecast visualization with Plotly
- Regime detection (BULL/BEAR/SIDEWAYS)
- Uncertainty bands (90% confidence interval)
- CSV export for forecasts and submissions

## 📁 Project Structure

```
fpt_streamlit_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── data/
    └── FPT_train.csv     # (Optional) Sample data
```

## 🛠️ Local Development

### 1. Clone or Download the Project

```bash
# Create project directory
mkdir fpt_streamlit_app
cd fpt_streamlit_app
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud

### Step 1: Prepare GitHub Repository

1. **Create a new GitHub repository**
   - Go to [github.com/new](https://github.com/new)
   - Name it: `fpt-stock-forecast` (or any name you prefer)
   - Set to **Public** (required for free Streamlit Cloud)
   - Click "Create repository"

2. **Upload project files**
   
   Option A: Using GitHub Web Interface
   - Click "Add file" → "Upload files"
   - Drag and drop all project files
   - Commit changes

   Option B: Using Git CLI
   ```bash
   cd fpt_streamlit_app
   git init
   git add .
   git commit -m "Initial commit: FPT Stock Forecast App"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/fpt-stock-forecast.git
   git push -u origin main
   ```

### Step 2: Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Click **"Sign in with GitHub"**

3. Click **"New app"**

4. Fill in the deployment form:
   - **Repository**: `YOUR_USERNAME/fpt-stock-forecast`
   - **Branch**: `main`
   - **Main file path**: `app.py`

5. Click **"Deploy!"**

### Step 3: Wait for Deployment

- Streamlit Cloud will install dependencies and build the app
- This usually takes 2-5 minutes
- Once done, you'll get a URL like: `https://your-app-name.streamlit.app`

---

## 📊 Data Format

The app expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `time` | datetime | Trading date |
| `open` | float | Opening price |
| `high` | float | Highest price |
| `low` | float | Lowest price |
| `close` | float | Closing price |
| `volume` | int/float | Trading volume |
| `symbol` | string | Stock symbol (optional) |

Example:
```csv
time,open,high,low,close,volume,symbol
2024-01-02,100.0,102.0,99.0,101.0,1000000,FPT
2024-01-03,101.5,103.0,100.5,102.5,1200000,FPT
```

---

## ⚙️ Configuration Parameters

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Forecast Horizon | 100 days | Number of days to forecast |
| STL Period | 20 | Seasonal decomposition period |

### Pricing Layer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Return Clip Quantile | 0.99 | Quantile for return clipping |
| Half-Life Days | 60 | Damping half-life |
| Mean Revert Alpha | 0.06 | Mean reversion strength |
| Mean Revert Start | 40 | Day to start mean reversion |
| Fair Up Mult | 1.4 | Upper bound multiplier |
| Fair Down Mult | 0.75 | Lower bound multiplier |
| Trend Lookback | 30 | Days for trend calculation |
| Trend Ret Thresh | 0.18 | Threshold for trend detection |

---

## 🔧 Customization

### Adding New Features

1. Edit `app.py` to add new feature engineering in `add_stl_ohlcv_features()`
2. Update `get_feature_cols()` to include new features

### Changing Model

1. Modify XGBoost parameters in `train_xgb_on_dfmodel()`
2. Adjust pricing layer in `apply_pricing_on_raw_path()`

### Styling

1. Edit `.streamlit/config.toml` for theme changes
2. Modify `app.py` for layout changes

---

## 📝 Troubleshooting

### Common Issues

1. **"Module not found" error**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data loading fails**
   - Check CSV format matches expected columns
   - Ensure dates are in parseable format

3. **Streamlit Cloud build fails**
   - Check `requirements.txt` for typos
   - Ensure all imports are in requirements

4. **Memory errors on Streamlit Cloud**
   - Reduce forecast horizon
   - Use smaller dataset for testing

---

## 📄 License

This project is for educational purposes. Use at your own risk for financial decisions.

---

## 🙏 Credits

- Model based on hybrid approach: Math Backbone + XGBoost + Pricing Layer
- Built with Streamlit, Plotly, XGBoost, and scikit-learn
