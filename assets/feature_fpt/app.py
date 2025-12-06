"""
FPT Stock Forecast - Streamlit App
===================================
Hybrid Model: Math Backbone + XGBoost Residual + Pricing Layer
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import io

from statsmodels.tsa.seasonal import STL
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FPT Stock Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
SEED = 98
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class PricingParams:
    ret_clip_quantile: float
    half_life_days: int
    mean_revert_alpha: float
    mean_revert_start: int
    fair_up_mult: float
    fair_down_mult: float
    trend_lookback: int
    trend_ret_thresh: float


@dataclass
class FinalBaseModel:
    xgb: XGBRegressor
    scaler_X: StandardScaler
    feature_cols: List[str]
    df_train: pd.DataFrame
    df_feat: pd.DataFrame
    train_end_date: pd.Timestamp
    resid_std: float
    resid_mean: float


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_stl_ohlcv_features(df_raw: pd.DataFrame, stl_period: int = 20) -> pd.DataFrame:
    """Feature engineering for FPT OHLCV data"""
    df = df_raw.copy()
    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column.")

    df["close"] = df["close"].astype(float)
    df["close_log"] = np.log(df["close"] + 1e-8)
    eps = 1e-6

    # Price action from OHLC
    has_ohlc = all(c in df.columns for c in ["open", "high", "low"])
    pa_cols = [
        "body", "range", "upper_shadow", "lower_shadow", "body_rel", "close_pos",
        "gap_oc", "gap_prev_close", "range_pct", "true_range", "atr_14", "park_vol",
        "range_ma_10", "range_expansion", "body_lag1", "body_lag2", "close_pos_lag1",
    ]
    
    if has_ohlc:
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)

        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]
        df["body_rel"] = df["body"] / (df["range"].abs() + eps)

        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["close_pos"] = (df["close"] - df["low"]) / (df["range"].abs() + eps)

        df["gap_oc"] = df["open"] - df["close"]
        df["gap_prev_close"] = df["open"] - df["close"].shift(1)

        prev_close = df["close"].shift(1)
        df["range_pct"] = df["range"] / (prev_close.abs() + eps)

        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = df["true_range"].rolling(14).mean()

        ratio_hl = (df["high"] / (df["low"] + eps)).clip(lower=eps)
        df["park_vol"] = (1.0 / (4.0 * np.log(2.0))) * (np.log(ratio_hl) ** 2)

        df["range_ma_10"] = df["range"].rolling(10).mean()
        df["range_expansion"] = (df["range"] > 1.5 * (df["range_ma_10"].abs() + eps)).astype(float)

        df["body_lag1"] = df["body"].shift(1)
        df["body_lag2"] = df["body"].shift(2)
        df["close_pos_lag1"] = df["close_pos"].shift(1)
    else:
        for c in pa_cols:
            df[c] = 0.0

    # Volume & money flow
    vol_cols = ["volume", "vol_ma_5", "vol_ma_20", "vol_ratio", "money_flow",
                "money_flow_5", "vol_z_20", "obv", "vol_ratio_lag1", "obv_diff_1d"]
    
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
        df["vol_ma_5"] = df["volume"].rolling(5).mean()
        df["vol_ma_20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_ma_20"].abs() + eps)
        df["money_flow"] = df["close"] * df["volume"]
        df["money_flow_5"] = df["money_flow"].rolling(5).mean()
        df["vol_z_20"] = (df["volume"] - df["vol_ma_20"]) / (df["vol_ma_20"].abs() + eps)

        sign = np.sign(df["close"].diff(1))
        df["obv"] = (sign * df["volume"]).fillna(0).cumsum()
        df["vol_ratio_lag1"] = df["vol_ratio"].shift(1)
        df["obv_diff_1d"] = df["obv"].diff(1)
    else:
        for c in vol_cols:
            df[c] = 0.0

    # STL on close_log
    close_log = df["close_log"].copy()
    if close_log.isna().any():
        close_log = close_log.ffill().bfill()

    res = STL(close_log.values, period=stl_period, robust=True).fit()
    df["trend_stl"] = res.trend
    df["seasonal_stl"] = res.seasonal
    df["resid_stl"] = res.resid

    # Returns
    df["ret_1d"] = df["close_log"].diff(1)
    df["ret_5d"] = df["close_log"].diff(5)
    df["ret_10d"] = df["close_log"].diff(10)

    # Patterns
    df["up_1d"] = (df["ret_1d"] > 0).astype(float)
    df["down_1d"] = (df["ret_1d"] < 0).astype(float)
    df["up_3streak"] = (
        (df["ret_1d"] > 0) & (df["ret_1d"].shift(1) > 0) & (df["ret_1d"].shift(2) > 0)
    ).astype(float)
    df["down_3streak"] = (
        (df["ret_1d"] < 0) & (df["ret_1d"].shift(1) < 0) & (df["ret_1d"].shift(2) < 0)
    ).astype(float)

    # Trend slopes & stats
    df["trend_slope_1"] = df["trend_stl"].diff(1)
    df["trend_slope_3"] = df["trend_stl"].diff(3)
    df["trend_slope_7"] = df["trend_stl"].diff(7)
    df["trend_accel_3"] = df["trend_slope_1"] - df["trend_slope_1"].shift(3)
    df["trend_mean_21"] = df["trend_stl"].rolling(21).mean()
    df["trend_std_21"] = df["trend_stl"].rolling(21).std(ddof=0)
    df["resid_std_10"] = df["resid_stl"].rolling(10).std(ddof=0)
    df["resid_std_20"] = df["resid_stl"].rolling(20).std(ddof=0)
    df["z_trend_21"] = (df["trend_stl"] - df["trend_mean_21"]) / (df["trend_std_21"] + eps)

    return df


def get_feature_cols() -> List[str]:
    return [
        "close_log", "ret_1d", "ret_5d", "ret_10d", "up_1d", "down_1d",
        "up_3streak", "down_3streak", "trend_stl", "trend_slope_1", "trend_slope_3",
        "trend_slope_7", "trend_accel_3", "trend_mean_21", "trend_std_21",
        "resid_std_10", "resid_std_20", "z_trend_21", "body", "range",
        "upper_shadow", "lower_shadow", "body_rel", "close_pos", "gap_oc",
        "gap_prev_close", "range_pct", "range_ma_10", "range_expansion",
        "body_lag1", "body_lag2", "close_pos_lag1", "true_range", "atr_14",
        "park_vol", "volume", "vol_ma_5", "vol_ma_20", "vol_ratio",
        "vol_ratio_lag1", "money_flow", "money_flow_5", "vol_z_20", "obv", "obv_diff_1d",
    ]


def build_modeling_df(df_slice: pd.DataFrame, stl_period: int, horizon: int):
    """Build feature + target dataframe"""
    df_slice = df_slice.sort_values("time").reset_index(drop=True)
    df_feat = add_stl_ohlcv_features(df_slice, stl_period=stl_period)
    feature_cols = get_feature_cols()

    df_feat["time_future"] = df_feat["time"].shift(-horizon)
    df_feat["close_log_future"] = df_feat["close_log"].shift(-horizon)
    df_feat["future_ret"] = df_feat["close_log_future"] - df_feat["close_log"]

    # Math backbone
    time_idx = np.arange(len(df_feat)).reshape(-1, 1)
    y_log = df_feat["close_log"].values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(time_idx, y_log)
    trend_log = lr.predict(time_idx).flatten()
    df_feat["trend_log"] = trend_log
    df_feat["trend_log_future"] = df_feat["trend_log"].shift(-horizon)
    df_feat["math_ret"] = df_feat["trend_log_future"] - df_feat["trend_log"]
    df_feat["resid_ret"] = df_feat["future_ret"] - df_feat["math_ret"]

    df_model = df_feat.dropna(
        subset=feature_cols + ["future_ret", "math_ret", "resid_ret", "time_future"]
    ).reset_index(drop=True)

    return df_feat, df_model, feature_cols


# ============================================================
# XGB TRAINING
# ============================================================
def train_xgb_on_dfmodel(df_model: pd.DataFrame, feature_cols: List[str]):
    """Train XGBoost on residual returns"""
    if len(df_model) < 200:
        raise ValueError(f"Too few samples: {len(df_model)}")

    X_all = df_model[feature_cols].values.astype(np.float32)
    y_resid = df_model["resid_ret"].values.astype(np.float32)

    N = len(df_model)
    train_ratio = 0.8
    val_ratio = 0.1

    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))

    X_train = X_all[:train_end]
    y_train = y_resid[:train_end]

    scaler_X = StandardScaler().fit(X_train)
    X_train_s = scaler_X.transform(X_train)

    xgb = XGBRegressor(
        n_estimators=450,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=3,
        objective="reg:squarederror",
        random_state=SEED,
    )
    xgb.fit(X_train_s, y_train)

    # Calculate residual stats
    y_pred_all = xgb.predict(scaler_X.transform(X_all))
    residuals = y_resid - y_pred_all
    resid_std = float(np.std(residuals, ddof=1))
    resid_mean = float(np.mean(residuals))

    return xgb, scaler_X, resid_std, resid_mean


# ============================================================
# REGIME DETECTION
# ============================================================
def detect_regime(hist_close: np.ndarray, df_feat_hist: pd.DataFrame) -> str:
    """Detect market regime: BULL, BEAR, or SIDEWAYS"""
    price_series = pd.Series(hist_close.astype(float))
    if len(price_series) >= 120:
        ma_long = price_series.rolling(120).mean().iloc[-1]
    else:
        ma_long = price_series.mean()

    price_last = price_series.iloc[-1]
    price_pos = price_last / (ma_long + 1e-8) - 1.0

    ret_1d = df_feat_hist["ret_1d"].dropna()
    if len(ret_1d) < 30:
        return "SIDEWAYS"

    vol_20 = ret_1d.rolling(20).std().iloc[-1]
    vol_all = ret_1d.std()
    vol_ratio = vol_20 / (vol_all + 1e-8)

    if price_pos < -0.05 and vol_ratio > 1.2:
        regime = "BEAR"
    elif price_pos > 0.05 and vol_ratio < 0.8:
        regime = "BULL"
    else:
        regime = "SIDEWAYS"

    return regime


# ============================================================
# RAW HYBRID PATH
# ============================================================
def build_raw_base_path_hybrid(
    df_hist: pd.DataFrame,
    xgb: XGBRegressor,
    scaler_X: StandardScaler,
    feature_cols: List[str],
    total_days: int,
    stl_period: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build recursive hybrid path: Math backbone + XGB residual"""
    df_state = df_hist.sort_values("time").reset_index(drop=True).copy()
    df_state["close"] = df_state["close"].astype(float)
    df_state["close_log"] = np.log(df_state["close"] + 1e-8)

    # Fit math backbone
    N_hist = len(df_state)
    time_idx_hist = np.arange(N_hist).reshape(-1, 1)
    y_log_hist = df_state["close_log"].values.reshape(-1, 1)
    lr_trend = LinearRegression()
    lr_trend.fit(time_idx_hist, y_log_hist)

    # Extend trend
    total_len = N_hist + total_days
    time_idx_full = np.arange(total_len).reshape(-1, 1)
    trend_log_full = lr_trend.predict(time_idx_full).flatten()

    # Precompute math_ret_forecast
    math_rets_forecast = np.zeros(total_days, dtype=float)
    for k in range(total_days):
        base_idx = N_hist - 1 + k
        if base_idx + 1 < len(trend_log_full):
            math_rets_forecast[k] = trend_log_full[base_idx + 1] - trend_log_full[base_idx]
        else:
            math_rets_forecast[k] = math_rets_forecast[k - 1] if k > 0 else 0.0

    raw_prices = []
    raw_rets = []

    for step_idx in range(total_days):
        last_close = df_state["close"].iloc[-1]
        last_time = df_state["time"].iloc[-1]

        new_row = {"time": last_time + BDay(1), "close": last_close}
        if "volume" in df_state.columns:
            new_row["volume"] = df_state["volume"].iloc[-1]
        if all(c in df_state.columns for c in ["open", "high", "low"]):
            new_row["open"] = last_close
            new_row["high"] = last_close
            new_row["low"] = last_close

        df_state = pd.concat([df_state, pd.DataFrame([new_row])], ignore_index=True)
        df_state["close"] = df_state["close"].astype(float)
        df_state["close_log"] = np.log(df_state["close"] + 1e-8)

        df_state_feat = add_stl_ohlcv_features(df_state, stl_period=stl_period)
        last_row = df_state_feat.iloc[-1]
        
        feat_vals = []
        for col in feature_cols:
            v = last_row.get(col, np.nan)
            if pd.isna(v):
                v = 0.0
            feat_vals.append(v)

        X_last = np.array(feat_vals, dtype=np.float32).reshape(1, -1)
        X_last_s = scaler_X.transform(X_last)
        resid_pred = float(xgb.predict(X_last_s)[0])

        math_ret = float(math_rets_forecast[step_idx])
        final_ret = math_ret + resid_pred

        last_log = df_state["close_log"].iloc[-2]
        next_log = last_log + final_ret
        next_price = float(np.exp(next_log))

        last_idx = df_state.index[-1]
        df_state.at[last_idx, "close"] = next_price
        df_state.at[last_idx, "close_log"] = np.log(next_price + 1e-8)
        if all(c in df_state.columns for c in ["open", "high", "low"]):
            df_state.at[last_idx, "open"] = next_price
            df_state.at[last_idx, "high"] = next_price
            df_state.at[last_idx, "low"] = next_price

        raw_prices.append(next_price)
        raw_rets.append(final_ret)

    return np.array(raw_prices, dtype=float), np.array(raw_rets, dtype=float)


# ============================================================
# PRICING LAYER
# ============================================================
def apply_pricing_on_raw_path(
    hist_close: np.ndarray,
    df_feat_hist: pd.DataFrame,
    raw_rets: np.ndarray,
    pricing: PricingParams,
) -> np.ndarray:
    """Apply pricing layer: clip, damping, mean-revert"""
    total_days = len(raw_rets)

    FAIR_MA_LEN = 60
    if len(hist_close) >= FAIR_MA_LEN:
        fair_level = float(hist_close[-FAIR_MA_LEN:].mean())
    else:
        fair_level = float(hist_close.mean())

    hist_abs_ret = df_feat_hist["ret_1d"].dropna().abs()
    if len(hist_abs_ret) == 0:
        base_ret_clip = 0.05
    else:
        base_ret_clip = float(hist_abs_ret.quantile(pricing.ret_clip_quantile))

    ret_1d = df_feat_hist["ret_1d"].dropna()
    if len(ret_1d) >= 30:
        vol_20 = ret_1d.rolling(20).std().iloc[-1]
        vol_all = ret_1d.std()
        vol_ratio = vol_20 / (vol_all + 1e-8)
    else:
        vol_ratio = 1.0

    regime = detect_regime(hist_close, df_feat_hist)

    # Regime-based scaling
    if regime == "BULL":
        clip_scale_regime = 1.2
        mr_alpha_scale_regime = 0.7
        damp_scale_regime = 0.85
        up_mult_scale_regime = 1.05
        down_mult_scale_regime = 1.0
    elif regime == "BEAR":
        clip_scale_regime = 0.95
        mr_alpha_scale_regime = 1.25
        damp_scale_regime = 1.15
        up_mult_scale_regime = 0.95
        down_mult_scale_regime = 0.95
    else:
        clip_scale_regime = 1.0
        mr_alpha_scale_regime = 1.0
        damp_scale_regime = 1.0
        up_mult_scale_regime = 1.0
        down_mult_scale_regime = 1.0

    clip_scale_vol = float(np.clip(vol_ratio, 0.8, 1.2))
    ret_clip = base_ret_clip * clip_scale_regime * clip_scale_vol
    ret_clip = float(np.clip(ret_clip, 0.01, 0.15))

    lambda_damp = np.log(2.0) / float(max(int(pricing.half_life_days * damp_scale_regime), 1))
    alpha_base = pricing.mean_revert_alpha * mr_alpha_scale_regime
    fair_up_mult = pricing.fair_up_mult * up_mult_scale_regime
    fair_down_mult = pricing.fair_down_mult * down_mult_scale_regime

    prices = np.empty(total_days, dtype=float)
    full_history = list(hist_close.astype(float))

    for step_idx in range(total_days):
        raw_ret = float(raw_rets[step_idx])

        # 1) Clip return
        pred_ret = np.clip(raw_ret, -ret_clip, ret_clip)

        # 2) Damping
        scale = np.exp(-lambda_damp * step_idx)
        pred_ret *= scale

        # 3) Convert to price
        last_price = full_history[-1]
        next_price = float(last_price * np.exp(pred_ret))

        # 4) Trend-based gating
        lookback = pricing.trend_lookback
        if len(full_history) > lookback:
            past_price = full_history[-lookback]
            current_price = full_history[-1]
            trend_ret = (current_price - past_price) / max(past_price, 1e-6)
        else:
            trend_ret = 0.0

        is_strong_uptrend = trend_ret > pricing.trend_ret_thresh
        is_strong_downtrend = trend_ret < -pricing.trend_ret_thresh

        # 5) Mean-revert
        if step_idx >= pricing.mean_revert_start:
            upper = fair_level * fair_up_mult
            lower = fair_level * fair_down_mult

            if (next_price > upper) and (not is_strong_uptrend):
                alpha_up = alpha_base * (0.7 if regime == "BULL" else 1.0)
                next_price = (1 - alpha_up) * next_price + alpha_up * upper
            elif next_price < lower:
                if regime == "BEAR" and is_strong_downtrend:
                    alpha_down = alpha_base * 0.7
                else:
                    alpha_down = alpha_base
                next_price = (1 - alpha_down) * next_price + alpha_down * lower

        prices[step_idx] = next_price
        full_history.append(next_price)

    return prices


# ============================================================
# UNCERTAINTY BAND
# ============================================================
def build_uncertainty_band(
    base_path: np.ndarray,
    resid_std: float,
    resid_mean: float = 0.0,
    z_score: float = 1.28155,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build uncertainty band (90% confidence)"""
    H = len(base_path)
    steps = np.arange(1, H + 1, dtype=float)
    base_log = np.log(base_path + 1e-8)

    sigma_h = resid_std * np.sqrt(steps)
    lower_log = base_log - z_score * sigma_h
    upper_log = base_log + z_score * sigma_h
    center_log = base_log

    return np.exp(center_log), np.exp(lower_log), np.exp(upper_log)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================
def train_final_base_model(df_train: pd.DataFrame, stl_period: int, horizon: int) -> FinalBaseModel:
    """Train final model on full training data"""
    df_feat, df_model, feature_cols = build_modeling_df(df_train, stl_period=stl_period, horizon=horizon)
    xgb, scaler_X, resid_std, resid_mean = train_xgb_on_dfmodel(df_model, feature_cols)

    return FinalBaseModel(
        xgb=xgb,
        scaler_X=scaler_X,
        feature_cols=feature_cols,
        df_train=df_train.sort_values("time").reset_index(drop=True),
        df_feat=df_feat,
        train_end_date=df_train["time"].max(),
        resid_std=resid_std,
        resid_mean=resid_mean,
    )


# ============================================================
# MAIN FORECAST FUNCTION
# ============================================================
@st.cache_data
def run_forecast(
    df_train: pd.DataFrame,
    total_predict_days: int,
    stl_period: int,
    pricing_params: dict,
    progress_callback=None,
):
    """Run the full forecast pipeline"""
    horizon = 1
    
    # Create pricing params
    pricing = PricingParams(**pricing_params)
    
    # Train final model
    if progress_callback:
        progress_callback(0.2, "Training XGBoost model...")
    
    final_base = train_final_base_model(
        df_train=df_train,
        stl_period=stl_period,
        horizon=horizon,
    )
    
    # Build raw hybrid path
    if progress_callback:
        progress_callback(0.5, "Building hybrid forecast path...")
    
    raw_base_prices, raw_rets = build_raw_base_path_hybrid(
        df_hist=final_base.df_train,
        xgb=final_base.xgb,
        scaler_X=final_base.scaler_X,
        feature_cols=final_base.feature_cols,
        total_days=total_predict_days,
        stl_period=stl_period,
    )
    
    hist_close = final_base.df_train["close"].values.astype(float)
    
    # Apply pricing layer
    if progress_callback:
        progress_callback(0.7, "Applying pricing layer...")
    
    base_path = apply_pricing_on_raw_path(
        hist_close=hist_close,
        df_feat_hist=final_base.df_feat,
        raw_rets=raw_rets,
        pricing=pricing,
    )
    
    # Trend model
    df_trend = final_base.df_train.sort_values("time").reset_index(drop=True)
    df_trend["time_idx"] = np.arange(len(df_trend))
    lr = LinearRegression()
    lr.fit(df_trend[["time_idx"]].values, df_trend["close"].values.astype(float))
    last_idx = int(df_trend["time_idx"].iloc[-1])
    future_idx = np.arange(last_idx + 1, last_idx + 1 + total_predict_days).reshape(-1, 1)
    trend_price = lr.predict(future_idx)
    
    # Uncertainty band
    if progress_callback:
        progress_callback(0.9, "Computing uncertainty bands...")
    
    uncert_center, uncert_lower, uncert_upper = build_uncertainty_band(
        base_path,
        final_base.resid_std,
        final_base.resid_mean,
    )
    
    # Central deterministic
    central_det = 0.7 * base_path + 0.25 * trend_price + 0.05 * uncert_center
    bull = np.maximum(base_path, trend_price)
    bear = np.minimum(base_path, trend_price)
    
    # Detect regime
    regime = detect_regime(hist_close, final_base.df_feat)
    
    # Future dates
    train_end_date = final_base.train_end_date
    start_date = train_end_date + BDay(1)
    future_dates = pd.bdate_range(start=start_date, periods=total_predict_days)
    
    # Create output dataframe
    forecast_df = pd.DataFrame({
        "time": future_dates,
        "base": base_path,
        "trend": trend_price.flatten(),
        "central_det": central_det,
        "uncert_center": uncert_center,
        "uncert_lower": uncert_lower,
        "uncert_upper": uncert_upper,
        "bull": bull,
        "bear": bear,
    })
    
    return {
        "forecast_df": forecast_df,
        "train_df": df_train,
        "train_end_date": train_end_date,
        "regime": regime,
        "resid_std": final_base.resid_std,
    }


# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    # Header
    st.title("📈 FPT Stock Price Forecast")
    st.markdown("""
    **Hybrid Model**: Math Backbone + XGBoost Residual + Pricing Layer (Regime-Aware)
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data upload
    st.sidebar.subheader("📁 Data Input")
    
    # Option to use sample data or upload
    data_option = st.sidebar.radio(
        "Data Source",
        ["Upload CSV", "Use Sample Data URL"]
    )
    
    df_train = None
    
    if data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload FPT_train.csv",
            type=["csv"],
            help="CSV file with columns: time, open, high, low, close, volume"
        )
        if uploaded_file is not None:
            df_train = pd.read_csv(uploaded_file)
            df_train["time"] = pd.to_datetime(df_train["time"])
            df_train = df_train.sort_values("time").reset_index(drop=True)
    else:
        # Sample data URL from Google Drive
        data_url = st.sidebar.text_input(
            "Data URL (Google Drive ID)",
            value="1l2TtEaWrp4yieMDWE4Cmehnf5mLx3rop",
            help="Google Drive file ID for FPT_train.csv"
        )
        if st.sidebar.button("Load Data"):
            try:
                import gdown
                url = f"https://drive.google.com/uc?id={data_url}"
                df_train = pd.read_csv(url)
                df_train["time"] = pd.to_datetime(df_train["time"])
                df_train = df_train.sort_values("time").reset_index(drop=True)
                st.sidebar.success("Data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")
    
    # Model Parameters
    st.sidebar.subheader("🔧 Model Parameters")
    
    total_predict_days = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=10, max_value=200, value=100, step=10
    )
    
    stl_period = st.sidebar.slider(
        "STL Period",
        min_value=10, max_value=40, value=20, step=5
    )
    
    # Pricing Layer Parameters
    st.sidebar.subheader("💰 Pricing Layer")
    
    with st.sidebar.expander("Advanced Pricing Parameters"):
        ret_clip_quantile = st.slider("Return Clip Quantile", 0.90, 0.999, 0.99, 0.005)
        half_life_days = st.slider("Half-Life (days)", 20, 150, 60, 5)
        mean_revert_alpha = st.slider("Mean Revert Alpha", 0.01, 0.15, 0.06, 0.01)
        mean_revert_start = st.slider("Mean Revert Start Day", 10, 80, 40, 5)
        fair_up_mult = st.slider("Fair Up Multiplier", 1.1, 1.8, 1.4, 0.05)
        fair_down_mult = st.slider("Fair Down Multiplier", 0.5, 0.95, 0.75, 0.05)
        trend_lookback = st.slider("Trend Lookback", 15, 80, 30, 5)
        trend_ret_thresh = st.slider("Trend Return Threshold", 0.05, 0.35, 0.18, 0.02)
    
    pricing_params = {
        "ret_clip_quantile": ret_clip_quantile,
        "half_life_days": half_life_days,
        "mean_revert_alpha": mean_revert_alpha,
        "mean_revert_start": mean_revert_start,
        "fair_up_mult": fair_up_mult,
        "fair_down_mult": fair_down_mult,
        "trend_lookback": trend_lookback,
        "trend_ret_thresh": trend_ret_thresh,
    }
    
    # Main content
    if df_train is not None:
        # Data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(df_train))
        with col2:
            st.metric("Start Date", df_train["time"].min().strftime("%Y-%m-%d"))
        with col3:
            st.metric("End Date", df_train["time"].max().strftime("%Y-%m-%d"))
        
        # Show sample data
        with st.expander("📊 Preview Training Data"):
            st.dataframe(df_train.tail(20))
        
        # Run forecast button
        if st.button("🚀 Run Forecast", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)
            
            try:
                results = run_forecast(
                    df_train=df_train,
                    total_predict_days=total_predict_days,
                    stl_period=stl_period,
                    pricing_params=pricing_params,
                    progress_callback=update_progress,
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Forecast complete!")
                
                # Store results in session state
                st.session_state["results"] = results
                
            except Exception as e:
                st.error(f"Error during forecast: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Display results if available
        if "results" in st.session_state:
            results = st.session_state["results"]
            forecast_df = results["forecast_df"]
            train_df = results["train_df"]
            
            st.success(f"**Detected Regime:** {results['regime']}")
            
            # Create interactive plot
            st.subheader("📈 Forecast Visualization")
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=train_df["time"],
                y=train_df["close"],
                mode="lines",
                name="Historical",
                line=dict(color="blue", width=1),
                opacity=0.6
            ))
            
            # Forecast lines
            fig.add_trace(go.Scatter(
                x=forecast_df["time"],
                y=forecast_df["central_det"],
                mode="lines",
                name="Central Forecast",
                line=dict(color="red", width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df["time"],
                y=forecast_df["base"],
                mode="lines",
                name="Base (Hybrid + Pricing)",
                line=dict(color="orange", width=1, dash="dot")
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df["time"],
                y=forecast_df["trend"],
                mode="lines",
                name="Trend (Linear)",
                line=dict(color="green", width=1, dash="dash")
            ))
            
            # Uncertainty band
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df["time"], forecast_df["time"][::-1]]),
                y=pd.concat([forecast_df["uncert_upper"], forecast_df["uncert_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="90% Confidence",
                showlegend=True
            ))
            
            # Train end line
            fig.add_vline(
                x=results["train_end_date"],
                line=dict(color="gray", dash="dash"),
                annotation_text="Train End"
            )
            
            fig.update_layout(
                title="FPT Stock Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price (VND)",
                hovermode="x unified",
                height=600,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast metrics
            st.subheader("📊 Forecast Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            last_price = train_df["close"].iloc[-1]
            forecast_50 = forecast_df["central_det"].iloc[49] if len(forecast_df) >= 50 else forecast_df["central_det"].iloc[-1]
            forecast_100 = forecast_df["central_det"].iloc[-1]
            
            with col1:
                st.metric("Last Price", f"{last_price:,.0f}")
            with col2:
                ret_50 = (forecast_50 / last_price - 1) * 100
                st.metric("50-Day Forecast", f"{forecast_50:,.0f}", f"{ret_50:+.1f}%")
            with col3:
                ret_100 = (forecast_100 / last_price - 1) * 100
                st.metric("100-Day Forecast", f"{forecast_100:,.0f}", f"{ret_100:+.1f}%")
            with col4:
                st.metric("Residual Std", f"{results['resid_std']:.4f}")
            
            # Download section
            st.subheader("📥 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name="FPT_forecast.csv",
                    mime="text/csv"
                )
            
            with col2:
                submission = pd.DataFrame({
                    "id": np.arange(1, len(forecast_df) + 1),
                    "close": forecast_df["central_det"].values
                })
                csv_sub = submission.to_csv(index=False)
                st.download_button(
                    label="Download Submission CSV",
                    data=csv_sub,
                    file_name="submission.csv",
                    mime="text/csv"
                )
            
            # Detailed forecast table
            with st.expander("📋 Detailed Forecast Data"):
                st.dataframe(forecast_df.round(2))
    
    else:
        st.info("👆 Please upload data or load from URL to start forecasting.")
        
        # Show example format
        st.subheader("📝 Expected Data Format")
        example_df = pd.DataFrame({
            "time": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [100.0, 101.5, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.5, 101.0],
            "close": [101.0, 102.5, 103.0],
            "volume": [1000000, 1200000, 1100000],
            "symbol": ["FPT", "FPT", "FPT"]
        })
        st.dataframe(example_df)


if __name__ == "__main__":
    main()
