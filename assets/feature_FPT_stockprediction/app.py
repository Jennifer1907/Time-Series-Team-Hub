"""
FPT Stock Prediction Demo App
===============================
Interactive Streamlit dashboard for FPT stock price forecasting
using Hybrid ML approach (Math Backbone + XGBoost Residual Learning)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import os
import sys
import subprocess

# Page configuration
st.set_page_config(
    page_title="FPT Stock Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_training_data():
    """Load historical training data"""
    if os.path.exists("FPT_train.csv"):
        df = pd.read_csv("FPT_train.csv")
        df['time'] = pd.to_datetime(df['time'])
        return df
    return None

@st.cache_data
def load_forecast_data():
    """Load forecast results"""
    if os.path.exists("FPT_forecast.csv"):
        df = pd.read_csv("FPT_forecast.csv")
        df['time'] = pd.to_datetime(df['time'])
        return df
    return None

@st.cache_data
def load_forecast_data2():
    """Load custom forecast results (FPT_forecast2.csv)"""
    if os.path.exists("FPT_forecast2.csv"):
        df = pd.read_csv("FPT_forecast2.csv")
        df['time'] = pd.to_datetime(df['time'])
        return df
    return None

@st.cache_data
def load_realdata(start_date=None, end_date=None):
    """Load real data for comparison if available

    Args:
        start_date: Optional start date to filter data (datetime or string)
        end_date: Optional end date to filter data (datetime or string)

    Returns:
        DataFrame with real stock data, optionally filtered by date range
    """
    # First try to load FPT_realdata.csv
    if os.path.exists("FPT_realdata.csv"):
        try:
            df = pd.read_csv("FPT_realdata.csv")
            # Check if it's the Vietnamese format
            if 'Ngày' in df.columns:
                # Convert Vietnamese format
                df['time'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y', errors='coerce')
                # Clean price data (remove commas and convert)
                df['close'] = df['Lần cuối'].str.replace(',', '').astype(float)
                df['open'] = df['Mở'].str.replace(',', '').astype(float)
                df['high'] = df['Cao'].str.replace(',', '').astype(float)
                df['low'] = df['Thấp'].str.replace(',', '').astype(float)
                df = df[['time', 'open', 'high', 'low', 'close']].dropna()
            elif 'time' in df.columns:
                # Standard format with time column
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                # Ensure numeric columns are float type
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                # Select only required columns if they exist
                available_cols = ['time'] + [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]
                df = df[available_cols].dropna(subset=['time', 'close'])

            # Apply date filtering if specified
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                df = df[df['time'] >= start_date]
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                df = df[df['time'] <= end_date]

            return df.sort_values('time').reset_index(drop=True)
        except Exception as e:
            st.warning(f"Could not load FPT_realdata.csv: {e}")

    # Fallback to FPT_train.csv
    if os.path.exists("FPT_train.csv"):
        try:
            df = pd.read_csv("FPT_train.csv")
            # Check if it's the Vietnamese format
            if 'Ngày' in df.columns:
                # Convert Vietnamese format
                df['time'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y', errors='coerce')
                # Clean price data (remove commas and convert)
                df['close'] = df['Lần cuối'].str.replace(',', '').astype(float)
                df['open'] = df['Mở'].str.replace(',', '').astype(float)
                df['high'] = df['Cao'].str.replace(',', '').astype(float)
                df['low'] = df['Thấp'].str.replace(',', '').astype(float)
                df = df[['time', 'open', 'high', 'low', 'close']].dropna()
            elif 'time' in df.columns:
                # Standard format with time column
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                # Ensure numeric columns are float type
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                # Select only required columns if they exist
                available_cols = ['time'] + [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]
                df = df[available_cols].dropna(subset=['time', 'close'])

            # Apply date filtering if specified
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                df = df[df['time'] >= start_date]
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                df = df[df['time'] <= end_date]

            return df.sort_values('time').reset_index(drop=True)
        except Exception as e:
            st.warning(f"Could not load real data: {e}")
            return None
    return None

@st.cache_data
def load_model_params():
    """Load saved model parameters"""
    if os.path.exists("saved_params.json"):
        with open("saved_params.json", "r") as f:
            return json.load(f)
    return None

def calculate_mse(forecast_df, real_df, forecast_col='central_det', real_col='close'):
    """Calculate Mean Squared Error between forecast and real data

    Args:
        forecast_df: DataFrame with forecast data
        real_df: DataFrame with real data
        forecast_col: Column name in forecast_df to compare
        real_col: Column name in real_df to compare

    Returns:
        dict: Dictionary with MSE, RMSE, MAE, MAPE and count of matching data points
    """
    if forecast_df is None or real_df is None:
        return None

    try:
        # Merge on time column to align dates
        merged = pd.merge(
            forecast_df[['time', forecast_col]],
            real_df[['time', real_col]],
            on='time',
            how='inner'
        )

        if len(merged) == 0:
            return {'mse': None, 'rmse': None, 'mae': None, 'mape': None, 'count': 0, 'message': 'No overlapping dates'}

        # Calculate MSE and RMSE
        mse = np.mean((merged[forecast_col] - merged[real_col]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(merged[forecast_col] - merged[real_col]))
        mape = np.mean(np.abs((merged[real_col] - merged[forecast_col]) / merged[real_col])) * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'count': len(merged),
            'merged_df': merged
        }
    except Exception as e:
        return {'mse': None, 'rmse': None, 'mae': None, 'mape': None, 'count': 0, 'message': f'Error: {str(e)}'}

# ============================================================
# FORECAST RE-RUN FUNCTIONS
# ============================================================

def save_custom_params(params_dict):
    """Save custom parameters to a temporary file for the model to use"""
    temp_params = {
        "my_config": {
            "horizon": params_dict.get("horizon", 100),
            "ret_clip_quantile": params_dict.get("ret_clip_quantile", 0.97),
            "half_life": params_dict.get("half_life_days", 60),
            "mean_revert_alpha": params_dict.get("mean_revert_alpha", 0.02),
            "mean_revert_start": params_dict.get("mean_revert_start", 40),
            "fair_up_mult": params_dict.get("fair_up_mult", 1.4),
            "fair_down_mult": params_dict.get("fair_down_mult", 0.75),
            "trend_lookback": params_dict.get("trend_lookback", 30),
            "trend_ret_thresh": params_dict.get("trend_ret_thresh", 0.18),
        }
    }
    with open("custom_params.json", "w") as f:
        json.dump(temp_params, f, indent=2)

def run_forecast_with_params(params_dict, status_text=None, progress_bar=None):
    """
    Run the forecast model with custom parameters

    Args:
        params_dict: Dictionary containing pricing layer parameters
        status_text: Streamlit text object for status updates
        progress_bar: Streamlit progress bar object (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Save parameters
        if status_text:
            status_text.text("📝 Saving custom parameters...")
        if progress_bar:
            progress_bar.progress(0.05)
        save_custom_params(params_dict)

        # Add the current directory to Python path if needed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        if status_text:
            status_text.text("🚀 Starting forecast model...")
        if progress_bar:
            progress_bar.progress(0.1)

        # Execute the forecast script with real-time output
        import threading
        import time

        process = subprocess.Popen(
            [sys.executable, "finalpm6_kaggle.py"],
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Track progress based on output
        progress_stages = {
            "Loading": 0.15,
            "FE": 0.25,
            "XGB": 0.40,
            "CV": 0.55,
            "cutoff": 0.60,
            "Train": 0.65,
            "FINAL": 0.75,
            "forecast": 0.85,
            "Saved": 0.95
        }

        current_progress = 0.1
        output_lines = []

        # Read output in real-time
        if process.stdout:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    output_lines.append(line.strip())

                    # Update progress based on keywords in output
                    for keyword, progress_val in progress_stages.items():
                        if keyword in line and progress_val > current_progress:
                            current_progress = progress_val
                            if progress_bar:
                                progress_bar.progress(current_progress)
                            break

                    # Update status text with last meaningful line
                    if status_text and line.strip() and not line.strip().startswith('['):
                        # Show condensed status
                        if "XGB" in line:
                            status_text.text("🤖 Training XGBoost model...")
                        elif "CV" in line or "cutoff" in line:
                            status_text.text("📊 Running cross-validation...")
                        elif "FINAL" in line or "Train" in line:
                            status_text.text("🎯 Training final model...")
                        elif "forecast" in line.lower() or "path" in line.lower():
                            status_text.text("🔮 Generating forecast...")
                        elif "Saved" in line:
                            status_text.text("💾 Saving results...")        # Wait for process to complete
        return_code = process.wait(timeout=300)

        if progress_bar:
            progress_bar.progress(1.0)

        if return_code == 0:
            if status_text:
                status_text.text("✅ Forecast completed successfully!")
            # Clear cached data to reload new forecast
            st.cache_data.clear()
            return True
        else:
            if status_text:
                status_text.text("❌ Forecast failed!")
            # Show last few lines of output for debugging
            st.error(f"Forecast failed. Last output:\n" + "\n".join(output_lines[-10:]))
            return False

    except subprocess.TimeoutExpired:
        if status_text:
            status_text.text("⏱️ Forecast timed out!")
        st.error("Forecast timed out after 5 minutes. Please try again with different parameters.")
        return False
    except Exception as e:
        if status_text:
            status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Error running forecast: {str(e)}")
        return False# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_forecast_plot(train_df, forecast_df, real_df=None, show_uncertainty=True, show_real_overlay=False):
    """Create main forecast visualization with historical and predicted data

    Args:
        train_df: Historical training data
        forecast_df: Forecast predictions
        real_df: Real data for comparison (optional)
        show_uncertainty: Show uncertainty bands
        show_real_overlay: Show real data overlay on forecast period
    """
    fig = go.Figure()

    # Historical data (last 200 days for context)
    if train_df is not None and len(train_df) > 0:
        train_subset = train_df.tail(200)
        fig.add_trace(go.Scatter(
            x=train_subset['time'],
            y=train_subset['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: %{y:.2f}<extra></extra>'
        ))

    # Forecast data
    if forecast_df is not None:
        # Central prediction
        fig.add_trace(go.Scatter(
            x=forecast_df['time'],
            y=forecast_df['central_det'],
            mode='lines',
            name='Forecast (Central)',
            line=dict(color='#ff7f0e', width=3, dash='solid'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: %{y:.2f}<extra></extra>'
        ))

        # Uncertainty bands
        if show_uncertainty and 'uncert_lower_ana' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['time'],
                y=forecast_df['uncert_upper_ana'],
                mode='lines',
                name='Upper Bound (90%)',
                line=dict(width=0),
                showlegend=False,
                hovertemplate='<b>Upper</b>: %{y:.2f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['time'],
                y=forecast_df['uncert_lower_ana'],
                mode='lines',
                name='Lower Bound (90%)',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                hovertemplate='<b>Lower</b>: %{y:.2f}<extra></extra>'
            ))

        # Base path (Hybrid + Pricing)
        fig.add_trace(go.Scatter(
            x=forecast_df['time'],
            y=forecast_df['base'],
            mode='lines',
            name='Base Path (Hybrid)',
            line=dict(color='#2ca02c', width=1.5, dash='dot'),
            hovertemplate='<b>Base</b>: %{y:.2f}<extra></extra>'
        ))

        # Trend line
        fig.add_trace(go.Scatter(
            x=forecast_df['time'],
            y=forecast_df['trend'],
            mode='lines',
            name='Trend (Linear)',
            line=dict(color='#d62728', width=1.5, dash='dash'),
            hovertemplate='<b>Trend</b>: %{y:.2f}<extra></extra>'
        ))

    # Real data overlay (if available and enabled)
    if show_real_overlay and real_df is not None and forecast_df is not None:
        # Filter real data to match forecast date range
        forecast_start = forecast_df['time'].min()
        forecast_end = forecast_df['time'].max()
        real_filtered = real_df[
            (real_df['time'] >= forecast_start) &
            (real_df['time'] <= forecast_end)
        ]

        if len(real_filtered) > 0:
            fig.add_trace(go.Scatter(
                x=real_filtered['time'],
                y=real_filtered['close'],
                mode='lines+markers',
                name='Actual Price (Real Data)',
                line=dict(color='#9467bd', width=3),
                marker=dict(size=6, symbol='circle'),
                hovertemplate='<b>Actual</b>: %{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title='FPT Stock Price Forecast - 100 Days Ahead',
        xaxis_title='Date',
        yaxis_title='Price (VND)',
        hovermode='x unified',
        height=600,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        dragmode='pan'
    )

    # Enable scroll zoom
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    return fig

def create_forecast_candlestick(train_df, forecast_df, real_df=None, show_uncertainty=True, timeframe='1D', show_real_overlay=False):
    """Create candlestick forecast visualization with OHLC data and predictions

    Args:
        train_df: Training dataframe with OHLC data
        forecast_df: Forecast dataframe with predictions
        real_df: Real data for comparison (optional)
        show_uncertainty: Show uncertainty bands
        timeframe: '1D' (daily), '1W' (weekly), '1M' (monthly), '3M' (quarterly)
        show_real_overlay: Show real data overlay on forecast period
    """
    # Aggregate OHLCV data based on timeframe
    def aggregate_ohlcv(df, freq):
        df_copy = df.copy()
        df_copy.set_index('time', inplace=True)
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }
        if 'volume' in df_copy.columns:
            agg_dict['volume'] = 'sum'
        aggregated = df_copy.resample(freq).agg(agg_dict).dropna()
        aggregated.reset_index(inplace=True)
        return aggregated

    # Map timeframe to pandas frequency
    freq_map = {
        '1D': 'D',
        '1W': 'W',
        '1M': 'M',
        '3M': 'Q'
    }

    timeframe_labels = {
        '1D': 'Daily',
        '1W': 'Weekly',
        '1M': 'Monthly',
        '3M': 'Quarterly'
    }

    freq = freq_map.get(timeframe, 'D')

    fig = go.Figure()

    # Historical candlestick data (last 200 days context)
    if train_df is not None and len(train_df) > 0:
        if all(col in train_df.columns for col in ['open', 'high', 'low', 'close']):
            train_subset = train_df.tail(200)

            # Aggregate if needed
            if timeframe != '1D':
                train_subset = aggregate_ohlcv(train_subset, freq)

            fig.add_trace(go.Candlestick(
                x=train_subset['time'],
                open=train_subset['open'],
                high=train_subset['high'],
                low=train_subset['low'],
                close=train_subset['close'],
                name=f'Historical ({timeframe_labels[timeframe]})',
                increasing=dict(line=dict(color='#00CC96', width=1), fillcolor='#00CC96'),
                decreasing=dict(line=dict(color='#EF553B', width=1), fillcolor='#EF553B'),
                whiskerwidth=0
            ))

    # Forecast data overlays
    if forecast_df is not None:
        # Aggregate forecast data if needed
        if timeframe != '1D':
            forecast_copy = forecast_df.copy()
            forecast_copy.set_index('time', inplace=True)
            forecast_agg = forecast_copy.resample(freq).agg({
                'central_det': 'mean',
                'base': 'mean',
                'trend': 'mean',
                'uncert_lower_ana': 'min',
                'uncert_upper_ana': 'max'
            }).dropna()
            forecast_agg.reset_index(inplace=True)
            forecast_data = forecast_agg
        else:
            forecast_data = forecast_df

        # Central prediction line
        fig.add_trace(go.Scatter(
            x=forecast_data['time'],
            y=forecast_data['central_det'],
            mode='lines',
            name='Forecast (Central)',
            line=dict(color='#ff7f0e', width=3, dash='solid'),
            hovertemplate='<b>Forecast</b>: %{y:.2f}<extra></extra>'
        ))

        # Uncertainty bands
        if show_uncertainty and 'uncert_lower_ana' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['time'],
                y=forecast_data['uncert_upper_ana'],
                mode='lines',
                name='Upper Bound (90%)',
                line=dict(width=0),
                showlegend=False,
                hovertemplate='<b>Upper</b>: %{y:.2f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['time'],
                y=forecast_data['uncert_lower_ana'],
                mode='lines',
                name='Lower Bound (90%)',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                hovertemplate='<b>Lower</b>: %{y:.2f}<extra></extra>'
            ))

    # Real data overlay (if available and enabled)
    if show_real_overlay and real_df is not None and forecast_df is not None:
        # Filter real data to match forecast date range
        forecast_start = forecast_df['time'].min()
        forecast_end = forecast_df['time'].max()
        real_filtered = real_df[
            (real_df['time'] >= forecast_start) &
            (real_df['time'] <= forecast_end)
        ]

        if len(real_filtered) > 0 and all(col in real_filtered.columns for col in ['open', 'high', 'low', 'close']):
            # Aggregate real data if needed
            if timeframe != '1D':
                real_filtered = aggregate_ohlcv(real_filtered, freq)

            # Add candlestick overlay for real data
            fig.add_trace(go.Candlestick(
                x=real_filtered['time'],
                open=real_filtered['open'],
                high=real_filtered['high'],
                low=real_filtered['low'],
                close=real_filtered['close'],
                name='Actual Price (Real Data)',
                increasing=dict(line=dict(color='#9467bd', width=2), fillcolor='rgba(148, 103, 189, 0.5)'),
                decreasing=dict(line=dict(color='#c44e52', width=2), fillcolor='rgba(196, 78, 82, 0.5)'),
                whiskerwidth=0.5
            ))

    # Add volume if available (only for historical data)
    if train_df is not None and 'volume' in train_df.columns:
        train_subset = train_df.tail(200)
        if timeframe != '1D':
            train_subset = aggregate_ohlcv(train_subset, freq)

        fig.add_trace(go.Bar(
            x=train_subset['time'],
            y=train_subset['volume'],
            name='Volume',
            marker=dict(color='rgba(128, 128, 128, 0.25)'),
            yaxis='y2',
            showlegend=True
        ))

    fig.update_layout(
        title=f'FPT Stock Forecast - {timeframe_labels[timeframe]} Candlesticks (100 Days Ahead)',
        xaxis_title='Date',
        yaxis_title='Price (VND)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=650,
        template='plotly_white',
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='date'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        dragmode='zoom',
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )

    # Enable scroll zoom
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    return fig

def create_line_chart(train_df, real_df=None):
    """Create optimized line chart showing historical price data"""
    if train_df is None or 'close' not in train_df.columns:
        return None

    # Optimize: Sample data to reduce rendering load
    total_points = len(train_df)
    if total_points > 500:
        # Keep last 300 points at full resolution, sample earlier data
        recent_data = train_df.tail(300).copy()
        older_data = train_df.head(total_points - 300).iloc[::2].copy()
        optimized_df = pd.concat([older_data, recent_data]).reset_index(drop=True)
    else:
        optimized_df = train_df.copy()

    fig = go.Figure()

    # Main price line from training data
    fig.add_trace(go.Scatter(
        x=optimized_df['time'],
        y=optimized_df['close'],
        mode='lines',
        name='FPT Stock Price',
        line=dict(color='#1f77b4', width=1.5, shape='linear'),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: %{y:.2f}<extra></extra>'
    ))

    # Add moving averages (calculated on optimized data)
    if len(optimized_df) >= 20:
        optimized_df['MA20'] = optimized_df['close'].rolling(window=20).mean()
        optimized_df['MA60'] = optimized_df['close'].rolling(window=60).mean()

        fig.add_trace(go.Scatter(
            x=optimized_df['time'],
            y=optimized_df['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='#ff7f0e', width=1, dash='dash', shape='linear'),
            hovertemplate='<b>MA20</b>: %{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=optimized_df['time'],
            y=optimized_df['MA60'],
            mode='lines',
            name='MA60',
            line=dict(color='#2ca02c', width=1, dash='dot', shape='linear'),
            hovertemplate='<b>MA60</b>: %{y:.2f}<extra></extra>'
        ))


    # Simplified volume (sample more aggressively)
    if 'volume' in optimized_df.columns:
        volume_sample = optimized_df.iloc[::3] if len(optimized_df) > 300 else optimized_df
        fig.add_trace(go.Bar(
            x=volume_sample['time'],
            y=volume_sample['volume'],
            name='Volume',
            marker=dict(color='rgba(128, 128, 128, 0.25)'),
            yaxis='y2',
            showlegend=True
        ))

    # Update layout - optimized for performance
    fig.update_layout(
        title='FPT Stock Price History (2020-2025) - Line Chart',
        xaxis_title='Date',
        yaxis_title='Price (VND)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=650,  # Slightly reduced height
        template='plotly_white',
        hovermode='x unified',
        xaxis_rangeslider_visible=False,  # Disable rangeslider for better performance
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(label="All", step="all")
                ]),
                bgcolor='rgba(150, 150, 150, 0.1)',
                font=dict(size=10)
            ),
            type='date'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        dragmode='zoom',
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )

    # Enable scroll zoom
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    return fig

def create_candlestick_chart(train_df, real_df=None, timeframe='1D'):
    """Create optimized candlestick chart showing historical OHLC data

    Args:
        train_df: Training dataframe with OHLC data
        timeframe: '1D' (daily), '1W' (weekly), '1M' (monthly), '3M' (quarterly)
    """
    if train_df is None or not all(col in train_df.columns for col in ['open', 'high', 'low', 'close']):
        return None

    # Aggregate data based on timeframe
    def aggregate_ohlcv(df, freq):
        """Aggregate OHLCV data to specified frequency"""
        df_copy = df.copy()
        df_copy.set_index('time', inplace=True)

        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }

        if 'volume' in df_copy.columns:
            agg_dict['volume'] = 'sum'

        aggregated = df_copy.resample(freq).agg(agg_dict).dropna()
        aggregated.reset_index(inplace=True)
        return aggregated

    # Map timeframe to pandas frequency
    freq_map = {
        '1D': 'D',   # Daily (no aggregation)
        '1W': 'W',   # Weekly
        '1M': 'M',   # Monthly (end of month)
        '3M': 'Q'    # Quarterly
    }

    timeframe_labels = {
        '1D': 'Daily',
        '1W': 'Weekly',
        '1M': 'Monthly',
        '3M': 'Quarterly'
    }

    freq = freq_map.get(timeframe, 'D')

    # Aggregate training data
    if timeframe != '1D':
        optimized_df = aggregate_ohlcv(train_df, freq)
    else:
        # For daily, apply original optimization
        total_points = len(train_df)
        if total_points > 500:
            recent_data = train_df.tail(300).copy()
            older_data = train_df.head(total_points - 300).iloc[::2].copy()
            optimized_df = pd.concat([older_data, recent_data]).reset_index(drop=True)
        else:
            optimized_df = train_df.copy()

    fig = go.Figure()

    # Main candlestick from training data
    fig.add_trace(go.Candlestick(
        x=optimized_df['time'],
        open=optimized_df['open'],
        high=optimized_df['high'],
        low=optimized_df['low'],
        close=optimized_df['close'],
        name=f'FPT Stock ({timeframe_labels[timeframe]})',
        increasing=dict(line=dict(color='#00CC96', width=1), fillcolor='#00CC96'),
        decreasing=dict(line=dict(color='#EF553B', width=1), fillcolor='#EF553B'),
        whiskerwidth=0
    ))

    # Add volume bars
    if 'volume' in optimized_df.columns:
        # Sample volume based on timeframe
        if timeframe == '1D' and len(optimized_df) > 300:
            volume_sample = optimized_df.iloc[::3]
        else:
            volume_sample = optimized_df

        fig.add_trace(go.Bar(
            x=volume_sample['time'],
            y=volume_sample['volume'],
            name='Volume',
            marker=dict(color='rgba(128, 128, 128, 0.25)'),
            yaxis='y2',
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        title=f'FPT Stock Price History - {timeframe_labels[timeframe]} Candlesticks',
        xaxis_title='Date',
        yaxis_title='Price (VND)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=650,
        template='plotly_white',
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='date'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        dragmode='pan',
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )

    # Enable scroll zoom
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    return fig

def create_regime_plot(forecast_df):
    """Visualize bull/bear regime predictions"""
    if forecast_df is None or 'bull' not in forecast_df.columns:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Bull Signal', 'Bear Signal'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # Bull signal
    fig.add_trace(go.Scatter(
        x=forecast_df['time'],
        y=forecast_df['bull'],
        mode='lines',
        name='Bull',
        line=dict(color='green', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.2)'
    ), row=1, col=1)

    # Bear signal
    fig.add_trace(go.Scatter(
        x=forecast_df['time'],
        y=forecast_df['bear'],
        mode='lines',
        name='Bear',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white',
        title_text='Market Regime Indicators'
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)

    return fig

def create_price_distribution(forecast_df):
    """Show distribution of predicted prices"""
    if forecast_df is None:
        return None

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=forecast_df['central_det'],
        nbinsx=30,
        name='Central Prediction',
        marker_color='#ff7f0e',
        opacity=0.7
    ))

    if 'base' in forecast_df.columns:
        fig.add_trace(go.Histogram(
            x=forecast_df['base'],
            nbinsx=30,
            name='Base Path',
            marker_color='#2ca02c',
            opacity=0.5
        ))

    fig.update_layout(
        title='Distribution of Forecasted Prices',
        xaxis_title='Price (VND)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400,
        template='plotly_white'
    )

    return fig

def create_volatility_plot(train_df):
    """Calculate and plot rolling volatility"""
    if train_df is None or len(train_df) < 30:
        return None

    df = train_df.copy()
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252) * 100
    df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252) * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['time'].tail(200),
        y=df['volatility_20'].tail(200),
        mode='lines',
        name='20-Day Volatility',
        line=dict(color='orange', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['time'].tail(200),
        y=df['volatility_60'].tail(200),
        mode='lines',
        name='60-Day Volatility',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title='Historical Volatility Analysis',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        height=400,
        template='plotly_white'
    )

    return fig

def create_comparison_table(forecast_df):
    """Create comparison table for different prediction methods"""
    if forecast_df is None:
        return None

    summary = {
        'Method': ['Central Det', 'Base Path', 'Trend', 'Lower Bound', 'Upper Bound'],
        'Min Price': [
            forecast_df['central_det'].min(),
            forecast_df['base'].min(),
            forecast_df['trend'].min(),
            forecast_df['uncert_lower_ana'].min() if 'uncert_lower_ana' in forecast_df.columns else None,
            forecast_df['uncert_upper_ana'].min() if 'uncert_upper_ana' in forecast_df.columns else None
        ],
        'Max Price': [
            forecast_df['central_det'].max(),
            forecast_df['base'].max(),
            forecast_df['trend'].max(),
            forecast_df['uncert_lower_ana'].max() if 'uncert_lower_ana' in forecast_df.columns else None,
            forecast_df['uncert_upper_ana'].max() if 'uncert_upper_ana' in forecast_df.columns else None
        ],
        'Mean Price': [
            forecast_df['central_det'].mean(),
            forecast_df['base'].mean(),
            forecast_df['trend'].mean(),
            forecast_df['uncert_lower_ana'].mean() if 'uncert_lower_ana' in forecast_df.columns else None,
            forecast_df['uncert_upper_ana'].mean() if 'uncert_upper_ana' in forecast_df.columns else None
        ],
        'Final Price': [
            forecast_df['central_det'].iloc[-1],
            forecast_df['base'].iloc[-1],
            forecast_df['trend'].iloc[-1],
            forecast_df['uncert_lower_ana'].iloc[-1] if 'uncert_lower_ana' in forecast_df.columns else None,
            forecast_df['uncert_upper_ana'].iloc[-1] if 'uncert_upper_ana' in forecast_df.columns else None
        ]
    }

    return pd.DataFrame(summary)

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<div class="main-header">📈 FPT Stock Price Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hybrid ML Approach: Math Backbone + XGBoost Residual Learning</div>', unsafe_allow_html=True)

    # Load data
    train_df = load_training_data()
    forecast_df = load_forecast_data()
    real_df = load_realdata()
    params = load_model_params()

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Settings")

        # Model info
        st.subheader("Model Configuration")
        if params and 'my_config' in params:
            config = params['my_config']
            st.write(f"**Horizon:** {config.get('horizon', 100)} days")
            st.write(f"**Damping:** {config.get('damp', 0.98):.2f}")
            st.write(f"**Mean Revert α:** {config.get('mr_alpha', 0.02):.2f}")
            st.write(f"**Clip Quantile:** {config.get('clip_q', 0.97):.2f}")
            st.write(f"**Half-life:** {config.get('half_life', 60)} days")

        st.divider()

        # Display options
        st.subheader("Display Options")

        # Chart type selector
        chart_type = st.radio(
            "Historical Chart Type",
            options=["Candlestick", "Line Chart"],
            index=0,
            help="Choose between candlestick (OHLC) or line chart view"
        )

        # Timeframe selector
        timeframe_labels = ["1 Day", "1 Week", "1 Month", "3 Months"]
        timeframe_values = ["1D", "1W", "1M", "3M"]
        timeframe_idx = st.radio(
            "Candlestick Timeframe",
            options=range(len(timeframe_labels)),
            format_func=lambda x: timeframe_labels[x],
            index=0,
            help="Each candlestick represents the selected time period"
        )
        timeframe = timeframe_values[timeframe_idx]

        st.divider()
        st.subheader("Forecast Options")

        forecast_chart_type = st.radio(
            "Forecast Chart Type",
            options=["Line Chart", "Candlestick"],
            index=0,
            help="Choose visualization for 100-day forecast"
        )

        # Forecast Candlestick Timeframe selector (only shown for candlestick)
        if forecast_chart_type == "Candlestick":
            forecast_timeframe_idx = st.radio(
                "Forecast Candlestick Timeframe",
                options=range(len(timeframe_labels)),
                format_func=lambda x: timeframe_labels[x],
                index=1,
                help="Timeframe for forecast candlesticks"
            )
            forecast_timeframe = timeframe_values[forecast_timeframe_idx]
        else:
            forecast_timeframe = '1W'
            forecast_timeframe_idx = 1

        show_uncertainty = st.checkbox("Show Uncertainty Bands", value=True)
        show_regime = st.checkbox("Show Bull/Bear Regime", value=False)
        show_real_overlay = st.checkbox(
            "Show Real Data Overlay",
            value=False,
            help="Overlay actual prices from FPT_realdata.csv on forecast period for accuracy comparison"
        )

        st.divider()

        # ============================================================
        # CUSTOM FORECAST PARAMETERS IN SIDEBAR
        # ============================================================
        st.subheader("🔧 Custom Forecast Parameters")
        st.markdown("*Adjust parameters and re-run forecast*")

        with st.expander("📊 Pricing Layer", expanded=False):
            st.markdown("#### 📈 Return & Volatility")
            ret_clip_quantile = st.slider(
                "Return Clip Quantile",
                min_value=0.95,
                max_value=0.999,
                value=0.97,
                step=0.001,
                help="Quantile for clipping extreme returns"
            )

            half_life_days = st.slider(
                "Half-Life (days)",
                min_value=10,
                max_value=120,
                value=60,
                step=5,
                help="Half-life for exponential damping"
            )

            mean_revert_alpha = st.slider(
                "Mean Revert Alpha",
                min_value=0.02,
                max_value=0.10,
                value=0.06,
                step=0.01,
                help="Mean reversion strength coefficient"
            )

            mean_revert_start = st.slider(
                "Mean Revert Start Day",
                min_value=10,
                max_value=65,
                value=40,
                step=5,
                help="Day to start applying mean reversion"
            )

            st.markdown("#### 🎯 Fair Value & Trend")
            fair_up_mult = st.slider(
                "Fair Up Multiplier",
                min_value=1.25,
                max_value=1.60,
                value=1.40,
                step=0.05,
                help="Multiplier for upward fair value adjustment"
            )

            fair_down_mult = st.slider(
                "Fair Down Multiplier",
                min_value=0.65,
                max_value=0.90,
                value=0.75,
                step=0.05,
                help="Multiplier for downward fair value adjustment"
            )

            trend_lookback = st.slider(
                "Trend Lookback",
                min_value=25,
                max_value=60,
                value=30,
                step=5,
                help="Days to look back for trend analysis"
            )

            trend_ret_thresh = st.slider(
                "Trend Return Threshold",
                min_value=0.08,
                max_value=0.25,
                value=0.18,
                step=0.01,
                help="Minimum return to classify as strong trend"
            )

            st.markdown("#### ⚙️ Advanced")
            forecast_horizon = st.number_input(
                "Forecast Horizon (days)",
                min_value=30,
                max_value=200,
                value=100,
                step=10,
                help="Number of days to forecast ahead"
            )

            st.divider()

            # Run forecast button in sidebar
            run_forecast_btn = st.button(
                "🚀 Run Forecast",
                type="primary",
                use_container_width=True,
                help="Run the forecast model with custom parameters"
            )

            if run_forecast_btn:
                # Prepare parameters dictionary
                custom_params = {
                    "horizon": forecast_horizon,
                    "ret_clip_quantile": ret_clip_quantile,
                    "half_life_days": half_life_days,
                    "mean_revert_alpha": mean_revert_alpha,
                    "mean_revert_start": mean_revert_start,
                    "fair_up_mult": fair_up_mult,
                    "fair_down_mult": fair_down_mult,
                    "trend_lookback": trend_lookback,
                    "trend_ret_thresh": trend_ret_thresh,
                }

                # Create containers for status updates
                status_container = st.empty()
                progress_container = st.empty()

                with status_container.container():
                    status_text = st.empty()
                    status_text.text("🚀 Initializing forecast...")

                with progress_container.container():
                    progress_bar = st.progress(0)

                # Run forecast with real-time updates
                success = run_forecast_with_params(custom_params, status_text, progress_bar)

                if success:
                    status_text.text("✅ Forecast completed successfully!")
                    st.success("✅ Forecast completed! Page will refresh in 2 seconds...")
                    st.balloons()
                    import time
                    time.sleep(2)
                    # Clear containers
                    status_container.empty()
                    progress_container.empty()
                    # Rerun to refresh data
                    st.rerun()
                else:
                    st.error("❌ Forecast failed. Please check the error messages above.")
                    # Keep status visible for debugging        st.divider()

        # Data info
        st.subheader("📊 Data Info")
        if train_df is not None:
            st.write(f"**Training samples:** {len(train_df):,}")
            st.write(f"**Period:** {train_df['time'].min().date()} to {train_df['time'].max().date()}")
        if forecast_df is not None:
            st.write(f"**Forecast days:** {len(forecast_df)}")

    # Check if data is loaded
    if train_df is None or forecast_df is None:
        st.error("❌ Data files not found! Please run the model first to generate forecast data.")
        st.info("Expected files: FPT_train.csv, FPT_forecast.csv")
        return

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Analytics", "🔍 Model Insights", "📁 Data Explorer"])

    with tab1:
        st.header("Price Forecast Visualization")

        # Historical Chart - Full Data (2020-2025)
        chart_title = "📊 Full Stock History (2020-2025)"

        # Add timeframe info to subtitle for candlestick
        if chart_type == "Candlestick":
            timeframe_desc = {
                "1D": "Daily",
                "1W": "Weekly",
                "1M": "Monthly",
                "3M": "Quarterly"
            }
            chart_subtitle = f"*{timeframe_desc[timeframe]} candlesticks - Complete Open, High, Low, and Close data from training set FPT_train.csv"
        else:
            chart_subtitle = "*Price data with moving averages (MA20, MA60)*"

        st.subheader(f"{chart_title} - {chart_type}")
        st.markdown(chart_subtitle)

        # Display selected chart type
        if chart_type == "Candlestick":
            historical_fig = create_candlestick_chart(train_df, real_df, timeframe)
        else:
            historical_fig = create_line_chart(train_df, real_df)

        if historical_fig:
            st.plotly_chart(
                historical_fig,
                width='stretch',
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'responsive': True,
                    'doubleClick': 'reset',
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'fpt_stock_chart',
                        'height': 650,
                        'width': 1200,
                        'scale': 1
                    }
                }
            )

        st.divider()

        st.divider()

        # Forecast plot with chart type selection
        st.subheader("🔮 100-Day Price Forecast")

        if forecast_chart_type == "Candlestick":
            forecast_subtitle = f"*{['Daily', 'Weekly', 'Monthly', 'Quarterly'][forecast_timeframe_idx]} candlesticks with forecast overlay*"
        else:
            forecast_subtitle = "*Line chart with historical context and prediction bands*"

        st.markdown(forecast_subtitle)

        # Main forecast plot
        if forecast_chart_type == "Candlestick":
            fig_forecast = create_forecast_candlestick(train_df, forecast_df, real_df, show_uncertainty, timeframe=forecast_timeframe, show_real_overlay=show_real_overlay)
        else:
            fig_forecast = create_forecast_plot(train_df, forecast_df, real_df, show_uncertainty, show_real_overlay=show_real_overlay)

        st.plotly_chart(
            fig_forecast,
            width='stretch',
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'responsive': True,
                'doubleClick': 'reset',
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'fpt_forecast_chart',
                    'height': 650,
                    'width': 1200,
                    'scale': 1
                }
            }
        )

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        current_price = train_df['close'].iloc[-1]
        forecast_final = forecast_df['central_det'].iloc[-1]
        forecast_change = ((forecast_final - current_price) / current_price) * 100

        with col1:
            st.metric("Current Price", f"{current_price:,.2f} VND", help="Last known price from training data")
        with col2:
            st.metric("100-Day Forecast", f"{forecast_final:,.2f} VND", f"{forecast_change:+.2f}%")
        with col3:
            if 'uncert_lower_ana' in forecast_df.columns:
                lower_bound = forecast_df['uncert_lower_ana'].iloc[-1]
                st.metric("Lower Bound (90%)", f"{lower_bound:,.2f} VND")
        with col4:
            if 'uncert_upper_ana' in forecast_df.columns:
                upper_bound = forecast_df['uncert_upper_ana'].iloc[-1]
                st.metric("Upper Bound (90%)", f"{upper_bound:,.2f} VND")

        # MSE metrics (if real data overlay is enabled)
        if show_real_overlay and real_df is not None:
            st.divider()
            st.subheader("📊 Forecast Accuracy Metrics compare to Real FPT Stock")

            # Calculate MSE for forecast period
            forecast_start = forecast_df['time'].min()
            forecast_end = forecast_df['time'].max()
            real_filtered = real_df[
                (real_df['time'] >= forecast_start) &
                (real_df['time'] <= forecast_end)
            ]

            mse_result = calculate_mse(forecast_df, real_filtered, 'central_det', 'close')

            if mse_result and mse_result['count'] > 0:
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("MSE", f"{mse_result['mse']:,.2f}")
                with col2:
                    st.metric("RMSE", f"{mse_result['rmse']:,.2f} VND")
                with col3:
                    st.metric("MAE", f"{mse_result['mae']:,.2f} VND")
                with col4:
                    st.metric("MAPE", f"{mse_result['mape']:.2f}%")
                with col5:
                    st.metric("Data Points", f"{mse_result['count']}")

                st.info(f"📅 Comparison period: {forecast_start.strftime('%Y-%m-%d')} to {forecast_end.strftime('%Y-%m-%d')}")
            else:
                st.warning("⚠️ No overlapping dates between forecast and real data for accuracy calculation.")

        # Regime indicators
        if show_regime:
            st.divider()
            regime_fig = create_regime_plot(forecast_df)
            if regime_fig:
                st.plotly_chart(regime_fig, width='stretch')

        # ============================================================
        # CUSTOM FORECAST DISPLAY (FPT_forecast2.csv)
        # ============================================================
        forecast_df2 = load_forecast_data2()

        if forecast_df2 is not None:
            st.divider()
            st.subheader("🔮 Custom Forecast (With Adjusted Parameters)")

            if forecast_chart_type == "Candlestick":
                forecast_subtitle2 = f"*{['Daily', 'Weekly', 'Monthly', 'Quarterly'][forecast_timeframe_idx]} candlesticks with custom parameters*"
            else:
                forecast_subtitle2 = "*Custom forecast using your adjusted pricing layer parameters*"

            st.markdown(forecast_subtitle2)

            # Create custom forecast plot
            if forecast_chart_type == "Candlestick":
                fig_forecast2 = create_forecast_candlestick(train_df, forecast_df2, real_df, show_uncertainty, timeframe=forecast_timeframe, show_real_overlay=show_real_overlay)
            else:
                fig_forecast2 = create_forecast_plot(train_df, forecast_df2, real_df, show_uncertainty, show_real_overlay=show_real_overlay)

            st.plotly_chart(
                fig_forecast2,
                width='stretch',
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'responsive': True,
                    'doubleClick': 'reset',
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'fpt_custom_forecast_chart',
                        'height': 650,
                        'width': 1200,
                        'scale': 1
                    }
                }
            )

            # Custom forecast metrics
            col1, col2, col3, col4 = st.columns(4)

            current_price2 = train_df['close'].iloc[-1]
            forecast_final2 = forecast_df2['central_det'].iloc[-1]
            forecast_change2 = ((forecast_final2 - current_price2) / current_price2) * 100

            with col1:
                st.metric("Current Price", f"{current_price2:,.2f} VND", help="Last known price from training data")
            with col2:
                delta_vs_original = forecast_final2 - forecast_final
                st.metric(
                    "Custom Forecast",
                    f"{forecast_final2:,.2f} VND",
                    f"{forecast_change2:+.2f}% ({delta_vs_original:+.2f} vs original)",
                    help="Custom forecast with your parameters"
                )
            with col3:
                if 'uncert_lower_ana' in forecast_df2.columns:
                    lower_bound2 = forecast_df2['uncert_lower_ana'].iloc[-1]
                    st.metric("Lower Bound (90%)", f"{lower_bound2:,.2f} VND")
            with col4:
                if 'uncert_upper_ana' in forecast_df2.columns:
                    upper_bound2 = forecast_df2['uncert_upper_ana'].iloc[-1]
                    st.metric("Upper Bound (90%)", f"{upper_bound2:,.2f} VND")

            # MSE metrics for custom forecast (if real data overlay is enabled)
            if show_real_overlay and real_df is not None:
                st.divider()
                st.subheader("📊 Custom Forecast Accuracy Metrics compare to Real FPT Stock")

                # Calculate MSE for custom forecast period
                custom_start = forecast_df2['time'].min()
                custom_end = forecast_df2['time'].max()
                real_filtered_custom = real_df[
                    (real_df['time'] >= custom_start) &
                    (real_df['time'] <= custom_end)
                ]

                mse_custom = calculate_mse(forecast_df2, real_filtered_custom, 'central_det', 'close')

                if mse_custom and mse_custom['count'] > 0:
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("MSE", f"{mse_custom['mse']:,.2f}")
                    with col2:
                        st.metric("RMSE", f"{mse_custom['rmse']:,.2f} VND")
                    with col3:
                        st.metric("MAE", f"{mse_custom['mae']:,.2f} VND")
                    with col4:
                        st.metric("MAPE", f"{mse_custom['mape']:.2f}%")
                    with col5:
                        st.metric("Data Points", f"{mse_custom['count']}")

                    st.info(f"📅 Comparison period: {custom_start.strftime('%Y-%m-%d')} to {custom_end.strftime('%Y-%m-%d')}")
                else:
                    st.warning("⚠️ No overlapping dates between custom forecast and real data for accuracy calculation.")

            # Comparison info
            st.info(f"💡 **Comparison:** Original forecast: {forecast_final:,.2f} VND | Custom forecast: {forecast_final2:,.2f} VND | Difference: {delta_vs_original:+.2f} VND ({((delta_vs_original/forecast_final)*100):+.2f}%)")

            # Download buttons for custom forecast files
            st.divider()
            st.markdown("#### 📥 Download Custom Forecast Files")

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                # Download FPT_forecast2.csv
                if os.path.exists("FPT_forecast2.csv"):
                    with open("FPT_forecast2.csv", "rb") as file:
                        st.download_button(
                            label="📊 Download Custom Forecast (CSV)",
                            data=file,
                            file_name="FPT_forecast2.csv",
                            mime="text/csv",
                            help="Download the custom forecast data (FPT_forecast2.csv)",
                            use_container_width=True
                        )
                else:
                    st.button(
                        "📊 Custom Forecast (Not Available)",
                        disabled=True,
                        use_container_width=True,
                        help="Run a custom forecast first"
                    )

            with col_dl2:
                # Download submission.csv
                if os.path.exists("submission.csv"):
                    with open("submission.csv", "rb") as file:
                        st.download_button(
                            label="📤 Download Submission File (CSV)",
                            data=file,
                            file_name="submission.csv",
                            mime="text/csv",
                            help="Download the submission file (competition format)",
                            use_container_width=True
                        )
                else:
                    st.button(
                        "📤 Submission File (Not Available)",
                        disabled=True,
                        use_container_width=True,
                        help="Run a forecast first"
                    )

    with tab2:
        st.header("Advanced Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Price distribution
            dist_fig = create_price_distribution(forecast_df)
            if dist_fig:
                st.plotly_chart(dist_fig, width='stretch')

        with col2:
            # Volatility analysis
            vol_fig = create_volatility_plot(train_df)
            if vol_fig:
                st.plotly_chart(vol_fig, width='stretch')

        # Comparison table
        st.subheader("Forecast Comparison Table")
        comparison_df = create_comparison_table(forecast_df)
        if comparison_df is not None:
            st.dataframe(comparison_df.style.format({
                'Min Price': '{:.2f}',
                'Max Price': '{:.2f}',
                'Mean Price': '{:.2f}',
                'Final Price': '{:.2f}'
            }), width='stretch')

    with tab3:
        st.header("Model Insights")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🧠 Model Architecture")
            st.markdown("""
            **Hybrid Approach:**
            1. **Math Backbone** - Linear trend on log-price
            2. **XGBoost Residual** - Learns residual patterns
            3. **Pricing Layer** - Regime-aware adjustments

            **Key Features:**
            - STL Decomposition (trend, seasonal, residual)
            - OHLCV Price Action patterns
            - Volume & Money Flow indicators
            - Technical indicators (ATR, Parkinson Vol)
            - Rolling statistics & Z-scores
            """)

            st.subheader("🎯 Model Parameters")
            if params and 'my_config' in params:
                config_df = pd.DataFrame([params['my_config']]).T
                config_df.columns = ['Value']
                st.dataframe(config_df)

        with col2:
            st.subheader("📈 Performance Metrics")

            # Calculate some basic metrics
            if real_df is not None and len(real_df) > 0:
                # If we have real data, calculate accuracy
                merged = pd.merge(forecast_df[['time', 'central_det']], real_df[['time', 'close']], on='time', how='inner')
                if len(merged) > 0:
                    mse = np.mean((merged['central_det'] - merged['close']) ** 2)
                    mae = np.mean(np.abs(merged['central_det'] - merged['close']))
                    mape = np.mean(np.abs((merged['close'] - merged['central_det']) / merged['close'])) * 100

                    st.metric("MSE", f"{mse:,.2f}")
                    st.metric("MAE", f"{mae:,.2f}")
                    st.metric("MAPE", f"{mape:.2f}%")
            else:
                st.info("Real data not available for validation")

            st.subheader("🔧 Feature Engineering")
            st.markdown("""
            **Technical Indicators:**
            - Body, Range, Shadows, Gaps
            - ATR (14-period)
            - Parkinson Volatility
            - Volume ratios & OBV
            - Rolling stats (5, 10, 20, 60-day)
            - Return patterns (1d, 5d, 10d)
            - 3-streak patterns
            """)

    with tab4:
        st.header("Data Explorer")

        # Data selector
        data_option = st.selectbox("Select Dataset", ["Training Data", "Forecast Data", "Real Data (if available)"])

        if data_option == "Training Data" and train_df is not None:
            st.subheader(f"Training Data ({len(train_df)} rows)")
            st.dataframe(train_df.tail(100), width='stretch')

            # Download button
            csv = train_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Training Data",
                data=csv,
                file_name="FPT_train.csv",
                mime="text/csv"
            )

        elif data_option == "Forecast Data" and forecast_df is not None:
            st.subheader(f"Forecast Data ({len(forecast_df)} rows)")
            st.dataframe(forecast_df, width='stretch')

            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data",
                data=csv,
                file_name="FPT_forecast.csv",
                mime="text/csv"
            )

        elif data_option == "Real Data (if available)":
            if real_df is not None:
                st.subheader(f"Real Data ({len(real_df)} rows)")
                st.dataframe(real_df, width='stretch')
            else:
                st.info("Real data not available")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚀 FPT Stock Forecasting System | Hybrid ML Model | 100-Day Prediction Horizon</p>
        <p><small>Model: Math Backbone + XGBoost Residual + Pricing Layer | Data: FPT Corporation (Vietnam)</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
