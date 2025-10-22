import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import os

# Import backend functions
from signals_backend import (
    NASDAQ100_STARTER, TICKER_SECTOR, SPLITS, run_simplified_pipeline
)

# Additional imports for stock chart
import yfinance as yf

st.set_page_config(
    page_title="AI Stock Signal Generator", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        color: #000000;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Force all text elements to be black */
    .stApp, .main, p, span, div, h1, h2, h3, h4, h5, h6, label {
        color: #000000 !important;
    }
    
    /* Streamlit specific overrides */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #000000 !important;
    }
    
    /* Radio button labels */
    .stRadio > label, .stRadio div > label, .stRadio div div label, .stRadio div div div label {
        color: #000000 !important;
    }
    
    /* Selectbox labels */
    .stSelectbox > label, .stSelectbox div > label {
        color: #000000 !important;
    }
    
    /* Metric labels and values */
    .stMetric > label, .stMetric div, .stMetric span {
        color: #000000 !important;
    }
    
    /* Alert boxes */
    .stAlert > div, .stSuccess > div, .stError > div, .stInfo > div {
        color: #000000 !important;
    }
    
    /* COMPREHENSIVE SELECTBOX STYLING - Force white background and black text */
    
    /* Main selectbox container */
    .stSelectbox {
        background-color: #ffffff !important;
    }
    
    /* Selectbox input and control */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #d1d5db !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Target by test ID */
    [data-testid="stSelectbox"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stSelectbox"] > div {
        background-color: #ffffff !important;
    }
    
    [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #d1d5db !important;
    }
    
    [data-testid="stSelectbox"] input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* BaseWeb Select component styling */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Dropdown menu when opened */
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    div[data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    div[data-baseweb="menu"] ul {
        background-color: #ffffff !important;
    }
    
    div[data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    div[data-baseweb="menu"] li:hover {
        background-color: #f3f4f6 !important;
        color: #000000 !important;
    }
    
    /* Option items */
    [role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [role="option"]:hover {
        background-color: #f3f4f6 !important;
        color: #000000 !important;
    }
    
    /* Listbox container */
    [role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* CSS class targeting (common Streamlit classes) */
    .css-1wa3eu0-placeholder,
    .css-1uccc91-singleValue,
    .css-g3gb2o-SingleValue {
        color: #000000 !important;
    }
    
    .css-26l3qy-menu,
    .css-1n7v3ny-option,
    .css-1pahdxg-control,
    .css-1hwfws3 {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .css-1n7v3ny-option:hover {
        background-color: #f3f4f6 !important;
        color: #000000 !important;
    }
    
    /* Force override any dark theme */
    .stSelectbox * {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Test-specific overrides */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    .element-container div, .element-container p, .element-container span {
        color: #000000 !important;
    }
    
    .card {
        background: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,.1);
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    
    .card-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 10px;
        color: #000000 !important;
    }
    
    .signal-buy {
        color: #16a34a;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    
    .signal-sell {
        color: #dc2626;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    
    .signal-hold {
        color: #ea580c;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    
    .metric-value {
        font-size: 14px;
        font-weight: 500;
        margin: 5px 0;
        color: #000000;
    }
    
    .metric-large {
        font-size: 16px;
        font-weight: 600;
        margin: 8px 0;
        color: #000000;
    }
    
    .stButton > button {
        width: 100%;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
    }
    
    .stDownloadButton > button {
        width: 100%;
        background: #ffffff;
        color: #000000;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
    }
    
    .stDownloadButton > button:hover {
        background: #f3f4f6;
        color: #000000;
    }
    
    .time-buttons {
        display: flex;
        gap: 8px;
        margin: 10px 0;
    }
    
    .time-button {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        color: #000000;
    }
    
    .mini-chart {
        height: 60px;
        margin: 10px 0;
    }
    
    .chart-label {
        font-size: 12px;
        font-weight: 500;
        color: #000000;
        margin-bottom: 5px;
    }
    
    h1 {
        color: #000000;
    }
    
    h2 {
        color: #000000;
    }
    
    h3 {
        color: #000000;
    }
    
    .stSelectbox label {
        color: #000000;
    }
    
    .stMetric label {
        color: #000000;
    }
    
    .stMarkdown {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript to force selectbox styling after page load
st.markdown("""
<script>
    // Function to apply white background to selectbox elements
    function forceSelectboxStyling() {
        // Wait for elements to load
        setTimeout(function() {
            // Target all potential selectbox elements
            const selectors = [
                '[data-testid="stSelectbox"]',
                '[data-baseweb="select"]',
                '[data-baseweb="menu"]',
                '[role="listbox"]',
                '[role="option"]',
                '.stSelectbox'
            ];
            
            selectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    el.style.backgroundColor = '#ffffff';
                    el.style.color = '#000000';
                    
                    // Also style all children
                    const children = el.querySelectorAll('*');
                    children.forEach(child => {
                        child.style.backgroundColor = '#ffffff';
                        child.style.color = '#000000';
                    });
                });
            });
            
            // Specifically target dropdown options
            const options = document.querySelectorAll('[role="option"], div[data-baseweb="menu"] li');
            options.forEach(option => {
                option.style.backgroundColor = '#ffffff';
                option.style.color = '#000000';
                
                option.addEventListener('mouseenter', function() {
                    this.style.backgroundColor = '#f3f4f6';
                    this.style.color = '#000000';
                });
                
                option.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = '#ffffff';
                    this.style.color = '#000000';
                });
            });
            
        }, 500);
    }
    
    // Run on page load
    document.addEventListener('DOMContentLoaded', forceSelectboxStyling);
    
    // Run on Streamlit rerun
    window.addEventListener('load', forceSelectboxStyling);
    
    // Also run periodically to catch dynamic content
    setInterval(forceSelectboxStyling, 1000);
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = NASDAQ100_STARTER[0]
if 'future_prediction_range' not in st.session_state:
    st.session_state['future_prediction_range'] = '1 day'

def format_signal_display(signal_value):
    """Format signal for display with appropriate styling"""
    # Handle both string and numeric signals
    if isinstance(signal_value, str):
        signal_text = signal_value
        signal_map_reverse = {"BUY": "signal-buy", "HOLD": "signal-hold", "SELL": "signal-sell"}
        signal_class = signal_map_reverse.get(signal_value, "signal-hold")
    else:
        # Legacy numeric format
        signal_map = {2: "BUY", 1: "HOLD", 0: "SELL", -1: "SELL"}
        signal_class_map = {2: "signal-buy", 1: "signal-hold", 0: "signal-sell", -1: "signal-sell"}
        signal_text = signal_map.get(signal_value, "HOLD")
        signal_class = signal_class_map.get(signal_value, "signal-hold")
    
    return f'<div class="{signal_class}">{signal_text}</div>'

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_ticker_data(ticker, period="1y"):
    """Load stock data for a given ticker"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

def create_stock_chart(ticker_data, ticker, time_range="1Y"):
    """Create an interactive stock chart"""
    if ticker_data.empty:
        st.warning(f"No data available for {ticker}")
        return None
    
    # Filter data based on time range
    end_date = ticker_data.index.max()
    
    if time_range == '1D':
        start_date = end_date - timedelta(days=1)
    elif time_range == '1W':
        start_date = end_date - timedelta(days=7)
    elif time_range == '1M':
        start_date = end_date - timedelta(days=30)
    elif time_range == '6M':
        start_date = end_date - timedelta(days=180)
    elif time_range == '1Y':
        start_date = end_date - timedelta(days=365)
    else:  # MAX
        start_date = ticker_data.index.min()
    
    filtered_data = ticker_data[ticker_data.index >= start_date]
    
    if filtered_data.empty:
        st.warning(f"No data available for {ticker} in the selected time range")
        return None
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#2563eb', width=2)
    ))
    
    # Customize layout with white background and black text
    fig.update_layout(
        title=f'{ticker} Stock Price ({time_range})',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        title_font=dict(color='black'),
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    return fig

def create_basic_price_chart(prices_data, ticker):
    """Create a basic price chart from price data"""
    if prices_data is None or len(prices_data) == 0:
        st.warning("No price data available")
        return None
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=list(range(len(prices_data))),
        y=prices_data,
        mode='lines',
        name='Price',
        line=dict(color='#2563eb', width=2)
    ))
    
    # Customize layout with white background and black text
    fig.update_layout(
        title=f'{ticker} Stock Price Chart',
        xaxis_title='Time',
        yaxis_title='Price ($)',
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        title_font=dict(color='black'),
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    return fig

def create_price_prediction_chart(ticker, result):
    """Create price prediction chart with true values and predictions"""
    try:
        # Get current stock data for true values
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period="2y")  # Get more historical data
        
        if historical_data.empty:
            return None
            
        # Get future predictions
        future_predictions = result.get('future_predictions', {})
        if not future_predictions or not future_predictions.get('success'):
            return None
            
        predictions_data = future_predictions.get('predictions', {})
        if not predictions_data:
            return None
            
        # Create the figure
        fig = go.Figure()
        
        # Add true values (blue line) - show last 6 months of historical data
        recent_data = historical_data.tail(180)  # Last ~6 months
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            mode='lines',
            name='True Value',
            line=dict(color='blue', width=2)
        ))
        
        # Get the last price and date
        last_date = recent_data.index[-1]
        last_price = recent_data['Close'].iloc[-1]
        
        # Create future dates for predictions
        future_dates = []
        future_prices = []
        
        # Process predictions for different timeframes using the predicted_price field
        for timeframe in ['1d', '7d', '30d']:
            if timeframe in predictions_data:
                pred_data = predictions_data[timeframe]
                predicted_price = pred_data.get('predicted_price')
                
                if predicted_price:  # Use the predicted price directly from backend
                    if timeframe == '1d':
                        future_date = last_date + timedelta(days=1)
                    elif timeframe == '7d':
                        future_date = last_date + timedelta(days=7)
                    else:  # 30d
                        future_date = last_date + timedelta(days=30)
                    
                    future_dates.append(future_date)
                    future_prices.append(predicted_price)
        
        # Add connecting line from last true price to first prediction
        if future_dates and future_prices:
            # Add prediction points (red)
            connection_dates = [last_date] + future_dates
            connection_prices = [last_price] + future_prices
            
            fig.add_trace(go.Scatter(
                x=connection_dates,
                y=connection_prices,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(color='red', size=8)
            ))
            
            # Add prediction labels
            for i, (date, price) in enumerate(zip(future_dates, future_prices)):
                timeframe = ['1D', '7D', '30D'][i]
                fig.add_annotation(
                    x=date,
                    y=price,
                    text=f"{timeframe}<br>${price:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(color="red", size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1
                )
        
        # Update layout with explicit white background and black text
        fig.update_layout(
            title=f"Stock Price Prediction - Next 1 Day, 7 Days, 30 Days",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            template="plotly_white",
            height=400,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color="black"),
            title_font=dict(color="black"),
            xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
            yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
            legend=dict(font=dict(color="black"))
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating price prediction chart: {e}")
        return None

def display_chart_image(chart_path, caption, full_width=False):
    """Display a chart image if it exists"""
    if chart_path and os.path.exists(chart_path):
        try:
            if full_width:
                st.image(chart_path, caption=caption, use_container_width=True)
            else:
                st.image(chart_path, caption=caption)
        except Exception as e:
            st.error(f"Error displaying chart: {e}")
    else:
        st.info(f"{caption} not available")

def main():
    # Header
    st.markdown("# AI Stock Signal Generator")
    st.markdown("---")
    
    # Top row - Ticker selection and time range
    col_top1, col_top2 = st.columns([1, 2])
    
    with col_top1:
        ticker = st.selectbox(
            "Select Stock Ticker",
            NASDAQ100_STARTER,
            index=NASDAQ100_STARTER.index(st.session_state.get('selected_ticker', NASDAQ100_STARTER[0]))
        )
        st.session_state['selected_ticker'] = ticker
    
    with col_top2:
        st.markdown("**Time Range:**")
        # Time range buttons
        time_ranges = ['1D', '1W', '1M', '6M', '1Y', 'MAX']
        
        cols_time = st.columns(len(time_ranges))
        for i, time_range in enumerate(time_ranges):
            with cols_time[i]:
                if st.button(time_range, key=f"time_{time_range}"):
                    st.session_state['selected_range'] = time_range
                    st.rerun()
        
        # Show selected range
        selected_range = st.session_state.get('selected_range', '1Y')
        st.markdown(f"**Selected Range:** {selected_range}")
    
    st.markdown("---")
    
    # Main 2-column layout matching the mockup
    col_left, col_right = st.columns([2, 1])
    
    # LEFT COLUMN - Charts
    with col_left:
        
        # Stock Chart - Always show when ticker is selected
        st.markdown("### Stock Chart")
        
        # Load and display stock chart immediately
        try:
            # Determine period for yfinance based on selected range
            period_map = {
                '1D': '5d',  # yfinance needs more data for 1D to show properly
                '1W': '1mo',
                '1M': '3mo', 
                '6M': '1y',
                '1Y': '2y',
                'MAX': '10y'
            }
            
            period = period_map.get(selected_range, '1y')
            ticker_data = load_ticker_data(ticker, period)
            
            if not ticker_data.empty:
                stock_fig = create_stock_chart(ticker_data, ticker, selected_range)
                if stock_fig:
                    st.plotly_chart(stock_fig, use_container_width=True)
                
                # Show latest price info
                latest_price = ticker_data['Close'].iloc[-1]
                prev_price = ticker_data['Close'].iloc[-2] if len(ticker_data) > 1 else latest_price
                price_change = latest_price - prev_price
                pct_change = (price_change / prev_price * 100) if prev_price != 0 else 0
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Latest Price", f"${latest_price:.2f}")
                with col_p2:
                    st.metric("Change", f"${price_change:.2f}", f"{pct_change:+.2f}%")
                with col_p3:
                    sector = TICKER_SECTOR.get(ticker, "Unknown")
                    st.metric("Sector", sector)
            else:
                st.error("Unable to load stock data")
                
        except Exception as e:
            st.error(f"Error loading stock chart: {e}")
        
        # Price Prediction Chart - Add this right after stock chart
        st.markdown("### Price Predictions")
        result = st.session_state.get('prediction_result')
        if result and result.get('success', False) and result.get('ticker') == ticker:
            prediction_fig = create_price_prediction_chart(ticker, result)
            if prediction_fig:
                st.plotly_chart(prediction_fig, use_container_width=True)
            else:
                st.info("Price prediction chart will appear here after running prediction")
        else:
            st.info("Price prediction chart will appear here after running prediction")
        
        # Get prediction result if available
        result = st.session_state.get('prediction_result')
        
        # Actual vs Predicted
        st.markdown("### Actual vs Predicted")
        if result and result.get('success', False) and result.get('ticker') == ticker:
            # Create dynamic chart using predicted_prices from backend
            if 'predicted_prices' in result and 'test_prices' in result:
                predicted_prices = result['predicted_prices']
                test_prices = result['test_prices']
                
                if predicted_prices and test_prices and len(predicted_prices) == len(test_prices):
                    # Create interactive chart with actual fixed predicted prices
                    fig = go.Figure()
                    
                    # Add actual prices (blue line)
                    fig.add_trace(go.Scatter(
                        x=list(range(len(test_prices))),
                        y=test_prices,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add predicted prices (red line) - using our FIXED algorithm
                    fig.add_trace(go.Scatter(
                        x=list(range(len(predicted_prices))),
                        y=predicted_prices,
                        mode='lines',
                        name='Prediction',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} - Actual vs Predicted Price (Test Split)",
                        xaxis_title="Time Period",
                        yaxis_title="Price ($)",
                        template="plotly_white",
                        height=500,
                        showlegend=True,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='black'),
                        title_font=dict(color='black'),
                        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
                        legend=dict(font=dict(color='black'))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Regression Metrics Comparison Table
                    st.markdown("#### Regression Metrics Comparison")
                    
                    # Get regression metrics from backend results
                    regression_metrics = result.get('metrics_regression', {})
                    
                    if regression_metrics:
                        # Extract metrics for each model
                        model_metrics = regression_metrics.get('model', {})
                        persistence_metrics = regression_metrics.get('persistence', {})
                        ema_metrics = regression_metrics.get('ema', {})
                        
                        # Create comparison table data
                        table_data = {
                            'Model': [
                                'Our Model',
                                'Persistence',
                                'EMA(10/50)'
                            ],
                            'RMSE': [
                                f"{model_metrics.get('rmse', float('nan')):.3f}" if not np.isnan(model_metrics.get('rmse', float('nan'))) else "—",
                                f"{persistence_metrics.get('rmse', float('nan')):.3f}" if not np.isnan(persistence_metrics.get('rmse', float('nan'))) else "—",
                                f"{ema_metrics.get('rmse', float('nan')):.3f}" if not np.isnan(ema_metrics.get('rmse', float('nan'))) else "—"
                            ],
                            'MAE': [
                                f"{model_metrics.get('mae', float('nan')):.3f}" if not np.isnan(model_metrics.get('mae', float('nan'))) else "—",
                                f"{persistence_metrics.get('mae', float('nan')):.3f}" if not np.isnan(persistence_metrics.get('mae', float('nan'))) else "—",
                                f"{ema_metrics.get('mae', float('nan')):.3f}" if not np.isnan(ema_metrics.get('mae', float('nan'))) else "—"
                            ],
                            'R²': [
                                f"{model_metrics.get('r2', float('nan')):.3f}" if not np.isnan(model_metrics.get('r2', float('nan'))) else "—",
                                f"{persistence_metrics.get('r2', float('nan')):.3f}" if not np.isnan(persistence_metrics.get('r2', float('nan'))) else "—",
                                f"{ema_metrics.get('r2', float('nan')):.3f}" if not np.isnan(ema_metrics.get('r2', float('nan'))) else "—"
                            ],
                            'MAPE': [
                                f"{model_metrics.get('mape', float('nan')):.3f}%" if not np.isnan(model_metrics.get('mape', float('nan'))) else "—",
                                f"{persistence_metrics.get('mape', float('nan')):.3f}%" if not np.isnan(persistence_metrics.get('mape', float('nan'))) else "—",
                                f"{ema_metrics.get('mape', float('nan')):.3f}%" if not np.isnan(ema_metrics.get('mape', float('nan'))) else "—"
                            ]
                        }
                        
                        # Create DataFrame and display as table
                        import pandas as pd
                        df_metrics = pd.DataFrame(table_data)
                        
                        # Style the table with custom CSS for compactness
                        st.markdown("""
                        <style>
                        .metrics-table {
                            font-size: 12px;
                            margin: 10px 0;
                        }
                        .metrics-table th {
                            background-color: #f0f2f6;
                            color: black;
                            font-weight: bold;
                            text-align: center;
                            padding: 8px;
                        }
                        .metrics-table td {
                            text-align: center;
                            padding: 6px;
                            color: black;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Display table using Streamlit's dataframe with custom styling
                        st.dataframe(
                            df_metrics,
                            use_container_width=True,
                            hide_index=True,
                            height=140  # Compact height
                        )
                        
                        # Add brief explanation
                        st.markdown('<p style="color: black; font-size: 11px; font-style: italic;">Lower RMSE, MAE, and MAPE indicate better accuracy. Higher R² indicates better fit.</p>', unsafe_allow_html=True)
                    
                    else:
                        # Fallback to simple metrics if regression_metrics not available
                        st.markdown("#### Regression/Forecasting Metrics")
                        import numpy as np
                        correlation = np.corrcoef(test_prices, predicted_prices)[0, 1]
                        mae = np.mean(np.abs(np.array(test_prices) - np.array(predicted_prices)))
                        rmse = np.sqrt(np.mean((np.array(test_prices) - np.array(predicted_prices))**2))
                        r_squared = correlation**2
                        
                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error - Average error in price predictions")
                        with col_r2:
                            st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error - Penalizes large errors more heavily")  
                        with col_r3:
                            st.metric("R²", f"{r_squared:.3f}", help="R² (Coefficient of Determination) - How well predictions fit the data")
                else:
                    st.error("Unable to load predicted prices data")
            else:
                # Fallback to static image if predicted_prices not available
                chart_paths = result.get('chart_paths', {})
                actual_vs_pred_path = chart_paths.get('actual_vs_predicted')
                if actual_vs_pred_path:
                    display_chart_image(actual_vs_pred_path, "Actual vs Predicted", full_width=True)
                else:
                    st.info("Actual vs predicted chart will appear here")
        else:
            st.info("Actual vs predicted chart will appear here after running prediction")
        
        # Strategy Performance
        st.markdown("### Strategy Performance")
        if result and result.get('success', False) and result.get('ticker') == ticker:
            chart_paths = result.get('chart_paths', {})
            strategy_perf_path = chart_paths.get('strategy_performance')
            if strategy_perf_path:
                display_chart_image(strategy_perf_path, "Strategy Performance", full_width=True)
                
                # Add strategy performance text analysis
                metrics = result.get('metrics', {})
                strategy_metrics = metrics.get('strategy', {})
                buy_hold_metrics = metrics.get('buy_hold', {})
                
                if strategy_metrics and buy_hold_metrics:
                    st.markdown("#### Strategy Analysis")
                    
                    # Use the correct field names from backend
                    strategy_return = strategy_metrics.get('total_return_pct', 0) / 100.0  # Convert from percentage
                    buy_hold_return = buy_hold_metrics.get('total_return_pct', 0) / 100.0  # Convert from percentage
                    outperformance = strategy_return - buy_hold_return
                    
                    if outperformance > 0:
                        performance_text = f"**Strategy Outperformance**: +{outperformance:.2%}"
                        st.success(performance_text)
                    else:
                        performance_text = f"**Strategy Underperformance**: {outperformance:.2%}"
                        st.error(performance_text)
                    
                    # Additional metrics
                    col_s1, col_s2, col_s3 = st.columns(3)
                    
                    with col_s1:
                        strategy_sharpe = strategy_metrics.get('sharpe_ratio', 0)
                        st.metric("Strategy Sharpe", f"{strategy_sharpe:.3f}")
                    
                    with col_s2:
                        strategy_vol = strategy_metrics.get('volatility_pct', 0) / 100.0  # Convert from percentage
                        st.metric("Strategy Volatility", f"{strategy_vol:.2%}")
                    
                    with col_s3:
                        max_drawdown = abs(strategy_metrics.get('max_drawdown_pct', 0)) / 100.0  # Convert from percentage and make positive
                        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                    
                    # Performance summary text
                    st.markdown('<p style="color: black;"><strong>Performance Summary:</strong></p>', unsafe_allow_html=True)
                    summary_text = f"""
                    <p style="color: black;">
                    The AI strategy generated a <strong>{strategy_return:.2%}</strong> total return compared to 
                    <strong>{buy_hold_return:.2%}</strong> for buy & hold, resulting in <strong>{outperformance:+.2%}</strong> 
                    {"outperformance" if outperformance > 0 else "underperformance"}. 
                    The strategy achieved a Sharpe ratio of <strong>{strategy_sharpe:.3f}</strong> with 
                    <strong>{strategy_vol:.2%}</strong> volatility and maximum drawdown of <strong>{max_drawdown:.2%}</strong>.
                    </p>
                    """
                    st.markdown(summary_text, unsafe_allow_html=True)
            else:
                st.info("Strategy performance chart will appear here")
        else:
            st.info("Strategy performance analysis will appear here after running prediction")
        
        # GARCH Analysis
        st.markdown("### GARCH Analysis")
        if result and result.get('success', False) and result.get('ticker') == ticker:
            chart_paths = result.get('chart_paths', {})
            garch_chart_path = chart_paths.get('garch_analysis')
            if garch_chart_path:
                display_chart_image(garch_chart_path, "GARCH Volatility Analysis", full_width=True)
            else:
                st.info("GARCH analysis chart will appear here")
        else:
            st.info("GARCH analysis chart will appear here after running prediction")
        
        # Technical Indicators
        st.markdown("### Technical Indicators")
        if result and result.get('success', False) and result.get('ticker') == ticker:
            chart_paths = result.get('chart_paths', {})
            tech_indicators_path = chart_paths.get('technical_indicators')
            if tech_indicators_path:
                display_chart_image(tech_indicators_path, "Technical Indicators", full_width=True)
            else:
                st.info("Technical indicators chart will appear here")
        else:
            st.info("Technical indicators chart will appear here after running prediction")
    
    # RIGHT COLUMN - Signal and Analysis Cards
    with col_right:
        
        # SIGNAL Card (Top Card)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">SIGNAL</div>', unsafe_allow_html=True)
        
        # Live Predict Button
        if st.button("Live Predict", key="predict_button"):
            try:
                with st.spinner("Training and predicting... This may take a few minutes."):
                    result = run_simplified_pipeline(ticker)
                    st.session_state['prediction_result'] = result
                    
                st.success("Prediction completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        
        # Display prediction results if available
        if result and result.get('success', False) and result.get('ticker') == ticker:
            # Get latest signal
            signals = result.get('signals')
            if signals is not None and len(signals) > 0:
                if isinstance(signals, pd.Series):
                    latest_signal = signals.iloc[-1]
                else:
                    latest_signal = signals[-1] if isinstance(signals, list) else signals
            else:
                latest_signal = result.get('signal', 1)
            
            # Display signal
            st.markdown(format_signal_display(latest_signal), unsafe_allow_html=True)
            
            # Expected and Confidence
            expected_pct = result.get('expected_pct', 0.0)
            confidence_pct = result.get('confidence_pct', 0.0)
            
            col_exp, col_conf = st.columns(2)
            with col_exp:
                st.markdown("**Expected**")
                st.markdown(f"**{expected_pct:+.2f}%**")
            with col_conf:
                st.markdown("**Confidence**")
                st.markdown(f"**{confidence_pct:.0f}%**")
        else:
            st.info("Click 'Live Predict' to generate signal")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Technical Indicators Card  
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Technical Indicators</div>', unsafe_allow_html=True)
        
        if result and result.get('success', False) and result.get('ticker') == ticker:
            tech_analysis = result.get('technical_analysis', {})
            
            if tech_analysis:
                # Display technical indicators in table format
                indicators_data = []
                
                rsi = tech_analysis.get('latest_rsi')
                if rsi:
                    indicators_data.append(['RSI(14)', f'{rsi:.1f}'])
                
                macd = tech_analysis.get('latest_macd')  
                if macd:
                    indicators_data.append(['MACD', f'{macd:.3f}'])
                
                ema_20 = tech_analysis.get('latest_ema_20')
                if ema_20:
                    indicators_data.append(['EMA 20', f'{ema_20:.2f}'])
                
                ema_50 = tech_analysis.get('latest_ema_50')
                if ema_50:
                    indicators_data.append(['EMA 50', f'{ema_50:.2f}'])
                
                bb_width = tech_analysis.get('latest_bb_width')
                if bb_width:
                    indicators_data.append(['BB Width(20)', f'{bb_width:.3f}'])
                
                # Create a simple table display with black text
                for indicator in indicators_data:
                    col_ind1, col_ind2 = st.columns([1, 1])
                    with col_ind1:
                        st.markdown(f'<p style="color: black;"><strong>{indicator[0]}</strong></p>', unsafe_allow_html=True)
                    with col_ind2:
                        st.markdown(f'<p style="color: black;">{indicator[1]}</p>', unsafe_allow_html=True)
                
                # Add technical analysis text
                st.markdown('<p style="color: black;"><strong>Technical Analysis:</strong></p>', unsafe_allow_html=True)
                
                # Generate analysis based on RSI and MACD
                analysis_text = ""
                if rsi:
                    if rsi > 70:
                        analysis_text += f"RSI indicates overbought conditions at {rsi:.1f}, suggesting potential downward pressure. "
                    elif rsi < 30:
                        analysis_text += f"RSI shows oversold conditions at {rsi:.1f}, indicating potential buying opportunity. "
                    else:
                        analysis_text += f"RSI at {rsi:.1f} suggests neutral momentum. "
                
                if macd:
                    if macd > 0:
                        analysis_text += f"MACD is positive at {macd:.3f}, indicating bullish momentum. "
                    else:
                        analysis_text += f"MACD is negative at {macd:.3f}, showing bearish momentum. "
                
                # Add Bollinger Band analysis
                bb_position = tech_analysis.get('bb_position', 'middle')
                if bb_position == 'upper':
                    analysis_text += "Price is trading above upper Bollinger Band, indicating potential overbought conditions. "
                elif bb_position == 'lower':
                    analysis_text += "Price is trading below lower Bollinger Band, suggesting oversold conditions. "
                else:
                    analysis_text += "Price is within normal Bollinger Band range. "
                
                # Add strategy recommendation
                latest_signal = result.get('signal', 'HOLD')
                if isinstance(latest_signal, (int, float)):
                    signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
                    signal_text = signal_map.get(int(latest_signal), "HOLD")
                else:
                    signal_text = str(latest_signal)
                
                analysis_text += f"Combined with model predictions, current recommendation is {signal_text}."
                
                st.markdown(f'<p style="color: black; font-size: 12px;">{analysis_text}</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: black;">Technical indicators will appear here</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: black;">Technical indicators will appear here</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Performance Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Model Performance</div>', unsafe_allow_html=True)
        
        if result and result.get('success', False) and result.get('ticker') == ticker:
            metrics = result.get('metrics', {})
            classification_metrics = metrics.get('classification', {})
            
            # Model Accuracy
            accuracy = classification_metrics.get('accuracy', 0.0)
            st.markdown('<p style="color: black;"><strong>Model Accuracy</strong></p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: black; font-size: 18px;"><strong>{accuracy:.1%}</strong></p>', unsafe_allow_html=True)
            
            # F1 Score  
            f1_score = classification_metrics.get('f1_score', 0.0)
            st.markdown('<p style="color: black;"><strong>F1 Score</strong></p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: black; font-size: 18px;"><strong>{f1_score:.3f}</strong></p>', unsafe_allow_html=True)
            
            # Classification Metrics (for Buy/Hold/Sell signals)
            st.markdown('<p style="color: black;"><strong>Classification Metrics</strong></p>', unsafe_allow_html=True)
            
            precision = classification_metrics.get('precision', 0.0)
            recall = classification_metrics.get('recall', 0.0)
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown(f'<p style="color: black;"><strong>Precision:</strong> {precision:.3f}</p>', unsafe_allow_html=True)
            with col_c2:
                st.markdown(f'<p style="color: black;"><strong>Recall:</strong> {recall:.3f}</p>', unsafe_allow_html=True)
            
        else:
            st.markdown('<p style="color: black;">Model performance will appear here</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Future Predictions Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Future Predictions</div>', unsafe_allow_html=True)
        
        if result and result.get('success', False) and result.get('ticker') == ticker:
            future_predictions = result.get('future_predictions', {})
            
            if future_predictions and future_predictions.get('success'):
                # Prediction timeframe selector
                pred_timeframe = st.radio(
                    "Select prediction timeframe:",
                    ["1 Day", "7 Days", "30 Days"],
                    horizontal=True,
                    key=f"pred_timeframe_{ticker}"  # Make key unique per ticker
                )
                
                # Map selection to key
                timeframe_map = {"1 Day": "1d", "7 Days": "7d", "30 Days": "30d"}
                selected_key = timeframe_map[pred_timeframe]
                
                predictions_data = future_predictions.get('predictions', {})
                future_data = predictions_data.get(selected_key, {})
                
                if future_data:
                    signal = future_data.get('signal', 1)
                    confidence = future_data.get('confidence', 0.0)
                    expected_return = future_data.get('expected_return_pct', 0.0)
                    signal_name = future_data.get('signal_name', 'HOLD')
                    
                    # Use signal_name directly (it's already in correct format)
                    signal_colors = {"BUY": "#28a745", "HOLD": "#ffc107", "SELL": "#dc3545"}
                    
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.markdown('<p style="color: black; margin-bottom: 5px;"><strong>Signal</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: {signal_colors.get(signal_name, "#ffc107")}; font-size: 18px; font-weight: bold; margin-top: 0px;"><strong>{signal_name}</strong></p>', unsafe_allow_html=True)
                    with col_f2:
                        st.markdown('<p style="color: black; margin-bottom: 5px;"><strong>Confidence</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: black; font-size: 18px; font-weight: bold; margin-top: 0px;">{confidence:.1%}</p>', unsafe_allow_html=True)
                    
                    # Expected return
                    if expected_return != 0:
                        return_color = "#28a745" if expected_return > 0 else "#dc3545"
                        st.markdown(f'<p style="color: black;"><strong>Expected Return:</strong> <span style="color: {return_color};">{expected_return:+.2f}%</span></p>', unsafe_allow_html=True)
                    
                    # Predicted price if available
                    predicted_price = future_data.get('predicted_price')
                    current_price = future_data.get('current_price')
                    if predicted_price and current_price:
                        price_color = "#28a745" if predicted_price > current_price else "#dc3545"
                        st.markdown(f'<p style="color: black;"><strong>Current Price:</strong> ${current_price:.2f}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: black;"><strong>Predicted Price:</strong> <span style="color: {price_color};">${predicted_price:.2f}</span></p>', unsafe_allow_html=True)
                
                else:
                    st.markdown('<p style="color: black;">No predictions available for selected timeframe</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: black;">Future predictions will appear here</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: black;">Future predictions will appear here after running prediction</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download CSV Button
        if result and result.get('success', False) and result.get('ticker') == ticker:
            csv_path = result.get('csv_path')
            if csv_path and os.path.exists(csv_path):
                try:
                    with open(csv_path, 'rb') as f:
                        csv_data = f.read()
                    
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_data,
                        file_name=f"{ticker}_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error preparing CSV download: {e}")

if __name__ == "__main__":
    main()