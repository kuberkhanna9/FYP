import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import sys
import os
from pathlib import Path

# Get absolute paths
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
data_dir = project_root / "data"
models_dir = project_root / "models"

sys.path.append(str(project_root))

# Page config
st.set_page_config(
    page_title="AI Stock Forecaster | KK",
    page_icon="✌️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .explanation-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    .explanation-box h3 {
        color: #E0E0E0;
        margin-bottom: 15px;
    }
    .explanation-box p {
        color: white;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the stock data"""
    data_file = data_dir / "enriched_merged_stocks.csv"
    try:
        # Read CSV and force Date column to datetime
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'].astype(str))
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {data_file}. Please ensure 'enriched_merged_stocks.csv' exists in the data directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(f"Error type: {type(e)}")
        st.error(f"Error args: {e.args}")
        return None

@st.cache_resource
def load_model():
    """Load and cache the XGBoost model"""
    model_file = models_dir / "xgb_model.pkl"
    try:
        model = joblib.load(model_file)
        # Define feature names used for prediction
        model.feature_names_ = [
            # Technical Indicators
            'RSI',
            'EMA10',
            'EMA50',
            'MACD',
            'MACD_Signal',
            'MACD_Hist',
            
            # Price and Volume
            'Volume',
            'Close',
            'High',
            'Low',
            
            # Derived Features
            'HL_Range',
            'Volume_1d_chg',
            'Price_1d_chg',
            'future_return'
        ]
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_file}. Please ensure 'xgb_model.pkl' exists in the models directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_shap_values():
    """Load pre-computed SHAP values if available"""
    try:
        shap_values = joblib.load(models_dir / "shap_values.pkl")
        return shap_values
    except FileNotFoundError:
        return None

def plot_stock_chart(df, ticker):
    """Create stock price chart with EMAs"""
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.1,
                       row_heights=[0.7, 0.3])
    
    # Price and EMAs
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['EMA10'],
                  name='EMA10', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['EMA50'],
                  name='EMA50', line=dict(color='orange')),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI'],
                  name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Add RSI zones
    fig.add_hline(y=70, line_dash="dash", line_color="red",
                  annotation_text="Overbought", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green",
                  annotation_text="Oversold", row=2, col=1)
    
    fig.update_layout(
        title=f'{ticker} Stock Price and Indicators',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis2_title='RSI',
        height=800
    )
    
    return fig

def get_prediction_color(prediction):
    """Return color based on prediction"""
    colors = {
        'Buy': '#00C853',  # Brighter green
        'Sell': '#FF1744',  # Brighter red
        'Hold': '#FFB300'   # Brighter orange/amber
    }
    return colors.get(prediction, '#757575')  # Default to a neutral gray

def plot_prediction_probabilities(probabilities, classes):
    """Plot prediction probabilities as horizontal bars"""
    fig = go.Figure()
    colors = ['#FF1744', '#FFB300', '#00C853']  # Match prediction colors
    
    for prob, label, color in zip(probabilities, classes, colors):
        fig.add_trace(go.Bar(
            y=[label],
            x=[prob * 100],
            orientation='h',
            marker_color=color,
            name=label,
            text=[f"{prob * 100:.1f}%"],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Probability (%)',
        yaxis_title='Signal',
        height=200,
        showlegend=False,
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            range=[0, 100],
            tickfont=dict(color='white'),
        ),
        yaxis=dict(
            tickfont=dict(color='white'),
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def generate_signal_explanation(latest_data, prediction):
    """Generate explanation text based on technical indicators"""
    explanation = []
    
    # RSI-based explanation
    rsi = latest_data['RSI']
    if rsi < 30:
        explanation.append(f"RSI is oversold ({rsi:.1f} < 30), suggesting potential upward reversal.")
    elif rsi > 70:
        explanation.append(f"RSI is overbought ({rsi:.1f} > 70), suggesting potential downward reversal.")
    else:
        explanation.append(f"RSI is neutral at {rsi:.1f}.")
    
    # EMA crossover explanation
    ema10 = latest_data['EMA10']
    ema50 = latest_data['EMA50']
    if ema10 > ema50:
        explanation.append("Short-term trend (EMA10) is above long-term trend (EMA50), indicating bullish momentum.")
    else:
        explanation.append("Short-term trend (EMA10) is below long-term trend (EMA50), indicating bearish momentum.")
    
    # MACD explanation
    if 'MACD' in latest_data and 'MACD_Signal' in latest_data:
        macd = latest_data['MACD']
        macd_signal = latest_data['MACD_Signal']
        macd_hist = latest_data['MACD_Hist']
        if macd > macd_signal:
            explanation.append(f"MACD ({macd:.2f}) is above signal line ({macd_signal:.2f}), suggesting upward momentum.")
        else:
            explanation.append(f"MACD ({macd:.2f}) is below signal line ({macd_signal:.2f}), suggesting downward momentum.")
        
        if macd_hist > 0:
            explanation.append("MACD histogram is positive, indicating bullish momentum is building.")
        else:
            explanation.append("MACD histogram is negative, indicating bearish momentum is building.")
    
    # Price action
    price_chg = latest_data['Price_1d_chg'] * 100
    if abs(price_chg) > 0.1:  # Only mention if change is significant
        explanation.append(f"Price changed by {price_chg:.1f}% in the last day.")
    
    # Volume analysis
    vol_chg = latest_data['Volume_1d_chg'] * 100
    if abs(vol_chg) > 10:  # Only mention if change is significant
        explanation.append(f"Trading volume changed by {vol_chg:.1f}% compared to previous day.")
    
    return " ".join(explanation)

def main():
    # Load data and model first
    df = load_data()
    model = load_model()
    shap_values = load_shap_values()
    
    if df is None or model is None:
        return
        
    # Sidebar
    with st.sidebar:
        st.title('📈 AI Stock Forecaster')
        
        # Stock selector
        ticker = st.selectbox(
            'Select Stock',
            ['AAPL', 'TSLA', 'MSFT'],
            index=0
        )
        
        # Date range selector with proper validation
        today = datetime.now().date()
        default_end_date = min(df['Date'].max().date(), today)
        default_start_date = default_end_date - timedelta(days=365)  # 1 year of data by default
        
        date_range = st.date_input(
            'Select Date Range',
            value=(default_start_date, default_end_date),
            min_value=df['Date'].min().date(),
            max_value=default_end_date
        )
        
        predict_btn = st.button('Predict', type='primary')
    
    # Filter data for selected stock and date range
    stock_data = df[df['Ticker'] == ticker].copy()
    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
        # Convert Date column to datetime explicitly after making the copy, handling timezone
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.tz_localize(None)
        mask = (stock_data['Date'].dt.normalize() >= start_date) & \
               (stock_data['Date'].dt.normalize() <= end_date)
        stock_data = stock_data[mask].sort_values('Date')
        
        # Calculate derived features
        stock_data['HL_Range'] = (stock_data['High'] - stock_data['Low']) / stock_data['Close']
        stock_data['Volume_1d_chg'] = stock_data['Volume'].pct_change()
        stock_data['Price_1d_chg'] = stock_data['Close'].pct_change()
        stock_data['future_return'] = stock_data['Close'].shift(-1) / stock_data['Close'] - 1
        
        # Fill NaN values
        stock_data = stock_data.fillna(method='ffill')
        stock_data = stock_data.fillna(0)
    
    # Main panel
    st.title('AI-Driven Stock Trend Forecasting')
    
    # Validate we have data to display
    if stock_data.empty:
        st.error("No data available for the selected date range. Please adjust the dates.")
        return
        
    # Current info
    try:
        latest_data = stock_data.iloc[-1]
        
        # Format metrics with proper styling
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div style='background-color: #1E1E1E; padding: 20px; border-radius: 5px; text-align: center;'>
                    <h4 style='margin: 0; color: #9E9E9E;'>Selected Stock</h4>
                    <h2 style='margin: 10px 0 0 0; color: white;'>{}</h2>
                </div>
            """.format(ticker), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div style='background-color: #1E1E1E; padding: 20px; border-radius: 5px; text-align: center;'>
                    <h4 style='margin: 0; color: #9E9E9E;'>Current Price</h4>
                    <h2 style='margin: 10px 0 0 0; color: white;'>${:.2f}</h2>
                </div>
            """.format(latest_data['Close']), unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
                <div style='background-color: #1E1E1E; padding: 20px; border-radius: 5px; text-align: center;'>
                    <h4 style='margin: 0; color: #9E9E9E;'>Date</h4>
                    <h2 style='margin: 10px 0 0 0; color: white;'>{}</h2>
                </div>
            """.format(latest_data['Date'].strftime('%Y-%m-%d')), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")
        return
    
    # Show prediction if button clicked
    if predict_btn:
        try:
            # Get latest features for prediction
            latest_features = stock_data.iloc[-1][model.feature_names_]
            
            # Make prediction and map numeric labels to string labels
            numeric_prediction = model.predict([latest_features])[0]
            label_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
            prediction = label_map[numeric_prediction]
            probabilities = model.predict_proba([latest_features])[0]
            
            # Display prediction
            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                st.markdown(f"""
                    <div style='background-color: {get_prediction_color(prediction)}; 
                              padding: 20px; border-radius: 5px; color: white;
                              text-align: center; font-size: 24px;'>
                        Signal: {prediction}
                    </div>
                """, unsafe_allow_html=True)
            
            # Plot prediction probabilities
            with pred_col2:
                prob_fig = plot_prediction_probabilities(
                    probabilities, 
                    ['Sell', 'Hold', 'Buy']  # Use string labels instead of numeric
                )
                st.plotly_chart(prob_fig, use_container_width=True)
            
            # Signal Explanation & Feature Impact section
            st.header("Signal Explanation & Feature Impact")
            
            # Display signal explanation
            explanation = generate_signal_explanation(latest_data, prediction)
            st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                    <h3 style='color: #E0E0E0; margin-bottom: 15px;'>Why this signal?</h3>
                    <p style='color: #FFFFFF; line-height: 1.6;'>{explanation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display SHAP values if available
            if shap_values is not None:
                try:
                    import shap
                    st.subheader("Top Features Influencing Prediction")
                    
                    # Get SHAP values for current prediction
                    feature_importance = pd.DataFrame({
                        'Feature': model.feature_names_,
                        'Importance': np.abs(shap_values[0])
                    })
                    feature_importance = feature_importance.sort_values(
                        'Importance', ascending=True
                    ).tail(5)
                    
                    # Plot SHAP values
                    fig_shap = go.Figure(go.Bar(
                        x=feature_importance['Importance'],
                        y=feature_importance['Feature'],
                        orientation='h'
                    ))
                    fig_shap.update_layout(
                        title='Top 5 Features Impact',
                        xaxis_title='SHAP Value (Impact)',
                        yaxis_title='Feature',
                        height=300,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)',
                            tickfont=dict(color='white')
                        ),
                        yaxis=dict(
                            tickfont=dict(color='white')
                        )
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.info("SHAP visualization not available")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Charts
    st.subheader('Technical Analysis')
    fig = plot_stock_chart(stock_data, ticker)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 