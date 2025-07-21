"""
Technical Analysis Indicators Module

This module provides custom implementations of popular technical analysis indicators
without relying on external TA libraries. Each function is implemented based on
standard technical analysis formulas and best practices.
"""

import numpy as np
import pandas as pd


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA) for a given series.
    
    The EMA gives more weight to recent prices compared to the simple moving average.
    Formula: EMA_t = α * Price_t + (1 - α) * EMA_{t-1}
    where α = 2 / (window + 1)
    
    Args:
        series (pd.Series): Price series (typically closing prices)
        window (int): Number of periods for EMA calculation
        
    Returns:
        pd.Series: EMA values with same length as input series
    """
    alpha = 2 / (window + 1)
    # Initialize with SMA for first window periods
    ema = pd.Series(index=series.index, dtype=float)
    ema.iloc[:window] = series.iloc[:window].mean()
    
    # Calculate EMA for remaining periods
    for i in range(window, len(series)):
        ema.iloc[i] = series.iloc[i] * alpha + ema.iloc[i-1] * (1 - alpha)
    
    return ema


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) for a given series.
    
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.
    
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = (Average Gain / Average Loss) over n periods
    
    Args:
        series (pd.Series): Price series (typically closing prices)
        window (int): Number of periods for RSI calculation (default: 14)
        
    Returns:
        pd.Series: RSI values ranging from 0 to 100
    """
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Initialize average gains and losses with simple means
    avg_gains = pd.Series(index=series.index, dtype=float)
    avg_losses = pd.Series(index=series.index, dtype=float)
    
    # First values are NaN for window-1 periods
    avg_gains.iloc[:window] = np.nan
    avg_losses.iloc[:window] = np.nan
    
    # Initial averages
    avg_gains.iloc[window] = gains.iloc[1:window+1].mean()
    avg_losses.iloc[window] = losses.iloc[1:window+1].mean()
    
    # Calculate smoothed averages
    for i in range(window + 1, len(series)):
        avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (window-1) + gains.iloc[i]) / window
        avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (window-1) + losses.iloc[i]) / window
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(series: pd.Series, 
                  fast_window: int = 12,
                  slow_window: int = 26,
                  signal_window: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD) components.
    
    MACD measures the relationship between two EMAs of different lengths.
    Components:
    - MACD Line: Difference between fast and slow EMAs
    - Signal Line: EMA of MACD Line
    - Histogram: Difference between MACD Line and Signal Line
    
    Args:
        series (pd.Series): Price series (typically closing prices)
        fast_window (int): Periods for fast EMA (default: 12)
        slow_window (int): Periods for slow EMA (default: 26)
        signal_window (int): Periods for signal line EMA (default: 9)
        
    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: MACD line, signal line, and histogram
    """
    # Calculate fast and slow EMAs
    ema_fast = calculate_ema(series, fast_window)
    ema_slow = calculate_ema(series, slow_window)
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = calculate_ema(macd_line, signal_window)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram 