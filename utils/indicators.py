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


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV is a momentum indicator that uses volume flow to predict changes in stock price.
    Formula:
    If closing price > previous close: OBV = previous OBV + current volume
    If closing price < previous close: OBV = previous OBV - current volume
    If closing price = previous close: OBV = previous OBV
    
    Args:
        close (pd.Series): Closing prices
        volume (pd.Series): Trading volumes
        
    Returns:
        pd.Series: OBV values
    """
    price_change = close.diff()
    
    # Initialize OBV series
    obv = pd.Series(0, index=close.index)
    
    # First value is NaN since we don't have previous close
    obv.iloc[0] = np.nan
    
    # Calculate OBV based on price changes
    for i in range(1, len(close)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_bollinger_bands(close: pd.Series, 
                            window: int = 20,
                            num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of:
    - Middle Band: n-period simple moving average (SMA)
    - Upper Band: SMA + (standard deviation × num_std)
    - Lower Band: SMA - (standard deviation × num_std)
    
    Args:
        close (pd.Series): Closing prices
        window (int): Number of periods for moving average (default: 20)
        num_std (float): Number of standard deviations for bands (default: 2.0)
        
    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: Upper band, middle band, lower band
    """
    # Calculate middle band (SMA)
    middle_band = close.rolling(window=window).mean()
    
    # Calculate standard deviation
    std = close.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_adx(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures the strength of a trend (regardless of direction).
    Components:
    - +DI: Positive Directional Indicator
    - -DI: Negative Directional Indicator
    - ADX: Smoothed average of DX (Directional Index)
    
    Formula:
    1. Calculate +DM and -DM (Directional Movement)
    2. Calculate TR (True Range)
    3. Calculate +DI and -DI
    4. Calculate DX = |+DI - -DI| / (+DI + -DI) × 100
    5. ADX = EMA of DX
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Closing prices
        window (int): Number of periods (default: 14)
        
    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: ADX, +DI, -DI
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    pos_dm = pd.Series(0, index=high.index)
    neg_dm = pd.Series(0, index=high.index)
    
    pos_dm[up_move > down_move] = up_move[up_move > down_move]
    pos_dm[up_move <= 0] = 0
    neg_dm[down_move > up_move] = down_move[down_move > up_move]
    neg_dm[down_move <= 0] = 0
    
    # Calculate smoothed TR and DM
    tr_smooth = pd.Series(index=tr.index, dtype=float)
    pos_dm_smooth = pd.Series(index=tr.index, dtype=float)
    neg_dm_smooth = pd.Series(index=tr.index, dtype=float)
    
    # Initialize first values
    tr_smooth.iloc[:window] = np.nan
    pos_dm_smooth.iloc[:window] = np.nan
    neg_dm_smooth.iloc[:window] = np.nan
    
    # First smoothed values
    tr_smooth.iloc[window] = tr.iloc[1:window+1].sum()
    pos_dm_smooth.iloc[window] = pos_dm.iloc[1:window+1].sum()
    neg_dm_smooth.iloc[window] = neg_dm.iloc[1:window+1].sum()
    
    # Calculate subsequent values
    for i in range(window + 1, len(tr)):
        tr_smooth.iloc[i] = tr_smooth.iloc[i-1] - (tr_smooth.iloc[i-1]/window) + tr.iloc[i]
        pos_dm_smooth.iloc[i] = pos_dm_smooth.iloc[i-1] - (pos_dm_smooth.iloc[i-1]/window) + pos_dm.iloc[i]
        neg_dm_smooth.iloc[i] = neg_dm_smooth.iloc[i-1] - (neg_dm_smooth.iloc[i-1]/window) + neg_dm.iloc[i]
    
    # Calculate +DI and -DI
    pos_di = 100 * (pos_dm_smooth / tr_smooth)
    neg_di = 100 * (neg_dm_smooth / tr_smooth)
    
    # Calculate DX
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    
    # Calculate ADX
    adx = pd.Series(index=dx.index, dtype=float)
    adx.iloc[:2*window-1] = np.nan
    adx.iloc[2*window-1] = dx.iloc[window:2*window].mean()
    
    for i in range(2*window, len(dx)):
        adx.iloc[i] = (adx.iloc[i-1] * (window-1) + dx.iloc[i]) / window
    
    return adx, pos_di, neg_di


def calculate_cci(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 window: int = 20,
                 constant: float = 0.015) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    CCI measures the difference between the current price and historical average price.
    Formula:
    CCI = (Typical Price - SMA(Typical Price)) / (constant × Mean Deviation)
    where:
    - Typical Price = (High + Low + Close) / 3
    - Mean Deviation = mean(abs(Typical Price - SMA(Typical Price)))
    - constant = 0.015 (traditional)
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Closing prices
        window (int): Number of periods (default: 20)
        constant (float): Scaling constant (default: 0.015)
        
    Returns:
        pd.Series: CCI values
    """
    # Calculate typical price
    tp = (high + low + close) / 3
    
    # Calculate SMA of typical price
    tp_sma = tp.rolling(window=window).mean()
    
    # Calculate mean deviation
    mean_dev = pd.Series(index=tp.index)
    for i in range(window-1, len(tp)):
        mean_dev.iloc[i] = abs(tp.iloc[i-window+1:i+1] - tp_sma.iloc[i]).mean()
    
    # Calculate CCI
    cci = (tp - tp_sma) / (constant * mean_dev)
    
    # Set NaN values for the first window-1 periods
    cci.iloc[:window-1] = np.nan
    
    return cci


if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'High': None,
        'Low': None,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Calculate High and Low based on Close with some random variation
    sample_data['High'] = sample_data['Close'] + abs(np.random.randn(len(dates))) * 0.5
    sample_data['Low'] = sample_data['Close'] - abs(np.random.randn(len(dates))) * 0.5
    
    # Test OBV
    obv = calculate_obv(sample_data['Close'], sample_data['Volume'])
    print("\nOn-Balance Volume (first 5 values):")
    print(obv.head())
    
    # Test Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(sample_data['Close'])
    print("\nBollinger Bands (first 5 values):")
    print("Upper:", upper.head())
    print("Middle:", middle.head())
    print("Lower:", lower.head())
    
    # Test ADX
    adx, plus_di, minus_di = calculate_adx(
        sample_data['High'],
        sample_data['Low'],
        sample_data['Close']
    )
    print("\nADX Components (first 5 values):")
    print("ADX:", adx.head())
    print("+DI:", plus_di.head())
    print("-DI:", minus_di.head())
    
    # Test CCI
    cci = calculate_cci(
        sample_data['High'],
        sample_data['Low'],
        sample_data['Close']
    )
    print("\nCCI (first 5 values):")
    print(cci.head()) 