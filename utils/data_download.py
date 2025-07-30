"""
Data Download Utility Module

This module provides functions for downloading and validating financial data
from yfinance. It includes error handling, data validation, and logging.
"""

import pandas as pd
import yfinance as yf
import logging
from typing import Optional, Dict, List, Union, Tuple
from datetime import datetime
import time
import random
from pathlib import Path
from tqdm import tqdm

def download_yf_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = '1d'
) -> Optional[pd.DataFrame]:
    """
    Download historical data for a single ticker using yfinance.
    
    Args:
        ticker (str): Stock/ETF ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        interval (str): Data interval (default: '1d')
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with historical data or None if error
    """
    try:
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        
        if df.empty:
            logging.error(f"No data downloaded for {ticker}")
            return None
            
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Add ticker column
        df['Symbol'] = ticker
        
        # Check for Adj Close column and create if missing
        if 'Adj Close' not in df.columns:
            logging.warning(f"'Adj Close' missing for {ticker}, fallback to 'Close' used.")
            df['Adj Close'] = df['Close']
            
        return df
        
    except Exception as e:
        logging.error(f"Error downloading {ticker}: {str(e)}")
        return None

def validate_downloaded_data(
    df: pd.DataFrame,
    required_columns: List[str]
) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate downloaded data against required columns and check for issues.
    
    Args:
        df (pd.DataFrame): Downloaded data to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        Dict[str, Union[bool, List[str]]]: Validation results with status and issues
    """
    validation_result = {
        'is_valid': True,
        'issues': []
    }
    
    try:
        # Check if DataFrame is empty
        if df is None or df.empty:
            validation_result['is_valid'] = False
            validation_result['issues'].append("DataFrame is empty or None")
            return validation_result
        
        # Check for missing columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Missing columns: {missing_columns}")
        
        # Check for missing values
        null_counts = df[required_columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if not columns_with_nulls.empty:
            validation_result['is_valid'] = False
            for col, count in columns_with_nulls.items():
                validation_result['issues'].append(f"{col}: {count} missing values")
        
        # Check date range
        if 'Date' in df.columns and len(df) > 0:
            date_range = f"{df['Date'].min()} to {df['Date'].max()}"
            validation_result['date_range'] = date_range
            
        # Check data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"{col} is not numeric")
        
        return validation_result
        
    except Exception as e:
        validation_result['is_valid'] = False
        validation_result['issues'].append(f"Validation error: {str(e)}")
        return validation_result

def batch_download_data(
    tickers: Dict[str, str],
    start_date: str,
    end_date: str,
    output_dir: str,
    required_columns: List[str],
    interval: str = '1d',
    delay_range: Tuple[float, float] = (0.5, 1.5)
) -> List[str]:
    """
    Download data for multiple tickers with progress tracking and validation.
    
    Args:
        tickers (Dict[str, str]): Dictionary of ticker symbols to names/sectors
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_dir (str): Directory to save downloaded data
        required_columns (List[str]): List of required columns for validation
        interval (str): Data interval (default: '1d')
        delay_range (Tuple[float, float]): Range for random delay between downloads
        
    Returns:
        List[str]: List of failed downloads
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Track failed downloads
    failed_downloads = []
    
    # Download data for each ticker
    for ticker, name in tqdm(tickers.items(), desc="Downloading data"):
        try:
            logging.info(f"Downloading data for {ticker} ({name})")
            
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(*delay_range))
            
            # Download data
            df = download_yf_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            if df is not None:
                # Validate data
                validation_result = validate_downloaded_data(df, required_columns)
                
                if validation_result['is_valid']:
                    # Save to CSV
                    output_path = f'{output_dir}/{name}.csv'
                    df.to_csv(output_path, index=False)
                    logging.info(f"Saved {name} data to {output_path}")
                    
                    # Print summary
                    print(f"\n{ticker} ({name}) Summary:")
                    print(f"- Date Range: {validation_result.get('date_range', 'Not available')}")
                    print(f"- Trading Days: {len(df)}")
                    print(f"- File saved: {output_path}")
                else:
                    logging.error(f"Validation failed for {ticker}: {validation_result['issues']}")
                    failed_downloads.append(ticker)
            else:
                failed_downloads.append(ticker)
                
        except Exception as e:
            logging.error(f"Unexpected error processing {ticker}: {str(e)}")
            failed_downloads.append(ticker)
            continue
    
    # Log download statistics
    if failed_downloads:
        logging.warning(f"Failed to download {len(failed_downloads)} tickers: {', '.join(failed_downloads)}")
    else:
        logging.info("All downloads completed successfully")
        
    return failed_downloads 