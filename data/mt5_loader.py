"""
MT5 Loader — Direct MetaTrader 5 Data Fetcher.

Replaces yfinance by pulling the exact broker-specific historical rates.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

def initialize_mt5() -> bool:
    """Initialize connection to MT5 terminal."""
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    return True

def get_mt5_data(symbol: str, timeframe: str, num_bars: int) -> pd.DataFrame:
    """
    Fetch historical data from MT5.
    
    Parameters
    ----------
    symbol : str
        e.g., "EURUSD"
    timeframe : str
        "15m", "1h", etc.
    num_bars : int
        Number of bars to fetch (e.g. 5000)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and ['open', 'high', 'low', 'close', 'volume']
    """
    # Map timeframe string to MT5 timeframe
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    mt5_tf = tf_map.get(timeframe)
    if mt5_tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
        
    # Get rates
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, num_bars)
    
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
        
    # Convert to pandas DataFrame
    df = pd.DataFrame(rates)
    
    # MT5 returns time as integer seconds since epoch
    # Convert to datetime and make it timezone-aware (UTC)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.index.name = "timestamp"
    
    # Rename tick_volume to volume for compatibility with our ML features
    df = df.rename(columns={'tick_volume': 'volume'})
    
    # Keep only required columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    return df
