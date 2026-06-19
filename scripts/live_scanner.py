"""
Live Market Scanner — Forward test the ML models on today's live market.

This script loops through all configured major pairs, downloads the latest
real-time 15m data, and evaluates the ML model to generate live trading signals.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

from config.settings import (
    PRIMARY_TIMEFRAME, DATA_LOOKBACK_DAYS, YFINANCE_TICKERS,
    ATR_MULTIPLIER, REWARD_RISK_RATIO, ML_LONG_THRESHOLD, ML_SHORT_THRESHOLD,
    INITIAL_BALANCE, RISK_PER_TRADE_PCT, PIP_SIZES, SPREADS
)
from data.loader import clean_data
from ml.features import compute_all_features, extract_feature_matrix, get_supertrend_direction
from ml.predictor import MLPredictor
from risk.session_filter import SessionFilter
from strategy.position_sizer import calculate_position_size

def get_live_data(pair: str) -> pd.DataFrame:
    """Download live data using yfinance."""
    ticker = YFINANCE_TICKERS.get(pair)
    if not ticker:
        raise ValueError(f"Unknown ticker for {pair}")
        
    df_raw = yf.download(
        tickers=ticker,
        period=f"{DATA_LOOKBACK_DAYS}d",
        interval=PRIMARY_TIMEFRAME,
        auto_adjust=True,
        progress=False
    )
    if df_raw.empty:
        return df_raw
        
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    df_raw.index.name = "timestamp"
    df_raw = df_raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low", 
        "Close": "close", "Volume": "volume"
    })
    
    return clean_data(df_raw)

def generate_signals():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
    session_filter = SessionFilter(require_overlap_only=False)
    
    print("\n" + "=" * 90)
    print("  LIVE MARKET SCANNER - PROP FIRM ML SYSTEM")
    print(f"  Timeframe: {PRIMARY_TIMEFRAME} | Model: RandomForest HFT | Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 90)
    
    signals = []
    
    for pair in pairs:
        try:
            # 1. Get Live Data
            df = get_live_data(pair)
            if df.empty:
                continue
                
            # 2. Compute Features
            df = compute_all_features(df)
            
            # Get the very last bar (the currently active/forming bar)
            current_bar = df.iloc[-1]
            current_time = df.index[-1]
            current_price = current_bar['close']
            
            # 3. Check Session
            session_status = ""
            if not session_filter.is_tradeable(current_time):
                session_status = " (Off-Session)"
                
            # 4. Load Model and Predict
            predictor = MLPredictor(pair, PRIMARY_TIMEFRAME)
            probas_df = predictor.predict_proba_batch(df.iloc[[-1]])
            
            prob_long = probas_df['prob_1'].iloc[0] if 'prob_1' in probas_df else 0.0
            prob_short = probas_df['prob_-1'].iloc[0] if 'prob_-1' in probas_df else 0.0
            
            st_dir = get_supertrend_direction(df).iloc[-1]
            
            # 5. Generate Signal
            signal = "NEUTRAL"
            confidence = 0.0
            
            if prob_long >= ML_LONG_THRESHOLD and st_dir == 1:
                signal = "LONG"
                confidence = prob_long
            elif prob_short >= ML_SHORT_THRESHOLD and st_dir == -1:
                signal = "SHORT"
                confidence = prob_short
                
            if signal == "NEUTRAL":
                signals.append([pair, round(current_price, 5), "NEUTRAL", "-", "-", "-", "-", f"WAITING{session_status}"])
                continue
                
            # 6. Calculate Trade Parameters
            atr = df['high'].iloc[-14:] - df['low'].iloc[-14:]
            atr_val = atr.mean() # Approximate recent ATR
            sl_dist = atr_val * ATR_MULTIPLIER
            tp_dist = sl_dist * REWARD_RISK_RATIO
            
            if signal == "LONG":
                entry = current_price
                sl = entry - sl_dist
                tp = entry + tp_dist
            else:
                entry = current_price
                sl = entry + sl_dist
                tp = entry - tp_dist
                
            # Calculate lot size
            lots = calculate_position_size(
                account_balance=INITIAL_BALANCE,
                stop_distance_price=sl_dist,
                pair=pair,
                risk_pct=RISK_PER_TRADE_PCT
            )
            
            signals.append([
                pair, 
                round(current_price, 5), 
                f"{signal} ({confidence:.0%})", 
                round(entry, 5), 
                round(sl, 5), 
                round(tp, 5), 
                f"{lots} Lots",
                f"ACTIVE{session_status}"
            ])
            
        except Exception as e:
            signals.append([pair, "ERROR", str(e)[:20], "-", "-", "-", "-", "-"])
            
        time.sleep(1) # Prevent rate limiting

    # Print Table
    headers = ["Pair", "Current Price", "ML Signal", "Entry", "Stop Loss", "Take Profit", "Size ($100k)", "Status"]
    print(tabulate(signals, headers=headers, tablefmt="simple"))
    print("\n* Note: Trade size is calculated using 1.5% Kelly-scaled risk on a $100k account.")
    print("=" * 90)

if __name__ == "__main__":
    generate_signals()
