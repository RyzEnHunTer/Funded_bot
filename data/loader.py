"""
Data Loader -- load, validate, and preprocess Forex candle data.

Handles both raw CSV/parquet files downloaded by the download script
and provides clean DataFrames for feature engineering and backtesting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DIR, PROCESSED_DIR, PIP_SIZES


def load_raw_data(pair: str, timeframe: str = "1h") -> pd.DataFrame:
    """
    Load raw candle data for a pair/timeframe from the data/raw directory.

    Parameters
    ----------
    pair : str
        Currency pair (e.g., "EURUSD").
    timeframe : str
        Timeframe (e.g., "1h", "15m").

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    filename = f"{pair}_{timeframe}.csv"
    filepath = RAW_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Run 'python scripts/download_data.py' to download data first."
        )

    df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")

    # Ensure correct column names (lowercase)
    df.columns = [c.lower().strip() for c in df.columns]

    # Validate required columns
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {filename}: {missing}")

    # Add volume if missing (Forex doesn't always have real volume)
    if "volume" not in df.columns:
        df["volume"] = 0

    # Sort by time
    df.sort_index(inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    return df


def validate_data(df: pd.DataFrame, pair: str) -> dict:
    """
    Validate candle data integrity.

    Returns a dict with validation results and any issues found.
    """
    issues = []

    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.any():
        for col, count in nan_counts.items():
            if count > 0:
                issues.append(f"  {col}: {count} NaN values")

    # Check OHLC relationships: high >= low, high >= open/close, low <= open/close
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        issues.append(f"  {bad_hl} bars where high < low")

    bad_ho = (df["high"] < df["open"]).sum()
    if bad_ho > 0:
        issues.append(f"  {bad_ho} bars where high < open")

    bad_hc = (df["high"] < df["close"]).sum()
    if bad_hc > 0:
        issues.append(f"  {bad_hc} bars where high < close")

    bad_lo = (df["low"] > df["open"]).sum()
    if bad_lo > 0:
        issues.append(f"  {bad_lo} bars where low > open")

    bad_lc = (df["low"] > df["close"]).sum()
    if bad_lc > 0:
        issues.append(f"  {bad_lc} bars where low > close")

    # Check for zero or negative prices
    zero_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if zero_prices > 0:
        issues.append(f"  {zero_prices} bars with zero/negative prices")

    # Check for gaps (expected on weekends for Forex)
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        large_gaps = (time_diffs > median_diff * 10).sum()
        if large_gaps > 0:
            issues.append(f"  {large_gaps} large time gaps (>10x median interval, expected for weekends)")

    result = {
        "pair": pair,
        "rows": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "days": (df.index[-1] - df.index[0]).days,
        "valid": len(issues) == 0,
        "issues": issues,
    }

    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean candle data by handling NaN values and obvious errors.

    - Forward-fills NaN values (last valid price)
    - Fixes OHLC violations
    - Removes bars with zero prices
    """
    df = df.copy()

    # Remove bars with zero/negative prices
    valid_mask = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    df = df[valid_mask]

    # Forward-fill NaN values
    df.ffill(inplace=True)

    # Drop any remaining NaN (at the start)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Fix OHLC violations: ensure high is the max and low is the min
    ohlc = df[["open", "high", "low", "close"]]
    df["high"] = ohlc.max(axis=1)
    df["low"]  = ohlc.min(axis=1)

    return df


def save_processed(df: pd.DataFrame, pair: str, timeframe: str) -> Path:
    """Save cleaned data to the processed directory."""
    filepath = PROCESSED_DIR / f"{pair}_{timeframe}.csv"
    df.to_csv(filepath)
    return filepath


def load_processed(pair: str, timeframe: str) -> pd.DataFrame:
    """Load previously cleaned data from the processed directory."""
    filepath = PROCESSED_DIR / f"{pair}_{timeframe}.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run the data pipeline first."
        )
    df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
    return df
