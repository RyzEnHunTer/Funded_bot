"""
Data Download Script -- Fetch Forex candle data from yfinance.

Downloads OHLCV data for configured pairs and saves to data/raw/.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --pair EURUSD --timeframe 1h --days 720
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf
import pandas as pd

from config.settings import (
    PAIRS, PRIMARY_PAIR, PRIMARY_TIMEFRAME,
    YFINANCE_TICKERS, RAW_DIR, DATA_LOOKBACK_DAYS,
)
from data.loader import validate_data, clean_data, save_processed


def download_pair(pair: str, timeframe: str = "1h", days: int = 720) -> pd.DataFrame:
    """
    Download Forex candle data from yfinance.

    Parameters
    ----------
    pair : str
        Currency pair (e.g., "EURUSD").
    timeframe : str
        Candle timeframe (e.g., "1h", "15m", "1d").
    days : int
        Number of days of historical data to fetch.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    """
    ticker = YFINANCE_TICKERS.get(pair)
    if not ticker:
        raise ValueError(f"Unknown pair: {pair}. Available: {list(YFINANCE_TICKERS.keys())}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"  Downloading {pair} ({ticker}) -- {timeframe} -- last {days} days...")

    # yfinance interval mapping
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d",
    }

    interval = interval_map.get(timeframe)
    if not interval:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # yfinance has limits on how far back intraday data goes:
    # 1m: 7 days, 5m/15m/30m: 60 days, 1h: 730 days, 1d: unlimited
    max_days = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730, "4h": 730, "1d": 10000}
    actual_days = min(days, max_days.get(timeframe, 730))

    if actual_days < days:
        print(f"  WARNING yfinance limits {timeframe} data to {actual_days} days (requested {days})")

    start_date = end_date - timedelta(days=actual_days)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {pair} ({ticker})")

    # Flatten MultiIndex columns if present (yfinance sometimes returns multi-level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename index
    df.index.name = "timestamp"

    # Keep only OHLCV columns
    keep_cols = ["open", "high", "low", "close", "volume"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available]

    # Add volume column if missing
    if "volume" not in df.columns:
        df["volume"] = 0

    return df


def save_raw(df: pd.DataFrame, pair: str, timeframe: str) -> Path:
    """Save raw data to CSV."""
    filepath = RAW_DIR / f"{pair}_{timeframe}.csv"
    df.to_csv(filepath)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Download Forex data from yfinance")
    parser.add_argument("--pair", type=str, default=None,
                        help="Specific pair to download (default: all configured pairs)")
    parser.add_argument("--timeframe", type=str, default=PRIMARY_TIMEFRAME,
                        help=f"Timeframe (default: {PRIMARY_TIMEFRAME})")
    parser.add_argument("--days", type=int, default=DATA_LOOKBACK_DAYS,
                        help=f"Days of history (default: {DATA_LOOKBACK_DAYS})")
    args = parser.parse_args()

    pairs_to_download = [args.pair] if args.pair else [PRIMARY_PAIR]

    print("=" * 64)
    print("  FOREX DATA DOWNLOADER")
    print("=" * 64)
    print(f"  Timeframe : {args.timeframe}")
    print(f"  Lookback  : {args.days} days")
    print(f"  Pairs     : {', '.join(pairs_to_download)}")
    print(f"  Output    : {RAW_DIR}")
    print("=" * 64)

    for pair in pairs_to_download:
        print(f"\n{'-' * 48}")
        print(f"  {pair}")
        print(f"{'-' * 48}")

        try:
            # Download
            df = download_pair(pair, args.timeframe, args.days)

            # Save raw
            raw_path = save_raw(df, pair, args.timeframe)
            print(f"  OK Raw data saved: {raw_path.name}")
            print(f"    Rows: {len(df):,}")
            print(f"    Range: {df.index[0]} -> {df.index[-1]}")

            # Validate
            report = validate_data(df, pair)
            if report["valid"]:
                print(f"  OK Validation passed ({report['days']} days of data)")
            else:
                print(f"  WARNING Validation issues:")
                for issue in report["issues"]:
                    print(f"    {issue}")

            # Clean and save processed
            df_clean = clean_data(df)
            proc_path = save_processed(df_clean, pair, args.timeframe)
            print(f"  OK Processed data saved: {proc_path.name}")
            print(f"    Clean rows: {len(df_clean):,}")

        except Exception as e:
            print(f"  X Error: {e}")

    print(f"\n{'=' * 64}")
    print("  DOWNLOAD COMPLETE")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
