"""
Feature Engineering — Compute 15 dimensionless features from OHLCV data.

Uses the `ta` library (pure Python, no numba dependency) for indicators.
All features are normalised to be dimensionless (ratios, log returns,
centred scores) so the model generalises across different price levels,
pairs, and volatility regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    EMA_FAST, EMA_MED, EMA_SLOW, EMA_TREND,
    RSI_PERIOD, ADX_PERIOD, ATR_PERIOD,
    BB_PERIOD, BB_STD,
    KELTNER_PERIOD, KELTNER_ATR_MULT,
    STOCH_K_PERIOD, STOCH_D_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER,
)

# Small epsilon to prevent division by zero
EPS = 1e-9

# Ordered list of feature names (alphabetical — must match training order)
FEATURE_NAMES = sorted([
    "adx_centered",
    "atr_expansion",
    "atr_pct",
    "bb_position",
    "ema21_50_ratio",
    "ema9_21_ratio",
    "ema9_dist",
    "keltner_pos",
    "log_return_1",
    "log_return_20",
    "log_return_5",
    "macd_signal_dist",
    "rsi_centered",
    "session_cos",
    "session_sin",
    "stoch_centered",
    "supertrend_dist",
    "wick_body_ratio",
])


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100 - (100 / (1 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = _atr(high, low, close, period)

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / (atr + EPS))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / (atr + EPS))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx


def _stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Stochastic %K."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    return 100 * (close - lowest_low) / (highest_high - lowest_low + EPS)


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute Supertrend indicator.

    Returns DataFrame with columns: 'trend' (the supertrend line) and 'direction' (+1/-1).
    """
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    n = len(close)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    supertrend[0] = upper_band.iloc[0]
    direction[0] = 1

    for i in range(1, n):
        if close.iloc[i] > supertrend[i-1]:
            # Bullish
            supertrend[i] = max(lower_band.iloc[i], supertrend[i-1]) if direction[i-1] == 1 else lower_band.iloc[i]
            direction[i] = 1
        else:
            # Bearish
            supertrend[i] = min(upper_band.iloc[i], supertrend[i-1]) if direction[i-1] == -1 else upper_band.iloc[i]
            direction[i] = -1

    return pd.DataFrame({
        "trend": supertrend,
        "direction": direction,
    }, index=close.index)


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 15 features for every bar in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
        Must have columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 15 new feature columns appended.
        Rows where indicators haven't warmed up will contain NaN.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ── Exponential Moving Averages ───────────────────────────────────────────
    ema9   = _ema(close, EMA_FAST)
    ema21  = _ema(close, EMA_MED)
    ema50  = _ema(close, EMA_SLOW)

    # EMA ratios — dimensionless trend alignment measures
    df["ema9_21_ratio"]  = (ema9 - ema21) / (ema21 + EPS)
    df["ema21_50_ratio"] = (ema21 - ema50) / (ema50 + EPS)

    # Distance from fast EMA — mean reversion signal
    df["ema9_dist"] = (close - ema9) / (ema9 + EPS)

    # ── ATR (Average True Range) & Expansion ──────────────────────────────────
    atr = _atr(high, low, close, ATR_PERIOD)
    atr_fast = _atr(high, low, close, 5)
    df["atr_pct"] = atr / (close + EPS)
    df["atr_expansion"] = atr_fast / (atr + EPS)

    # ── RSI (Relative Strength Index) ─────────────────────────────────────────
    rsi = _rsi(close, RSI_PERIOD)
    df["rsi_centered"] = (rsi - 50) / 50

    # ── ADX (Average Directional Index) ───────────────────────────────────────
    adx = _adx(high, low, close, ADX_PERIOD)
    df["adx_centered"] = (adx - 25) / 25

    # ── Stochastic Oscillator ─────────────────────────────────────────────────
    stoch_k = _stochastic_k(high, low, close, STOCH_K_PERIOD)
    df["stoch_centered"] = (stoch_k - 50) / 50

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_line = _ema(close, MACD_FAST) - _ema(close, MACD_SLOW)
    signal_line = _ema(macd_line, MACD_SIGNAL)
    df["macd_signal_dist"] = (macd_line - signal_line) / (atr + EPS)

    # ── Bollinger Bands Position ──────────────────────────────────────────────
    bb_mid = close.rolling(window=BB_PERIOD).mean()
    bb_std = close.rolling(window=BB_PERIOD).std()
    bb_upper = bb_mid + BB_STD * bb_std
    bb_lower = bb_mid - BB_STD * bb_std
    bb_width = bb_upper - bb_lower + EPS
    df["bb_position"] = (close - bb_lower) / bb_width

    # ── Keltner Channel Position ──────────────────────────────────────────────
    kc_mid = ema21
    kc_upper = kc_mid + KELTNER_ATR_MULT * atr
    kc_lower = kc_mid - KELTNER_ATR_MULT * atr
    kc_width = kc_upper - kc_lower + EPS
    df["keltner_pos"] = (close - kc_lower) / kc_width

    # ── Log Returns ───────────────────────────────────────────────────────────
    df["log_return_1"]  = np.log(close / close.shift(1).replace(0, np.nan))
    df["log_return_5"]  = np.log(close / close.shift(5).replace(0, np.nan))
    df["log_return_20"] = np.log(close / close.shift(20).replace(0, np.nan))

    # ── Supertrend Distance ───────────────────────────────────────────────────
    st = _supertrend(high, low, close, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    df["supertrend_dist"] = (close - st["trend"]) / (atr + EPS)

    # ── Microstructure & Liquidity Features ───────────────────────────────────
    body = (df["open"] - close).abs()
    wick = high - low
    df["wick_body_ratio"] = wick / (body + EPS)

    hours = df.index.hour + df.index.minute / 60.0
    df["session_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["session_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Store supertrend direction and line for signal generation and trailing stops
    df["_supertrend_dir"] = st["direction"]
    df["_supertrend_line"] = st["trend"]

    return df


def extract_feature_matrix(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Extract only the feature columns from a DataFrame with computed features.
    """
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Run compute_all_features() first.")

    features = df[FEATURE_NAMES].copy()
    if dropna:
        features.dropna(inplace=True)
    return features


def get_supertrend_direction(df: pd.DataFrame) -> pd.Series:
    """
    Get the Supertrend direction signal for each bar.

    Returns +1 for bullish, -1 for bearish.
    """
    if "_supertrend_dir" in df.columns:
        return df["_supertrend_dir"]

    # Compute if not already done
    close = df["close"]
    high = df["high"]
    low = df["low"]
    st = _supertrend(high, low, close, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    return st["direction"]
