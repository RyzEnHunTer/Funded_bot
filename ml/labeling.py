"""
Triple-Barrier Labeling Engine.

Implements the triple-barrier method from Marcos López de Prado's
'Advances in Financial Machine Learning'. Each bar is labeled based
on which of three barriers is hit first:

  +1 — Upper barrier (profit target hit)
  -1 — Lower barrier (stop-loss hit)
   0 — Vertical barrier (time expiry — no clear direction)
"""

import numpy as np
import pandas as pd
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    ATR_MULTIPLIER, MIN_DISTANCE_PCT, VERTICAL_BARRIER,
    ATR_PERIOD, WARMUP_BARS,
)
from ml.features import compute_all_features, FEATURE_NAMES, _atr


def compute_barrier_distance(close: pd.Series, high: pd.Series,
                              low: pd.Series, atr_multiplier: float = ATR_MULTIPLIER,
                              min_distance_pct: float = MIN_DISTANCE_PCT) -> pd.Series:
    """
    Compute the barrier distance for each bar.

    distance = max(ATR * atr_multiplier, close * min_distance_pct)

    This scales with volatility while maintaining a floor to prevent
    near-zero distances in very low-volatility environments.
    """
    atr = _atr(high, low, close, ATR_PERIOD)
    atr_distance = atr * atr_multiplier
    price_floor = close * min_distance_pct
    return pd.DataFrame({"atr": atr_distance, "floor": price_floor}).max(axis=1)


def label_triple_barrier(df: pd.DataFrame,
                          atr_multiplier: float = ATR_MULTIPLIER,
                          min_distance_pct: float = MIN_DISTANCE_PCT,
                          vertical_bars: int = VERTICAL_BARRIER) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to every valid bar.

    For each bar i:
      1. Compute barrier distance from ATR
      2. Set upper barrier = close[i] + distance
      3. Set lower barrier = close[i] - distance
      4. Look forward up to vertical_bars bars
      5. Label based on which barrier is hit first:
         - Upper hit first -> +1
         - Lower hit first -> -1
         - Neither hit within vertical_bars -> 0

    Uses HIGH and LOW of future bars (not just close) to check barrier hits,
    which is more realistic — intrabar price action can hit SL/TP.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with features already computed.
    atr_multiplier : float
        Multiplier for ATR to set barrier distance.
    min_distance_pct : float
        Minimum barrier distance as fraction of price.
    vertical_bars : int
        Maximum bars to look forward before vertical barrier fires.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [features..., label, barrier_distance, timestamp]
        Only rows with valid labels are included.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    # Compute barrier distances
    distances = compute_barrier_distance(
        df["close"], df["high"], df["low"],
        atr_multiplier, min_distance_pct
    ).values

    labels = np.full(n, np.nan)
    barrier_dists = np.full(n, np.nan)

    # For each bar, look forward to find which barrier is hit first
    for i in range(WARMUP_BARS, n - vertical_bars):
        dist = distances[i]

        if np.isnan(dist) or dist <= 0:
            continue

        entry_price = close[i]
        upper_barrier = entry_price + dist
        lower_barrier = entry_price - dist

        label = 0  # Default: vertical barrier (time expiry)

        for j in range(i + 1, min(i + vertical_bars + 1, n)):
            # Check if HIGH touches upper barrier (profit target)
            upper_hit = high[j] >= upper_barrier
            # Check if LOW touches lower barrier (stop-loss)
            lower_hit = low[j] <= lower_barrier

            if upper_hit and lower_hit:
                # Both hit in the same bar — use close to decide
                # This is ambiguous; label based on which side close is closer to
                if close[j] >= entry_price:
                    label = 1
                else:
                    label = -1
                break
            elif upper_hit:
                label = 1
                break
            elif lower_hit:
                label = -1
                break

        labels[i] = label
        barrier_dists[i] = dist

    df = df.copy()
    df["label"] = labels
    df["barrier_distance"] = barrier_dists

    # Drop rows without labels (warmup period and tail)
    df.dropna(subset=["label"], inplace=True)

    # Also drop rows where any feature is NaN
    df.dropna(subset=FEATURE_NAMES, inplace=True)

    # Convert label to int
    df["label"] = df["label"].astype(int)

    return df


def create_labeled_dataset(df: pd.DataFrame,
                            atr_multiplier: float = ATR_MULTIPLIER,
                            min_distance_pct: float = MIN_DISTANCE_PCT,
                            vertical_bars: int = VERTICAL_BARRIER) -> Tuple[pd.DataFrame, dict]:
    """
    Full pipeline: compute features -> label -> return clean dataset.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        - DataFrame with feature columns + 'label' column
        - Summary statistics dict
    """
    # Step 1: Compute features
    print("  Computing features...")
    df_feat = compute_all_features(df)

    # Step 2: Apply triple-barrier labeling
    print("  Applying triple-barrier labeling...")
    df_labeled = label_triple_barrier(df_feat, atr_multiplier, min_distance_pct, vertical_bars)

    # Step 3: Extract clean dataset
    feature_cols = FEATURE_NAMES + ["label", "barrier_distance"]
    dataset = df_labeled[feature_cols].copy()

    # Summary statistics
    label_counts = dataset["label"].value_counts().sort_index()
    stats = {
        "total_samples": len(dataset),
        "label_distribution": {
            "+1 (profit)": int(label_counts.get(1, 0)),
            " 0 (timeout)": int(label_counts.get(0, 0)),
            "-1 (stoploss)": int(label_counts.get(-1, 0)),
        },
        "avg_barrier_distance": float(dataset["barrier_distance"].mean()),
        "date_range": f"{df_labeled.index[0]} -> {df_labeled.index[-1]}",
    }

    return dataset, stats


def save_labeled_data(dataset: pd.DataFrame, pair: str, timeframe: str,
                       output_dir: Path = None) -> Path:
    """Save labeled dataset to CSV."""
    if output_dir is None:
        from config.settings import FEATURES_DIR
        output_dir = FEATURES_DIR

    filepath = output_dir / f"{pair}_{timeframe}_labeled.csv"
    dataset.to_csv(filepath)
    return filepath


def load_labeled_data(pair: str, timeframe: str,
                       data_dir: Path = None) -> pd.DataFrame:
    """Load previously saved labeled dataset."""
    if data_dir is None:
        from config.settings import FEATURES_DIR
        data_dir = FEATURES_DIR

    filepath = data_dir / f"{pair}_{timeframe}_labeled.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Labeled data not found: {filepath}")

    return pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
