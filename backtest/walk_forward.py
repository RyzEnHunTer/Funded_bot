"""
Walk-Forward Validation -- The gold standard for ML trading systems.

Retrains the model on rolling windows and tests on truly out-of-sample
forward periods. This prevents lookahead bias and gives a realistic
estimate of live performance.

Schema:
  |--- Train Window (6 months) ---|--- Test Window (2 months) ---|
                      |--- Train Window (6 months) ---|--- Test (2 mo) ---|
                                          |--- Train Window (6 mo) ---|---...
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import INITIAL_BALANCE, WARMUP_BARS
from config.prop_firm_rules import PropFirmRuleSet, DEFAULT_RULES
from ml.features import compute_all_features, FEATURE_NAMES
from ml.labeling import create_labeled_dataset
from ml.trainer import train_and_evaluate
from ml.predictor import MLPredictor
from backtest.engine import BacktestEngine


warnings.filterwarnings("ignore", category=UserWarning)


def walk_forward_validate(
    df: pd.DataFrame,
    pair: str = "EURUSD",
    timeframe: str = "1h",
    train_months: int = 6,
    test_months: int = 2,
    rules: Optional[PropFirmRuleSet] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run walk-forward validation on historical data.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV dataset.
    pair, timeframe : str
        Identifiers for model saving.
    train_months, test_months : int
        Rolling window sizes.
    rules : PropFirmRuleSet
        Prop firm rules for backtest.
    verbose : bool
        Print progress.

    Returns
    -------
    Dict with aggregated results across all windows.
    """
    rules = rules or DEFAULT_RULES
    start_date = df.index[WARMUP_BARS]
    end_date = df.index[-1]

    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  WALK-FORWARD VALIDATION")
        print(f"{'=' * 64}")
        print(f"  Pair      : {pair}")
        print(f"  Data      : {start_date.date()} -> {end_date.date()}")
        print(f"  Train     : {train_months} months")
        print(f"  Test      : {test_months} months")
        print(f"{'=' * 64}")

    # Generate windows
    windows = []
    current = start_date

    while True:
        train_end = current + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        windows.append({
            "train_start": current,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
        })

        current = current + pd.DateOffset(months=test_months)

    if verbose:
        print(f"  Windows   : {len(windows)}")

    # Run each window
    all_trades = []
    all_equity_changes = []
    window_results = []

    for w_idx, window in enumerate(windows):
        if verbose:
            print(f"\n{'-' * 64}")
            print(f"  Window {w_idx + 1}/{len(windows)}")
            print(f"  Train: {window['train_start'].date()} -> {window['train_end'].date()}")
            print(f"  Test:  {window['test_start'].date()} -> {window['test_end'].date()}")
            print(f"{'-' * 64}")

        # Extract train and test data
        train_df = df[(df.index >= window["train_start"]) & (df.index < window["train_end"])].copy()
        test_df = df[(df.index >= window["test_start"]) & (df.index < window["test_end"])].copy()

        if len(train_df) < 500 or len(test_df) < 100:
            if verbose:
                print(f"  Skipping: insufficient data (train={len(train_df)}, test={len(test_df)})")
            continue

        # Train on this window
        try:
            dataset, stats = create_labeled_dataset(train_df)
            if stats["total_samples"] < 100:
                if verbose:
                    print(f"  Skipping: too few labeled samples ({stats['total_samples']})")
                continue

            result = train_and_evaluate(dataset, "RandomForest", pair, timeframe)
            mcc = result["metrics"]["mcc"]
            acc = result["metrics"]["test_accuracy"]

            if verbose:
                print(f"  Model: MCC={mcc:.4f}, Acc={acc:.2%}")

        except Exception as e:
            if verbose:
                print(f"  Training error: {e}")
            continue

        # Backtest on test window
        try:
            predictor = MLPredictor(pair, timeframe)
            engine = BacktestEngine(pair, timeframe, rules=rules, initial_balance=INITIAL_BALANCE)
            bt_result = engine.run(test_df, predictor, verbose=False)

            n_trades = bt_result.get("trades", 0)
            ret_pct = bt_result.get("return_pct", 0)
            max_dd = bt_result.get("max_drawdown_pct", 0)
            win_rate = bt_result.get("win_rate", 0)
            pf = bt_result.get("profit_factor", 0)

            window_results.append({
                "window": w_idx + 1,
                "test_start": window["test_start"].date(),
                "test_end": window["test_end"].date(),
                "trades": n_trades,
                "return_pct": ret_pct,
                "max_dd": max_dd,
                "win_rate": win_rate,
                "profit_factor": pf,
                "mcc": mcc,
            })

            if "closed_trades" in bt_result:
                all_trades.extend(bt_result["closed_trades"])

            if verbose:
                print(f"  Result: {n_trades} trades, Return={ret_pct:.2%}, DD={max_dd:.2%}, WR={win_rate:.2%}")

        except Exception as e:
            if verbose:
                print(f"  Backtest error: {e}")
            continue

    # Aggregate results
    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  WALK-FORWARD RESULTS SUMMARY")
        print(f"{'=' * 64}")

    if not window_results:
        if verbose:
            print("  No valid windows completed.")
        return {"windows": 0, "status": "NO DATA"}

    df_results = pd.DataFrame(window_results)

    total_trades = df_results["trades"].sum()
    avg_return = df_results["return_pct"].mean()
    total_return = (1 + df_results["return_pct"]).prod() - 1
    worst_dd = df_results["max_dd"].max()
    avg_win_rate = df_results["win_rate"].mean()
    avg_pf = df_results[df_results["profit_factor"] > 0]["profit_factor"].mean()
    avg_mcc = df_results["mcc"].mean()
    winning_windows = (df_results["return_pct"] > 0).sum()

    summary = {
        "windows": len(window_results),
        "winning_windows": int(winning_windows),
        "total_trades": int(total_trades),
        "avg_return_per_window": avg_return,
        "compounded_return": total_return,
        "worst_window_dd": worst_dd,
        "avg_win_rate": avg_win_rate,
        "avg_profit_factor": avg_pf,
        "avg_mcc": avg_mcc,
        "window_details": window_results,
    }

    if verbose:
        print(f"  Windows completed  : {len(window_results)}")
        print(f"  Winning windows    : {winning_windows}/{len(window_results)}")
        print(f"  Total trades       : {total_trades}")
        print(f"  Avg return/window  : {avg_return:.2%}")
        print(f"  Compounded return  : {total_return:.2%}")
        print(f"  Worst window DD    : {worst_dd:.2%}")
        print(f"  Avg win rate       : {avg_win_rate:.2%}")
        print(f"  Avg profit factor  : {avg_pf:.2f}")
        print(f"  Avg model MCC      : {avg_mcc:.4f}")
        print(f"\n  Per-window breakdown:")
        print(f"  {'Win':>3} | {'Period':<25} | {'Trades':>6} | {'Return':>8} | {'MaxDD':>7} | {'WR':>6} | {'PF':>6}")
        print(f"  {'-' * 3} | {'-' * 25} | {'-' * 6} | {'-' * 8} | {'-' * 7} | {'-' * 6} | {'-' * 6}")
        for w in window_results:
            print(f"  {w['window']:>3} | {str(w['test_start'])} -> {str(w['test_end']):<10} | {w['trades']:>6} | {w['return_pct']:>7.2%} | {w['max_dd']:>6.2%} | {w['win_rate']:>5.1%} | {w['profit_factor']:>5.2f}")

        print(f"\n{'=' * 64}")

    return summary
