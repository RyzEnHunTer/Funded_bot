"""
Full Pipeline Script -- End-to-end: Download -> Features -> Label -> Train -> Backtest

This is the master script that runs the complete ML trading pipeline:
  Phase 1: Download Forex data (or load existing)
  Phase 2: Compute features + triple-barrier labeling -> Train ML model
  Phase 3: Run backtest with prop firm rules

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --pair EURUSD --timeframe 1h --skip-download
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from config.settings import (
    PRIMARY_PAIR, PRIMARY_TIMEFRAME, DATA_LOOKBACK_DAYS,
    INITIAL_BALANCE, MODELS_DIR,
)
from config.prop_firm_rules import DEFAULT_RULES, get_rules


def run_pipeline(pair: str = PRIMARY_PAIR,
                  timeframe: str = PRIMARY_TIMEFRAME,
                  skip_download: bool = False,
                  rules_name: str = "ftmo_challenge"):
    """Run the complete pipeline."""

    print("=" * 64)
    print("  PROP FIRM ML FOREX TRADING SYSTEM -- FULL PIPELINE")
    print("=" * 64)
    print(f"  Pair      : {pair}")
    print(f"  Timeframe : {timeframe}")
    print(f"  Rules     : {rules_name}")
    print(f"  Balance   : ${INITIAL_BALANCE:,.2f}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)

    # ==========================================================================
    # PHASE 1: DATA ACQUISITION
    # ==========================================================================
    print(f"\n{'-' * 64}")
    print("  PHASE 1: DATA ACQUISITION")
    print(f"{'-' * 64}")

    if not skip_download:
        from scripts.download_data import download_pair, save_raw
        from data.loader import clean_data, save_processed, validate_data

        try:
            df_raw = download_pair(pair, timeframe, DATA_LOOKBACK_DAYS)
            save_raw(df_raw, pair, timeframe)
            report = validate_data(df_raw, pair)
            print(f"  OK Downloaded {len(df_raw):,} bars")
            print(f"    Range: {df_raw.index[0]} -> {df_raw.index[-1]}")
            print(f"    Days:  {report['days']}")

            df = clean_data(df_raw)
            save_processed(df, pair, timeframe)
            print(f"  OK Cleaned: {len(df):,} bars")
        except Exception as e:
            print(f"  X Download failed: {e}")
            print("  Attempting to load existing data...")
            from data.loader import load_processed
            df = load_processed(pair, timeframe)
            print(f"  OK Loaded existing data: {len(df):,} bars")
    else:
        from data.loader import load_processed
        df = load_processed(pair, timeframe)
        print(f"  OK Loaded existing data: {len(df):,} bars")

    # ==========================================================================
    # PHASE 2: FEATURE ENGINEERING + LABELING + TRAINING
    # ==========================================================================
    print(f"\n{'-' * 64}")
    print("  PHASE 2: ML MODEL TRAINING")
    print(f"{'-' * 64}")

    from ml.labeling import create_labeled_dataset, save_labeled_data
    from ml.trainer import train_and_evaluate, compare_models

    # Create labeled dataset
    dataset, stats = create_labeled_dataset(df)
    print(f"\n  Labeling Results:")
    print(f"    Total samples: {stats['total_samples']:,}")
    print(f"    Label distribution:")
    for label, count in stats['label_distribution'].items():
        pct = count / stats['total_samples'] * 100
        print(f"      {label}: {count:,} ({pct:.1f}%)")
    print(f"    Avg barrier distance: {stats['avg_barrier_distance']:.6f}")
    print(f"    Date range: {stats['date_range']}")

    # Save labeled data
    save_labeled_data(dataset, pair, timeframe)
    print(f"  OK Labeled data saved")

    if stats['total_samples'] < 100:
        print(f"\n  WARNING WARNING: Only {stats['total_samples']} samples. Need at least 100 for meaningful training.")
        print(f"    Consider using more historical data or a shorter timeframe.")

    # Train and compare models
    print(f"\n  Training models...")
    result = train_and_evaluate(dataset, "RandomForest", pair, timeframe)
    metrics = result["metrics"]

    # ==========================================================================
    # PHASE 2.5: MODEL QUALITY CHECK
    # ==========================================================================
    print(f"\n{'-' * 64}")
    print("  MODEL QUALITY CHECK")
    print(f"{'-' * 64}")

    mcc = metrics["mcc"]
    test_acc = metrics["test_accuracy"]
    train_acc = metrics["train_accuracy"]
    overfit_gap = train_acc - test_acc

    checks = []
    checks.append(("MCC > 0.05 (beats random)", mcc > 0.05, f"{mcc:.4f}"))
    checks.append(("Test Accuracy > 36%", test_acc > 0.36, f"{test_acc:.2%}"))
    checks.append(("Overfit gap < 20%", overfit_gap < 0.20, f"{overfit_gap:.2%}"))

    all_passed = True
    for desc, passed, val in checks:
        status = "OK" if passed else "X"
        print(f"    {status} {desc} -> {val}")
        if not passed:
            all_passed = False

    if not all_passed:
        print(f"\n  WARNING Model quality check FAILED.")
        print(f"    The model may not have a meaningful edge.")
        print(f"    Consider: more data, different features, different estimator.")
        print(f"    Proceeding with backtest anyway for evaluation...\n")
    else:
        print(f"\n  OK Model quality check PASSED. Proceeding to backtest.\n")

    # ==========================================================================
    # PHASE 3: BACKTEST WITH PROP FIRM RULES
    # ==========================================================================
    print(f"{'-' * 64}")
    print("  PHASE 3: BACKTEST WITH PROP FIRM RULES")
    print(f"{'-' * 64}")

    from ml.predictor import MLPredictor
    from backtest.engine import BacktestEngine

    # Load the trained model
    predictor = MLPredictor(pair, timeframe)

    # Run backtest
    rules = get_rules(rules_name)
    engine = BacktestEngine(pair, timeframe, rules=rules, initial_balance=INITIAL_BALANCE)
    results = engine.run(df, predictor, verbose=True)

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print(f"\n{'=' * 64}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 64}")
    print(f"  ML Model    : {metrics['estimator']} (MCC={mcc:.4f}, Acc={test_acc:.2%})")
    print(f"  Challenge   : {results.get('status', 'N/A')}")

    if results.get('trades', 0) > 0:
        print(f"  Trades      : {results['trades']}")
        print(f"  Win Rate    : {results['win_rate']:.2%}")
        print(f"  Return      : {results['return_pct']:.2%}")
        print(f"  Max DD      : {results['max_drawdown_pct']:.2%}")
        print(f"  Sharpe      : {results['sharpe_ratio']:.2f}")
    else:
        print(f"  WARNING No trades executed -- check signal thresholds and session filters")

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 64}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the full ML trading pipeline")
    parser.add_argument("--pair", type=str, default=PRIMARY_PAIR,
                        help=f"Currency pair (default: {PRIMARY_PAIR})")
    parser.add_argument("--timeframe", type=str, default=PRIMARY_TIMEFRAME,
                        help=f"Timeframe (default: {PRIMARY_TIMEFRAME})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download, use existing data")
    parser.add_argument("--rules", type=str, default="ftmo_challenge",
                        help="Prop firm rules to use (default: ftmo_challenge)")
    args = parser.parse_args()

    run_pipeline(
        pair=args.pair,
        timeframe=args.timeframe,
        skip_download=args.skip_download,
        rules_name=args.rules,
    )


if __name__ == "__main__":
    main()
