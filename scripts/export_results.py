"""
Export Backtest Results — Run after the pipeline to generate dashboard data.

Usage:
    python scripts/export_results.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

from config.settings import PRIMARY_PAIR, PRIMARY_TIMEFRAME, INITIAL_BALANCE
from config.prop_firm_rules import get_rules
from data.loader import load_processed
from ml.features import compute_all_features
from ml.predictor import MLPredictor
from backtest.engine import BacktestEngine
from backtest.metrics import export_results


def main():
    pair = PRIMARY_PAIR
    timeframe = PRIMARY_TIMEFRAME

    print("Loading data...")
    df = load_processed(pair, timeframe)

    print("Loading model...")
    predictor = MLPredictor(pair, timeframe)

    print("Running backtest...")
    rules = get_rules("ftmo_challenge")
    engine = BacktestEngine(pair, timeframe, rules=rules, initial_balance=INITIAL_BALANCE)
    results = engine.run(df, predictor, verbose=True)

    print("\nExporting to dashboard...")
    export_results(results, pair, timeframe)
    print("Done! Open dashboard/index.html to view results.")


if __name__ == "__main__":
    main()
