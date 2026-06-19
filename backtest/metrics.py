"""
Export backtest results to JSON for the dashboard.

Generates a JSON file with equity curve, trade list, and metrics
that the HTML dashboard reads to render charts.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PROJECT_ROOT


def export_results(results: Dict, pair: str, timeframe: str,
                    output_dir: Path = None) -> Path:
    """
    Export backtest results to a JSON file for the dashboard.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert equity curve timestamps to strings
    equity_data = []
    if "equity_curve" in results and "equity_timestamps" in results:
        for ts, eq in zip(results["equity_timestamps"], results["equity_curve"]):
            equity_data.append({
                "time": str(ts),
                "equity": round(eq, 2),
            })

    # Convert trades
    trades_data = []
    if "closed_trades" in results:
        for t in results["closed_trades"]:
            trades_data.append({
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_price": round(t.entry_price, 5),
                "exit_price": round(t.exit_price, 5),
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "lots": t.lots,
                "pnl": round(t.pnl, 2),
                "pnl_pips": round(t.pnl_pips, 1),
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            })

    # Build export object
    export = {
        "pair": pair,
        "timeframe": timeframe,
        "generated_at": str(datetime.now()),
        "metrics": {
            "status": results.get("status", "N/A"),
            "initial_balance": results.get("initial_balance", 0),
            "final_equity": round(results.get("final_equity", 0), 2),
            "total_pnl": round(results.get("total_pnl", 0), 2),
            "return_pct": round(results.get("return_pct", 0) * 100, 2),
            "trades": results.get("trades", 0),
            "winners": results.get("winners", 0),
            "losers": results.get("losers", 0),
            "win_rate": round(results.get("win_rate", 0) * 100, 1),
            "profit_factor": round(results.get("profit_factor", 0), 2),
            "avg_win": round(results.get("avg_win", 0), 2),
            "avg_loss": round(results.get("avg_loss", 0), 2),
            "max_drawdown_pct": round(results.get("max_drawdown_pct", 0) * 100, 2),
            "max_consecutive_losses": results.get("max_consecutive_losses", 0),
            "sharpe_ratio": round(results.get("sharpe_ratio", 0), 2),
            "recovery_factor": round(results.get("recovery_factor", 0), 2),
            "avg_bars_held": round(results.get("avg_bars_held", 0), 1),
            "trading_days": results.get("trading_days", 0),
        },
        "equity_curve": equity_data,
        "trades": trades_data,
    }

    filepath = output_dir / "results.json"
    with open(filepath, "w") as f:
        json.dump(export, f, indent=2)

    print(f"  Dashboard data exported to: {filepath}")
    return filepath
