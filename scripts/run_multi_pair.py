"""
Multi-Pair Batch Testing Script.

Runs the 15m HFT pipeline across multiple pairs to validate the strategy's
edge on the most popular, liquid, and stable pairs recommended for Prop Firms.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

from scripts.run_pipeline import run_pipeline

def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
    timeframe = "15m"
    
    print(f"Starting Multi-Pair Batch Test on {timeframe}")
    print(f"Pairs to test: {pairs}")
    print("-" * 50)
    
    results_summary = []
    
    for i, pair in enumerate(pairs):
        print(f"\n\n{'#' * 64}")
        print(f"# TESTING PAIR {i+1}/{len(pairs)}: {pair}")
        print(f"{'#' * 64}")
        
        try:
            # We must download data for the new pairs
            # For EURUSD it will overwrite, which is fine
            # We add a small sleep to avoid yfinance rate limits
            time.sleep(2)
            
            # Disable noisy prints in the run_pipeline if possible, 
            # but it doesn't have a verbose flag, so it will print.
            res = run_pipeline(pair=pair, timeframe=timeframe, skip_download=False)
            
            if res and "status" in res:
                results_summary.append({
                    "Pair": pair,
                    "Status": res.get("status", "FAILED"),
                    "Return": res.get("return_pct", 0) * 100,
                    "Max DD": res.get("max_drawdown_pct", 0) * 100,
                    "Win Rate": res.get("win_rate", 0) * 100,
                    "Trades": res.get("trades", 0),
                    "Days": res.get("trading_days", 0),
                })
            else:
                 results_summary.append({
                    "Pair": pair,
                    "Status": "ERROR",
                    "Return": 0.0,
                    "Max DD": 0.0,
                    "Win Rate": 0.0,
                    "Trades": 0,
                    "Days": 0,
                })
                
        except Exception as e:
            print(f"Error testing {pair}: {e}")
            results_summary.append({
                    "Pair": pair,
                    "Status": "EXCEPTION",
                    "Return": 0.0,
                    "Max DD": 0.0,
                    "Win Rate": 0.0,
                    "Trades": 0,
                    "Days": 0,
                })

    # Print Summary Table
    print("\n\n" + "=" * 85)
    print("  MULTI-PAIR PERFORMANCE SUMMARY (15m HFT)")
    print("=" * 85)
    print(f"{'Pair':<10} | {'Status':<15} | {'Return':<8} | {'Max DD':<8} | {'Win Rate':<10} | {'Trades':<8} | {'Days':<6}")
    print("-" * 85)
    
    for r in results_summary:
        ret_str = f"{r['Return']:+.2f}%" if r['Status'] not in ["ERROR", "EXCEPTION"] else "N/A"
        dd_str = f"{r['Max DD']:.2f}%" if r['Status'] not in ["ERROR", "EXCEPTION"] else "N/A"
        wr_str = f"{r['Win Rate']:.1f}%" if r['Status'] not in ["ERROR", "EXCEPTION"] else "N/A"
        
        print(f"{r['Pair']:<10} | {r['Status']:<15} | {ret_str:<8} | {dd_str:<8} | {wr_str:<10} | {r['Trades']:<8} | {r['Days']:<6}")
    
    print("=" * 85)

if __name__ == "__main__":
    main()
