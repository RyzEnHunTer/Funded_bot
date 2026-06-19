"""
Dynamic Portfolio Simulation Engine
Simulates the live_bot.py behavior by executing a synchronized, chronological 
backtest across multiple pairs, enforcing global constraints (Max 1 Trade at a time)
and picking only the absolute highest probability trade per bar.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    INITIAL_BALANCE, RISK_PER_TRADE_PCT, MAX_CONCURRENT_POSITIONS, MAX_TRADES_PER_DAY,
    ML_LONG_THRESHOLD, ML_SHORT_THRESHOLD, REWARD_RISK_RATIO, ATR_MULTIPLIER,
    COMMISSION_PER_LOT, PIP_VALUES, PIP_SIZES, SPREADS
)
from data.loader import load_processed
from ml.features import compute_all_features, get_supertrend_direction
from ml.predictor import MLPredictor
from ml.labeling import compute_barrier_distance
from risk.prop_firm_manager import PropFirmManager
from backtest.engine import Position, ClosedTrade, Signal, generate_signal
from risk.session_filter import SessionFilter

def run_portfolio_sim():
    pairs = ["USDCAD", "EURUSD", "NQ=F"]
    timeframe = "15m"
    
    print("\n" + "=" * 80)
    print("  INITIALIZING SYNCHRONOUS PORTFOLIO SIMULATION")
    print("=" * 80)
    
    # ─── 1. LOAD AND PRE-PROCESS ALL DATA ───
    dataframes = {}
    predictors = {}
    all_probs = {}
    st_dirs = {}
    barrier_dists = {}
    
    unified_index = pd.DatetimeIndex([])
    
    for pair in pairs:
        print(f"  Loading {pair}...")
        try:
            df = load_processed(pair, timeframe)
            df = compute_all_features(df)
            
            # Predictor
            predictor = MLPredictor(pair, timeframe)
            
            # Pre-compute probabilities
            valid_mask = df.notna().all(axis=1)
            valid_indices = df.index[valid_mask]
            
            probs = pd.DataFrame()
            if len(valid_indices) > 0:
                probs = predictor.predict_proba_batch(df.loc[valid_mask])
            
            # Pre-compute ST and Barrier
            st_dir = get_supertrend_direction(df)
            barrier_dist = compute_barrier_distance(df["close"], df["high"], df["low"])
            
            dataframes[pair] = df
            predictors[pair] = predictor
            all_probs[pair] = probs
            st_dirs[pair] = st_dir
            barrier_dists[pair] = barrier_dist
            
            unified_index = unified_index.union(df.index)
        except Exception as e:
            print(f"  [ERROR] Failed to load {pair}: {e}")
            
    unified_index = unified_index.sort_values()
    print(f"\n  Synchronized Timeline: {len(unified_index)} total bars")
    print(f"  Start: {unified_index[0]}")
    print(f"  End:   {unified_index[-1]}\n")
    
    # ─── 2. SETUP PORTFOLIO ENGINE ───
    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    open_positions = []
    closed_trades = []
    
    risk_mgr = PropFirmManager(initial_balance=INITIAL_BALANCE)
    session_filter = SessionFilter(require_overlap_only=False)
    
    daily_trades_count = 0
    last_trade_date = None
    
    # Track equity curve
    equity_curve = []
    equity_timestamps = []
    
    print("  Beginning Chronological Simulation...\n")
    
    for i, timestamp in enumerate(unified_index):
        # Midnight reset
        current_date = timestamp.date()
        if last_trade_date is None or current_date != last_trade_date:
            last_trade_date = current_date
            daily_trades_count = 0
            
        # ─── 3. UPDATE OPEN POSITIONS ───
        positions_to_close = []
        floating_pnl_total = 0.0
        
        for pos in open_positions:
            pair = pos.pair
            df = dataframes[pair]
            
            if timestamp not in df.index:
                continue
                
            bar = df.loc[timestamp]
            close = bar['close']
            high = bar['high']
            low = bar['low']
            
            pip_size = PIP_SIZES.get(pair, 0.0001)
            pip_value = PIP_VALUES.get(pair, 10.0)
            
            # --- Breakeven + Runner Logic ---
            st_line = df.loc[timestamp, "_supertrend_line"] if "_supertrend_line" in df.columns else None
            
            # 1:1 Scale-out check
            if not pos.breakeven_locked:
                hit_1_to_1 = False
                if pos.direction == 1 and high >= pos.entry_price + pos.original_sl_dist:
                    hit_1_to_1 = True
                elif pos.direction == -1 and low <= pos.entry_price - pos.original_sl_dist:
                    hit_1_to_1 = True
                    
                if hit_1_to_1:
                    # Close 50% for profit
                    scale_lots = pos.lots / 2.0
                    pos.lots -= scale_lots
                    pnl_pips = (pos.original_sl_dist / pip_size)  # Exact 1 R
                    pnl_gross = pnl_pips * pip_value * scale_lots
                    commission = COMMISSION_PER_LOT * scale_lots
                    pnl_net = pnl_gross - commission
                    
                    balance += pnl_net
                    risk_mgr.record_trade_close(pnl_net, timestamp)
                    
                    closed_trades.append({
                        "pair": pair,
                        "direction": pos.direction,
                        "entry_time": pos.entry_time,
                        "exit_time": timestamp,
                        "pnl": pnl_net,
                        "reason": "partial_profit"
                    })
                    
                    # Move SL to Breakeven
                    pos.stop_loss = pos.entry_price
                    pos.breakeven_locked = True
                    # print(f"[{timestamp}] SCALE-OUT + BREAKEVEN locked for {pair}")

            # Supertrend Trailing (if Runner)
            if pos.breakeven_locked and st_line is not None and not pd.isna(st_line):
                if pos.direction == 1 and close > pos.stop_loss and st_line > pos.stop_loss:
                    pos.stop_loss = st_line
                elif pos.direction == -1 and close < pos.stop_loss and st_line < pos.stop_loss:
                    pos.stop_loss = st_line

            sl_hit = False
            tp_hit = False
            
            if pos.direction == 1:
                sl_hit = low <= pos.stop_loss
                tp_hit = high >= pos.take_profit
            else:
                sl_hit = high >= pos.stop_loss
                tp_hit = low <= pos.take_profit
                
            if sl_hit or tp_hit:
                # Close trade
                exit_price = pos.stop_loss if sl_hit else pos.take_profit
                exit_reason = "stop_loss" if sl_hit else "take_profit"
                
                price_diff = (exit_price - pos.entry_price) * pos.direction
                pnl_pips = price_diff / pip_size
                pnl_gross = pnl_pips * pip_value * pos.lots
                commission = COMMISSION_PER_LOT * pos.lots
                pnl_net = pnl_gross - commission
                
                balance += pnl_net
                risk_mgr.record_trade_close(pnl_net, timestamp)
                
                closed_trades.append({
                    "pair": pair,
                    "direction": pos.direction,
                    "entry_time": pos.entry_time,
                    "exit_time": timestamp,
                    "pnl": pnl_net,
                    "reason": exit_reason
                })
                positions_to_close.append(pos)
            else:
                # Update floating
                floating_pnl_total += pos.unrealised_pnl(close, pip_size, pip_value)
                
        for pos in positions_to_close:
            open_positions.remove(pos)
            
        equity = balance + floating_pnl_total
        risk_mgr.update(equity, timestamp)
        
        equity_curve.append(equity)
        equity_timestamps.append(timestamp)
        
        if risk_mgr.challenge_status == "FAILED":
            print(f"\n  [X] CHALLENGE FAILED AT {timestamp}")
            break
        if risk_mgr.challenge_status == "PASSED":
            print(f"\n  [OK] CHALLENGE PASSED AT {timestamp}")
            break
            
        # ─── 4. DYNAMIC PORTFOLIO SELECTION ───
        if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
            continue
            
        if daily_trades_count >= MAX_TRADES_PER_DAY:
            continue
            
        potential_trades = []
        
        for pair in pairs:
            if timestamp not in all_probs[pair].index:
                continue
                
            # Session filter
            if not session_filter.is_tradeable(timestamp, symbol=pair):
                continue
                
            prob_row = all_probs[pair].loc[timestamp]
            st_dir = int(st_dirs[pair].loc[timestamp])
            
            prob_long = prob_row.get("prob_1", 0.0)
            prob_short = prob_row.get("prob_-1", 0.0)
            
            signal = Signal.FLAT
            if prob_long >= ML_LONG_THRESHOLD and st_dir == 1:
                signal = Signal.LONG
            elif prob_short >= ML_SHORT_THRESHOLD and st_dir == -1:
                signal = Signal.SHORT
                
            if signal == Signal.FLAT:
                continue
                
            distance = float(barrier_dists[pair].loc[timestamp])
            if np.isnan(distance) or distance <= 0:
                continue
                
            check = risk_mgr.pre_trade_check(pair, distance, timestamp)
            if not check.allowed:
                continue
                
            confidence = float(prob_long) if signal == Signal.LONG else float(prob_short)
            
            potential_trades.append({
                "pair": pair,
                "signal": signal,
                "confidence": confidence,
                "distance": distance,
                "lots": check.position_size_lots
            })
            
        # Execute only highest confidence
        if potential_trades:
            potential_trades.sort(key=lambda x: x["confidence"], reverse=True)
            best = potential_trades[0]
            
            pair = best["pair"]
            df = dataframes[pair]
            close = df.loc[timestamp]['close']
            
            spread_pips = SPREADS.get(pair, 2.0)
            pip_size = PIP_SIZES.get(pair, 0.0001)
            spread_cost = spread_pips * pip_size / 2
            
            if best["signal"] == Signal.LONG:
                entry_price = close + spread_cost
                sl = entry_price - best["distance"]
                tp = entry_price + best["distance"] * REWARD_RISK_RATIO
            else:
                entry_price = close - spread_cost
                sl = entry_price + best["distance"]
                tp = entry_price - best["distance"] * REWARD_RISK_RATIO
                
            pos = Position(
                direction=int(best["signal"]),
                entry_price=entry_price,
                entry_time=timestamp,
                lots=best["lots"],
                stop_loss=sl,
                take_profit=tp,
                pair=pair,
                bar_index=0,
                breakeven_locked=False,
                original_sl_dist=best["distance"]
            )
            open_positions.append(pos)
            risk_mgr.record_trade_open()
            daily_trades_count += 1
            
            # Print for visual logging
            dir_str = "LONG" if best["signal"] == Signal.LONG else "SHORT"
            print(f"[{timestamp}] Executing {dir_str} {pair} (Conf: {best['confidence']:.2%}) | Lots: {best['lots']}")

    # ─── 5. SUMMARY ───
    print("\n" + "=" * 60)
    print("  PORTFOLIO SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Status        : {risk_mgr.challenge_status}")
    print(f"  Final Equity  : ${risk_mgr.current_equity:,.2f}")
    print(f"  Total Trades  : {risk_mgr.total_trades}")
    print(f"  Win Rate      : {risk_mgr.win_rate:.2%}")
    print(f"  Max Total DD  : {risk_mgr.dd_governor.max_drawdown_pct_recorded:.2%}")
    print(f"  Max Daily DD  : {risk_mgr.daily_limiter.max_daily_loss_pct_recorded:.2%}")
    print(f"  Trading Days  : {len(risk_mgr.trading_days)}")
    print("-" * 60)
    
    # Pair by Pair Breakdown
    pair_stats = {p: {'trades': 0, 'wins': 0, 'pnl': 0.0} for p in pairs}
    for t in closed_trades:
        p = t['pair']
        pair_stats[p]['trades'] += 1
        if t['pnl'] > 0:
            pair_stats[p]['wins'] += 1
        pair_stats[p]['pnl'] += t['pnl']
        
    print(f"  {'Pair':<8} | {'Trades':<8} | {'Win Rate':<10} | {'Total P&L':<10}")
    print("-" * 60)
    for p, stats in pair_stats.items():
        wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0.0
        print(f"  {p:<8} | {stats['trades']:<8} | {wr:>5.1f}%     | ${stats['pnl']:>8.2f}")
    print("=" * 60)

if __name__ == "__main__":
    run_portfolio_sim()
