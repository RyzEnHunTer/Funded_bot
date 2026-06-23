"""
GBPUSD Integration Test
  Run 1: GBPUSD Solo (with Exhaustion Detection)
  Run 2: Full 4-Pair Portfolio (USDCAD, EURUSD, NQ=F, GBPUSD) with Exhaustion Detection

Uses the same funded challenge rules: $5K balance, 14% target, 4% daily DD, 10% max DD.
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

import warnings
warnings.filterwarnings("ignore")

# Exhaustion Config
EXHAUSTION_ML_FLIP_THRESHOLD = 0.45
EXHAUSTION_TP_LOCK_PCT       = 0.85
EXHAUSTION_LOCK_PROFIT_PCT   = 0.70


def close_position(pos, exit_price, exit_reason, timestamp, balance, risk_mgr, closed_trades, pip_size, pip_value):
    price_diff = (exit_price - pos.entry_price) * pos.direction
    pnl_pips = price_diff / pip_size
    pnl_gross = pnl_pips * pip_value * pos.lots
    commission = COMMISSION_PER_LOT * pos.lots
    pnl_net = pnl_gross - commission
    balance += pnl_net
    risk_mgr.record_trade_close(pnl_net, timestamp)
    closed_trades.append({
        "pair": pos.pair, "direction": pos.direction,
        "entry_price": pos.entry_price, "exit_price": exit_price,
        "entry_time": pos.entry_time, "exit_time": timestamp,
        "lots": pos.lots, "pnl": pnl_net, "pnl_pips": pnl_pips,
        "reason": exit_reason, "balance_after": balance
    })
    return balance, pnl_net


def run_sim(pairs, label):
    timeframe = "15m"
    
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Pairs: {', '.join(pairs)}")
    print(f"  Exhaustion Detection: ENABLED")
    print(f"{'=' * 70}")
    
    dataframes = {}
    all_probs = {}
    st_dirs = {}
    barrier_dists = {}
    unified_index = pd.DatetimeIndex([])
    
    for pair in pairs:
        print(f"  Loading {pair}...")
        try:
            df = load_processed(pair, timeframe)
            df = compute_all_features(df)
            predictor = MLPredictor(pair, timeframe)
            valid_mask = df.notna().all(axis=1)
            probs = pd.DataFrame()
            if len(df.index[valid_mask]) > 0:
                probs = predictor.predict_proba_batch(df.loc[valid_mask])
            st_dir = get_supertrend_direction(df)
            barrier_dist = compute_barrier_distance(df["close"], df["high"], df["low"])
            dataframes[pair] = df
            all_probs[pair] = probs
            st_dirs[pair] = st_dir
            barrier_dists[pair] = barrier_dist
            unified_index = unified_index.union(df.index)
        except Exception as e:
            print(f"  [ERROR] Failed to load {pair}: {e}")
            
    unified_index = unified_index.sort_values()
    print(f"  Timeline: {len(unified_index)} bars ({unified_index[0]} -> {unified_index[-1]})\n")
    
    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    open_positions = []
    closed_trades = []
    risk_mgr = PropFirmManager(initial_balance=INITIAL_BALANCE)
    session_filter = SessionFilter(require_overlap_only=False)
    daily_trades_count = 0
    last_trade_date = None
    equity_curve = []
    ml_flip_exits = 0
    tp_lock_exits = 0
    
    for i, timestamp in enumerate(unified_index):
        current_date = timestamp.date()
        if last_trade_date is None or current_date != last_trade_date:
            last_trade_date = current_date
            daily_trades_count = 0
            
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
            st_line = df.loc[timestamp, "_supertrend_line"] if "_supertrend_line" in df.columns else None
            
            # 1:1 Scale-out
            if not pos.breakeven_locked:
                hit = False
                if pos.direction == 1 and high >= pos.entry_price + pos.original_sl_dist:
                    hit = True
                elif pos.direction == -1 and low <= pos.entry_price - pos.original_sl_dist:
                    hit = True
                if hit:
                    scale_lots = pos.lots / 2.0
                    pos.lots -= scale_lots
                    pnl_pips = pos.original_sl_dist / pip_size
                    pnl_gross = pnl_pips * pip_value * scale_lots
                    commission = COMMISSION_PER_LOT * scale_lots
                    pnl_net = pnl_gross - commission
                    balance += pnl_net
                    risk_mgr.record_trade_close(pnl_net, timestamp)
                    closed_trades.append({
                        "pair": pair, "direction": pos.direction,
                        "entry_price": pos.entry_price,
                        "exit_price": pos.entry_price + pos.original_sl_dist * pos.direction,
                        "entry_time": pos.entry_time, "exit_time": timestamp,
                        "lots": scale_lots, "pnl": pnl_net, "pnl_pips": pnl_pips,
                        "reason": "partial_profit", "balance_after": balance
                    })
                    pos.stop_loss = pos.entry_price
                    pos.breakeven_locked = True

            # Supertrend Trailing
            if pos.breakeven_locked and st_line is not None and not pd.isna(st_line):
                if pos.direction == 1 and close > pos.stop_loss and st_line > pos.stop_loss:
                    pos.stop_loss = st_line
                elif pos.direction == -1 and close < pos.stop_loss and st_line < pos.stop_loss:
                    pos.stop_loss = st_line

            # Exhaustion #1: ML Flip
            if pos.breakeven_locked and timestamp in all_probs[pair].index:
                prob_row = all_probs[pair].loc[timestamp]
                prob_long = prob_row.get("prob_1", 0.0)
                prob_short = prob_row.get("prob_-1", 0.0)
                if pos.direction == 1 and prob_short >= EXHAUSTION_ML_FLIP_THRESHOLD and close > pos.entry_price:
                    balance, pnl = close_position(pos, close, "ml_flip_exit", timestamp, balance, risk_mgr, closed_trades, pip_size, pip_value)
                    positions_to_close.append(pos)
                    ml_flip_exits += 1
                    continue
                elif pos.direction == -1 and prob_long >= EXHAUSTION_ML_FLIP_THRESHOLD and close < pos.entry_price:
                    balance, pnl = close_position(pos, close, "ml_flip_exit", timestamp, balance, risk_mgr, closed_trades, pip_size, pip_value)
                    positions_to_close.append(pos)
                    ml_flip_exits += 1
                    continue

            # Exhaustion #2: 90% Lock
            if pos.breakeven_locked:
                total_tp_dist = abs(pos.take_profit - pos.entry_price)
                current_move = (close - pos.entry_price) * pos.direction
                if total_tp_dist > 0 and current_move > 0:
                    progress_pct = current_move / total_tp_dist
                    if progress_pct >= EXHAUSTION_TP_LOCK_PCT:
                        locked_move = current_move * EXHAUSTION_LOCK_PROFIT_PCT
                        if pos.direction == 1:
                            new_sl = pos.entry_price + locked_move
                        else:
                            new_sl = pos.entry_price - locked_move
                        if pos.direction == 1 and new_sl > pos.stop_loss:
                            pos.stop_loss = new_sl
                            tp_lock_exits += 1
                        elif pos.direction == -1 and new_sl < pos.stop_loss:
                            pos.stop_loss = new_sl
                            tp_lock_exits += 1

            # SL/TP check
            sl_hit = (low <= pos.stop_loss) if pos.direction == 1 else (high >= pos.stop_loss)
            tp_hit = (high >= pos.take_profit) if pos.direction == 1 else (low <= pos.take_profit)
            if sl_hit or tp_hit:
                exit_price = pos.stop_loss if sl_hit else pos.take_profit
                exit_reason = "stop_loss" if sl_hit else "take_profit"
                balance, pnl = close_position(pos, exit_price, exit_reason, timestamp, balance, risk_mgr, closed_trades, pip_size, pip_value)
                positions_to_close.append(pos)
            else:
                floating_pnl_total += pos.unrealised_pnl(close, pip_size, pip_value)
                
        for pos in positions_to_close:
            if pos in open_positions:
                open_positions.remove(pos)
            
        equity = balance + floating_pnl_total
        risk_mgr.update(equity, timestamp)
        equity_curve.append(equity)
        
        if risk_mgr.challenge_status == "FAILED":
            print(f"\n  [X] CHALLENGE FAILED AT {timestamp}")
            break
        if risk_mgr.challenge_status == "PASSED":
            print(f"\n  [OK] CHALLENGE PASSED AT {timestamp}")
            break
            
        # Trade selection
        if len(open_positions) >= MAX_CONCURRENT_POSITIONS or daily_trades_count >= MAX_TRADES_PER_DAY:
            continue
            
        potential_trades = []
        for pair in pairs:
            if timestamp not in all_probs[pair].index:
                continue
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
                "pair": pair, "signal": signal, "confidence": confidence,
                "distance": distance, "lots": check.position_size_lots
            })
            
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
                direction=int(best["signal"]), entry_price=entry_price,
                entry_time=timestamp, lots=best["lots"], stop_loss=sl,
                take_profit=tp, pair=pair, bar_index=0,
                breakeven_locked=False, original_sl_dist=best["distance"]
            )
            open_positions.append(pos)
            risk_mgr.record_trade_open()
            daily_trades_count += 1
            dir_str = "LONG" if best["signal"] == Signal.LONG else "SHORT"
            print(f"  [{timestamp}] {dir_str} {pair} (Conf: {best['confidence']:.2%}) | Lots: {best['lots']}")

    # Results
    n_trades = len(closed_trades)
    winners = [t for t in closed_trades if t['pnl'] > 0]
    losers = [t for t in closed_trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in closed_trades)
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    gross_profit = sum(t['pnl'] for t in winners) if winners else 0
    gross_loss = abs(sum(t['pnl'] for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {label}")
    print(f"{'=' * 70}")
    print(f"  Status          : {risk_mgr.challenge_status}")
    print(f"  Final Equity    : ${equity_curve[-1]:,.2f}" if equity_curve else "  N/A")
    print(f"  Total P&L       : ${total_pnl:,.2f}")
    print(f"  Return          : {(total_pnl / INITIAL_BALANCE) * 100:.2f}%")
    print(f"  Total Trades    : {n_trades}")
    print(f"  Win Rate        : {win_rate:.2%}")
    print(f"  Profit Factor   : {profit_factor:.2f}")
    print(f"  Max Drawdown    : {max_dd:.2%}")
    print(f"  Trading Days    : {len(risk_mgr.trading_days)}")
    print(f"  ML Flip Exits   : {ml_flip_exits}")
    print(f"  TP Lock Ratchets: {tp_lock_exits}")
    
    # Exit reasons
    reasons = {}
    for t in closed_trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    print(f"\n  --- Exit Reasons ---")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason:<20}: {count}")
    
    # Pair breakdown
    pair_stats = {p: {'trades': 0, 'wins': 0, 'pnl': 0.0} for p in pairs}
    for t in closed_trades:
        p = t['pair']
        pair_stats[p]['trades'] += 1
        if t['pnl'] > 0:
            pair_stats[p]['wins'] += 1
        pair_stats[p]['pnl'] += t['pnl']
    print(f"\n  {'Pair':<8} | {'Trades':<8} | {'Win Rate':<10} | {'Total P&L':<10}")
    print(f"  {'-' * 50}")
    for p, stats in pair_stats.items():
        wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0.0
        print(f"  {p:<8} | {stats['trades']:<8} | {wr:>5.1f}%     | ${stats['pnl']:>8.2f}")
    print(f"{'=' * 70}")
    
    return {
        "label": label, "final_equity": equity_curve[-1] if equity_curve else INITIAL_BALANCE,
        "total_pnl": total_pnl, "n_trades": n_trades, "win_rate": win_rate,
        "profit_factor": profit_factor, "max_dd": max_dd, "status": risk_mgr.challenge_status,
        "days": len(risk_mgr.trading_days), "ml_flips": ml_flip_exits, "tp_locks": tp_lock_exits
    }


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  GBPUSD INTEGRATION TEST")
    print("#  Solo + 4-Pair Portfolio (with Exhaustion Detection)")
    print("#" * 70)
    
    # Run 1: GBPUSD Solo
    solo = run_sim(["GBPUSD"], "GBPUSD SOLO TEST")
    
    # Run 2: Original 3 pairs (for reference)
    trio = run_sim(["USDCAD", "EURUSD", "NQ=F"], "ORIGINAL 3-PAIR PORTFOLIO")
    
    # Run 3: Full 4-pair portfolio
    quad = run_sim(["USDCAD", "EURUSD", "NQ=F", "GBPUSD"], "FULL 4-PAIR PORTFOLIO")
    
    # Comparison
    print("\n\n" + "=" * 90)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 90)
    print(f"  {'Metric':<22} | {'GBPUSD Solo':<18} | {'3 Pairs':<18} | {'4 Pairs (+GBP)':<18}")
    print(f"  {'-' * 80}")
    print(f"  {'Status':<22} | {solo['status']:<18} | {trio['status']:<18} | {quad['status']:<18}")
    print(f"  {'Final Equity':<22} | ${solo['final_equity']:<16,.2f} | ${trio['final_equity']:<16,.2f} | ${quad['final_equity']:<16,.2f}")
    print(f"  {'Total P&L':<22} | ${solo['total_pnl']:<16,.2f} | ${trio['total_pnl']:<16,.2f} | ${quad['total_pnl']:<16,.2f}")
    print(f"  {'Total Trades':<22} | {solo['n_trades']:<18} | {trio['n_trades']:<18} | {quad['n_trades']:<18}")
    print(f"  {'Win Rate':<22} | {solo['win_rate']:<18.2%} | {trio['win_rate']:<18.2%} | {quad['win_rate']:<18.2%}")
    print(f"  {'Profit Factor':<22} | {solo['profit_factor']:<18.2f} | {trio['profit_factor']:<18.2f} | {quad['profit_factor']:<18.2f}")
    print(f"  {'Max Drawdown':<22} | {solo['max_dd']:<18.2%} | {trio['max_dd']:<18.2%} | {quad['max_dd']:<18.2%}")
    print(f"  {'Trading Days':<22} | {solo['days']:<18} | {trio['days']:<18} | {quad['days']:<18}")
    print(f"  {'ML Flip Exits':<22} | {solo['ml_flips']:<18} | {trio['ml_flips']:<18} | {quad['ml_flips']:<18}")
    print("=" * 90)
