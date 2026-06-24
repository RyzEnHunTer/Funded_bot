"""
Exhaustion Detection Simulation Engine
Tests two new features against the baseline portfolio sim:
  1. ML Confidence Decay — close trades when the model flips against you.
  2. 90% Trailing Lock — aggressively ratchet SL when price hits 85-90% of TP.

Run:  python scripts/run_exhaustion_sim.py
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

# ─── EXHAUSTION CONFIGURATION ─────────────────────────────────────────────────
EXHAUSTION_ML_FLIP_THRESHOLD = 0.45   # If counter-signal prob exceeds this, close trade
EXHAUSTION_TP_LOCK_PCT       = 0.85   # When price reaches 85% of TP distance...
EXHAUSTION_LOCK_PROFIT_PCT   = 0.70   # ...lock in 70% of current floating profit as new SL
# ──────────────────────────────────────────────────────────────────────────────

def close_position(pos, exit_price, exit_reason, timestamp, balance, risk_mgr, closed_trades, pip_size, pip_value):
    """Helper to close a position and return updated balance."""
    price_diff = (exit_price - pos.entry_price) * pos.direction
    pnl_pips = price_diff / pip_size
    pnl_gross = pnl_pips * pip_value * pos.lots
    commission = COMMISSION_PER_LOT * pos.lots
    pnl_net = pnl_gross - commission
    
    balance += pnl_net
    risk_mgr.record_trade_close(pnl_net, timestamp)
    
    closed_trades.append({
        "pair": pos.pair,
        "direction": pos.direction,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "entry_time": pos.entry_time,
        "exit_time": timestamp,
        "lots": pos.lots,
        "pnl": pnl_net,
        "pnl_pips": pnl_pips,
        "reason": exit_reason,
        "balance_after": balance
    })
    return balance, pnl_net


def run_sim(enable_exhaustion: bool, label: str):
    """Run a full portfolio simulation with or without exhaustion features."""
    pairs = ["USDCAD", "EURUSD", "NQ=F"]
    timeframe = "15m"
    
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Exhaustion Detection: {'ENABLED' if enable_exhaustion else 'DISABLED (Baseline)'}")
    print(f"{'=' * 70}")
    
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
            
            predictor = MLPredictor(pair, timeframe)
            valid_mask = df.notna().all(axis=1)
            valid_indices = df.index[valid_mask]
            
            probs = pd.DataFrame()
            if len(valid_indices) > 0:
                probs = predictor.predict_proba_batch(df.loc[valid_mask])
            
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
    print(f"  Timeline: {len(unified_index)} bars ({unified_index[0]} -> {unified_index[-1]})\n")
    
    # ─── 2. SETUP ENGINE ───
    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    open_positions = []
    closed_trades = []
    
    risk_mgr = PropFirmManager(initial_balance=INITIAL_BALANCE)
    session_filter = SessionFilter(require_overlap_only=False)
    
    daily_trades_count = 0
    last_trade_date = None
    
    equity_curve = []
    equity_timestamps = []
    
    # Exhaustion event counters
    ml_flip_exits = 0
    tp_lock_exits = 0
    
    for i, timestamp in enumerate(unified_index):
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
            
            # --- Standard Breakeven + Runner Logic ---
            st_line = df.loc[timestamp, "_supertrend_line"] if "_supertrend_line" in df.columns else None
            
            # 1:1 Scale-out check
            if not pos.breakeven_locked:
                hit_1_to_1 = False
                if pos.direction == 1 and high >= pos.entry_price + pos.original_sl_dist:
                    hit_1_to_1 = True
                elif pos.direction == -1 and low <= pos.entry_price - pos.original_sl_dist:
                    hit_1_to_1 = True
                    
                if hit_1_to_1:
                    scale_lots = pos.lots / 2.0
                    pos.lots -= scale_lots
                    pnl_pips = (pos.original_sl_dist / pip_size)
                    pnl_gross = pnl_pips * pip_value * scale_lots
                    commission = COMMISSION_PER_LOT * scale_lots
                    pnl_net = pnl_gross - commission
                    
                    balance += pnl_net
                    risk_mgr.record_trade_close(pnl_net, timestamp)
                    
                    closed_trades.append({
                        "pair": pair,
                        "direction": pos.direction,
                        "entry_price": pos.entry_price,
                        "exit_price": pos.entry_price + pos.original_sl_dist * pos.direction,
                        "entry_time": pos.entry_time,
                        "exit_time": timestamp,
                        "lots": scale_lots,
                        "pnl": pnl_net,
                        "pnl_pips": pnl_pips,
                        "reason": "partial_profit",
                        "balance_after": balance
                    })
                    
                    pos.stop_loss = pos.entry_price
                    pos.breakeven_locked = True

            # Supertrend Trailing (if Runner)
            if pos.breakeven_locked and st_line is not None and not pd.isna(st_line):
                if pos.direction == 1 and close > pos.stop_loss and st_line > pos.stop_loss:
                    pos.stop_loss = st_line
                elif pos.direction == -1 and close < pos.stop_loss and st_line < pos.stop_loss:
                    pos.stop_loss = st_line

            # ════════════════════════════════════════════════════════════════════
            # ██  EXHAUSTION FEATURE #1: ML Confidence Decay (Reversal Brain)  ██
            # ════════════════════════════════════════════════════════════════════
            if enable_exhaustion and pos.breakeven_locked:
                if timestamp in all_probs[pair].index:
                    prob_row = all_probs[pair].loc[timestamp]
                    prob_long = prob_row.get("prob_1", 0.0)
                    prob_short = prob_row.get("prob_-1", 0.0)
                    
                    # If we are LONG and the ML now screams SHORT
                    if pos.direction == 1 and prob_short >= EXHAUSTION_ML_FLIP_THRESHOLD:
                        # Only close if we are actually in profit
                        if close > pos.entry_price:
                            balance, pnl = close_position(pos, close, "ml_flip_exit", timestamp, balance, risk_mgr, closed_trades, pip_size, pip_value)
                            positions_to_close.append(pos)
                            ml_flip_exits += 1
                            print(f"  [{timestamp}] [BRAIN] ML FLIP EXIT: Closed LONG {pair} (prob_short={prob_short:.2%})")
                            continue
                    # If we are SHORT and the ML now screams LONG
                    elif pos.direction == -1 and prob_long >= EXHAUSTION_ML_FLIP_THRESHOLD:
                        if close < pos.entry_price:
                            balance, pnl = close_position(pos, close, "ml_flip_exit", timestamp, balance, risk_mgr, closed_trades, pip_size, pip_value)
                            positions_to_close.append(pos)
                            ml_flip_exits += 1
                            print(f"  [{timestamp}] [BRAIN] ML FLIP EXIT: Closed SHORT {pair} (prob_long={prob_long:.2%})")
                            continue

            # ════════════════════════════════════════════════════════════════════
            # ██  EXHAUSTION FEATURE #2: 90% Trailing Lock                     ██
            # ════════════════════════════════════════════════════════════════════
            if enable_exhaustion and pos.breakeven_locked:
                total_tp_dist = abs(pos.take_profit - pos.entry_price)
                current_move = (close - pos.entry_price) * pos.direction
                
                if total_tp_dist > 0 and current_move > 0:
                    progress_pct = current_move / total_tp_dist
                    
                    if progress_pct >= EXHAUSTION_TP_LOCK_PCT:
                        # Lock in 70% of the current floating profit as the new SL
                        locked_move = current_move * EXHAUSTION_LOCK_PROFIT_PCT
                        if pos.direction == 1:
                            new_sl = pos.entry_price + locked_move
                        else:
                            new_sl = pos.entry_price - locked_move
                            
                        # Only ratchet forward, never backward
                        if pos.direction == 1 and new_sl > pos.stop_loss:
                            pos.stop_loss = new_sl
                            tp_lock_exits += 1
                        elif pos.direction == -1 and new_sl < pos.stop_loss:
                            pos.stop_loss = new_sl
                            tp_lock_exits += 1

            # Standard SL/TP check
            sl_hit = False
            tp_hit = False
            
            if pos.direction == 1:
                sl_hit = low <= pos.stop_loss
                tp_hit = high >= pos.take_profit
            else:
                sl_hit = high >= pos.stop_loss
                tp_hit = low <= pos.take_profit
                
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
            # ─── ANTI-STACKING FILTER: 1 TRADE PER PAIR ───
            pair_open_positions = [p for p in open_positions if p.pair == pair]
            if len(pair_open_positions) > 0:
                continue
                
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
                "pair": pair,
                "signal": signal,
                "confidence": confidence,
                "distance": distance,
                "lots": check.position_size_lots
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
            
            dir_str = "LONG" if best["signal"] == Signal.LONG else "SHORT"
            print(f"  [{timestamp}] {dir_str} {pair} (Conf: {best['confidence']:.2%}) | Lots: {best['lots']}")

    # ─── 5. RESULTS ───
    n_trades = len(closed_trades)
    winners = [t for t in closed_trades if t['pnl'] > 0]
    losers = [t for t in closed_trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in closed_trades)
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    
    gross_profit = sum(t['pnl'] for t in winners) if winners else 0
    gross_loss = abs(sum(t['pnl'] for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {label}")
    print(f"{'=' * 70}")
    print(f"  Status          : {risk_mgr.challenge_status}")
    print(f"  Final Equity    : ${equity_curve[-1]:,.2f}" if equity_curve else "  Final Equity    : N/A")
    print(f"  Total P&L       : ${total_pnl:,.2f}")
    print(f"  Return          : {(total_pnl / INITIAL_BALANCE) * 100:.2f}%")
    print(f"  Total Trades    : {n_trades}")
    print(f"  Win Rate        : {win_rate:.2%}")
    print(f"  Profit Factor   : {profit_factor:.2f}")
    print(f"  Max Drawdown    : {max_dd:.2%}")
    print(f"  Trading Days    : {len(risk_mgr.trading_days)}")
    
    if enable_exhaustion:
        print(f"\n  --- Exhaustion Stats ---")
        print(f"  ML Flip Exits       : {ml_flip_exits}")
        print(f"  90% TP Lock Ratchets: {tp_lock_exits}")
    
    # Exit reason breakdown
    reasons = {}
    for t in closed_trades:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1
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
        "label": label,
        "final_equity": equity_curve[-1] if equity_curve else INITIAL_BALANCE,
        "total_pnl": total_pnl,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "status": risk_mgr.challenge_status,
        "ml_flip_exits": ml_flip_exits,
        "tp_lock_exits": tp_lock_exits,
        "closed_trades": closed_trades,
    }


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  EXHAUSTION DETECTION A/B TEST")
    print("#  Baseline vs. ML Flip + 90% TP Lock")
    print("#" * 70)
    
    # Run baseline (no exhaustion)
    baseline = run_sim(enable_exhaustion=False, label="BASELINE (No Exhaustion)")
    
    # Run with exhaustion features
    exhaustion = run_sim(enable_exhaustion=True, label="EXHAUSTION DETECTION (ML Flip + 90% Lock)")
    
    # ─── HEAD-TO-HEAD COMPARISON ───
    print("\n\n" + "=" * 70)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<25} | {'Baseline':<20} | {'Exhaustion':<20}")
    print(f"  {'-' * 65}")
    print(f"  {'Status':<25} | {baseline['status']:<20} | {exhaustion['status']:<20}")
    print(f"  {'Final Equity':<25} | ${baseline['final_equity']:<18,.2f} | ${exhaustion['final_equity']:<18,.2f}")
    print(f"  {'Total P&L':<25} | ${baseline['total_pnl']:<18,.2f} | ${exhaustion['total_pnl']:<18,.2f}")
    print(f"  {'Total Trades':<25} | {baseline['n_trades']:<20} | {exhaustion['n_trades']:<20}")
    print(f"  {'Win Rate':<25} | {baseline['win_rate']:<20.2%} | {exhaustion['win_rate']:<20.2%}")
    print(f"  {'Profit Factor':<25} | {baseline['profit_factor']:<20.2f} | {exhaustion['profit_factor']:<20.2f}")
    print(f"  {'Max Drawdown':<25} | {baseline['max_dd']:<20.2%} | {exhaustion['max_dd']:<20.2%}")
    
    pnl_diff = exhaustion['total_pnl'] - baseline['total_pnl']
    print(f"\n  [$] Exhaustion P&L Edge: ${pnl_diff:+,.2f}")
    print(f"  [BRAIN] ML Flip Exits Saved: {exhaustion['ml_flip_exits']}")
    print(f"  [LOCK] 90% TP Lock Ratchets: {exhaustion['tp_lock_exits']}")
    print("=" * 70)
    
    # ─── DETAILED TRADE JOURNAL ───
    print("\n\n" + "=" * 120)
    print("  DETAILED EXHAUSTION TRADE JOURNAL (Every Trade)")
    print("=" * 120)
    print(f"  {'#':<4} {'Pair':<8} {'Dir':<6} {'Entry Time':<22} {'Exit Time':<22} {'Entry':<10} {'Exit':<10} {'Lots':<6} {'P&L':<10} {'Pips':<8} {'Reason':<18} {'Balance'}")
    print(f"  {'-' * 118}")
    
    for idx, t in enumerate(exhaustion['closed_trades'], 1):
        d = "LONG" if t['direction'] == 1 else "SHORT"
        entry_t = str(t['entry_time'])[:19]
        exit_t = str(t['exit_time'])[:19]
        entry_p = t.get('entry_price', 0)
        exit_p = t.get('exit_price', 0)
        lots = t.get('lots', 0)
        pips = t.get('pnl_pips', 0)
        bal = t.get('balance_after', 0)
        reason = t['reason']
        pnl = t['pnl']
        
        # Mark exhaustion exits
        tag = ""
        if reason == "ml_flip_exit":
            tag = " <<< BRAIN"
        elif reason == "partial_profit":
            tag = ""
        
        print(f"  {idx:<4} {t['pair']:<8} {d:<6} {entry_t:<22} {exit_t:<22} {entry_p:<10.5f} {exit_p:<10.5f} {lots:<6.2f} ${pnl:<9.2f} {pips:<8.1f} {reason:<18} ${bal:,.2f}{tag}")
    
    print(f"\n  Total Trades: {len(exhaustion['closed_trades'])}")
    
    # ─── ML FLIP EXITS ONLY ───
    ml_flips = [t for t in exhaustion['closed_trades'] if t['reason'] == 'ml_flip_exit']
    if ml_flips:
        print("\n\n" + "=" * 100)
        print("  ML FLIP EXITS ONLY (Trades saved from potential SL reversal)")
        print("=" * 100)
        total_saved = sum(t['pnl'] for t in ml_flips)
        for idx, t in enumerate(ml_flips, 1):
            d = "LONG" if t['direction'] == 1 else "SHORT"
            held = t['exit_time'] - t['entry_time']
            hours = held.total_seconds() / 3600
            print(f"  {idx}. {t['pair']} {d} | Entered: {str(t['entry_time'])[:16]} | Brain Exit: {str(t['exit_time'])[:16]} | Held: {hours:.1f}h | P&L: ${t['pnl']:+.2f}")
        print(f"\n  Total P&L from ML Flip Exits: ${total_saved:+,.2f}")
        print(f"  Average P&L per ML Flip: ${total_saved/len(ml_flips):+,.2f}")
    print("=" * 100)
