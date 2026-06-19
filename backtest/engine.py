"""
Backtest Engine — Event-driven bar-by-bar backtester.

Simulates the full trading system: ML prediction → signal → prop firm check
→ position management → P&L tracking. Accounts for spread, slippage,
and commission.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    INITIAL_BALANCE, SPREADS, SLIPPAGE_PIPS, PIP_SIZES,
    COMMISSION_PER_LOT, REWARD_RISK_RATIO, SIGNAL_COOLDOWN,
)
from config.prop_firm_rules import PropFirmRuleSet, DEFAULT_RULES
from ml.features import compute_all_features, FEATURE_NAMES, get_supertrend_direction
from ml.predictor import MLPredictor
from strategy.signal_generator import generate_signal, Signal
from risk.prop_firm_manager import PropFirmManager


@dataclass
class Position:
    """An open trade position."""
    direction: int          # +1 for long, -1 for short
    entry_price: float
    entry_time: datetime
    lots: float
    stop_loss: float
    take_profit: float
    pair: str
    bar_index: int
    breakeven_locked: bool = False
    original_sl_dist: float = 0.0

    def unrealised_pnl(self, current_price: float, pip_size: float, pip_value: float) -> float:
        """Calculate floating P&L in dollars."""
        price_diff = (current_price - self.entry_price) * self.direction
        pips = price_diff / pip_size
        return pips * pip_value * self.lots


@dataclass
class ClosedTrade:
    """A completed trade with full P&L record."""
    direction: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    lots: float
    pnl: float
    pnl_pips: float
    exit_reason: str        # "stop_loss", "take_profit", "signal"
    bars_held: int


class BacktestEngine:
    """
    Event-driven backtester with full prop firm rule enforcement.

    Iterates bar-by-bar through historical data:
      1. Update open positions (check SL/TP hits)
      2. Update risk modules (equity, daily loss, drawdown)
      3. If no open position → compute features → ML predict → signal
      4. If signal + pre-trade check passes → open position
      5. Track equity curve, trades, metrics
    """

    def __init__(self, pair: str, timeframe: str,
                 rules: Optional[PropFirmRuleSet] = None,
                 initial_balance: float = INITIAL_BALANCE):
        self.pair = pair
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.rules = rules or DEFAULT_RULES

        self.pip_size = PIP_SIZES.get(pair, 0.0001)
        self.pip_value = 10.0  # USD per pip per lot (for USD-quoted pairs)
        self.spread_pips = SPREADS.get(pair, 2.0)

        # State
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self.equity_curve: List[float] = []
        self.equity_timestamps: List[datetime] = []

        # Risk manager
        self.risk_mgr = PropFirmManager(initial_balance, self.rules)

        # Signal cooldown tracker
        self._last_trade_bar = -SIGNAL_COOLDOWN - 1

    def run(self, df: pd.DataFrame, predictor: MLPredictor,
            verbose: bool = True) -> Dict:
        """
        Run the backtest on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex. Features will be computed.
        predictor : MLPredictor
            Trained ML model for predictions.
        verbose : bool
            Print progress updates.

        Returns
        -------
        Dict with backtest results.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  BACKTESTING -- {self.pair} {self.timeframe}")
            print(f"  Rules: {self.rules.name}")
            print(f"  Balance: ${self.initial_balance:,.2f}")
            print(f"  Data: {len(df):,} bars ({df.index[0]} -> {df.index[-1]})")
            print(f"{'=' * 60}")

        # Suppress sklearn warnings for clean output
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        # Compute features on the full dataset
        if verbose:
            print("  Computing features...")
        df = compute_all_features(df)

        # Get supertrend direction
        st_direction = get_supertrend_direction(df)

        # Pre-compute ALL predictions in batch (MUCH faster than per-bar)
        if verbose:
            print("  Batch-predicting all bars...")
        valid_mask = df[FEATURE_NAMES].notna().all(axis=1)
        valid_indices = df.index[valid_mask]
        if len(valid_indices) > 0:
            all_probs = predictor.predict_proba_batch(df.loc[valid_mask])
            if verbose:
                print(f"  Predicted {len(all_probs):,} bars in batch")
        else:
            all_probs = pd.DataFrame()

        # Pre-compute barrier distances in batch
        if verbose:
            print("  Computing barrier distances...")
        from ml.labeling import compute_barrier_distance
        barrier_distances = compute_barrier_distance(df["close"], df["high"], df["low"])

        n = len(df)
        skipped_signals = 0
        blocked_by_risk = 0

        for i in range(210, n):  # Skip warmup
            bar = df.iloc[i]
            timestamp = df.index[i]
            close = bar["close"]
            high = bar["high"]
            low = bar["low"]

            # ── 1. Update open positions ──────────────────────────────────────
            positions_to_close = []
            for pos in self.positions:
                # Check stop-loss (using low for longs, high for shorts)
                sl_hit = False
                tp_hit = False

                if pos.direction == 1:  # Long
                    sl_hit = low <= pos.stop_loss
                    tp_hit = high >= pos.take_profit
                else:  # Short
                    sl_hit = high >= pos.stop_loss
                    tp_hit = low <= pos.take_profit

                if sl_hit and tp_hit:
                    # Both hit — assume SL hit first (conservative)
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                elif sl_hit:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                elif tp_hit:
                    exit_price = pos.take_profit
                    exit_reason = "take_profit"
                else:
                    continue

                # Calculate P&L
                price_diff = (exit_price - pos.entry_price) * pos.direction
                pnl_pips = price_diff / self.pip_size
                pnl_gross = pnl_pips * self.pip_value * pos.lots
                commission = COMMISSION_PER_LOT * pos.lots
                pnl_net = pnl_gross - commission

                # Record closed trade
                trade = ClosedTrade(
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    entry_time=pos.entry_time,
                    exit_time=timestamp,
                    lots=pos.lots,
                    pnl=pnl_net,
                    pnl_pips=pnl_pips,
                    exit_reason=exit_reason,
                    bars_held=i - pos.bar_index,
                )
                self.closed_trades.append(trade)
                positions_to_close.append(pos)

                # Update balance
                self.balance += pnl_net
                self.risk_mgr.record_trade_close(pnl_net, timestamp)

            # Remove closed positions
            for pos in positions_to_close:
                self.positions.remove(pos)

            # ── 2. Calculate equity (balance + floating P&L) ──────────────────
            floating_pnl = sum(
                pos.unrealised_pnl(close, self.pip_size, self.pip_value)
                for pos in self.positions
            )
            self.equity = self.balance + floating_pnl

            # Record equity curve
            self.equity_curve.append(self.equity)
            self.equity_timestamps.append(timestamp)

            # ── 3. Update risk manager ────────────────────────────────────────
            self.risk_mgr.update(self.equity, timestamp)

            # Check if challenge failed or passed
            if self.risk_mgr.challenge_status == "FAILED":
                if verbose:
                    print(f"  [X] CHALLENGE FAILED at bar {i} ({timestamp})")
                break
            if self.risk_mgr.challenge_status == "PASSED":
                if verbose:
                    print(f"  [OK] CHALLENGE PASSED at bar {i} ({timestamp})")
                break

            # ── 4. Signal generation (only if no open positions) ──────────────
            if len(self.positions) > 0:
                continue

            # Cooldown check
            if i - self._last_trade_bar < SIGNAL_COOLDOWN:
                continue

            # Check if we have valid features for this bar
            timestamp_key = df.index[i]
            if timestamp_key not in all_probs.index:
                continue

            # Look up pre-computed ML probabilities
            prob_row = all_probs.loc[timestamp_key]
            ml_probs = {}
            for col in prob_row.index:
                # col format: "prob_1", "prob_0", "prob_-1"
                cls = int(col.split("_")[1])
                ml_probs[cls] = float(prob_row[col])

            # Get supertrend direction
            st_dir = int(st_direction.iloc[i]) if i < len(st_direction) else 0

            # Generate signal
            signal = generate_signal(ml_probs, st_dir)

            if signal == Signal.FLAT:
                continue

            # ── 5. Pre-trade check ────────────────────────────────────────────
            # Look up pre-computed barrier distance
            distance = float(barrier_distances.iloc[i])

            if np.isnan(distance) or distance <= 0:
                continue

            check = self.risk_mgr.pre_trade_check(self.pair, distance, timestamp)

            if not check.allowed:
                blocked_by_risk += 1
                continue

            # ── 6. Open position ──────────────────────────────────────────────
            entry_price = close
            # Apply spread (entry gets worse by half the spread)
            spread_cost = self.spread_pips * self.pip_size / 2
            if signal == Signal.LONG:
                entry_price += spread_cost  # Buy at ask
                stop_loss = entry_price - distance
                take_profit = entry_price + distance * REWARD_RISK_RATIO
            else:
                entry_price -= spread_cost  # Sell at bid
                stop_loss = entry_price + distance
                take_profit = entry_price - distance * REWARD_RISK_RATIO

            pos = Position(
                direction=int(signal),
                entry_price=entry_price,
                entry_time=timestamp,
                lots=check.position_size_lots,
                stop_loss=stop_loss,
                take_profit=take_profit,
                pair=self.pair,
                bar_index=i,
            )
            self.positions.append(pos)
            self.risk_mgr.record_trade_open()
            self._last_trade_bar = i

            # Apply slippage to balance (small cost)
            slippage_cost = SLIPPAGE_PIPS * self.pip_size * self.pip_value * pos.lots / 2
            self.balance -= slippage_cost

        # ── Build results ─────────────────────────────────────────────────────
        return self._build_results(verbose)

    def _build_results(self, verbose: bool) -> Dict:
        """Compile backtest results into a summary."""
        trades = self.closed_trades
        n_trades = len(trades)

        if n_trades == 0:
            if verbose:
                print("\n  WARNING: No trades were executed.")
            return {"trades": 0, "status": self.risk_mgr.challenge_status}

        # Trade stats
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
        win_rate = len(winners) / n_trades if n_trades > 0 else 0

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl for t in losers]) if losers else 0
        avg_trade_pnl = total_pnl / n_trades

        # Max consecutive losses
        max_consec_losses = 0
        current_streak = 0
        for t in trades:
            if t.pnl < 0:
                current_streak += 1
                max_consec_losses = max(max_consec_losses, current_streak)
            else:
                current_streak = 0

        # Drawdown from equity curve
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        # Average bars held
        avg_bars = np.mean([t.bars_held for t in trades])

        # Sharpe ratio (annualised, assuming 1h bars)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if np.std(returns) > 0:
                bars_per_year = 252 * 24  # ~6048 hours per year
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Recovery factor
        recovery_factor = total_pnl / (max_drawdown * self.initial_balance) if max_drawdown > 0 else 0

        results = {
            "status": self.risk_mgr.challenge_status,
            "initial_balance": self.initial_balance,
            "final_equity": self.equity,
            "total_pnl": total_pnl,
            "return_pct": (self.equity - self.initial_balance) / self.initial_balance,
            "trades": n_trades,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade_pnl": avg_trade_pnl,
            "max_drawdown_pct": max_drawdown,
            "max_consecutive_losses": max_consec_losses,
            "avg_bars_held": avg_bars,
            "sharpe_ratio": sharpe,
            "recovery_factor": recovery_factor,
            "trading_days": len(self.risk_mgr.trading_days),
            "equity_curve": self.equity_curve,
            "equity_timestamps": self.equity_timestamps,
            "closed_trades": trades,
        }

        if verbose:
            self._print_report(results)

        return results

    def _print_report(self, r: Dict) -> None:
        """Print a formatted backtest report."""
        print(f"\n{'=' * 60}")
        print(f"  BACKTEST RESULTS -- {self.pair} {self.timeframe}")
        print(f"{'=' * 60}")
        print(f"  Challenge Status : {r['status']}")
        print(f"  Rules            : {self.rules.name}")
        print(f"{'-' * 60}")
        print(f"  Initial Balance  : ${r['initial_balance']:>12,.2f}")
        print(f"  Final Equity     : ${r['final_equity']:>12,.2f}")
        print(f"  Total P&L        : ${r['total_pnl']:>12,.2f}")
        print(f"  Return           : {r['return_pct']:>12.2%}")
        print(f"{'-' * 60}")
        print(f"  Total Trades     : {r['trades']:>12}")
        print(f"  Winners          : {r['winners']:>12}")
        print(f"  Losers           : {r['losers']:>12}")
        print(f"  Win Rate         : {r['win_rate']:>12.2%}")
        print(f"  Profit Factor    : {r['profit_factor']:>12.2f}")
        print(f"{'-' * 60}")
        print(f"  Avg Win          : ${r['avg_win']:>12,.2f}")
        print(f"  Avg Loss         : ${r['avg_loss']:>12,.2f}")
        print(f"  Avg Trade P&L    : ${r['avg_trade_pnl']:>12,.2f}")
        print(f"{'-' * 60}")
        print(f"  Max Drawdown     : {r['max_drawdown_pct']:>12.2%}")
        print(f"  Max Consec Losses: {r['max_consecutive_losses']:>12}")
        print(f"  Avg Bars Held    : {r['avg_bars_held']:>12.1f}")
        print(f"  Sharpe Ratio     : {r['sharpe_ratio']:>12.2f}")
        print(f"  Recovery Factor  : {r['recovery_factor']:>12.2f}")
        print(f"  Trading Days     : {r['trading_days']:>12}")
        print(f"{'=' * 60}")

        # Prop firm compliance check
        print(f"\n  PROP FIRM COMPLIANCE:")
        dd_ok = r['max_drawdown_pct'] < self.rules.max_drawdown_pct
        print(f"    Max DD     : {r['max_drawdown_pct']:.2%} / {self.rules.max_drawdown_pct:.2%}  {'OK' if dd_ok else 'FAIL'}")

        if self.rules.profit_target_pct > 0:
            pt_ok = r['return_pct'] >= self.rules.profit_target_pct
            print(f"    Profit     : {r['return_pct']:.2%} / {self.rules.profit_target_pct:.2%}  {'OK' if pt_ok else 'FAIL'}")

        if self.rules.min_trading_days > 0:
            td_ok = r['trading_days'] >= self.rules.min_trading_days
            print(f"    Trade Days : {r['trading_days']} / {self.rules.min_trading_days}  {'OK' if td_ok else 'FAIL'}")

        print()
