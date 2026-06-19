"""
Prop Firm Manager — Unified risk orchestrator.

Coordinates all risk modules (drawdown governor, daily loss limiter,
session filter) into a single pre-trade check. Also tracks challenge
progress (profit target, trading days, phase status).
"""

from datetime import datetime, date
from typing import Optional, Dict
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.prop_firm_rules import PropFirmRuleSet, DEFAULT_RULES
from risk.drawdown_governor import DrawdownGovernor
from risk.daily_loss_limiter import DailyLossLimiter
from risk.session_filter import SessionFilter
from strategy.position_sizer import calculate_position_size, scale_for_drawdown
from config.settings import MAX_CONCURRENT_POSITIONS, RISK_PER_TRADE_PCT


@dataclass
class TradeCheckResult:
    """Result of a pre-trade check."""
    allowed: bool
    reason: str = ""
    position_size_lots: float = 0.0
    risk_score: float = 0.0   # 0.0 = lowest risk, 1.0 = highest risk


class PropFirmManager:
    """
    Unified prop firm rule orchestrator.

    Before any trade, checks:
      1. DrawdownGovernor — is overall DD within limit?
      2. DailyLossLimiter — is today's loss within limit?
      3. SessionFilter — are we in a tradeable session?
      4. Position count — not exceeding max concurrent positions
      5. Calculate position size (with drawdown scaling)

    Also tracks:
      - Profit target progress
      - Trading days count
      - Challenge status (passed / failed / in progress)
    """

    def __init__(self, initial_balance: float, rules: Optional[PropFirmRuleSet] = None):
        self.rules = rules or DEFAULT_RULES
        self.initial_balance = initial_balance

        # Initialise risk modules
        self.dd_governor = DrawdownGovernor(
            initial_balance=initial_balance,
            max_drawdown_pct=self.rules.max_drawdown_pct,
            drawdown_type=self.rules.drawdown_type,
        )

        self.daily_limiter = DailyLossLimiter(
            initial_balance=initial_balance,
            max_daily_loss_pct=self.rules.max_daily_loss_pct,
        )

        self.session_filter = SessionFilter()

        # Challenge tracking
        self.open_positions = 0
        self.trading_days: set = set()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.current_equity = initial_balance
        self._challenge_failed = False
        self._challenge_passed = False

    def update(self, equity: float, timestamp: datetime) -> None:
        """
        Update all risk modules with current equity and time.
        Call this on every bar, even if no trade is being placed.
        """
        self.current_equity = equity
        self.dd_governor.update(equity)
        self.daily_limiter.update(equity, timestamp)

        # Track trading days
        if self.total_trades > 0:
            self.trading_days.add(timestamp.date() if isinstance(timestamp, datetime) else timestamp)

        # Check if challenge is failed
        if not self.dd_governor.can_trade():
            self._challenge_failed = True

        # Check if profit target is reached
        if self.rules.profit_target_pct > 0:
            profit_pct = (equity - self.initial_balance) / self.initial_balance
            if (profit_pct >= self.rules.profit_target_pct
                    and len(self.trading_days) >= self.rules.min_trading_days):
                self._challenge_passed = True

    def pre_trade_check(self, pair: str, stop_distance: float,
                         timestamp: datetime) -> TradeCheckResult:
        """
        Run all pre-trade checks and return whether the trade is allowed.

        Parameters
        ----------
        pair : str
            Currency pair.
        stop_distance : float
            Stop-loss distance in price units.
        timestamp : datetime
            Current bar timestamp.

        Returns
        -------
        TradeCheckResult
            Contains: allowed, reason, position_size_lots, risk_score
        """
        # 1. Overall drawdown check
        if not self.dd_governor.can_trade():
            return TradeCheckResult(
                allowed=False,
                reason=f"HALTED: {self.dd_governor.halt_reason}",
            )

        # 2. Daily loss check
        if not self.daily_limiter.can_trade():
            return TradeCheckResult(
                allowed=False,
                reason=f"DAILY LIMIT: {self.daily_limiter.halt_reason}",
            )

        # 3. Session filter
        if not self.session_filter.is_tradeable(timestamp, symbol=pair):
            return TradeCheckResult(
                allowed=False,
                reason=f"OFF-SESSION: {self.session_filter.get_session_name(timestamp, symbol=pair)}",
            )

        # 4. Max concurrent positions
        if self.open_positions >= MAX_CONCURRENT_POSITIONS:
            return TradeCheckResult(
                allowed=False,
                reason=f"MAX POSITIONS: {self.open_positions}/{MAX_CONCURRENT_POSITIONS}",
            )

        # 5. Calculate position size
        base_lots = calculate_position_size(
            account_balance=self.current_equity,
            stop_distance_price=stop_distance,
            pair=pair,
            risk_pct=RISK_PER_TRADE_PCT,
        )

        # Scale down if approaching drawdown limit
        adjusted_lots = scale_for_drawdown(
            base_lots=base_lots,
            current_drawdown_pct=self.dd_governor.current_drawdown_pct,
            max_drawdown_pct=self.rules.max_drawdown_pct,
        )

        if adjusted_lots < 0.01:
            return TradeCheckResult(
                allowed=False,
                reason="Position size too small after drawdown scaling",
            )

        # Risk score (0 = safe, 1 = danger)
        risk_score = self.dd_governor.current_drawdown_pct / self.rules.max_drawdown_pct

        return TradeCheckResult(
            allowed=True,
            position_size_lots=adjusted_lots,
            risk_score=risk_score,
        )

    def record_trade_open(self) -> None:
        """Record that a new position was opened."""
        self.open_positions += 1
        self.total_trades += 1

    def record_trade_close(self, pnl: float, timestamp: datetime) -> None:
        """Record that a position was closed with the given P&L."""
        self.open_positions = max(0, self.open_positions - 1)
        self.total_pnl += pnl
        self.trading_days.add(timestamp.date() if isinstance(timestamp, datetime) else timestamp)

        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1

    @property
    def challenge_status(self) -> str:
        """Current challenge status."""
        if self._challenge_failed:
            return "FAILED"
        elif self._challenge_passed:
            return "PASSED"
        else:
            return "IN PROGRESS"

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction."""
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0

    @property
    def profit_pct(self) -> float:
        """Current profit as % of initial balance."""
        return (self.current_equity - self.initial_balance) / self.initial_balance

    def summary(self) -> Dict:
        """Get a summary of the current state."""
        return {
            "rules": self.rules.name,
            "status": self.challenge_status,
            "initial_balance": self.initial_balance,
            "current_equity": self.current_equity,
            "profit_pct": f"{self.profit_pct:.2%}",
            "profit_target": f"{self.rules.profit_target_pct:.2%}",
            "current_drawdown": f"{self.dd_governor.current_drawdown_pct:.2%}",
            "max_drawdown_limit": f"{self.rules.max_drawdown_pct:.2%}",
            "daily_loss": f"{self.daily_limiter.daily_loss_pct:.2%}",
            "daily_limit": f"{self.rules.max_daily_loss_pct:.2%}",
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.2%}",
            "trading_days": len(self.trading_days),
            "min_trading_days": self.rules.min_trading_days,
            "open_positions": self.open_positions,
        }
