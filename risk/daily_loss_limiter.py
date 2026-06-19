"""
Daily Loss Limiter — Enforce maximum daily loss limits.

Tracks P&L since the start of each trading day and halts trading
when the daily loss approaches the limit. Includes floating P&L
of open positions (many prop firms count unrealised losses).
"""

from datetime import datetime, date
from typing import Optional


class DailyLossLimiter:
    """
    Daily loss tracking and enforcement.

    Resets at the start of each new trading day (by server time).
    Includes a safety buffer to prevent breaching on execution.
    """

    def __init__(self, initial_balance: float, max_daily_loss_pct: float,
                 safety_buffer_pct: float = 0.005):
        """
        Parameters
        ----------
        initial_balance : float
            Starting balance (used to calculate the absolute daily limit).
        max_daily_loss_pct : float
            Maximum allowed daily loss (e.g., 0.05 = 5%).
        safety_buffer_pct : float
            Stop trading this much before the actual limit.
        """
        self.initial_balance = initial_balance
        self.max_daily_loss_pct = max_daily_loss_pct
        self.safety_buffer_pct = safety_buffer_pct

        self._current_date: Optional[date] = None
        self._day_start_equity: float = initial_balance
        self._current_equity: float = initial_balance
        self.is_halted: bool = False
        self._halt_reason: str = ""
        self.max_daily_loss_pct_recorded: float = 0.0

    def update(self, current_equity: float, timestamp: datetime) -> None:
        """
        Update with current equity and timestamp.

        Automatically resets when a new trading day starts.

        Parameters
        ----------
        current_equity : float
            Current account equity including floating P&L.
        timestamp : datetime
            Current bar timestamp (used to detect day boundaries).
        """
        current_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp

        # New day — reset
        if self._current_date is None or current_date != self._current_date:
            self._current_date = current_date
            self._day_start_equity = current_equity
            self.is_halted = False
            self._halt_reason = ""

        self._current_equity = current_equity
        
        # Track maximum daily loss ever recorded
        current_loss_pct = self.daily_loss_pct
        if current_loss_pct > self.max_daily_loss_pct_recorded:
            self.max_daily_loss_pct_recorded = current_loss_pct

        # Check daily loss
        # Most prop firms calculate daily loss from the INITIAL BALANCE, not day-start equity
        max_daily_loss_amount = self.initial_balance * self.max_daily_loss_pct
        buffer_amount = self.initial_balance * self.safety_buffer_pct
        effective_limit = max_daily_loss_amount - buffer_amount

        daily_pnl = current_equity - self._day_start_equity

        if daily_pnl < 0 and abs(daily_pnl) >= effective_limit:
            self.is_halted = True
            self._halt_reason = (
                f"Daily loss ${abs(daily_pnl):,.2f} approaching limit "
                f"${max_daily_loss_amount:,.2f} "
                f"({abs(daily_pnl)/self.initial_balance:.2%} of initial balance)"
            )

    @property
    def daily_pnl(self) -> float:
        """Current day's P&L (positive = profit, negative = loss)."""
        return self._current_equity - self._day_start_equity

    @property
    def daily_loss_pct(self) -> float:
        """Current day's loss as fraction of initial balance (0 if profitable)."""
        pnl = self.daily_pnl
        if pnl >= 0 or self.initial_balance <= 0:
            return 0.0
        return abs(pnl) / self.initial_balance

    @property
    def remaining_daily_loss(self) -> float:
        """Remaining daily loss budget in dollars."""
        max_loss = self.initial_balance * self.max_daily_loss_pct
        current_loss = max(0.0, -self.daily_pnl)
        return max(0.0, max_loss - current_loss)

    def can_trade(self) -> bool:
        """Whether trading is allowed today."""
        return not self.is_halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason if self.is_halted else ""
