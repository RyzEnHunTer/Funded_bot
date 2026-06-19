"""
Drawdown Governor — Enforce maximum drawdown limits.

Tracks equity vs initial balance (and optionally peak equity for
trailing drawdown) and halts trading when limits are approached.
"""

from typing import Optional


class DrawdownGovernor:
    """
    Real-time drawdown tracking and enforcement.

    Supports two modes:
      "balance"  — drawdown measured from initial balance (FTMO, MFF)
      "trailing" — drawdown measured from peak equity (TFT)

    Safety buffer: halts trading 1% before the actual limit to prevent
    breaching on slippage, spread, or gap risk.
    """

    def __init__(self, initial_balance: float, max_drawdown_pct: float,
                 drawdown_type: str = "balance", safety_buffer_pct: float = 0.01):
        """
        Parameters
        ----------
        initial_balance : float
            Starting account balance.
        max_drawdown_pct : float
            Maximum allowed drawdown (e.g., 0.10 = 10%).
        drawdown_type : str
            "balance" or "trailing".
        safety_buffer_pct : float
            Stop trading this much before the actual limit (e.g., 0.01 = 1%).
        """
        self.initial_balance = initial_balance
        self.max_drawdown_pct = max_drawdown_pct
        self.drawdown_type = drawdown_type
        self.safety_buffer_pct = safety_buffer_pct

        self.peak_equity = initial_balance
        self.current_equity = initial_balance
        self.is_halted = False
        self._halt_reason = ""
        self.max_drawdown_pct_recorded = 0.0

    def update(self, current_equity: float) -> None:
        """Update with current equity (including floating P&L)."""
        self.current_equity = current_equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Check drawdown
        if self.drawdown_type == "trailing":
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            reference = self.peak_equity
        else:
            drawdown = (self.initial_balance - current_equity) / self.initial_balance
            reference = self.initial_balance

        effective_limit = self.max_drawdown_pct - self.safety_buffer_pct

        if drawdown > self.max_drawdown_pct_recorded:
            self.max_drawdown_pct_recorded = drawdown

        if drawdown >= effective_limit:
            self.is_halted = True
            self._halt_reason = (
                f"Drawdown {drawdown:.2%} approaching limit "
                f"{self.max_drawdown_pct:.2%} (buffer: {self.safety_buffer_pct:.2%})"
            )

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown as a fraction (0.0 to 1.0)."""
        if self.drawdown_type == "trailing":
            if self.peak_equity <= 0:
                return 0.0
            return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)
        else:
            if self.initial_balance <= 0:
                return 0.0
            return max(0.0, (self.initial_balance - self.current_equity) / self.initial_balance)

    @property
    def remaining_drawdown_pct(self) -> float:
        """How much drawdown room is left before the limit."""
        return max(0.0, self.max_drawdown_pct - self.current_drawdown_pct)

    def can_trade(self) -> bool:
        """Whether trading is currently allowed."""
        return not self.is_halted

    @property
    def halt_reason(self) -> str:
        """Reason for halt (empty if not halted)."""
        return self._halt_reason if self.is_halted else ""

    def reset(self, new_balance: Optional[float] = None) -> None:
        """Reset the governor (e.g., for a new challenge phase)."""
        if new_balance is not None:
            self.initial_balance = new_balance
        self.peak_equity = self.initial_balance
        self.current_equity = self.initial_balance
        self.is_halted = False
        self._halt_reason = ""
