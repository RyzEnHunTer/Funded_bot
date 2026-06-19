"""
Prop Firm Rule Definitions.

Each prop firm has different rules for their challenge and funded phases.
This module defines them as dataclasses so the risk engine can enforce
whichever rule set is active.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PropFirmRuleSet:
    """
    Complete rule set for a single prop firm phase.

    Attributes
    ----------
    name : str
        Human-readable name (e.g., "FTMO Challenge Phase 1").
    max_drawdown_pct : float
        Maximum allowed drawdown from initial balance (e.g., 0.10 = 10%).
        Most firms measure this from the initial balance, not the peak.
    max_daily_loss_pct : float
        Maximum allowed loss in a single trading day (e.g., 0.05 = 5%).
        Includes floating P&L on open positions.
    profit_target_pct : float
        Profit target to pass the phase (e.g., 0.10 = 10%).
        Set to 0.0 for funded accounts (no target, just trade profitably).
    min_trading_days : int
        Minimum number of days with at least one trade.
        0 means no minimum (some firms removed this requirement).
    max_trading_days : Optional[int]
        Maximum days to reach the profit target.
        None means unlimited.
    max_position_lots : float
        Maximum position size in lots. Varies by account size.
    allow_weekend_holding : bool
        Whether positions can be held over the weekend.
    allow_news_trading : bool
        Whether trading during major news events is allowed.
    drawdown_type : str
        "balance" — DD measured from initial balance only.
        "trailing" — DD measured from peak equity (harder).
    """
    name: str
    max_drawdown_pct: float
    max_daily_loss_pct: float
    profit_target_pct: float
    min_trading_days: int = 0
    max_trading_days: Optional[int] = None
    max_position_lots: float = 20.0
    allow_weekend_holding: bool = True
    allow_news_trading: bool = True
    drawdown_type: str = "balance"  # "balance" or "trailing"


# ─── FTMO Rules ───────────────────────────────────────────────────────────────

FTMO_CHALLENGE = PropFirmRuleSet(
    name="FTMO Challenge",
    max_drawdown_pct=0.10,          # 10% max drawdown
    max_daily_loss_pct=0.05,        # 5% max daily loss
    profit_target_pct=0.10,         # 10% profit target
    min_trading_days=4,             # At least 4 trading days
    max_trading_days=None,          # No time limit (new rules)
    max_position_lots=20.0,
    allow_weekend_holding=True,
    allow_news_trading=False,       # Restricted during high-impact news
    drawdown_type="balance",
)

FTMO_VERIFICATION = PropFirmRuleSet(
    name="FTMO Verification",
    max_drawdown_pct=0.10,          # Same 10% max drawdown
    max_daily_loss_pct=0.05,        # Same 5% daily limit
    profit_target_pct=0.05,         # Only 5% profit target (easier)
    min_trading_days=4,
    max_trading_days=None,
    max_position_lots=20.0,
    allow_weekend_holding=True,
    allow_news_trading=False,
    drawdown_type="balance",
)

FTMO_FUNDED = PropFirmRuleSet(
    name="FTMO Funded",
    max_drawdown_pct=0.10,
    max_daily_loss_pct=0.05,
    profit_target_pct=0.0,          # No target — just stay profitable
    min_trading_days=0,
    max_trading_days=None,
    max_position_lots=20.0,
    allow_weekend_holding=True,
    allow_news_trading=False,
    drawdown_type="balance",
)

# ─── MyForexFunds (MFF) Rules ────────────────────────────────────────────────

MFF_CHALLENGE = PropFirmRuleSet(
    name="MFF Evaluation",
    max_drawdown_pct=0.12,          # 12% max drawdown
    max_daily_loss_pct=0.05,        # 5% daily limit
    profit_target_pct=0.08,         # 8% profit target
    min_trading_days=5,
    max_trading_days=None,
    max_position_lots=20.0,
    allow_weekend_holding=True,
    allow_news_trading=True,
    drawdown_type="balance",
)

# ─── The Funded Trader (TFT) Rules ───────────────────────────────────────────

TFT_CHALLENGE = PropFirmRuleSet(
    name="TFT Challenge",
    max_drawdown_pct=0.10,
    max_daily_loss_pct=0.05,
    profit_target_pct=0.10,
    min_trading_days=3,
    max_trading_days=None,
    max_position_lots=20.0,
    allow_weekend_holding=True,
    allow_news_trading=True,
    drawdown_type="trailing",       # TFT uses trailing drawdown!
)

# ─── Default Rule Set ────────────────────────────────────────────────────────

CUSTOM_CHALLENGE = PropFirmRuleSet(
    name="Custom 14% Challenge",
    max_drawdown_pct=0.10,          # 10% max drawdown
    max_daily_loss_pct=0.04,        # 4% max daily loss
    profit_target_pct=0.14,         # 14% profit target
    min_trading_days=0,             
    max_trading_days=None,          
    max_position_lots=20.0,
    allow_weekend_holding=True,
    allow_news_trading=True,       
    drawdown_type="balance",
)

# Default to Custom Challenge
DEFAULT_RULES = CUSTOM_CHALLENGE

# Registry for easy lookup
RULE_SETS = {
    "ftmo_challenge": FTMO_CHALLENGE,
    "ftmo_verification": FTMO_VERIFICATION,
    "ftmo_funded": FTMO_FUNDED,
    "mff_challenge": MFF_CHALLENGE,
    "tft_challenge": TFT_CHALLENGE,
}


def get_rules(name: str) -> PropFirmRuleSet:
    """Get a rule set by name. Raises KeyError if not found."""
    if name not in RULE_SETS:
        available = ", ".join(RULE_SETS.keys())
        raise KeyError(f"Unknown rule set '{name}'. Available: {available}")
    return RULE_SETS[name]
