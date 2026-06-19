"""
Position Sizer — Risk-based position sizing for prop firm accounts.

Calculates lot size based on:
  - Account balance
  - Risk per trade (% of balance)
  - Stop-loss distance (in pips)
  - Pip value (per lot)

Designed to keep per-trade risk low enough that a string of losers
won't breach the prop firm's daily or max drawdown limits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    RISK_PER_TRADE_PCT, PIP_VALUES, PIP_SIZES, SPREADS, SLIPPAGE_PIPS,
)


def calculate_position_size(
    account_balance: float,
    stop_distance_price: float,
    pair: str,
    risk_pct: float = RISK_PER_TRADE_PCT,
    max_lots: float = 20.0,
) -> float:
    """
    Calculate position size in lots based on risk parameters.

    Formula:
        risk_amount = account_balance * risk_pct
        stop_pips = stop_distance_price / pip_size
        total_stop_pips = stop_pips + spread + slippage
        lots = risk_amount / (total_stop_pips * pip_value)

    Example (EURUSD, $100k account, 0.5% risk, 25 pip stop):
        risk_amount = $100,000 * 0.005 = $500
        total_stop = 25 + 1.5 + 0.5 = 27 pips
        lots = $500 / (27 * $10) = 1.85 lots

    Parameters
    ----------
    account_balance : float
        Current account balance in USD.
    stop_distance_price : float
        Stop-loss distance in price units (e.g., 0.0025 for 25 pips EURUSD).
    pair : str
        Currency pair (e.g., "EURUSD").
    risk_pct : float
        Fraction of account to risk per trade (e.g., 0.005 = 0.5%).
    max_lots : float
        Maximum position size cap.

    Returns
    -------
    float
        Position size in standard lots (1 lot = 100,000 units).
    """
    pip_size = PIP_SIZES.get(pair, 0.0001)
    pip_value = PIP_VALUES.get(pair, 10.0)
    spread = SPREADS.get(pair, 2.0)

    # Convert stop distance from price to pips
    stop_pips = stop_distance_price / pip_size

    # Add spread and slippage to the stop
    total_stop_pips = stop_pips + spread + SLIPPAGE_PIPS

    if total_stop_pips <= 0:
        return 0.0

    # Calculate risk amount
    risk_amount = account_balance * risk_pct

    # Calculate lots
    lots = risk_amount / (total_stop_pips * pip_value)

    # Cap at maximum
    lots = min(lots, max_lots)

    # Round to 2 decimal places (0.01 lot = micro lot)
    lots = round(lots, 2)

    # Minimum lot size
    if lots < 0.01:
        lots = 0.0  # Too small to trade

    return lots


def scale_for_drawdown(
    base_lots: float,
    current_drawdown_pct: float,
    max_drawdown_pct: float,
) -> float:
    """
    Scale position size down when approaching the drawdown limit (Dynamic Drawdown Governor).

    - At < 4% DD: full size (1.5% risk)
    - At >= 4% DD: 50% size (0.75% risk)
    - At >= 7% DD: 16.6% size (0.25% risk)

    This implements the aggressive HFT scaling rule while preserving capital.
    """
    if max_drawdown_pct <= 0:
        return base_lots

    if current_drawdown_pct >= 0.07:
        scale = 0.166  # Drops to 0.25% risk
    elif current_drawdown_pct >= 0.04:
        scale = 0.50   # Drops to 0.75% risk
    else:
        scale = 1.0    # Full 1.5% risk

    return round(base_lots * scale, 2)
