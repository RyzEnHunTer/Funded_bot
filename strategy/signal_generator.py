"""
Signal Generator — Convert ML probabilities into trade signals.

Combines ML confidence thresholds with Supertrend directional filter
to produce discrete LONG / SHORT / FLAT signals.
"""

from enum import IntEnum
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    ML_LONG_THRESHOLD, ML_SHORT_THRESHOLD, CONFIDENCE_EDGE,
)


class Signal(IntEnum):
    """Trade signal types."""
    FLAT  = 0
    LONG  = 1
    SHORT = -1


def generate_signal(
    ml_probs: Dict[int, float],
    supertrend_direction: int,
    long_threshold: float = ML_LONG_THRESHOLD,
    short_threshold: float = ML_SHORT_THRESHOLD,
    confidence_edge: float = CONFIDENCE_EDGE,
) -> Signal:
    """
    Generate a trade signal from ML probabilities and Supertrend direction.

    Entry conditions:
      LONG:  supertrend bullish (+1)
             AND prob(+1) >= long_threshold
             AND prob(+1) > prob(-1) * confidence_edge
      SHORT: supertrend bearish (-1)
             AND prob(-1) >= short_threshold
             AND prob(-1) > prob(+1) * confidence_edge

    The confidence_edge (default 1.2) ensures we only trade when the model
    has a clear directional bias — not when it's ambiguous (e.g., 45% up / 42% down).

    Parameters
    ----------
    ml_probs : dict
        {1: prob_up, 0: prob_neutral, -1: prob_down}
    supertrend_direction : int
        +1 for bullish, -1 for bearish
    long_threshold : float
        Minimum prob(+1) for long entry
    short_threshold : float
        Minimum prob(-1) for short entry
    confidence_edge : float
        Required edge over opposing class (e.g., 1.2 = 20% more confident)

    Returns
    -------
    Signal
        LONG, SHORT, or FLAT
    """
    prob_up = ml_probs.get(1, 0.0)
    prob_dn = ml_probs.get(-1, 0.0)

    # Long conditions
    if (supertrend_direction > 0
            and prob_up >= long_threshold
            and prob_up > prob_dn * confidence_edge):
        return Signal.LONG

    # Short conditions
    if (supertrend_direction < 0
            and prob_dn >= short_threshold
            and prob_dn > prob_up * confidence_edge):
        return Signal.SHORT

    return Signal.FLAT
