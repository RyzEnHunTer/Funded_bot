"""
Global settings for the Prop Firm ML Forex Trading System.

All paths, pairs, timeframes, and tuning constants are centralised here
so every other module imports from a single source of truth.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR   = PROJECT_ROOT / "ml" / "models"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Trading Pairs ────────────────────────────────────────────────────────────

# Major pairs — tightest spreads, highest liquidity
PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
]

# Primary pair for initial development and testing
PRIMARY_PAIR = "EURUSD"

# yfinance ticker mapping (yfinance uses "EURUSD=X" format)
YFINANCE_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "NQ=F": "NQ=F",
}

# ─── Timeframes ───────────────────────────────────────────────────────────────

PRIMARY_TIMEFRAME = "15m"  # Shifted to M15 for high-frequency trading
# Note: yfinance provides up to ~730 days of 1h data for Forex.
# For 15m data, only ~60 days is available. We use 1h for the initial build
# and can switch to 15m when connecting to MT5 for higher-resolution data.

# ─── Data Settings ────────────────────────────────────────────────────────────

DATA_LOOKBACK_DAYS = 59     # yfinance limit for M15 is 60 days
WARMUP_BARS        = 60     # Reduced from 210 since EMA200 is removed

# ─── Feature Engineering ──────────────────────────────────────────────────────

# EMA periods
EMA_FAST   = 9
EMA_MED    = 21
EMA_SLOW   = 50
EMA_TREND  = 200

# Indicator periods
RSI_PERIOD     = 5
ADX_PERIOD     = 14
ATR_PERIOD     = 14
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
BB_PERIOD      = 20
BB_STD         = 2.0
KELTNER_PERIOD = 20
KELTNER_ATR_MULT = 1.5
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9

# ─── Triple-Barrier Labeling ─────────────────────────────────────────────────

ATR_MULTIPLIER     = 1.2     # Tighter stops for HFT
MIN_DISTANCE_PCT   = 0.0005  # Floor: 0.05% of price (~5 pips)
VERTICAL_BARRIER   = 24      # 24 bars = 6 hours on M15

# ─── ML Model ────────────────────────────────────────────────────────────────

TRAIN_TEST_SPLIT   = 0.80    # 80% train, 20% test (chronological)
RANDOM_STATE       = 42      # Reproducibility

# RandomForest defaults
RF_N_ESTIMATORS    = 200
RF_MAX_DEPTH       = 10
RF_MIN_SAMPLES_LEAF = 20     # Prevent overfitting on small leaf nodes

# XGBoost defaults
XGB_N_ESTIMATORS   = 200
XGB_MAX_DEPTH      = 6
XGB_LEARNING_RATE  = 0.1

# ─── Strategy ─────────────────────────────────────────────────────────────────

ML_LONG_THRESHOLD  = 0.45    # Minimum prob(+1) to enter long (Sniper filter)
ML_SHORT_THRESHOLD = 0.45    # Minimum prob(-1) to enter short (Sniper filter)
CONFIDENCE_EDGE    = 1.1     # prob(signal) must be > prob(opposite) * this

REWARD_RISK_RATIO  = 5.0     # Take-profit = SL distance * this (Massive safety net for Runner)
ATR_MULTIPLIER     = 1.5     # Stop-loss = ATR * this
SIGNAL_COOLDOWN    = 1       # Minimum bars between new trades (reduced)

# ─── Risk Management ─────────────────────────────────────────────────────────

RISK_PER_TRADE_PCT = 0.012   # 1.2% of account per trade (Safe for 4% daily loss)
MAX_CONCURRENT_POSITIONS = 2  # Max 2 trades open across the 3 pairs at once
MAX_TRADES_PER_DAY = 3       # Daily Governor: Shut down after 3 trades in a day

# ─── Pip Values ───────────────────────────────────────────────────────────────
# Pip value per standard lot (100,000 units) for USD-quoted accounts

PIP_VALUES = {
    "EURUSD": 10.0,     # 1 pip = $10 per lot
    "GBPUSD": 10.0,
    "USDJPY": 6.50,     # Approximate — depends on USDJPY rate
    "AUDUSD": 10.0,
    "USDCAD": 7.30,     # Approximate
    "USDCHF": 11.20,    # Approximate
    "NQ=F": 20.0,
}

# Pip size (minimum price increment that equals 1 pip)
PIP_SIZES = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDCHF": 0.0001,
    "NQ=F": 1.0,        # 1 whole point
}

# ─── Spread (in pips) — typical for prop firm accounts ────────────────────────

SPREADS = {
    "EURUSD": 1.5,
    "GBPUSD": 2.0,
    "USDJPY": 1.5,
    "AUDUSD": 2.0,
    "USDCAD": 2.0,
    "USDCHF": 2.0,
    "NQ=F": 1.5,        # 1.5 points spread
}

# ─── Backtesting ──────────────────────────────────────────────────────────────

SLIPPAGE_PIPS      = 0.5     # Simulated slippage per trade (in pips)
INITIAL_BALANCE    = 5000.0  # Prop Firm Challenge Balance
COMMISSION_PER_LOT = 7.0     # Round-trip commission per lot ($7 typical)
