"""
All tunable parameters for the news-filtered intraday reversal backtester.

Every other module imports from here; nothing else should hard-code a
parameter. Times are America/New_York wall-clock times.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BARS_DIR = DATA_DIR / "bars"          # data/bars/<SYMBOL>/<YYYY-MM>.parquet
NEWS_DIR = DATA_DIR / "news"          # data/news/<YYYY-MM>.parquet
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Alpaca API
# ---------------------------------------------------------------------------
# Keys come from the environment, or from a gitignored .env file in the repo root
# (lines like ALPACA_API_KEY=...). Never put them in this file.
API_KEY_ENV = "ALPACA_API_KEY"
API_SECRET_ENV = "ALPACA_SECRET_KEY"
DOTENV_PATH = ROOT / ".env"
DATA_BASE_URL = "https://data.alpaca.markets"
BARS_ENDPOINT = "/v2/stocks/bars"
NEWS_ENDPOINT = "/v1beta1/news"
BAR_TIMEFRAME = "5Min"
BAR_ADJUSTMENT = "all"
PREFERRED_FEED = "sip"                # falls back to FALLBACK_FEED on HTTP 403
FALLBACK_FEED = "iex"
BARS_PAGE_LIMIT = 10_000
NEWS_PAGE_LIMIT = 50
BAR_SYMBOLS_PER_REQUEST = 5           # ~2k 5-min bars per symbol-month (SIP, incl. extended hours)
FETCH_THREADS = 3                     # concurrent requests; the shared throttle still caps the total rate
MAX_REQUESTS_PER_MINUTE = 180         # free plan allows ~200; keep headroom
MAX_RETRIES = 6
HTTP_TIMEOUT_S = 60

# Default history: 2016-01-01 (start of Alpaca's SIP history) to yesterday.
HISTORY_START = dt.date(2016, 1, 1)


def default_history_end() -> dt.date:
    return dt.date.today() - dt.timedelta(days=1)


# Ticker changes: news before the rename is tagged with the old ticker.
# Bars are fine because Alpaca's `asof` symbol mapping handles renames.
NEWS_SYMBOL_ALIASES = {"META": ["FB"]}

# ---------------------------------------------------------------------------
# Universe: liquid US large caps -> SPDR sector ETF
# (today's large caps => survivorship bias; see README)
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "ORCL": "XLK", "CRM": "XLK",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF", "C": "XLF",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "EOG": "XLE", "SLB": "XLE",
    # Health care
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV",
    # Consumer discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY", "LOW": "XLY",
    # Consumer staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    # Industrials
    "CAT": "XLI", "HON": "XLI", "UNP": "XLI", "BA": "XLI", "GE": "XLI", "DE": "XLI",
    # Communication services
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC", "DIS": "XLC", "VZ": "XLC",
    # Utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
    # Materials
    "LIN": "XLB", "SHW": "XLB", "APD": "XLB", "FCX": "XLB", "NEM": "XLB",
    # Real estate
    "PLD": "XLRE", "AMT": "XLRE", "EQIX": "XLRE", "CCI": "XLRE", "SPG": "XLRE",
}
STOCKS = list(SECTOR_MAP)
ETFS = sorted(set(SECTOR_MAP.values()))

# Second universe for the out-of-universe robustness test (never used for tuning).
# Current GICS sectors are applied to the whole history (e.g. TGT moved XLY -> XLP in 2023).
OUT_OF_UNIVERSE_MAP = {
    "AMD": "XLK", "QCOM": "XLK", "TXN": "XLK", "INTC": "XLK", "IBM": "XLK", "AMAT": "XLK",
    "SCHW": "XLF", "AXP": "XLF", "USB": "XLF", "PNC": "XLF", "BK": "XLF", "MET": "XLF",
    "PSX": "XLE", "VLO": "XLE", "MPC": "XLE", "OXY": "XLE", "HAL": "XLE", "KMI": "XLE",
    "TMO": "XLV", "ABT": "XLV", "BMY": "XLV", "AMGN": "XLV", "GILD": "XLV", "CVS": "XLV",
    "SBUX": "XLY", "ROST": "XLY", "BKNG": "XLY", "TJX": "XLY", "F": "XLY", "GM": "XLY",
    "MDLZ": "XLP", "CL": "XLP", "MO": "XLP", "KMB": "XLP", "GIS": "XLP", "TGT": "XLP",
    "RTX": "XLI", "LMT": "XLI", "MMM": "XLI", "UPS": "XLI", "FDX": "XLI", "EMR": "XLI",
    "CMCSA": "XLC", "T": "XLC", "CHTR": "XLC", "EA": "XLC", "TMUS": "XLC",
    "EXC": "XLU", "XEL": "XLU", "SRE": "XLU", "PEG": "XLU", "ED": "XLU",
    "NUE": "XLB", "ECL": "XLB", "DD": "XLB", "PPG": "XLB", "VMC": "XLB",
    "PSA": "XLRE", "O": "XLRE", "WELL": "XLRE", "AVB": "XLRE", "EQR": "XLRE",
}
# Volatility-regime proxy (VIX itself is not on Alpaca): above/below its rolling median.
VIX_PROXY = "VIXY"
VIX_REGIME_LOOKBACK_DAYS = 252

# ---------------------------------------------------------------------------
# Session / bar grid
# ---------------------------------------------------------------------------
TIMEZONE = "America/New_York"
SESSION_OPEN = dt.time(9, 30)
BAR_MINUTES = 5
BARS_PER_DAY = 78                     # 09:30 ... 15:55 bar starts
# Signals are allowed when the bar's CLOSE time (start + 5 min) is in this window.
ENTRY_WINDOW_START = dt.time(10, 0)
ENTRY_WINDOW_END = dt.time(15, 0)
SKIP_EARLY_CLOSE_DAYS = True          # no trading on half days (stats still use them)
MIN_BARS_PER_DAY = 39                 # stock-days with fewer bars are ignored

# ---------------------------------------------------------------------------
# Daily universe filter (computed from data strictly before the day)
# ---------------------------------------------------------------------------
MIN_PRICE = 10.0
MIN_AVG_DOLLAR_VOLUME = 50e6
DOLLAR_VOLUME_LOOKBACK_DAYS = 20

# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
BETA_LOOKBACK_DAYS = 20               # β and σ use the previous N full trading days
MIN_LOOKBACK_DAYS = 15                # require at least this many valid days
LOOKBACK_BARS = 6                     # L: 6 x 5 min = 30 min
Z_ENTRY = 2.5
Z_EXIT = 0.5
HOLD_BARS = 12                        # H: 12 x 5 min = 60 min
STOP_Z_OFFSET = 2.0
# "signal": stop at |z at signal| + offset (default, keeps |z|>=4.5 entries alive)
# "threshold": stop at Z_ENTRY + offset (literal reading of the spec)
STOP_Z_REFERENCE = "signal"
USE_SECTOR_FILTER = True
SECTOR_Z_MAX = 2.0

# News: an article is company-specific when tagged with <= this many symbols.
MAX_NEWS_SYMBOLS = 3

# ---------------------------------------------------------------------------
# Portfolio / sizing / costs
# ---------------------------------------------------------------------------
CAPITAL = 100_000.0
MAX_POSITIONS = 10
USE_HEDGE = True
SIZING = "fixed"                      # "fixed" or "vol"
FIXED_NOTIONAL_FRAC = 0.10            # 10% of capital per stock leg
VOL_TARGET_RISK_USD = 15.0            # "vol": notional = this / σ(5-min residual)
VOL_MAX_NOTIONAL_FRAC = 0.20          # cap for vol-scaled sizing
MAX_FRAC_OF_BAR_DOLLAR_VOLUME = 0.01  # cap: 1% of avg 5-min bar dollar volume
COST_BPS_PER_SIDE = 2.0               # per leg per side
COMMISSION_BPS = 0.0
COST_MULTIPLIERS = (1.0, 2.0)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
IN_SAMPLE_FRAC = 0.60
Z_BUCKETS = [(2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, float("inf"))]
SWEEP_GRID = {
    "lookback_bars": [3, 6, 12],
    "z_entry": [2.0, 2.5, 3.0, 3.5],
    "z_exit": [0.0, 0.5, 1.0],
    "hold_bars": [6, 12, 24],
}

# ---------------------------------------------------------------------------
# Synthetic mode
# ---------------------------------------------------------------------------
SYNTH_SEED = 7
SYNTH_START = dt.date(2023, 1, 2)
SYNTH_DAYS = 250
SYNTH_SHOCK_PROB = 0.25               # per stock-day
SYNTH_NEWS_SHOCK_FRAC = 0.5           # share of shocks that come with news
SYNTH_RANDOM_NEWS_PROB = 0.15         # unrelated company news per stock-day
SYNTH_NEWS_AFTER_MOVE_FRAC = 0.25     # news shocks whose story lands just after the move
SYNTH_HALF_SPREAD_BPS = 1.5           # bid-ask bounce on observed prices

# ---------------------------------------------------------------------------
# Validation suite (validate.py)
# ---------------------------------------------------------------------------
VALIDATION_SEED = 42
N_WORKERS = 3                         # max parallel workers for light loops
N_WORKERS_HEAVY = 2                   # max workers for the permutation test (each rebuilds a full panel)
# Memory guard: worker counts are capped so this much RAM always stays free for the OS / editor,
# using the measured memory of the first task x the safety factor per worker.
MEM_RESERVE_MB = 1500
MEM_SAFETY_FACTOR = 1.3
VALIDATION_DIR = RESULTS_DIR / "validation"
VALIDATION_SYNTH_DIR = RESULTS_DIR / "validation_synthetic"
TRIAL_LOG_PATH = RESULTS_DIR / "trial_log.csv"
HOLDOUT_LOG_PATH = RESULTS_DIR / "holdout_unlocks.log"
CACHE_DIR = DATA_DIR / "cache"

# Splits (real data). The holdout start is frozen so the boundary does not drift
# day by day; it was "today - 12 months" when the suite was written (2026-10-21).
DEV_START, DEV_END = dt.date(2016, 1, 1), dt.date(2021, 12, 31)
WF_OOS_START = dt.date(2022, 1, 1)
HOLDOUT_START = dt.date(2025, 10, 21)
# Synthetic validation base data: same structure, splits by share of trading days.
VALIDATION_SYNTH_DAYS = 1000
VALIDATION_SYNTH_START = dt.date(2019, 1, 2)
VALIDATION_SYNTH_SPLITS = (0.50, 0.35, 0.15)   # dev, wf_oos, holdout

# stats
BOOTSTRAP_N = 10_000
BLOCK_LEN_DAYS = 5
RANDOM_ENTRY_N = 1_000
PLACEBO_N = 1_000
T_STAT_PASS = 2.0

# permutation
PERMUTATION_N = 500
PERM_MIN_TRADES = 30                  # sweep points with fewer trades can't be "best"

# synthetic controlled-truth scenarios
SCENARIO_RUNS = 200
SCENARIO_DAYS = 500
SCENARIO_FIT_DAYS = 250               # most recent dev days used to fit the generator
SCENARIO_SHOCK_PROB = 0.10            # planted edge: no-news shocks per stock-day
SCENARIO_SHOCK_SIGMA = (8.0, 12.0)    # shock size in residual sigmas
SCENARIO_REVERT_FRAC = (0.2, 0.6)     # share of the shock that reverts

# robustness
WF_TRAIN_DAYS = 504                   # ~2 years
WF_TEST_DAYS = 126                    # ~6 months (also the step)
WF_MIN_TRADES = 50
HEATMAP_GRID_A = {"lookback_bars": [2, 3, 4, 6, 8, 12], "z_entry": [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]}
HEATMAP_GRID_B = {"z_exit": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5], "hold_bars": [3, 6, 9, 12, 18, 24, 36]}
JITTER_N = 1_000
JITTER_FRAC = 0.20
TOP_TRADE_FRAC = 0.01

# execution
EXEC_REPS = 200                       # repetitions of each random stress
SLIPPAGE_DIST = "halfnormal"          # per fill, per leg: "halfnormal" | "exponential" | "none"
SLIPPAGE_SCALE_BPS = 1.0
MISSED_FILL_RATES = (0.05, 0.10, 0.20)
MISSED_FILL_BIAS = 2.0                # skip weight ~ (rank of trade net bps) ** bias -> best trades missed most
SHORT_BLOCK_RATE = 0.10
EXEC_COST_MULTIPLIERS = (1.0, 2.0, 3.0)

# risk
RISK_PATHS = 10_000
RISK_PATH_DAYS = 252
MAX_DRAWDOWN_LIMIT = 0.10             # of capital, at the 95th percentile
MAX_GROSS_LEVERAGE = 4.0              # cap on recommended size: all slots full, both legs
KILL_FRACTION = 0.5                   # kill if live mean net bps < this x backtest
KILL_POWER = 0.80

# coverage
COVERAGE_DEVIATION = 0.50             # flag years whose news coverage differs from the median by > 50%
