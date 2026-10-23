# News-filtered intraday reversal backtester

This project backtests one hypothesis: a large 30-minute stock move relative to its sector ETF
that comes **without company news** tends to revert, while a move that comes **with news** tends to persist.
It downloads 5-minute bars and news from Alpaca, builds the signals, and runs a signal study,
a portfolio backtest with position limits, an in-sample/out-of-sample split and a parameter sweep.
It does not trade, live or paper.

```
config.py        every tunable parameter + the universe (stock -> sector ETF)
fetch_data.py    Alpaca download, throttling/retry, parquet cache in data/
signals.py       (symbol, day, 5-min slot) panel, returns, β/σ, z, news flags, universe filter
backtest.py      trade simulation, portfolio, statistics, IS/OOS, sweep  (entry point)
report.py        results/ files and charts
synthetic.py     fake data with a planted effect (--synthetic)
validate.py      validation suite CLI (see "Validation" below)
validation/      one module per validation subcommand + shared plumbing
tests/           lookahead-bias tests, offline download/cache tests, validation-machinery tests
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env             # then fill in both keys; .env is gitignored
```

Keys are read from the environment variables `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`, or from `.env`
in the repo root. Never put them in `config.py`, which is committed.

## Running

```bash
# 1. Download (2016-01-01 .. yesterday by default). Re-runs only fetch missing months.
#    Default symbols: main universe + out-of-universe list + sector ETFs + VIXY.
python fetch_data.py
python fetch_data.py --start 2024-01-01 --symbols AAPL,MSFT    # subset (ETFs are added automatically)

# 2. Backtest on the cached data
python backtest.py              # add --no-sweep to skip the 108-point parameter grid

# Synthetic mode: no API keys needed; checks that the pipeline recovers a planted effect
python backtest.py --synthetic [--synth-days 250]

# Tests (lookahead checklist + offline fetch tests)
python -m pytest -q
```

A full download (2016 onward, about 135 symbols) takes roughly 2–3 hours on the free plan. Bars cost
about 8,700 requests, and news costs another 10–25k pages of 50 articles, all throttled to 180 requests per minute.
If the download is interrupted, re-running it resumes where it stopped. The backtest's 60/40 in-sample/out-of-sample split
now spans the full history. Use `validate.py` for the fixed development / walk-forward / holdout splits.

**Feed:** the code requests `feed=sip` and switches to `iex` for the rest of the run if the API returns 403.
IEX carries only about 2–3% of US volume, so IEX bars are sparser and noisier. Check
`data/manifest.json` to see which feed each chunk came from.

## Remote control (`ctl.py`)

`ctl.py` runs everything from an SSH session. It needs only the standard library, so it works before the virtualenv exists.
Jobs run detached: they keep running after you log out. They run at low CPU priority (`nice 10`),
log to `results/logs/<job>.log`, and keep their state in `results/run/`.

```bash
python3 ctl.py setup                         # create .venv, install requirements, run the tests
python3 ctl.py status                        # jobs, free RAM, downloaded months, report pass/fail counts
python3 ctl.py start fetch                   # download / resume Alpaca data
python3 ctl.py start validate-synthetic      # validate all --synthetic
python3 ctl.py start validate                # validate all on real data
python3 ctl.py start validate -- permutation --heavy-workers 3   # any validate.py arguments after --
python3 ctl.py start --nice 0 backtest       # ctl options go BEFORE the job name
python3 ctl.py logs validate -f              # follow a log (Ctrl-C stops following, not the job)
python3 ctl.py stop validate                 # stops the job and all its worker processes
python3 ctl.py report                        # print results/validation/REPORT.md (--synthetic for the other)
```

Jobs: `fetch`, `validate`, `validate-synthetic`, `backtest`, `backtest-synthetic`, `test`. `ctl.py` refuses to start a
second heavy job (any validate or backtest) while one is running; add `--force` to override.
`fetch` and the permutation test resume where they stopped, so stopping either one loses nothing.
`fetch` (network-bound) can run alongside a validation job.

## Moving to another machine

```bash
./migrate.sh user@host --dry-run       # shows what would happen, changes nothing
./migrate.sh user@host                 # copy + set up on the target
./migrate.sh user@host --dest ~/proj --port 2222 --no-env
./migrate.sh user@xxxxxxxx.onion --tor # every SSH connection through Tor (torsocks); required for .onion
```

The copy is **resumable**. It sends only files that are missing on the target or differ in size or modification time,
in batches of about 25 MB (`--batch-mb`). It retries a failed batch up to 5 times. If the connection still fails,
rerunning the same command resumes from where it stopped. This matters over Tor, where circuits drop and throughput is low.
After the copy, the script checks that every file is present with matching size and timestamp.

The script:

1. refuses to run while a fetch, validate or backtest job is running here;
2. sets up key login with `ssh-copy-id` if it doesn't work yet (you enter your password once);
3. checks that the target has Python 3.10+ and `tar`, and reports its CPUs and RAM;
4. streams the project over SSH, checks that the file counts match, and sets `.env` to mode 600;
5. runs `python3 ctl.py setup` there (virtualenv, requirements, tests).

It copies the code, `data/`, `results/` (including the trial log and saved permutations) and `.env`. The virtualenv,
caches and job state are rebuilt on the target. `--no-env` leaves the API keys behind, `--no-data` skips the download cache,
`--no-setup` only copies, and `--force` allows copying over an existing copy.

## Reading the output (`results/`)

| file | content |
|---|---|
| `summary.txt` | all tables (see below) |
| `trades.csv` | portfolio trades: signal/entry/exit times (New York), prices for both legs, z at signal and at exit, sector z, β, σ, news flag, exit reason, bars held, gross/cost/net bps, notional, $ P&L at 1× and 2× cost, IS/OOS label |
| `signal_study_trades.csv` | the simulated fade for **every** qualifying bar, news and no-news (`group` column) |
| `equity.png` | equity curves at 1× and 2× cost (out-of-sample shaded) and drawdown |
| `sweep.csv`, `sweep_sharpe.png`, `sweep_net_bps.png` | in-sample parameter grid |

`summary.txt` sections:

1. **Signal study.** Two views, for `no_news` vs `news`, at 1× and 2× cost:
   n, win rate, mean gross, mean net, median net bps, a plain t-stat, and a t-stat clustered by day.
   * `every_bar` is every bar with |z| ≥ z_entry, as specified. One shock usually keeps |z| high for
     several consecutive bars, so these trades overlap, and the plain t-stat overstates significance.
   * `first_per_stock_day` keeps only the first trigger per stock per day. It is closer to what
     the strategy can actually trade.
   * The hypothesis predicts `no_news` net > 0 and `news` ≤ 0.
2. **Portfolio** (no-news trades only; at most 10 positions; each stock at most once per day):
   annualized return on capital, Sharpe (daily × √252), max drawdown, average holding time,
   exit-reason shares. Reported for all dates, in-sample and out-of-sample.
   **Judge the strategy on the out-of-sample column.**
3. **Breakdowns** by year, sector, entry hour and |z| bucket, for each period.
4. **Sweep** (in-sample only): medians by parameter value and the top 10 settings.
   Look for a broad region of positive values in the heatmaps, not the single best cell.

## Validation

`validate.py` tries to break the strategy. Every check calls the same strategy code
(`backtest.run_strategy` / `simulate_trades` / `run_sweep`, `signals.compute_features`).

```bash
python validate.py <subcommand> [--synthetic] [--workers 3] [--heavy-workers 2] [--unlock-holdout]
python validate.py all            # everything + results/validation/REPORT.md
python validate.py report         # rebuild REPORT.md from saved results
```

| subcommand | what it does | real-data runtime* |
|---|---|---|
| `coverage` | news per stock per year, share of signals flagged as news, flagged years, missing bars | 1 min |
| `correctness` | planted-effect and null synthetic markets, lookahead traps (fill at signal close, news 5 min early), entry delay 0–3 bars, 20-trade audit, pytest | 3–5 min |
| `stats` | group stats, 10k day-block bootstrap, 1,000 random-entry runs, 1,000 placebo news filters, Deflated Sharpe | 10–20 min |
| `permutation` | 500 × (shuffle residuals within stock-days, move news to random days, rerun features + 108-point sweep); p-value of the best in-sample result | 2–4 h |
| `synthetic` | GARCH(1,1)-t generator fitted to the data; 200 runs × (no edge / planted edge / edge that vanishes halfway) | 10–15 min |
| `robustness` | walk-forward (2y train / 6m test), heatmaps, 1,000 jittered parameter sets, breakdowns incl. VIXY regime, leave-one-out, out-of-universe | 20–40 min |
| `execution` | 1/2/3× costs, break-even cost, pessimistic fills + slippage, missed fills biased to the best trades, random delay, blocked shorts | 5–10 min |
| `risk` | 10k one-year block-bootstrap paths: drawdown/streak/return percentiles, position size for a 10% 95th-pct drawdown, kill criteria | 1 min |

\*Estimates for 2016 → 2025-10 on a 4-core / 8 GB machine. The first real-data run also builds the feature cache
(`data/cache/`, a few minutes); later runs load it in seconds. `--synthetic` runs everything in about 35 minutes.

**Splits** (`config.py`): development 2016-01-01 … 2021-12-31; walk-forward out-of-sample 2022-01-01 … 2025-10-20;
holdout from `HOLDOUT_START` = 2025-10-21. The holdout start is frozen rather than computed from today, so the boundary does not drift.
The holdout is **never loaded** unless `--unlock-holdout` is passed. Each unlock prints a warning
and is appended to `results/holdout_unlocks.log`, and REPORT.md reports the holdout as spent. Headline numbers use the
walk-forward OOS period, and development-period numbers are reported next to them.

**Outputs:** `results/validation/<subcommand>/` (`summary.txt`, `results.json`, CSVs, PNGs), or
`results/validation_synthetic/` with `--synthetic`. `results/trial_log.csv` records every configuration ever
backtested (dataset, source, period, params, trades, net bps, Sharpe). The Deflated Sharpe uses the number of distinct
parameter sets on the same dataset. Permuted and generated null markets are not logged as trials.

**Reproducibility:** every random procedure draws from `VALIDATION_SEED` (42) plus a task key, so results do not depend
on the number of workers. The permutation test appends each finished permutation to `permutations.csv` and resumes after an interruption.

**Memory:** each permutation worker rebuilds a full development-period panel (~0.7 GB on real data). Keep
`--heavy-workers 2` on an 8 GB machine.

### Validation assumptions

* **Permutation test.** Residuals are `r_s − β·r_e`, with β estimated from the previous 20 days. They are shuffled across all 78
  slots of each stock-day, which also moves the high-volatility opening bars to random times.
  Each bar's open-vs-previous-close gap stays in place, and high/low keep their distance from max/min(open, close).
  Company-specific news items keep their time of day and move to a random trading day of the same stock;
  multi-ticker items are split per stock. "Best" means the best sweep point with at least 30 trades,
  and p-values are computed separately for best Sharpe and best mean net bps.
* **Walk-forward** picks the sweep point with the highest in-sample Sharpe among points with at least 50 trades.
* **Random-entry baseline** keeps each real trade's stock and direction. It picks a random tradable day in the period and a random
  signal bar in the window, and draws holding times by permuting the real trades' holding times. Only time and end-of-day exits apply.
* **Placebo news filter** shuffles the news flags across every qualifying signal bar, keeping the same news share.
* **Missed fills** are removed after position selection, and the freed slot is not reused. **Blocked shorts** block a
  stock-day's short entries. **Pessimistic fills** apply to both legs, on entry and exit.
* **Kill rule (a):** N is set so that if the edge has disappeared (true mean net = 0) the live mean falls below 50% of the
  backtest mean with 80% probability. The per-trade sd is inflated for within-day clustering.
  With a threshold halfway between the two cases, the false-kill probability when the edge is intact is also about 20%.
* **Position sizing** scales drawdowns linearly with size and is capped at 4× gross leverage (all 10 slots full, both legs).
* **VIX proxy:** VIXY's previous close against its rolling 252-day median. VIXY decays steadily (contango), so the regime is relative
  to recent history. A realized-volatility regime (from the sector ETFs' intraday volatility) is reported alongside it.
* **Coverage:** news counts are scaled to 252 trading days so partial years compare. The suite recommends excluding a year whose
  coverage is more than 50% *below* the median year, because there the news filter cannot work.
* **Controlled-truth generator:** fitted on the last 250 development days. It uses a market factor, sector ETFs and stock residuals,
  each GARCH(1,1) with Student-t shocks on seasonality-adjusted 5-min returns. News arrives at the fitted rate and
  time of day, and in-session news adds a persistent jump. Each scenario run is 500 days.
* **Synthetic validation base data:** 1,000 business days from 2019, split 50 / 35 / 15% into development / WF-OOS / holdout.
  Its microstructure is only a 1.5 bp bid-ask bounce, so the "fill at signal close" trap barely moves results there,
  and the suite flags that as a warning. Expect a large effect on real data.

## Timing conventions (how lookahead is avoided)

* Alpaca bar timestamps are bar **start** times. Slot *t* covers `[09:30+5t, 09:35+5t)`, and its
  close is only known at `09:35+5t`. Every feature indexed at *t* is known at that close time.
* Returns never span the overnight gap. The first bar of a day uses its own close/open, and later bars
  use close/close within the same day. A bar with no trades adds 0 (the close is forward-filled within the day only).
* β and σ for day *d* come from days *d−20 … d−1* (at least 15 valid days) and stay fixed during day *d*.
  β is the OLS slope with an intercept, and σ is the std of the residual `r_s − β r_e` in that window.
* The lookback sum `R_t` uses the last L bars of the **same day**. It is NaN until L bars exist,
  so L=12 cannot signal before 10:30.
* `HasNews_t` counts company-specific articles (≤ 3 tagged symbols) whose `created_at` falls in
  `(previous session close, close time of bar t]`. `updated_at` is never used.
* Universe filters for day *d* use the previous close and the 20-day average dollar volume of days before *d*.
* Entries fill at the open of the first bar **after** the signal bar (normally *t+1*), and the hedge
  fills at the ETF's open on that same bar. Exits are checked at each bar close starting with the entry
  bar. Exits 1–4 fill at the next bar's open, and the end-of-day exit fills at the 15:55 bar's close.

`tests/test_lookahead.py` has a test for each checklist item. Most are perturbation tests:
they scramble data that must be invisible at time *t* and check that nothing computed for *t* changes.

## Assumptions and choices

* **Stop-loss reference:** the stop is `|z| ≥ |z at signal| + 2` in the original direction.
  The literal reading, `z_entry + 2` (= 4.5), would stop every |z| ≥ 4.5 entry at the first bar.
  Set `STOP_Z_REFERENCE = "threshold"` in `config.py` for the literal rule.
* **Early-close days** (13:00 sessions) are not traded. Their bars still feed β/σ, and the news window
  for the next day starts at their 13:00 close.
* **Signal study** applies entry rules 1, 3 and 5 (|z|, time window, sector filter) and ignores news.
  Rule 4 (position limits / once per day) is approximated by the `first_per_stock_day` view.
* **Exit priority on the same bar:** reversion > time > stop > news > end of day.
  The time stop fires at the close of the H-th held bar, so H=12 means exactly 60 minutes.
  If any exit fires at the 15:55 bar, it fills at that bar's close.
* **Missing bars** (no trades in a 5-minute slot): an order fills at the next bar that does trade.
  The position's clock starts from that actual entry bar.
* **Hedge:** opposite position in the sector ETF with notional β × stock notional.
  Cost = `cost_bps × 2 sides × (1 + |β|)`.
* **Sizing:** $10k per stock leg (10% of $100k), capped at 1% of the stock's average 5-minute dollar volume
  over the previous 20 days. The hedge leg comes on top of that, so worst-case gross exposure is
  about 10 × $10k × (1 + β) ≈ 200% of capital. `SIZING = "vol"` sets notional = `VOL_TARGET_RISK_USD / σ`.
  P&L does not compound: return on capital = daily $ P&L / $100k.
* **Sharpe** uses every tradable day in the period, including days with no trades.
* **News:** Alpaca's news (mostly Benzinga) is not a complete record of company news, so some "no-news" moves
  will have had news elsewhere. The API sorts by `updated_at`, so monthly chunks may overlap;
  articles are de-duplicated by `id`. Facebook-era articles are tagged `FB`; they are requested and
  mapped to `META` (`NEWS_SYMBOL_ALIASES`).
* **Prices** are `adjustment=all` (split, dividend and spin-off adjusted as of today). That is fine for
  intraday returns but means historical price levels, and so the $10 price filter, are not exactly
  what was seen at the time.

## Known limitations

* **Survivorship bias:** the universe is a fixed list of today's large caps. Stocks that fell out of the
  large-cap set, or were delisted since 2021, are missing, so results will look better than a
  point-in-time universe would.
* **σ is one number per day, but intraday volatility is U-shaped.** As specified, z divides by a
  whole-day residual σ, so the first hour (the 10:00–11:00 entries) produces more |z| ≥ z_entry signals
  simply because volatility is higher then. Check the "by entry hour" breakdown before trusting
  early-morning trades. A time-of-day σ would be the natural next step.
* Costs are a flat bps charge per side. There is no spread model, queue position, borrow cost for shorts, or market impact beyond the 1% volume cap.
* Fills at the next bar's open assume you trade at the first print of that bar.
