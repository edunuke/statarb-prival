# Beta Test: S&P 500 Institutional Stat-Arb Pipeline

This directory hosts the live beta test for the **S&P 500 Institutional Statistical Arbitrage ("Stat-Arb") pipeline**. The core engine is implemented entirely in a single notebook, `live_test_1.0.ipynb`, which runs a walk-forward, out-of-sample (OOS) evaluation of a pairs-trading strategy and produces an executable next-morning order blotter.

## Project Overview

The pipeline discovers and trades **cointegrated pairs** of S&P 500 names. It couples a classic **statistical arbitrage** approach (long-short spread positions on mean-reverting, cointegrated pairs) with an **institutional-grade risk and execution layer** (net-beta neutrality, style-factor caps, NAV caps, and ADV liquidity participation limits).

The system is organized around a **Ghost → Eligible → Live** funnel:

1. **Ghost Universe**: Candidate pairs that pass the pairwise cointegration / mean-reversion pre-screen but are not yet entered.
2. **Eligible Universe**: Ghost pairs promoted past the adaptive entry gate, plus currently held (active) sleeves.
3. **Live Portfolio**: The final sleeve allocation chosen by the convex optimizer and carried forward.

## Key Components

| # | Component | Responsibility |
|---|-----------|----------------|
| 1 | `StatePersistenceEngine` | Pickle-based checkpoint save/load for crash recovery, persists Kalman states, LinUCB policy, and cooldown registries. |
| 2 | Mathematical utilities | Closed-form half-life, FFT GPH fractional-integration $d$, and continuous LinUCB reward calculations. |
| 3 | `FastKalmanFilter` | $O(1)$ online sticky Kalman state tracking (regression of asset A on asset B), producing innovations $e_t$ and variance $F_t$. |
| 4 | `FastBOCPDEngine` | Bayesian Online Change-Point Detection on innovation buffers to flag structural breaks. |
| 5 | `LinUCBContextEngine` | Contextual multi-armed bandit (MabWiser) that adaptively selects the entry Z-gate from arms $\{1.50,\, 1.85,\, 2.25\}\sigma$. |
| 6 | `UniverseScreener` | Maps tickers to GICS sectors (used for sector-neutrality filters). |
| 7 | `PairCointegrationEngine` | Tensor pre-screening (correlation, variance ratio) + fixed-window Engle–Granger cointegration + Hurst / fractional $d$ mean-reversion gates + LinUCB gate assignment. |
| 8 | `ConvexPortfolioOptimizer` | SLSQP optimizer with style-factor loading caps, net-beta neutrality, sleeve weights, and single-stock NAV caps. |
| 9 | `LiveMOOOrderGenerator` | Generates an actionable Market-On-Open (MOO) order blotter capped at **max 1.5% of 20-day ADV** participation. |
| 10 | Backtest engines | `CommitteeTearsheetEngine` (per-pair stats) and `VectorbtBacktestEngine` (global walk-forward net-portfolio simulation with T+1 execution lag and short borrow costs). |
| 11 | Orchestrator & plotter | `run_unified_pipeline` drives the full loop; `plot_diagnostic_tearsheet` renders the 5-panel diagnostic tearsheet. |

## Funnel & Adaptive Gating

- **Screening** (every 20 trading days) runs a Joblib-parallel cointegration sweep over the universe.
- **Rebalancing** (every 5 trading days) evaluates the current book, updates Kalman states, checks structural-break gates (CUSUM $S_t$, Ljung-Box, BOCPD $$P(r_t=0)$$), and runs the optimizer.
- **LinUCB** adapts the entry Z-threshold based on context `[vol_ratio, VIX proxy, CUSUM $S_t$, half-life]`, enabling tighter entry gates in calm markets and looser gates in volatile markets.
- **Exit rules** are triggered by target Z-spread reversion, hard stop-loss, time-stop ($t > 2.5 × half-life$), or a structural break. Exited/broken pairs are routed to a cooldown quarantine.

## Risk & Institutional Controls

- **Net-beta neutrality** $\le 0.05$ (long-short spread construction).
- **Style factor loading** $\le 0.10$ (momentum/value factor proxy).
- **Single-stock NAV cap** $\le 15\%$; **pair sleeve cap** $\le 25\%$; **gross exposure** $\le 1.8\times$.
- **Sector limits**: max 3 pairs per GICS sector.
- **ADV participation cap**: any order is limited to **1.5% of the 20-day average daily volume**.
- **Execution assumptions**: 1.5 basis-point fees, 50 bps annual borrow rate for short legs, Market-On-Open execution.

## Files in This Directory

| File | Role |
|------|------|
| `statarb_1.0.ipynb` | The complete pipeline implementation (engine + orchestrator + diagnostics). |
| `universe_daily_train.csv` | In-sample daily close-price universe for training/lookback. |
| `universe_daily_val.csv` | Out-of-sample daily close-price universe for OOS evaluation. |
| `stat_arb_state.pkl` | Live pipeline state checkpoint (active pairs, cooldown registry). |
| `TEST_CHECKLIST.md` | Live beta-testing checklist covering signal integrity, risk, microstructure, and ops. |
| `README.md` | This file. |

> Note: `run_unified_pipeline` also attempts to load `universe_daily_open.csv` (the `close` → `open` sibling) for open-price execution; if missing, it falls back to close prices.

## Changelog — Functional Evolution Across Versions

The pipeline evolved incrementally from `statarb_1.0.ipynb` through `statarb_1.4.ipynb`. Each version added concrete functional changes (cosmetic renames/comments are omitted). The artifacts for each run are written to `results/{n}/`:

### v1.0 → v1.1 — Dynamic dual-trigger engine + funnel attribution
- **Dynamic rebalancing**: replaced the fixed 5-day rebalance schedule with a **dual-trigger evaluator** — rebalances on any signal exit, new signal entry, L-infinity weight drift ≥ 5%, or the initial day.
- **Turnover penalty in optimizer**: SLSQP objective now deducts estimated round-trip turnover cost (fees + slippage) from net return; `Turnover_Cost_Drag` tracked per allocation.
- **Shadow funnel tracking**: added equal-weighted Ghost and Eligible master weight matrices alongside the live book; runs parallel shadow backtests and prints a **funnel conversion attribution report** (annualized return and Sortino per tier, plus marginal bandit and optimizer deltas).
- **Daily cooldown accounting**: cooldown timers now decrement every day rather than only on rebalance days.
- **Periodic screening counter**: universe screening cadence tracked via `days_since_screening` instead of a rebalance-step modulo.
- **Improved state persistence**: safe directory creation (`os.makedirs(..., exist_ok=True)`), checkpoint path moved under `artifacts/model_A/`, and walk-forward progress telemetry added to console output.
- **Diagnostics**: cumulative performance plot now overlays live, forecast, eligible-shadow, and ghost-shadow equity curves; added `compute_sortino_ratio` helper.

### v1.1 → v1.2 — Weight-drift evaluator extraction
- Extracted `ConvexPortfolioOptimizer.calculate_max_weight_drift()` (L-infinity norm `‖w_current − w_target‖∞`) used by the dual-trigger rebalancing gate.
- Console telemetry trimmed (cosmetic); parameters regrouped by theme.

### v1.2 → v1.3 — Min-variance optimizer + consecutive-loss quarantine
- **Removed return-forecast noise (mu) from the objective**: the allocator now optimizes **`risk + turnover` only** (pure min-variance under constraints), eliminating dependence on noisy expected-return estimates.
- **Full leverage scaling**: replaced the gross-exposure inequality with an **equality constraint** forcing deployment up to the gross cap (`effective_target_gross = min(max_gross, N × sleeve_cap)`); SLSQP failures fall back to analytic inverse-variance weights.
- **Inverse-volatility init**: optimizer warm-up uses inverse-volatility weights scaled to the target gross instead of a naive equal-weight guess.
- **Consecutive-loss quarantine gate**: persistent `loss_streak_tracker` (checkpointed) — any pair with 2 consecutive losing exits (or stop-loss/structural-break) is quarantined for **63 trading days** (up from the 15-day cooldown); half-life drift demotions now use the same quarantine window.
- `quarantine_days` config param added (default 63).

### v1.3 → v1.4 — Market-neutrality audit + screener speedup
- **`MarketNeutralityDiagnosticEngine`** (new): post-hoc governance audit of the live return series — single-factor CAPM regression (annualized alpha, β, t-stat, p-value, R²), asymmetric bull/bear down-tail beta, and optional multi-factor style regression (Fama-French/Barra); produces a PASS/FAIL audit vs bounds (|β| ≤ 0.03, R² ≤ 0.01, |factor loading| ≤ 0.05) and is invoked at the end of the pipeline.
- **Fast pair screener**: pair screening now passes lightweight NumPy arrays (not DataFrames) to the parallel worker (O(1) memory footprint), short-circuits bad pairs by computing the cheap sliding AB motif distance **before** the expensive Engle–Granger test, and replaces the O(N²) Python pair loop with **vectorized NumPy matrix masking / sector-match broadcasting**.
- Orchestrator integrates the neutrality audit report into terminal output.

## Parameters

| Parameter | Definition | Value / Range |
| :--- | :--- | :--- |
| `price_csv` | Path to historical in-sample price CSV data. | String (`"data/universe_daily_train.csv"`) |
| `oos_price_csv` | Path to out-of-sample validation price CSV data. | String (`"data/universe_daily_val.csv"`) |
| `constituents_path` | Path to universe constituent mapping CSV with sector taxonomy. | String (`"data/constituents.csv"`) |
| `state_persistence_file` | Path to serialized online model state checkpoint file (`.pkl`). | String (`"artifacts/stat_arb_state.pkl"`) |
| `lookback_days` | Total historical rolling window used for signal and parameter estimation. | Integer (`504` days) |
| `min_history_days` | Minimum required continuous price history for pair validation. | Integer (`100` days) |
| `rolling_window_days` | Standard window length for calculating rolling volatility, correlation, and beta. | Integer (`252` days) |
| `rebalance_freq_days` | Frequency (in trading days) between portfolio rebalancing cycles. | Integer (`5` days) |
| `screening_freq_days` | Frequency (in trading days) between full universe pairwise screening passes. | Integer (`20` days) |
| `cooldown_days` | Quarantine duration (in trading days) imposed on exited or demoted pairs. | Integer (`15` days) |
| `kalman_delta` | State transition covariance scalar regulating Kalman beta responsiveness ($\delta$). | Float (`1e-6` – `1e-3` \| Configured: `1e-4`) |
| `kalman_obs_noise` | Observation noise variance parameter ($V_\varepsilon$) for the Kalman Filter. | Float (`1e-4` – `1e-2` \| Configured: `1e-3`) |
| `kalman_z_window` | Rolling window length for calculating empirical innovation variance ($F_t$). | Integer (`10` – `63` days \| Configured: `21`) |
| `max_hurst_exponent` | Upper ceiling on Hurst exponent ($H$) to enforce mean-reverting spread behavior ($H < 0.50$). | Float (`0.40` – `0.55` \| Configured: `0.52`) |
| `min_correlation` | Minimum required asset-level return correlation threshold during pre-screening. | Float (`0.30` – `0.70` \| Configured: `0.45`) |
| `p_value_threshold` | Maximum allowed p-value threshold for Engle-Granger cointegration testing. | Float (`0.01` – `0.15` \| Configured: `0.10`) |
| `mp_window` | Subsequence length ($m$) for SciPy matrix profile distance evaluation. | Integer (`10` – `42` days \| Configured: `21`) |
| `max_mp_distance` | Upper limit on normalized Euclidean motif distance between price subsequences. | Float (`2.00` – `6.00` \| Configured: `4.50`) |
| `min_half_life` | Lower threshold for spread Ornstein-Uhlenbeck half-life in trading days. | Float (`1.0` – `5.0` days \| Configured: `3.0`) |
| `max_half_life` | Upper threshold for spread Ornstein-Uhlenbeck half-life in trading days. | Float (`15.0` – `60.0` days \| Configured: `30.0`) |
| `tau_hl_penalty` | Half-life penalty scaling parameter discounting expected returns on slow decay spreads. | Float (`2.0` – `10.0` \| Configured: `5.0`) |
| `base_min_z_spread` | Base Z-score threshold hurdle ($|e_t|$) required to trigger position entry. | Float (`1.50` – `2.50` $\sigma$ \| Configured: `1.85`) |
| `exit_z_spread` | Target Z-score threshold for mean-reversion profit taking. | Float (`0.00` – `0.50` $\sigma$ \| Configured: `0.20`) |
| `stop_loss_z` | Multiplier threshold on Z-score for triggering emergency stop-loss exits. | Float (`2.50` – `4.00` $\sigma$ \| Configured: `2.75`) |
| `bocpd_cp_threshold` | Bayesian Online Change-Point Detection probability threshold triggering structural exit. | Float (`0.20` – `0.50` \| Configured: `0.35`) |
| `ghost_eval_window_days` | Rolling evaluation horizon (in days) for monitoring non-allocated ghost pairs. | Integer (`21` – `126` days \| Configured: `63`) |
| `max_half_life_expansion_ratio` | Maximum allowed expansion ratio of current vs. entry half-life before demotion. | Float (`1.5` – `4.0` \| Configured: `2.5`) |
| `jump_lookback_days` | Lookback period (in days) for detecting single-stock idiosyncratic price jumps. | Integer (`5` – `30` days \| Configured: `15`) |
| `jump_threshold` | Standard deviation threshold for flagging idiosyncratic stock price jumps. | Float (`2.5` – `5.0` $\sigma$ \| Configured: `4.0`) |
| `max_macro_correlation` | Maximum allowed absolute linear correlation between spread returns and market returns. | Float (`0.05` – `0.30` \| Configured: `0.20`) |
| `demote_bottom_n` | Number of lowest expected return live pairs forced into liquidation each rebalance. | Integer (`0` – `5` \| Configured: `0`) |
| `promote_top_n` | Maximum number of top-ranking ghost pairs promoted to eligible tier per rebalance. | Integer (`5` – `50` \| Configured: `20`) |
| `obj_alpha_return` | Multiplier weight on expected return term in the SLSQP objective function. | Float (`0.5` – `2.0` \| Configured: `1.0`) |
| `obj_beta_risk` | Risk aversion coefficient / quadratic penalty multiplier on portfolio covariance. | Float (`0.1` – `2.0` \| Configured: `0.5`) |
| `max_factor_exposure` | Maximum allowed aggregate net loading ceiling on style factor proxies (e.g., momentum). | Float (`0.05` – `0.25` \| Configured: `0.10`) |
| `max_gross_exposure` | Maximum total portfolio gross leverage cap (sum of absolute asset weights). | Float (`1.0` – `2.5` \| Configured: `1.8`) |
| `max_pairs_in_book` | Hard ceiling on the total number of simultaneous active pair allocations. | Integer (`5` – `30` \| Configured: `12`) |
| `max_pairs_per_sector` | Maximum number of active pair allocations allowed within a single GICS sector. | Integer (`1` – `5` \| Configured: `3`) |
| `max_sleeve_weight` | Maximum portfolio gross weight allowed for any individual pair sleeve. | Float (`0.10` – `0.35` \| Configured: `0.25`) |
| `max_asset_cap` | Maximum absolute net NAV exposure cap on any single underlying ticker. | Float (`0.05` – `0.25` \| Configured: `0.15`) |
| `max_adv_participation` | Maximum allowed trade share volume capped as a fraction of 20-day ADV. | Float (`0.005` – `0.05` \| Configured: `0.015` / `1.5%`) |
| `initial_capital` | Initial portfolio NAV starting balance for backtesting and order sizing. | Float (`$100,000.0` – `$10,000,000.0` \| Configured: `$250,000.0`) |
| `exec_fee` | One-way transaction fee applied per trade leg (commissions + slippage). | Float (`0.00005` – `0.00100` \| Configured: `0.00015` / `1.5 bps`) |
| `borrow_bps` | Annual short borrow cost in basis points ($1 \text{ bps} = 0.01\%$). | Float (`10.0` – `200.0` bps \| Configured: `50.0` bps) |

## Outputs

Running `live_test_1.0.ipynb` end-to-end produces:

1. **Investment Committee Pair Review Tearsheet** — per-pair return, Sharpe/Sortino, max drawdown, win rate, trade count.
2. **Walk-Forward Net Portfolio Stats** — global `vbt.Portfolio` stats (net of fees, borrow costs, and T+1 execution lag), including max gross exposure.
3. **Diagnostic Tearsheet** — 5-panel plot: universe funnel sizing, forecast vs. realized returns, Kalman innovation Z / CUSUM, the LinUCB gate trajectory, and the cumulative performance horizon.
4. **Actionable MOO Order Blotter** — next-morning share orders (BUY/SELL, shares, est. USD, ADV participation %) for the following session.

## Dependencies

- `numpy`, `pandas`, `matplotlib`
- `statsmodels` (Engle–Granger cointegration, Ljung-Box)
- `scipy` (Kalman, SLSQP optimizer, chi-squared / Student-t distributions)
- `vectorbt` (portfolio simulation / tearsheets)
- `joblib` (parallel screening and pair evaluation)
- `mabwiser` (LinUCB contextual bandit)

## Quick Start

1. Place `universe_daily_train.csv`, `universe_daily_val.csv`, and (optionally) `data/constituents.csv` with GICS sector data in the project working directory.
2. Open `live_test_1.0.ipynb` and run all cells.
3. Review the committee tearsheet and net-portfolio stats printed to the console.
4. Feed the resulting `moo_blotter` to your brokerage (per the staging workflow in `TEST_CHECKLIST.md`).

For the live-signoff checklist that accompanies this notebook, see `TEST_CHECKLIST.md`.