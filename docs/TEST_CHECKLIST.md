# Live Beta Testing Checklist: Institutional Stat-Arb Pipeline

## 1. Algorithmic & Signal Integrity
- [ ] **Corporate Action Adjustment Alignment**: Verify that live price feeds apply point-in-time dividend and stock-split adjustments to prevent artificial Z-score innovation spikes ($|e_t| \ge 3.0\sigma$).
- [ ] **Context Vector Scaling**: Monitor LinUCB context inputs ($VIX$ proxy, $5d/63d$ volatility ratio, CUSUM $S_t$, half-life) to confirm live values match backtest distribution bounds.
- [ ] **Kalman State Continuity**: Compare online $O(1)$ state updates ($\hat{\theta}_t, P_t$) against full-window batch recalculations to ensure zero drift.
- [ ] **Change-Point & Risk Gate Audit**: Validate BOCPD change-point probabilities ($P(r_t=0) > 0.35$) and Ljung-Box residual p-values against false-positive triggers.
- [ ] **Fractional $d$ / Hurst Consistency**: Verify GPH log-periodogram ($d < 0.45$) and Hurst exponent ($H < 0.52$) outputs on rolling live 126/252-bar windows.

## 2. Risk Management & Portfolio Constraints
- [ ] **Net-Beta Neutrality Enforcement**: Audit SLSQP optimizer output daily to verify overall portfolio net-beta stays strictly within $\le 0.05$.
- [ ] **Single-Stock & Sleeve NAV Caps**: Confirm no individual stock allocation exceeds $15\%$ portfolio NAV and no pair sleeve exceeds $25\%$.
- [ ] **Style Factor Loading Bounds**: Track net portfolio momentum/value factor proxy loadings to ensure they remain bounded ($\le 0.10$).
- [ ] **Gross Exposure Management**: Ensure total gross leverage stays below the $1.8\times$ hard cap regardless of candidate pool size.
- [ ] **Quarantine Routing**: Verify that exited, stopped-out, or structurally broken pairs are routed to the 15-day cooldown tracker without leakage.

## 3. Microstructure, Slippage & Cost Controls
- [ ] **ADV Participation Capping**: Verify that staged order sizes are strictly capped at $\le 1.5\%$ of 20-day Average Daily Volume (ADV).
- [ ] **Short Locate Availability**: Query broker APIs at $T-1$ post-close to confirm locate availability and verify short leg rates are not marked Hard-To-Borrow (HTB).
- [ ] **Borrow Rate Reconciliation**: Track actual daily borrow costs against the 50 bps model assumption across all active short legs.
- [ ] **Slippage Benchmarking**: Measure real Market-On-Open (MOO) fill prices against previous close ($T-1$) and open ($T$) prices to quantify realized slippage.
- [ ] **All-In Transaction Cost Comparison**: Audit combined commission, locate fees, and market impact against the baseline $1.5\text{ bps}$ model execution drag.

## 4. Operational & Infrastructure Reliability
- [ ] **Broker API Staging (`ib_insync`)**: Test staging of actionable `moo_blotter` output directly to paper/live broker accounts without manual keying.
- [ ] **Order Filter Validation**: Confirm `LiveMOOOrderGenerator` correctly enforces minimum order size thresholds ($>\$500$) and round-lot rounding logic.
- [ ] **State Persistence & Crash Recovery**: Perform a cold-start recovery test by terminating the engine mid-session and confirming state reloads cleanly from `stat_arb_state.pkl`.
- [ ] **Data Feed Latency & Missing Bars**: Verify pipeline behavior when encountering missing price bars, exchange halts, or delayed $T-1$ daily closes.
- [ ] **Automated Telemetry & Alerting**: Establish real-time alerts (Slack/Email/PagerDuty) for unallocated cash, optimizer convergence failures, or state persistence errors.