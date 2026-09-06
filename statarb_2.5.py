# %%
# ==============================================================================
# S&P 500 STAT-ARB PIPELINE: HIGH-YIELD MARKET-NEUTRAL ENGINE (SPY-INTEGRATED)
# ==============================================================================
import os
import gc
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from scipy.optimize import minimize as scipy_minimize
import vectorbt as vbt
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 250)
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_id = "model_A_spy_integrated_neutral"

# ==============================================================================
# 1. INSTITUTIONAL PARAMETER CONFIGURATION
# ==============================================================================
run_params = {
    "price_csv": "data/universe_daily_train.csv",
    "oos_price_csv": "data/universe_daily_val.csv",
    "constituents_path": "data/constituents.csv",
    "state_persistence_file": f"artifacts/{model_id}/stat_arb_state_{timestamp}.pkl",

    "lookback_days": 504,             
    "min_history_days": 100,          
    "weight_drift_threshold": 0.05,     
    "screening_freq_days": 20,          
    "quarantine_days": 30,             
    
    "kalman_delta": 1e-4,              
    "kalman_obs_noise": 1e-3,          
    "max_hurst_exponent": 0.43,        # Guarantee mean-reversion
    "max_slope_tstat": 2.00,           # Momentum filter: reject entries if |t-stat| > 2.0
    
    "min_correlation": 0.40,           
    "p_value_threshold": 0.20,         
    
    "min_half_life": 1.5,             
    "max_half_life": 30.0,            # Capped at 30 days to kill structural trends
    "tau_hl_penalty": 5.0,             
    
    "entry_z_spread": 0.95,            # Entry dislocation threshold for high exposure
    "exit_z_spread": 0.30,             # Reversion profit-taking threshold
    "stop_loss_z": 2.80,               # Risk stop-loss threshold
    
    "promote_top_n": 50,               
    "ghost_truncation_quantile": 0.20, # Promotes top 80% of candidates
    
    # SLSQP RISK & NEUTRALITY BOUNDS
    "max_net_dollar_exposure": 0.03,   # Hard dollar neutrality cap (3%)
    "max_net_beta_exposure": 0.02,     # Hard SPY beta neutrality cap (2%)
    "max_sector_exposure": 0.08,       # Sector diversification limit (8%)
    "max_single_stock_gross": 0.15,    # Single stock leg cap across pairs (15%)
    "max_gross_exposure": 1.80,        # Portfolio gross capacity limit (180%)
    "max_sleeve_weight": 0.20,         # Sleeve cap to allow high-conviction scaling

    "max_adv_participation": 0.015,    
    "initial_capital": 250_000.0,
    "exec_fee": 0.00015,              
    "borrow_bps": 50.0
}

# ==============================================================================
# 2. STATE PERSISTENCE ENGINE
# ==============================================================================
class StatePersistenceEngine:
    @staticmethod
    def save_checkpoint(filepath: str, state_dict: dict):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(state_dict, f)
        except Exception:
            pass

    @staticmethod
    def load_checkpoint(filepath: str) -> Optional[dict]:
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

# ==============================================================================
# 3. VECTORIZED MATH KERNELS & STATISTICAL UTILITIES
# ==============================================================================
def compute_sortino_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
    """Calculates annualized Sortino ratio with downside deviation floor."""
    if len(returns) < 5 or returns.std() < 1e-8: return 0.0
    downside_returns = returns[returns < target_return]
    if len(downside_returns) < 2: return 0.0
    downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252.0)
    return float((returns.mean() * 252.0) / max(downside_std, 1e-6))

def get_rolling_half_life(spread: pd.Series, window: int = 126) -> pd.Series:
    """Vectorized OLS Half-Life using rolling covariance."""
    s_lag = spread.shift(1)
    s_diff = spread.diff()
    var_x = s_lag.rolling(window).var()
    cov_xy = s_lag.rolling(window).cov(s_diff)
    gamma = cov_xy / np.maximum(var_x, 1e-8)
    gamma = np.clip(gamma, -0.9999, -1e-5)
    hl = -np.log(2.0) / np.log(1.0 + gamma)
    return hl

def get_rolling_hurst(spread: pd.Series, window: int = 126, max_lag: int = 20) -> pd.Series:
    """Vectorized Hurst Exponent."""
    lags = np.arange(2, max_lag)
    log_lags = np.log(lags)
    
    tau_dict = {}
    for lag in lags:
        tau_dict[lag] = spread.diff(lag).rolling(window).std()
    tau_df = pd.DataFrame(tau_dict)
    
    log_tau = np.log(np.maximum(tau_df, 1e-8))
    mean_x = np.mean(log_lags)
    var_x = np.var(log_lags)
    
    mean_y = log_tau.mean(axis=1)
    cov_xy = ((log_tau.subtract(mean_y, axis=0)) * (log_lags - mean_x)).mean(axis=1)
    
    slope = cov_xy / var_x
    return slope * 2.0

def batch_kalman_filter_numpy(y: np.ndarray, x: np.ndarray, delta: float, obs_noise: float):
    """Native Python Kalman execution."""
    N = len(y)
    betas = np.zeros(N)
    innovations = np.zeros(N)
    F_arr = np.zeros(N)
    
    state_mean = np.zeros(2)
    state_cov = np.eye(2)
    Q = (delta / (1.0 - delta)) * np.eye(2)
    
    for t in range(N):
        H = np.array([1.0, x[t]])
        pred_cov = state_cov + Q
        v = y[t] - np.dot(H, state_mean)
        F = np.dot(H, np.dot(pred_cov, H)) + obs_noise
        K = np.dot(pred_cov, H) / F
        
        state_mean = state_mean + K * v
        state_cov = pred_cov - np.outer(K, np.dot(H, pred_cov))
        
        betas[t] = state_mean[1]
        innovations[t] = v
        F_arr[t] = F
        
    return betas, innovations, F_arr

# ==============================================================================
# 4. THE "ALPHA CUBE" PRE-COMPUTE WORKER
# ==============================================================================
def process_pair_history(t1: str, t2: str, price_s1: pd.Series, price_s2: pd.Series, market_ret_series: pd.Series, params: dict):
    """Generates statistical signals, real SPY CAPM betas, and returns for a single pair."""
    y = np.log(price_s1.values)
    x = np.log(price_s2.values)
    
    betas, innovations, F_arr = batch_kalman_filter_numpy(
        y, x, delta=params['kalman_delta'], obs_noise=params['kalman_obs_noise']
    )
    
    F_series = pd.Series(F_arr, index=price_s1.index).rolling(21, min_periods=5).mean().bfill().values
    spread_vol = np.sqrt(np.maximum(F_series, 1e-4))
    
    z_spread = np.clip(innovations / spread_vol, -5.0, 5.0)
    
    spread = pd.Series(y - betas * x, index=price_s1.index)
    hl = get_rolling_half_life(spread).fillna(25.0).clip(params['min_half_life'], params['max_half_life'])
    hurst = get_rolling_hurst(spread).fillna(0.50)
    
    # Vectorized 20-day Spread OLS Slope t-statistic (Trend / Breakout Filter)
    W_slope = 20
    t_idx = pd.Series(np.arange(len(spread)), index=price_s1.index)
    var_t = float((W_slope**2 - 1.0) / 12.0)
    cov_st = spread.rolling(W_slope).cov(t_idx)
    slope_s = cov_st / var_t
    var_s = spread.rolling(W_slope).var()
    se_slope = np.sqrt(np.maximum(1e-8, var_s - (slope_s**2 * var_t)) / ((W_slope - 2.0) * var_t))
    slope_tstats = (slope_s / np.maximum(se_slope, 1e-6)).fillna(0.0).values

    # True Rolling CAPM Betas against Real SPY Market Return
    W_beta = 63
    ret_A = price_s1.pct_change().fillna(0.0)
    ret_B = price_s2.pct_change().fillna(0.0)
    m_ret = market_ret_series.reindex(price_s1.index).fillna(0.0)
    
    var_m = m_ret.rolling(W_beta, min_periods=20).var().fillna(1e-4)
    cov_a_m = ret_A.rolling(W_beta, min_periods=20).cov(m_ret).fillna(0.0)
    cov_b_m = ret_B.rolling(W_beta, min_periods=20).cov(m_ret).fillna(0.0)
    
    beta_a_spy = (cov_a_m / np.maximum(var_m, 1e-6)).values
    beta_b_spy = (cov_b_m / np.maximum(var_m, 1e-6)).values
    
    # Net CAPM Market Beta of pair = Beta_A_SPY - (beta_coint * Beta_B_SPY)
    pair_spy_betas = beta_a_spy - (betas * beta_b_spy)
    
    exp_yield = np.maximum(0.0, np.abs(z_spread) - params['exit_z_spread']) * spread_vol
    alpha = exp_yield * (252.0 / hl.values) * (1.0 - np.exp(-hl.values / params['tau_hl_penalty']))
    
    pair_returns = (ret_A.values - betas * ret_B.values) / (1.0 + np.abs(betas))
    
    df = pd.DataFrame({
        'Beta': betas,
        'SPY_Beta': pair_spy_betas,
        'Slope_TStat': slope_tstats,
        'Z_Spread': z_spread,
        'Spread_Vol': spread_vol,
        'Half_Life': hl.values,
        'Hurst': hurst.values,
        'Parametric_Alpha': alpha,
        'Spread_Return': pair_returns
    }, index=price_s1.index)
    
    return (t1, t2), df

class UniverseScreener:
    def __init__(self, constituents_path: str):
        self.constituents_path = constituents_path

    def get_sectors(self, tickers: List[str]) -> pd.Series:
        if os.path.exists(self.constituents_path):
            const = pd.read_csv(self.constituents_path)
            sym_col = "Symbol" if "Symbol" in const.columns else "Ticker"
            return pd.Series(tickers, index=tickers).map(dict(zip(const[sym_col], const["GICS Sector"]))).fillna("Unknown")
        return pd.Series("Unknown", index=tickers)

# ==============================================================================
# 5. SLSQP OPTIMIZER (EMPIRICAL COVARIANCE + REAL SPY BETA NEUTRALITY)
# ==============================================================================
class ConvexPortfolioOptimizer:
    def __init__(self, params: dict):
        self.params = params

    @staticmethod
    def calculate_max_weight_drift(current_sleeve_weights: Dict[Tuple[str, str], float], target_sleeve_weights: Dict[Tuple[str, str], float]) -> float:
        all_keys = set(current_sleeve_weights.keys()).union(target_sleeve_weights.keys())
        if not all_keys: return 0.0
        return float(max([abs(current_sleeve_weights.get(pk, 0.0) - target_sleeve_weights.get(pk, 0.0)) for pk in all_keys]))

    def allocate(self, live_df: pd.DataFrame, sectors: pd.Series, current_weights: dict, alpha_cube: dict, current_idx: int):
        if live_df.empty: return {}, {}, {}

        N = len(live_df)
        pair_keys = [(r["Asset_A"], r["Asset_B"]) for _, r in live_df.iterrows()]
        
        returns_list = []
        for pk in pair_keys:
            s_ret = alpha_cube[pk]["Spread_Return"].iloc[max(0, current_idx-63):current_idx].values
            if len(s_ret) < 63:
                s_ret = np.pad(s_ret, (63 - len(s_ret), 0), 'constant')
            returns_list.append(s_ret)
            
        returns_mat = np.column_stack(returns_list)
        cov_matrix = np.cov(returns_mat, rowvar=False) * 252.0
        if N == 1: cov_matrix = np.array([[cov_matrix]])
        cov_matrix += np.eye(N) * 1e-5 

        pair_betas = live_df["Beta"].values
        pair_spy_betas = live_df["SPY_Beta"].values
        pair_zs = live_df["Z_Spread"].values
        alphas = live_df["Parametric_Alpha"].values

        w_prev = np.array([current_weights.get(pk, 0.0) for pk in pair_keys])
        turnover_bps = (self.params["exec_fee"] * 2.0) + 0.0005 

        dollar_mults = np.where(pair_zs > 0, pair_betas - 1.0, 1.0 - pair_betas)
        # Using Real CAPM SPY Betas for benchmark market neutrality
        spy_beta_mults = np.where(pair_zs > 0, -pair_spy_betas, pair_spy_betas)

        pair_secs = np.array([sectors.get(live_df.iloc[i]["Asset_A"], "Unknown") for i in range(N)])
        unique_secs = np.unique(pair_secs)

        def objective(w):
            risk = np.dot(w.T, np.dot(cov_matrix, w))
            turnover = np.sum(np.abs(w - w_prev)) * turnover_bps
            return risk + turnover

        target_gross = min(self.params["max_gross_exposure"], N * self.params["max_sleeve_weight"])
        min_gross = max(1.0, target_gross * 0.5) 
        
        max_net_dollar = self.params["max_net_dollar_exposure"]
        max_net_beta = self.params["max_net_beta_exposure"]
        max_sec = self.params["max_sector_exposure"]
        
        constraints = [
            {'type': 'ineq', 'fun': lambda w: target_gross - np.sum(w)}, 
            {'type': 'ineq', 'fun': lambda w: np.sum(w) - min_gross},    
            {'type': 'ineq', 'fun': lambda w: max_net_dollar - np.abs(np.sum(w * dollar_mults))},
            {'type': 'ineq', 'fun': lambda w: max_net_beta - np.abs(np.sum(w * spy_beta_mults))}
        ]
        
        for s in unique_secs:
            constraints.append({'type': 'ineq', 'fun': lambda w, sec=s: max_sec - np.abs(np.sum(w[pair_secs == sec] * dollar_mults[pair_secs == sec]))})

        all_assets = list(set(live_df["Asset_A"]).union(live_df["Asset_B"]))
        for asset in all_assets:
            asset_mask = np.zeros(N)
            for i, r in live_df.iterrows():
                idx_pos = live_df.index.get_loc(i)
                if r["Asset_A"] == asset:
                    asset_mask[idx_pos] += 1.0
                if r["Asset_B"] == asset:
                    asset_mask[idx_pos] += abs(r["Beta"])
            constraints.append({'type': 'ineq', 'fun': lambda w, mask=asset_mask: self.params["max_single_stock_gross"] - np.sum(w * mask)})

        base_cap = self.params["max_sleeve_weight"]
        alpha_median = max(float(np.median(alphas)), 1e-6)
        bounds = [(0.0, base_cap * np.clip(float(alphas[i]) / alpha_median, 0.20, 1.00)) for i in range(N)]
        
        spread_vols = live_df["Spread_Vol"].values
        init_w = (1.0 / np.maximum(spread_vols, 1e-4))
        init_w = np.minimum((init_w / np.sum(init_w)) * target_gross, [b[1] for b in bounds])

        res = scipy_minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        best_w = res.x if res.success else init_w
        best_w[best_w < 0.01] = 0.0

        allocs, memory = {}, {}
        for i, pk in enumerate(pair_keys):
            if best_w[i] > 0:
                row = live_df.iloc[i]
                w_a = -best_w[i] if row["Z_Spread"] > 0 else best_w[i]
                w_b = best_w[i] * row["Beta"] if row["Z_Spread"] > 0 else -best_w[i] * row["Beta"]
                allocs[pk] = {row["Asset_A"]: w_a, row["Asset_B"]: w_b, "Sleeve_Weight": best_w[i]}
                memory[pk] = {
                    "Beta": row["Beta"], "Entry_Z": row["Entry_Z"] if row["Is_Active"] else row["Z_Spread"],
                    "Parametric_Alpha": row["Parametric_Alpha"], 
                    "Days_Held": row.get("Days_Held", 0)
                }

        metrics = {"Portfolio_Variance": np.dot(best_w.T, np.dot(cov_matrix, best_w)), "Turnover_Cost": np.sum(np.abs(best_w - w_prev)) * turnover_bps, "Effective_Gross": np.sum(best_w)}
        return allocs, memory, metrics

# ==============================================================================
# 6. UNIFIED 2-STAGE ORCHESTRATOR
# ==============================================================================
def run_unified_pipeline(params: dict):
    print("=" * 85, flush=True)
    print("STARTING ROBUST 2-STAGE STAT-ARB ENGINE (TENSOR-DAG PARAMETRIC SLSQP)", flush=True)
    print("=" * 85, flush=True)
    
    try:
        prices_train = pd.read_csv(params["price_csv"], index_col=0, parse_dates=True)
        prices_val = pd.read_csv(params["oos_price_csv"], index_col=0, parse_dates=True)
        prices_full = pd.concat([prices_train, prices_val]).dropna(axis=1, how="all").ffill()
    except Exception:
        prices_train = pd.read_csv(params["price_csv"], index_col=0, parse_dates=True)
        prices_full = prices_train.ffill()

    prices_full = prices_full[~prices_full.index.duplicated()].sort_index().ffill()

    try:
        open_train = pd.read_csv(params["price_csv"].replace("close", "open"), index_col=0, parse_dates=True)
        open_val = pd.read_csv(params["oos_price_csv"].replace("close", "open"), index_col=0, parse_dates=True)
        open_full = pd.concat([open_train, open_val]).dropna(axis=1, how="all").ffill()
        open_full = open_full[~open_full.index.duplicated(keep='first')].sort_index()
    except FileNotFoundError:
        open_full = prices_full.copy()

    # DETECT REAL SPY TICKER vs. SYNTHETIC PROXY FALLBACK
    if 'SPY' in prices_full.columns:
        print("[Pipeline Info] 'SPY' ticker detected. Using real SPY price series for CAPM market benchmark.", flush=True)
        market_ret = prices_full['SPY'].pct_change().fillna(0.0)
        tradeable_prices = prices_full.drop(columns=['SPY'])
    else:
        print("[Pipeline Info] 'SPY' ticker not detected. Falling back to synthetic equal-weighted cross-sectional index.", flush=True)
        tradeable_prices = prices_full

    print("[DAG Pre-Compute] Initializing Vectorized Matrix Space...", flush=True)
    
    dag_df = tradeable_prices.dropna(axis=1)
    valid_tickers = dag_df.columns
    
    if len(valid_tickers) < 2:
        raise ValueError(f"[DAG Pre-Compute Error] Insufficient valid tickers ({len(valid_tickers)} remaining after dropna).")

    if 'SPY' not in prices_full.columns:
        market_ret = dag_df.pct_change().mean(axis=1)

    screener = UniverseScreener(params["constituents_path"])
    sectors = screener.get_sectors(valid_tickers.tolist())

    log_prices = np.log(dag_df.values)
    corr_matrix = np.corrcoef(np.diff(log_prices, axis=0), rowvar=False)
    
    sector_arr = sectors.reindex(valid_tickers).values
    sec_match = sector_arr[:, None] == sector_arr[None, :]
    mask = np.triu(sec_match & (corr_matrix >= params["min_correlation"]), k=1)
    
    candidate_pairs = [(valid_tickers[i], valid_tickers[j]) for i, j in zip(*np.where(mask))]
    print(f"[DAG Pre-Compute] Processing {len(candidate_pairs)} Sector-Matched Correlated Pairs across {len(dag_df)} days...", flush=True)
    
    with Parallel(n_jobs=-1, backend="loky") as parallel:
        results = parallel(delayed(process_pair_history)(t1, t2, dag_df[t1], dag_df[t2], market_ret, params) for t1, t2 in candidate_pairs)
    
    alpha_cube = {pair: df for pair, df in results}
    print("[DAG Pre-Compute] Completed successfully. Executing Portfolio Routing Loop.", flush=True)
    
    engine_alloc = ConvexPortfolioOptimizer(params)
    
    state = StatePersistenceEngine.load_checkpoint(params["state_persistence_file"]) or {}
    active_pairs, cooldown_tracker = state.get("active_pairs", {}), state.get("cooldown_tracker", {})
    
    start_idx = min(params["lookback_days"], len(prices_train))
    total_steps = len(prices_full) - start_idx

    master_weights = pd.DataFrame(np.nan, index=prices_full.index, columns=tradeable_prices.columns, dtype=np.float64)
    ghost_weights = pd.DataFrame(np.nan, index=prices_full.index, columns=tradeable_prices.columns, dtype=np.float64)
    promoted_weights = pd.DataFrame(np.nan, index=prices_full.index, columns=tradeable_prices.columns, dtype=np.float64)
    
    pair_weight_matrices, current_sleeve_weights, target_sleeve_weights = {}, {}, {}
    metrics_history, cached_keys = [], []
    days_since_screening = 0

    for i in range(start_idx, len(prices_full)):
        current_date = prices_full.index[i]
        step_num = i - start_idx + 1
        progress_pct = (step_num / total_steps) * 100.0

        has_exit = False
        for p in list(active_pairs.keys()):
            mem = active_pairs[p]
            current_state = alpha_cube[p].iloc[i]
            
            e_t = current_state["Z_Spread"]
            mem.update({"Days_Held": mem["Days_Held"] + 1})

            # EXIT CONDITION: Reversion <= exit_z_spread OR Stop Loss >= stop_loss_z OR Half-life timeout
            if abs(e_t) <= params["exit_z_spread"] or abs(e_t) >= params["stop_loss_z"] or mem["Days_Held"] > (current_state["Half_Life"] * 2.0):
                has_exit = True
                master_weights.loc[current_date, p[0]], master_weights.loc[current_date, p[1]] = 0.0, 0.0
                if p in pair_weight_matrices:
                    pair_weight_matrices[p].loc[current_date, p[0]] = 0.0
                    pair_weight_matrices[p].loc[current_date, p[1]] = 0.0

                if abs(e_t) >= params["stop_loss_z"]: cooldown_tracker[p] = params["quarantine_days"]
                del active_pairs[p]
                if p in current_sleeve_weights: del current_sleeve_weights[p]

        if not active_pairs:
            current_sleeve_weights = {}
            target_sleeve_weights = {}

        for p in list(cooldown_tracker.keys()):
            cooldown_tracker[p] -= 1
            if cooldown_tracker[p] <= 0: del cooldown_tracker[p]

        if days_since_screening >= params["screening_freq_days"] or not cached_keys:
            lb_prices = tradeable_prices.iloc[i - params["lookback_days"] : i]
            def _check_coint(t1, t2):
                p1, p2 = lb_prices[t1].values, lb_prices[t2].values
                if coint(p1, p2)[1] <= params["p_value_threshold"]: return (t1, t2)
                return None
            cached_keys = [res for res in [_check_coint(p[0], p[1]) for p in candidate_pairs] if res is not None]
            days_since_screening = 0
        days_since_screening += 1

        ghost_records = []
        for p in list(active_pairs.keys()) + [k for k in cached_keys if k not in active_pairs and k not in cooldown_tracker]:
            if p not in alpha_cube: continue
            state_t = alpha_cube[p].iloc[i]
            if state_t.isna().any(): continue
            
            is_active = p in active_pairs
            
            # Stationarity & Momentum Trend Filters
            if not is_active and state_t["Hurst"] >= params["max_hurst_exponent"]: continue
            if not is_active and state_t["Half_Life"] == params["max_half_life"]: continue
            if not is_active and abs(state_t["Slope_TStat"]) >= params.get("max_slope_tstat", 2.00): continue
            
            rec = {
                "Asset_A": p[0], "Asset_B": p[1], "Beta": state_t["Beta"], "SPY_Beta": state_t["SPY_Beta"],
                "Half_Life_Days": state_t["Half_Life"], "Z_Spread": state_t["Z_Spread"],
                "Parametric_Alpha": state_t["Parametric_Alpha"], "Spread_Vol": state_t["Spread_Vol"], 
                "Days_Held": active_pairs[p]["Days_Held"] if is_active else 0,
                "Is_Active": is_active, "Entry_Z": active_pairs[p]["Entry_Z"] if is_active else state_t["Z_Spread"]
            }
            ghost_records.append(rec)
            
        ghost_df = pd.DataFrame(ghost_records)

        promoted_ghosts = pd.DataFrame()
        if not ghost_df.empty:
            ghost_pool = ghost_df[~ghost_df["Is_Active"]]
            
            if not ghost_pool.empty:
                eligible_entries = ghost_pool[abs(ghost_pool["Z_Spread"]) >= params["entry_z_spread"]]
                
                if not eligible_entries.empty:
                    min_score = max(1e-6, eligible_entries["Parametric_Alpha"].quantile(params["ghost_truncation_quantile"]))
                    promoted_ghosts = eligible_entries[eligible_entries["Parametric_Alpha"] >= min_score]
                    promoted_ghosts = promoted_ghosts.sort_values("Parametric_Alpha", ascending=False).head(params["promote_top_n"])

            if len(ghost_pool) > 0:
                g_alloc = 1.0 / len(ghost_pool)
                for _, r in ghost_pool.iterrows():
                    ghost_weights.loc[current_date, r["Asset_A"]] = -g_alloc if r["Z_Spread"] > 0 else g_alloc
                    ghost_weights.loc[current_date, r["Asset_B"]] = g_alloc * r["Beta"] if r["Z_Spread"] > 0 else -g_alloc * r["Beta"]
            
            if len(promoted_ghosts) > 0:
                p_alloc = 1.0 / len(promoted_ghosts)
                for _, r in promoted_ghosts.iterrows():
                    promoted_weights.loc[current_date, r["Asset_A"]] = -p_alloc if r["Z_Spread"] > 0 else p_alloc
                    promoted_weights.loc[current_date, r["Asset_B"]] = p_alloc * r["Beta"] if r["Z_Spread"] > 0 else -p_alloc * r["Beta"]

        current_max_drift = ConvexPortfolioOptimizer.calculate_max_weight_drift(current_sleeve_weights, target_sleeve_weights)
        has_drift_trigger = current_max_drift >= params["weight_drift_threshold"]
        is_initial_day = (i == start_idx)

        should_rebalance = has_exit or len(promoted_ghosts) > 0 or has_drift_trigger or is_initial_day

        if should_rebalance:
            trigger_cause = (
                "INITIAL_ENTRY" if is_initial_day else
                ("SIGNAL_EXIT" if has_exit else
                ("SIGNAL_ENTRY" if not promoted_ghosts.empty else f"WEIGHT_DRIFT ({current_max_drift:.1%})"))
            )

            print(f"\n[Dynamic Rebalance @ {current_date.date()} | Progress: {progress_pct:5.1f}% ({step_num}/{total_steps})] Cause: {trigger_cause}", flush=True)
            
            if ghost_df.empty:
                print(f"  --> Pipeline Funnel : Ghost=0 | Promoted=0 | Live={len(active_pairs)}", flush=True)
                metrics_history.append({"Date": current_date, "Ghost_Size": 0, "Promoted_Size": 0, "Live_Size": len(active_pairs), "Opt_Exp_Variance": 0.0, "Effective_Gross": 0.0, "Mean_Innovation_Z": 0.0, "Mean_Parametric_Alpha": 0.0, "rebalanced": 1})
                continue

            live_candidates = ghost_df[ghost_df["Is_Active"]]
            live_df = pd.concat([live_candidates, promoted_ghosts]).drop_duplicates(subset=["Asset_A", "Asset_B"])
            
            allocs, active_pairs_updates, opt = engine_alloc.allocate(live_df, sectors, current_sleeve_weights, alpha_cube, i)
            
            dropped = set(active_pairs.keys()) - set(allocs.keys())
            for p in dropped:
                master_weights.loc[current_date, p[0]], master_weights.loc[current_date, p[1]] = 0.0, 0.0
                if p in pair_weight_matrices:
                    pair_weight_matrices[p].loc[current_date, p[0]] = 0.0
                    pair_weight_matrices[p].loc[current_date, p[1]] = 0.0
                cooldown_tracker[p] = params["quarantine_days"]

            active_pairs = active_pairs_updates
            current_sleeve_weights = {p: w["Sleeve_Weight"] for p, w in allocs.items()}
            target_sleeve_weights = current_sleeve_weights.copy()

            for p, w in allocs.items():
                if p not in pair_weight_matrices: pair_weight_matrices[p] = pd.DataFrame(np.nan, index=prices_full.index, columns=[p[0], p[1]])
                pair_weight_matrices[p].loc[current_date, p[0]] = w[p[0]]
                pair_weight_matrices[p].loc[current_date, p[1]] = w[p[1]]
                
                curr_w_a = master_weights.at[current_date, p[0]]
                curr_w_b = master_weights.at[current_date, p[1]]
                
                master_weights.at[current_date, p[0]] = (0.0 if np.isnan(curr_w_a) else curr_w_a) + w[p[0]]
                master_weights.at[current_date, p[1]] = (0.0 if np.isnan(curr_w_b) else curr_w_b) + w[p[1]]

            avg_inno_z = np.mean([abs(mem['Entry_Z']) for mem in active_pairs.values()]) if active_pairs else 0.0
            avg_alpha = live_df["Parametric_Alpha"].mean() if not live_df.empty else 0.0
            
            metrics_history.append({"Date": current_date, "Ghost_Size": len(ghost_pool) if 'ghost_pool' in locals() else 0, "Promoted_Size": len(promoted_ghosts), "Live_Size": len(active_pairs), "Opt_Exp_Variance": opt.get("Portfolio_Variance", 0.0), "Effective_Gross": opt.get("Effective_Gross", 0.0), "Mean_Innovation_Z": avg_inno_z, "Mean_Parametric_Alpha": avg_alpha, "rebalanced": 1})

            print(f"  --> Pipeline Funnel : Ghost={len(ghost_pool) if 'ghost_pool' in locals() else 0} | Promoted={len(promoted_ghosts)} | Live={len(active_pairs)}", flush=True)
            if active_pairs:
                print(f"  --> SLSQP Metrics   : Eff Gross={opt.get('Effective_Gross', 0):.2f} | Port Vol={np.sqrt(max(0, opt.get('Portfolio_Variance', 0))):.4f} | Turnover Drag={opt.get('Turnover_Cost', 0):.6f}", flush=True)
                print(f"  --> Top Allocations :", flush=True)
                sorted_allocs = sorted(active_pairs.keys(), key=lambda x: abs(allocs[x]['Sleeve_Weight']), reverse=True)[:3]
                for p in sorted_allocs:
                    w_data = allocs[p]
                    mem = active_pairs[p]
                    print(f"      {p[0]}/{p[1]} | W_A: {w_data[p[0]]:.1%} | W_B: {w_data[p[1]]:.1%} | Inno Z: {mem['Entry_Z']:.2f} | Parametric Alpha: {mem['Parametric_Alpha']:.4f}", flush=True)
            print("-" * 60, flush=True)

            StatePersistenceEngine.save_checkpoint(params["state_persistence_file"], {
                "active_pairs": active_pairs,
                "cooldown_tracker": cooldown_tracker
            })

    # ==============================================================================
    # EXHAUSTIVE BACKTEST & FUNNEL ATTRIBUTION LOGIC
    # ==============================================================================
    cap = params["max_gross_exposure"]
    for df_w in [master_weights, ghost_weights, promoted_weights]:
        abs_sum = df_w.abs().sum(axis=1)
        exceed_mask = abs_sum > cap
        if exceed_mask.any(): 
            df_w.loc[exceed_mask] = df_w.loc[exceed_mask].div(abs_sum[exceed_mask], axis=0) * cap

    exec_weights = master_weights.shift(1).iloc[start_idx:]
    exec_ghost_weights = ghost_weights.shift(1).iloc[start_idx:]
    exec_promoted_weights = promoted_weights.shift(1).iloc[start_idx:]

    for p in pair_weight_matrices: 
        pair_weight_matrices[p] = pair_weight_matrices[p].ffill().fillna(0.0).shift(1)

    rebalance_metrics_df = pd.DataFrame(metrics_history).set_index("Date")

    exec_close_prices = tradeable_prices.iloc[start_idx:]
    exec_open_prices = open_full[tradeable_prices.columns].iloc[start_idx:]

    metrics_df = pd.DataFrame(index=exec_close_prices.index)
    metrics_df = metrics_df.join(rebalance_metrics_df, how="left")
    metrics_df["rebalanced"] = metrics_df["rebalanced"].fillna(0).astype(int)

    ffill_cols = ["Ghost_Size", "Promoted_Size", "Live_Size", "Opt_Exp_Variance", "Effective_Gross", "Mean_Innovation_Z", "Mean_Parametric_Alpha"]
    metrics_df[ffill_cols] = metrics_df[ffill_cols].ffill().fillna(0.0)

    global_engine = VectorbtBacktestEngine(params)
    net_portfolio_live, val_live, true_gross_exp_series = global_engine.run_backtest(exec_close_prices, exec_open_prices, exec_weights)
    _, val_ghost, _ = global_engine.run_backtest(exec_close_prices, exec_open_prices, exec_ghost_weights)
    _, val_promoted, _ = global_engine.run_backtest(exec_close_prices, exec_open_prices, exec_promoted_weights)

    ret_live = val_live.pct_change().fillna(0.0)
    ret_ghost = val_ghost.pct_change().fillna(0.0)
    ret_promoted = val_promoted.pct_change().fillna(0.0)

    metrics_df["Live_Realized_Return"] = ret_live
    metrics_df["Ghost_Realized_Return"] = ret_ghost
    metrics_df["Promoted_Realized_Return"] = ret_promoted

    n_days = max(1, len(ret_live))
    ann_factor = 252.0 / n_days

    tot_ret_ghost = (val_ghost.iloc[-1] / params["initial_capital"]) - 1.0
    tot_ret_promoted = (val_promoted.iloc[-1] / params["initial_capital"]) - 1.0
    tot_ret_live = (val_live.iloc[-1] / params["initial_capital"]) - 1.0

    ann_ret_ghost = (((1.0 + tot_ret_ghost) ** ann_factor) - 1.0) * 100.0
    ann_ret_promoted = (((1.0 + tot_ret_promoted) ** ann_factor) - 1.0) * 100.0
    ann_ret_live = (((1.0 + tot_ret_live) ** ann_factor) - 1.0) * 100.0

    sortino_ghost = compute_sortino_ratio(ret_ghost)
    sortino_promoted = compute_sortino_ratio(ret_promoted)
    sortino_live = compute_sortino_ratio(ret_live)

    ret_delta_parametric = ann_ret_promoted - ann_ret_ghost
    ret_delta_optimizer = ann_ret_live - ann_ret_promoted

    sortino_delta_parametric = sortino_promoted - sortino_ghost
    sortino_delta_optimizer = sortino_live - sortino_promoted

    benchmark_returns = market_ret.iloc[start_idx:]
    neutrality_engine = MarketNeutralityDiagnosticEngine(max_beta=0.03, max_r2=0.01)
    neutrality_metrics_df, verdict = neutrality_engine.analyze(strategy_returns=ret_live, benchmark_returns=benchmark_returns)

    tearsheet_engine = CommitteeTearsheetEngine(params)
    committee_report = tearsheet_engine.generate_tearsheets(tradeable_prices, pair_weight_matrices)

    stats_df = net_portfolio_live.stats()
    stats_df["Max Gross Exposure [%]"] = float(true_gross_exp_series.max() * 100.0)

    order_gen = LiveMOOOrderGenerator(min_order_usd=500.0, max_adv_part=params.get("max_adv_participation", 0.015))
    rolling_adv_shares = (tradeable_prices.iloc[-20:].mean() * 50_000).fillna(1_000_000) 
    final_target_weights = master_weights.ffill().fillna(0.0).iloc[-1]
    
    moo_blotter = order_gen.generate_blotter(
        target_weights=final_target_weights,
        current_positions={},
        latest_prices=tradeable_prices.iloc[-1],
        rolling_adv_shares=rolling_adv_shares,
        portfolio_nav=params["initial_capital"]
    )

    summary_data = {
        "stats_df": stats_df,
        "ann_ret_ghost": ann_ret_ghost,
        "sortino_ghost": sortino_ghost,
        "ann_ret_promoted": ann_ret_promoted,
        "sortino_promoted": sortino_promoted,
        "ann_ret_live": ann_ret_live,
        "sortino_live": sortino_live,
        "ret_delta_parametric": ret_delta_parametric,
        "sortino_delta_parametric": sortino_delta_parametric,
        "ret_delta_optimizer": ret_delta_optimizer,
        "sortino_delta_optimizer": sortino_delta_optimizer,
        "verdict": verdict,
        "neutrality_metrics_df": neutrality_metrics_df
    }

    return committee_report, net_portfolio_live, metrics_df, moo_blotter, summary_data

# ==============================================================================
# 7. BACKTEST ENGINES & DIAGNOSTICS
# ==============================================================================
class CommitteeTearsheetEngine:
    def __init__(self, params: dict):
        self.initial_capital = params.get("initial_capital", 250_000.0)
        self.exec_fee = params.get("exec_fee", 0.00015)
        self.daily_borrow_rate = (params.get("borrow_bps", 50.0) / 10000.0) / 252.0

    def generate_tearsheets(self, prices_df: pd.DataFrame, pair_weight_matrices: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
        results = []
        for pair_key, w_df in pair_weight_matrices.items():
            a, b = pair_key
            active_dates = w_df.dropna(how='all').index
            if len(active_dates) == 0: continue
            start_date = active_dates[0]
            p_oos = prices_df[[a, b]].loc[start_date:]
            w_target = w_df.loc[start_date:].copy()
            w_target.iloc[-1] = 0.0 
            
            pf_gross = vbt.Portfolio.from_orders(
                close=p_oos, size=w_target, size_type='targetpercent',
                group_by=True, cash_sharing=True, init_cash=self.initial_capital, fees=self.exec_fee
            )
            asset_values = pf_gross.asset_value(group_by=False)
            short_exposure = asset_values.where(asset_values < 0, 0).abs()
            daily_short_cost = short_exposure.sum(axis=1) * self.daily_borrow_rate
            net_val = pf_gross.value() - daily_short_cost.cumsum()
            net_ret = net_val.pct_change().fillna(0)
            
            tot_ret = (net_val.iloc[-1] / self.initial_capital) - 1.0
            mean_ret, std_ret = net_ret.mean(), net_ret.std()
            down_std = np.sqrt((net_ret[net_ret < 0] ** 2).mean())
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
            sortino = (mean_ret / down_std) * np.sqrt(252) if down_std > 0 else 0.0
            max_dd = (1 - net_val / net_val.cummax()).max() if not net_val.empty else 0.0
            trades = pf_gross.trades.count()
            win_rate = (pf_gross.trades.winning.count() / trades) if trades > 0 else 0.0
            
            results.append({
                "Pair_Legs": f"{a} / {b}", "Total_Return_[%]": tot_ret * 100,
                "Sharpe_Ratio": sharpe, "Sortino_Ratio": sortino,
                "Max_DD_[%]": max_dd * 100, "Win_Rate_[%]": win_rate * 100, "Total_Trades": trades
            })
            del pf_gross
            gc.collect()

        if not results: return pd.DataFrame()
        return pd.DataFrame(results).sort_values("Sharpe_Ratio", ascending=False).reset_index(drop=True)

class VectorbtBacktestEngine:
    def __init__(self, params: dict):
        self.initial_capital = params.get("initial_capital", 250_000.0)
        self.exec_fee = params.get("exec_fee", 0.00015)
        self.daily_borrow_rate = (params.get("borrow_bps", 50.0) / 10000.0) / 252.0

    def run_backtest(self, close_prices_df: pd.DataFrame, open_prices_df: pd.DataFrame, weights_df: pd.DataFrame) -> Tuple[vbt.Portfolio, pd.Series, pd.Series]:
        tradable_assets = weights_df.columns.intersection(close_prices_df.columns)
        p_close_oos, p_open_oos = close_prices_df[tradable_assets].ffill(), open_prices_df[tradable_assets].ffill()
        sparse_weights = weights_df[tradable_assets]

        pf_gross = vbt.Portfolio.from_orders(
            close=p_close_oos, price=p_open_oos, size=sparse_weights, size_type='targetpercent',
            group_by=True, cash_sharing=True, init_cash=self.initial_capital, fees=self.exec_fee
        )
        
        asset_values = pf_gross.asset_value(group_by=False)
        short_exposure = asset_values.where(asset_values < 0, 0).abs()
        daily_short_cost = short_exposure.sum(axis=1) * self.daily_borrow_rate
        net_portfolio_value = pf_gross.value() - daily_short_cost.cumsum()
        
        total_abs_asset_val = asset_values.abs().sum(axis=1)
        true_gross_exposure = (total_abs_asset_val / net_portfolio_value.replace(0.0, np.nan)).fillna(0.0)
        return pf_gross, net_portfolio_value, true_gross_exposure

class MarketNeutralityDiagnosticEngine:
    def __init__(self, max_beta: float = 0.03, max_r2: float = 0.01):
        self.max_beta, self.max_r2 = max_beta, max_r2

    def analyze(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[pd.DataFrame, dict]:
        aligned_df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        aligned_df.columns = ["Strategy", "Benchmark"]
        r_p, r_m = aligned_df["Strategy"], aligned_df["Benchmark"]
        
        capm_model = sm.OLS(r_p, sm.add_constant(r_m)).fit()
        alpha_ann = capm_model.params.get("const", 0.0) * 252.0
        beta_m = capm_model.params.get("Benchmark", 0.0)
        p_val_beta = capm_model.pvalues.get("Benchmark", 1.0)
        t_stat_beta = capm_model.tvalues.get("Benchmark", 0.0)
        r2_capm = capm_model.rsquared

        bull_mask, bear_mask = r_m > 0, r_m < 0
        beta_bull = sm.OLS(r_p[bull_mask], sm.add_constant(r_m[bull_mask])).fit().params.get("Benchmark", 0.0) if bull_mask.sum() > 10 else 0.0
        beta_bear = sm.OLS(r_p[bear_mask], sm.add_constant(r_m[bear_mask])).fit().params.get("Benchmark", 0.0) if bear_mask.sum() > 10 else 0.0

        metrics = [
            {"Metric": "Annualized Alpha", "Value": f"{alpha_ann:.2%}", "Threshold": "N/A", "Status": "INFO"},
            {"Metric": "Market Beta (β_m)", "Value": f"{beta_m:.4f}", "Threshold": f"< |{self.max_beta}|", "Status": "PASS" if abs(beta_m) <= self.max_beta else "FAIL"},
            {"Metric": "Beta p-value", "Value": f"{p_val_beta:.4f}", "Threshold": ">= 0.05", "Status": "PASS" if p_val_beta >= 0.05 else "FAIL"},
            {"Metric": "Beta t-statistic", "Value": f"{t_stat_beta:.4f}", "Threshold": "< |1.96|", "Status": "PASS" if abs(t_stat_beta) < 1.96 else "FAIL"},
            {"Metric": "Variance Explained (R²)", "Value": f"{r2_capm:.2%}", "Threshold": f"< {self.max_r2:.1%}", "Status": "PASS" if r2_capm <= self.max_r2 else "FAIL"},
            {"Metric": "Bull Market Beta (R_m > 0)", "Value": f"{beta_bull:.4f}", "Threshold": f"< |{self.max_beta}|", "Status": "PASS" if abs(beta_bull) <= self.max_beta else "FAIL"},
            {"Metric": "Bear Market Beta (R_m < 0)", "Value": f"{beta_bear:.4f}", "Threshold": f"< |{self.max_beta}|", "Status": "PASS" if abs(beta_bear) <= self.max_beta else "FAIL"}
        ]
        metrics_df = pd.DataFrame(metrics)
        all_passed = (metrics_df["Status"] != "FAIL").all()
        verdict = {"Market_Neutral": "TRUE" if all_passed else "FALSE", "Core_Failure_Reason": "None" if all_passed else ", ".join(metrics_df[metrics_df["Status"] == "FAIL"]["Metric"].tolist())}
        return metrics_df, verdict

class LiveMOOOrderGenerator:
    def __init__(self, min_order_usd: float = 250.0, max_adv_part: float = 0.015, round_lots: bool = False):
        self.min_order_usd, self.max_adv_part, self.round_lots = min_order_usd, max_adv_part, round_lots

    def generate_blotter(self, target_weights: pd.Series, current_positions: Dict[str, int], latest_prices: pd.Series, rolling_adv_shares: pd.Series, portfolio_nav: float) -> pd.DataFrame:
        blotter = []
        all_tickers = set(target_weights.index).union(current_positions.keys())
        for ticker in all_tickers:
            t_weight, price, adv_shares = float(target_weights.get(ticker, 0.0)), float(latest_prices.get(ticker, np.nan)), float(rolling_adv_shares.get(ticker, 1_000_000.0))
            if np.isnan(price) or price <= 0: continue

            curr_shares = int(current_positions.get(ticker, 0))
            raw_target_shares = int((t_weight * portfolio_nav) / price)
            max_allowed_delta = int(adv_shares * self.max_adv_part)
            desired_delta = raw_target_shares - curr_shares
            
            target_shares = curr_shares + (max_allowed_delta if desired_delta > max_allowed_delta else (-max_allowed_delta if desired_delta < -max_allowed_delta else desired_delta))
            if self.round_lots: target_shares = (target_shares // 100) * 100

            delta_shares = target_shares - curr_shares
            delta_value = delta_shares * price

            if abs(delta_value) < self.min_order_usd and target_shares != 0: continue

            if delta_shares != 0:
                blotter.append({
                    "Ticker": ticker, "Action": "BUY" if delta_shares > 0 else "SELL", "Order_Type": "MOO",
                    "Delta_Shares": abs(delta_shares), "Target_Shares": target_shares, "Current_Shares": curr_shares,
                    "Est_Order_USD": round(abs(delta_value), 2), "Target_Weight_%": round((target_shares * price / portfolio_nav) * 100, 2),
                    "ADV_Participation_%": round((abs(delta_shares) / max(1.0, adv_shares)) * 100, 3), "Price_Ref": round(price, 2)
                })

        df_blotter = pd.DataFrame(blotter)
        if df_blotter.empty: return pd.DataFrame(columns=["Ticker", "Action", "Order_Type", "Delta_Shares", "Target_Shares", "Current_Shares", "Est_Order_USD", "Target_Weight_%", "ADV_Participation_%", "Price_Ref"])
        return df_blotter.sort_values("Est_Order_USD", ascending=False).reset_index(drop=True)

# ==============================================================================
# 8. DIAGNOSTIC TEARSHEET PLOTTER & REPORTING ENGINE
# ==============================================================================
def _get_next_run_dir(base_dir: str = "results") -> str:
    os.makedirs(base_dir, exist_ok=True)
    existing_ids = [int(d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    run_id = max(existing_ids) + 1 if existing_ids else 1
    run_dir = os.path.join(base_dir, str(run_id))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def format_independent_axis(ax, rebalance_dates):
    for r_date in rebalance_dates:
        ax.axvline(x=r_date, color="red", linestyle=":", alpha=0.4, linewidth=1.2)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def print_execution_summary(
    metrics_df: pd.DataFrame, 
    stats_df: pd.DataFrame, 
    committee_report: pd.DataFrame, 
    moo_blotter: pd.DataFrame,
    ann_ret_ghost: float, sortino_ghost: float,
    ann_ret_promoted: float, sortino_promoted: float,
    ann_ret_live: float, sortino_live: float,
    ret_delta_parametric: float, sortino_delta_parametric: float,
    ret_delta_optimizer: float, sortino_delta_optimizer: float,
    verdict: dict, neutrality_metrics_df: pd.DataFrame
):
    print("\n" + "=" * 85, flush=True)
    print("FUNNEL CONVERSION ATTRIBUTION & MARGINAL PERFORMANCE REPORT", flush=True)
    print("=" * 85, flush=True)
    print(f"  {'Funnel Tier / Portfolio Stage':<38} | {'Ann. Return [%]':<18} | {'Sortino Ratio':<18}", flush=True)
    print("-" * 85, flush=True)
    print(f"  {'Ghost Portfolio (Raw Universe)':<38} | {ann_ret_ghost:17.2f}% | {sortino_ghost:18.4f}", flush=True)
    print(f"  {'Promoted Portfolio (Parametric Gated)':<38} | {ann_ret_promoted:17.2f}% | {sortino_promoted:18.4f}", flush=True)
    print(f"  {'Live Portfolio (SLSQP Optimized)':<38} | {ann_ret_live:17.2f}% | {sortino_live:18.4f}", flush=True)
    print("-" * 85, flush=True)
    print(f"  {'--> Parametric Selection Delta':<38} | {ret_delta_parametric:+17.2f}% | {sortino_delta_parametric:+18.4f}", flush=True)
    print(f"  {'--> Optimizer Allocation Delta':<38} | {ret_delta_optimizer:+17.2f}% | {sortino_delta_optimizer:+18.4f}", flush=True)
    print("=" * 85, flush=True)

    print("\n" + "=" * 85, flush=True)
    print("QUANTITATIVE MARKET NEUTRALITY & FACTOR DIAGNOSTIC AUDIT", flush=True)
    print("=" * 85, flush=True)
    print(neutrality_metrics_df.to_string(index=False), flush=True)
    print("-" * 85, flush=True)
    print(f"  VERDICT: Market Neutral = {verdict['Market_Neutral']} | Failures: {verdict['Core_Failure_Reason']}", flush=True)
    print("=" * 85, flush=True)

    print("\n" + "=" * 85, flush=True)
    if not committee_report.empty:
        print("INVESTMENT COMMITTEE PAIR REVIEW TEARSHEET", flush=True)
        print("=" * 85, flush=True)
        print(committee_report.to_string(index=False), flush=True)
    else:
        print("INVESTMENT COMMITTEE PAIR REVIEW TEARSHEET: NO TRADES COMPLETED.", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("WALK-FORWARD NET PORTFOLIO STATS (POST-FEES & EXECUTION LAG)", flush=True)
    print("=" * 85, flush=True)
    print(stats_df.to_string(), flush=True)

    print("\n" + "=" * 85, flush=True)
    print("ACTIONABLE MARKET-ON-OPEN (MOO) ORDER BLOTTER FOR NEXT SESSION", flush=True)
    print("=" * 85, flush=True)
    print(moo_blotter.to_string(index=False), flush=True)

def plot_diagnostic_tearsheet(
    metrics_df: pd.DataFrame,
    initial_capital: float = 250_000.0,
    committee_report: Optional[pd.DataFrame] = None,
    net_portfolio: Optional[vbt.Portfolio] = None,
    moo_blotter: Optional[pd.DataFrame] = None,
    save: bool = True,
    prefix: str = "dev_"
):
    if metrics_df.empty: return
    df = metrics_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
    rebalance_dates = df[df["rebalanced"] == 1].index if "rebalanced" in df.columns else []

    output_dir = None
    if save:
        output_dir = _get_next_run_dir("results")
        print(f"\n[Export Engine] Saving artifacts to: '{output_dir}/'", flush=True)
        metrics_df.to_csv(os.path.join(output_dir, f"{prefix}metrics.csv"))
        if committee_report is not None and not committee_report.empty: committee_report.to_csv(os.path.join(output_dir, f"{prefix}committee_report.csv"), index=False)
        if moo_blotter is not None: moo_blotter.to_csv(os.path.join(output_dir, f"{prefix}moo_blotter.csv"), index=False)
        if net_portfolio is not None:
            try: net_portfolio.trades.records_readable.to_csv(os.path.join(output_dir, f"{prefix}trade_records.csv"), index=False)
            except Exception: pass
            try: net_portfolio.stats().to_csv(os.path.join(output_dir, f"{prefix}portfolio_stats.csv"))
            except Exception: pass

    fig1, ax1 = plt.subplots(figsize=(16, 4))
    if "Ghost_Size" in df.columns:
        ax1.plot(df.index, df["Ghost_Size"], label="Ghost Universe", color="#2b5c8f", linestyle="--", linewidth=1.5)
    if "Promoted_Size" in df.columns:
        ax1.plot(df.index, df["Promoted_Size"], label="Promoted Universe (Parametric Gated)", color="#e07a5f", linestyle="-.", linewidth=1.5)
    if "Live_Size" in df.columns:
        ax1.step(df.index, df["Live_Size"], label="Live Portfolio", color="#2a9d8f", where="post", linewidth=2.5)
    ax1.set_title("Universe Funnel Sizing (Pair Counts)", fontsize=11, fontweight="bold", loc="left")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=True)
    format_independent_axis(ax1, rebalance_dates)
    plt.tight_layout()
    if save and output_dir: fig1.savefig(os.path.join(output_dir, f"{prefix}funnel_sizing.png"), dpi=300, bbox_inches="tight")
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(16, 4))
    if "Mean_Parametric_Alpha" in df.columns:
        ax4.plot(df.index, df["Mean_Parametric_Alpha"], label="Mean Parametric Alpha", color="#457b9d", linewidth=2.0)
    ax4.set_title("Parametric Expected Yield Trajectory", fontsize=11, fontweight="bold", loc="left")
    ax4.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=True)
    format_independent_axis(ax4, rebalance_dates)
    plt.tight_layout()
    if save and output_dir: fig4.savefig(os.path.join(output_dir, f"{prefix}parametric_alpha.png"), dpi=300, bbox_inches="tight")
    plt.show()

    fig5, ax5 = plt.subplots(figsize=(16, 5))
    live_cum = (1.0 + df.get("Live_Realized_Return", pd.Series(0.0, index=df.index)).fillna(0.0)).cumprod() * initial_capital
    promoted_cum = (1.0 + df.get("Promoted_Realized_Return", pd.Series(0.0, index=df.index)).fillna(0.0)).cumprod() * initial_capital
    ghost_cum = (1.0 + df.get("Ghost_Realized_Return", pd.Series(0.0, index=df.index)).fillna(0.0)).cumprod() * initial_capital

    ax5.plot(df.index, live_cum, label="Live Realized Value ($)", color="#2a9d8f", linewidth=2.5)
    ax5.plot(df.index, promoted_cum, label="Promoted Shadow Value ($)", color="#e07a5f", linestyle="-.", linewidth=1.5)
    ax5.plot(df.index, ghost_cum, label="Ghost Shadow Value ($)", color="#2b5c8f", linestyle="--", linewidth=1.5)
    
    ax5.axhline(initial_capital, color="black", linewidth=0.8)
    ax5.set_title("Cumulative Performance Horizon & Funnel Attribution ($)", fontsize=11, fontweight="bold", loc="left")
    ax5.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax5.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=True)
    format_independent_axis(ax5, rebalance_dates)
    plt.tight_layout()
    if save and output_dir: fig5.savefig(os.path.join(output_dir, f"{prefix}cumulative_performance.png"), dpi=300, bbox_inches="tight")
    plt.show()

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    committee_report, net_portfolio, metrics, moo_blotter, summary_data = run_unified_pipeline(params=run_params)
    
    if metrics is not None and not metrics.empty:
        print_execution_summary(
            metrics_df=metrics,
            committee_report=committee_report,
            moo_blotter=moo_blotter,
            **summary_data
        )
        
        plot_diagnostic_tearsheet(
            metrics_df=metrics,
            initial_capital=run_params.get("initial_capital", 250_000.0),
            committee_report=committee_report,
            net_portfolio=net_portfolio,
            moo_blotter=moo_blotter,
            save=True,
            prefix="dev_"
        )


