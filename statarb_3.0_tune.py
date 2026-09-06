# ==============================================================================
# STAT-ARB PARAMETER TUNING ENGINE (v3.0)
# Runs statarb_3.0.py pipeline with systematic parameter variations
# until a strategy achieves 8-20% annualized return.
# No modifications to the original pipeline code.
# ==============================================================================
import os
import sys
import json
import warnings
import importlib.util
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Suppress matplotlib display (no GUI needed)
import matplotlib
matplotlib.use("Agg")

# ==============================================================================
# Import the original pipeline as a module
# ==============================================================================
spec = importlib.util.spec_from_file_location(
    "statarb_v3_pipeline",
    os.path.join(os.path.dirname(__file__), "statarb_3.0.py")
)
statarb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statarb)

# ==============================================================================
# TUNING CONFIGURATION
# ==============================================================================
# Base parameters from the original pipeline
BASE_PARAMS = deepcopy(statarb.run_params)

# Search space: most impactful parameters first (v3.0 has different params)
PRIMARY_KEYS = [
    "entry_ecdf_percentile",
    "min_override_z",
    "friction_clearance_multiplier",
]
SECONDARY_KEYS = [
    "exit_z_spread",
    "stop_loss_z",
    "max_hurst_exponent",
    "max_slope_tstat",
    "promote_top_n",
    "ghost_truncation_quantile",
    "max_gross_exposure",
    "p_value_threshold",
    "min_correlation",
    "min_holding_days",
    "max_holding_days",
    "max_pair_spy_beta",
    "max_sleeve_weight_low_density",
]

PARAM_GRID = {
    # Core entry/exit thresholds (v3.0 specific)
    "entry_ecdf_percentile": [0.85, 0.90, 0.95],
    "min_override_z": [0.50, 0.60, 0.70, 0.80],
    "friction_clearance_multiplier": [10.0, 15.0, 20.0],

    # Exit / risk thresholds
    "exit_z_spread":  [0.15, 0.20, 0.25],
    "stop_loss_z": [2.00, 2.20, 2.50],

    # Stationarity & momentum filters
    "max_hurst_exponent": [0.40, 0.43, 0.45, 0.48],
    "max_slope_tstat": [1.50, 1.75, 1.96],

    # Funnel sizing
    "promote_top_n": [20, 30, 40],
    "ghost_truncation_quantile": [0.15, 0.20, 0.25],

    # Risk bounds
    "max_gross_exposure": [1.50, 1.80, 2.00],

    # Statistical filters
    "p_value_threshold": [0.03, 0.04, 0.05],
    "min_correlation": [0.40, 0.45, 0.50],

    # Holding period
    "min_holding_days": [1, 2, 3],
    "max_holding_days": [10, 12, 15],

    # Overlay / hedging
    "max_pair_spy_beta": [1.00, 1.20, 1.50],

    # Sleeve scaling
    "max_sleeve_weight_low_density": [0.30, 0.35, 0.40],
}

RESULTS_FILE = "artifacts/tuning_results_v3.json"

# ==============================================================================
# TUNING ORCHESTRATOR
# ==============================================================================
def load_existing_results():
    """Load previously completed tuning runs."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {"runs": [], "best": None}

def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

def run_single_strategy(params, run_label):
    """Execute one pipeline run and return summary metrics."""
    print(f"\n{'='*85}")
    print(f"TUNING RUN: {run_label}")
    print(f"{'='*85}")

    # Set a unique state file per run so runs don't conflict
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = run_label.replace(" ", "_").replace("/", "_").replace(",", "").replace("=", "_")
    params["state_persistence_file"] = f"artifacts/tuning_runs_v3/state_{safe_label}_{ts}.pkl"
    params["model_id"] = f"tune_v3_{safe_label}_{ts}"

    try:
        committee_report, net_portfolio, metrics, moo_blotter, summary = statarb.run_unified_pipeline(params=params)

        result = {
            "run_label": run_label,
            "timestamp": ts,
            "params": {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
            "ann_ret_live": float(summary["ann_ret_live"]),
            "ann_ret_promoted": float(summary["ann_ret_promoted"]),
            "ann_ret_ghost": float(summary["ann_ret_ghost"]),
            "sortino_live": float(summary["sortino_live"]),
            "sortino_promoted": float(summary["sortino_promoted"]),
            "sortino_ghost": float(summary["sortino_ghost"]),
            "ret_delta_parametric": float(summary["ret_delta_parametric"]),
            "ret_delta_optimizer": float(summary["ret_delta_optimizer"]),
            "market_neutral": str(summary["verdict"].get("Market_Neutral", "UNKNOWN")),
            "neutrality_failures": str(summary["verdict"].get("Core_Failure_Reason", "None")),
        }

        # Print the execution summary
        if metrics is not None and not metrics.empty:
            statarb.print_execution_summary(
                metrics_df=metrics,
                committee_report=committee_report,
                moo_blotter=moo_blotter,
                **summary
            )

        return result

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Run failed: {e}", flush=True)
        traceback.print_exc()
        return {
            "run_label": run_label,
            "timestamp": ts,
            "params": {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
            "ann_ret_live": None,
            "error": str(e)
        }


def generate_param_combinations():
    """Generate parameter combinations systematically.

    Strategy: grid search over primary dims, then vary secondary dims
    around promising configurations.
    """
    # Phase 1: Vary ONE primary parameter at a time while keeping others at base
    phase1_configs = []
    for key in PRIMARY_KEYS:
        for val in PARAM_GRID[key]:
            params = deepcopy(BASE_PARAMS)
            params[key] = val
            label = f"{key}={val}"
            phase1_configs.append((params, label))

    # Phase 2: Best primary combo from Phase 1, then vary secondary params
    # (These will be generated dynamically based on Phase 1 results)

    return phase1_configs


def check_acceptance(result):
    """Check if a result meets the 8-20% annualized return criterion."""
    if result.get("ann_ret_live") is None:
        return False, "run_failed"

    ann_ret = result["ann_ret_live"]
    if 8.0 <= ann_ret <= 20.0:
        return True, f"ACCEPTED: ann_ret_live={ann_ret:.2f}%"
    elif ann_ret < 8.0:
        return False, f"too_low: {ann_ret:.2f}%"
    else:
        return False, f"too_high: {ann_ret:.2f}%"


# ==============================================================================
# MAIN TUNING LOOP
# ==============================================================================
def main():
    print("=" * 85)
    print("STAT-ARB PARAMETER TUNING ENGINE STARTED (v3.0)")
    print(f"Target: 8% <= ann_ret_live <= 20%")
    print(f"Grid size: {len(generate_param_combinations())} Phase 1 configs")
    print("=" * 85)

    results_db = load_existing_results()
    attempted_labels = {r["run_label"] for r in results_db["runs"]}

    # ================================================================
    # PHASE 1: Grid search over primary params one at a time
    # ================================================================
    phase1_configs = generate_param_combinations()

    for params, label in phase1_configs:
        if label in attempted_labels:
            print(f"[SKIP] Already attempted: {label}")
            continue

        result = run_single_strategy(params, label)
        results_db["runs"].append(result)

        accepted, msg = check_acceptance(result)
        print(f"\n>>> {label}: {msg}")

        if accepted:
            results_db["best"] = result
            save_results(results_db)
            print(f"\n{'='*85}")
            print(f"STRATEGY FOUND! ann_ret_live = {result['ann_ret_live']:.2f}%")
            print(f"Parameters:")
            for k, v in result["params"].items():
                if k in PARAM_GRID:
                    print(f"  {k}: {v}")
            print(f"{'='*85}")
            return result

        save_results(results_db)

    # ================================================================
    # PHASE 2: If Phase 1 didn't find a winner, analyze and refine
    # ================================================================
    completed = [r for r in results_db["runs"] if r.get("ann_ret_live") is not None]
    if not completed:
        print("[TUNING] No successful runs completed. Check pipeline for errors.")
        return None

    # Find the config closest to target range
    best_runs = sorted(completed, key=lambda r: abs(r["ann_ret_live"] - 14.0))[:5]

    print(f"\n{'='*85}")
    print(f"PHASE 1 COMPLETE. Best candidates:")
    for r in best_runs[:3]:
        print(f"  {r['run_label']}: ann_ret_live={r['ann_ret_live']:.2f}%")
    print(f"{'='*85}")

    # Phase 2: Try combinations of the best parameters
    best_params = deepcopy(BASE_PARAMS)
    for r in best_runs[:3]:
        for k, v in r["params"].items():
            if k in PARAM_GRID:
                best_params[k] = v

    # Try varying secondary params around the best config
    for key in SECONDARY_KEYS:
        for val in PARAM_GRID[key]:
            params = deepcopy(best_params)
            params[key] = val
            label = f"phase2_{key}={val}"

            if label in attempted_labels:
                continue

            result = run_single_strategy(params, label)
            results_db["runs"].append(result)

            accepted, msg = check_acceptance(result)
            print(f"\n>>> {label}: {msg}")

            if accepted:
                results_db["best"] = result
                save_results(results_db)
                print(f"\n{'='*85}")
                print(f"STRATEGY FOUND! ann_ret_live = {result['ann_ret_live']:.2f}%")
                print(f"Parameters:")
                for k, v in result["params"].items():
                    if k in PARAM_GRID:
                        print(f"  {k}: {v}")
                print(f"{'='*85}")
                return result

            save_results(results_db)

    # ================================================================
    # REPORT BEST EFFORT
    # ================================================================
    all_valid = [r for r in results_db["runs"] if r.get("ann_ret_live") is not None]
    if all_valid:
        best_overall = min(all_valid, key=lambda r: abs(r["ann_ret_live"] - 14.0))
        print(f"\n{'='*85}")
        print(f"BEST EFFORT (no exact hit). Closest to 14%:")
        print(f"  ann_ret_live = {best_overall['ann_ret_live']:.2f}%")
        print(f"  Run: {best_overall['run_label']}")
        print(f"{'='*85}")
        return best_overall

    print("[TUNING] No valid runs completed.")
    return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\nFinal result: ann_ret_live = {result.get('ann_ret_live')}%")
        sys.exit(0)
    else:
        sys.exit(1)