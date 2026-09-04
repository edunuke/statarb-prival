#!/usr/bin/env python3
"""
download_data.py
================
Downloads S&P 500 daily close prices and GICS constituents from Yahoo Finance.

Produces three files inside the dev/data/ directory:
  1. universe_daily_train.csv  — historical closes before the validation window
  2. universe_daily_val.csv    — exactly 1 year (252 trading days) of daily closes
  3. constituents.csv          — current S&P 500 ticker ↔ GICS sector mapping

Constraints
-----------
* Train and validation date ranges never overlap.
* Validation contains exactly 1 year (252 trading days) of data.
* Column headers are bare ticker symbols (e.g. AAPL, MSFT, etc.).
* Index is a DatetimeIndex of trading-day dates.
* The train end-date is the business day immediately before val start-date.

Usage
-----
    python download_data.py                       # defaults: val ends today
    python download_data.py --val-end 2026-08-25  # explicit val end
    python download_data.py --val-days 252        # override trading-day count
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# S&P 500 constituents from Wikipedia
# ---------------------------------------------------------------------------

def fetch_sp500_constituents() -> pd.DataFrame:
    """Return a DataFrame with columns ['Symbol', 'GICS Sector']."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    import io
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    # Standardise column names
    sym_col = "Symbol" if "Symbol" in df.columns else df.columns[1]
    sec_col = "GICS Sector" if "GICS Sector" in df.columns else df.columns[2]
    out = df[[sym_col, sec_col]].copy()
    out.columns = ["Symbol", "GICS Sector"]
    # Yahoo uses '.' for share-class suffixes (BRK.B etc.) — keep as-is.
    out["Symbol"] = out["Symbol"].str.strip()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Price download
# ---------------------------------------------------------------------------

def download_close_prices(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 80,
    pause: float = 1.5,
) -> pd.DataFrame:
    """
    Download Adjusted Close prices from Yahoo Finance in batches.

    Returns a DataFrame with DatetimeIndex and one column per ticker.
    NaN columns (tickers that failed) are dropped.
    """
    all_frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        tag = f"[batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}]"
        print(f"  {tag} Downloading {len(batch)} tickers ({batch[0]}..{batch[-1]}) ...", end=" ", flush=True)
        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as exc:
            print(f"FAILED ({exc})")
            time.sleep(pause)
            continue

        # yf.download with group_by='ticker' gives MultiIndex columns
        # when there are multiple tickers.
        if isinstance(data.columns, pd.MultiIndex):
            # Take the "Close" level for each ticker
            if "Close" in data.columns.get_level_values(-1):
                close = data.xs("Close", axis=1, level=-1)
            elif "Close" in data.columns.get_level_values(0):
                close = data["Close"]
            else:
                # Fallback: just grab the first-level columns if single-level
                close = data
        else:
            close = data

        # Ensure DatetimeIndex
        if not isinstance(close.index, pd.DatetimeIndex):
            close.index = pd.to_datetime(close.index)

        n_cols = close.shape[1]
        print(f"got {n_cols} tickers.")
        all_frames.append(close)
        time.sleep(pause)

    if not all_frames:
        print("[ERROR] No data downloaded.", file=sys.stderr)
        sys.exit(1)

    # Merge all batches
    prices = pd.concat(all_frames, axis=1)

    # Drop columns that are entirely NaN (failed downloads)
    before = prices.shape[1]
    prices = prices.dropna(axis=1, how="all")
    after = prices.shape[1]
    if before != after:
        print(f"  Dropped {before - after} tickers with no data ({after} kept).")

    # Forward-fill small gaps (holidays, halted days)
    prices = prices.ffill()

    # Sort
    prices = prices.sort_index()
    prices.index.name = None  # notebook expects no index name

    return prices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download S&P 500 daily price data for stat-arb pipeline")
    parser.add_argument(
        "--val-end",
        type=str,
        default=None,
        help="Last trading day of the validation window (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=252,
        help="Number of trading days in the validation window (default: 252 ≈ 1 year).",
    )
    parser.add_argument(
        "--train-extra-days",
        type=int,
        default=60,
        help="Extra trading days of training data before val window (default: 60, ensures lookback >= 100).",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=504,
        help="Minimum training history in trading days (default: 504 ≈ 2 years).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory relative to dev/ (default: data/).",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Constituents ----
    print("Fetching S&P 500 constituents from Wikipedia ...")
    constituents = fetch_sp500_constituents()
    tickers = constituents["Symbol"].tolist()
    print(f"  Found {len(tickers)} tickers.")
    constituents.to_csv(output_dir / "constituents.csv", index=False)
    print(f"  Saved -> {output_dir / 'constituents.csv'}")
    print()

    # ---- Download ALL prices first, then split by actual trading days ----
    # Download a generous window: 4+ years back from val_end
    val_end_approx = pd.Timestamp(args.val_end) if args.val_end else pd.Timestamp.today().normalize()
    download_start = (val_end_approx - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
    download_end = (val_end_approx + pd.offsets.BDay(2)).strftime("%Y-%m-%d")

    print(f"Downloading daily closes ({download_start} -> {download_end}) ...")
    prices = download_close_prices(tickers, start=download_start, end=download_end)
    print(f"  Downloaded {prices.shape[1]} tickers x {prices.shape[0]} days.")
    print()

    # ---- Split from actual trading days ----
    # Find the actual val_end as the last trading day <= the requested date
    actual_val_end = prices.index[prices.index <= val_end_approx][-1]
    # Take the last val_days trading days as validation
    val_start_idx = len(prices) - args.val_days
    if val_start_idx <= 0:
        print(f"[ERROR] Not enough data: only {len(prices)} days available, need > {args.val_days}", file=sys.stderr)
        sys.exit(1)

    prices_val = prices.iloc[val_start_idx:].copy()
    prices_train = prices.iloc[:val_start_idx].copy()

    # Ensure no overlap
    assert prices_train.index[-1] < prices_val.index[0], (
        f"Overlap detected! Train ends {prices_train.index[-1]}, Val starts {prices_val.index[0]}"
    )

    # Ensure enough training history
    assert len(prices_train) >= args.min_history, (
        f"Training has only {len(prices_train)} days, need >= {args.min_history}"
    )

    print(f"Train shape: {prices_train.shape}  ({prices_train.index[0].date()} -> {prices_train.index[-1].date()})")
    print(f"Val   shape: {prices_val.shape}  ({prices_val.index[0].date()} -> {prices_val.index[-1].date()})")
    print(f"Val trading days: {len(prices_val)}")
    assert len(prices_val) >= args.val_days, (
        f"Validation has only {len(prices_val)} days, expected >= {args.val_days}"
    )

    # ---- Save ----
    train_path = output_dir / "universe_daily_train.csv"
    val_path = output_dir / "universe_daily_val.csv"

    prices_train.to_csv(train_path)
    prices_val.to_csv(val_path)

    print()
    print(f"Saved -> {train_path}  ({train_path.stat().st_size / 1024:.0f} KB)")
    print(f"Saved -> {val_path}    ({val_path.stat().st_size / 1024:.0f} KB)")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
