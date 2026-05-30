import math
import pathlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

INITIAL_CAPITAL = 3_000_000
POSITION_SIZE = 1_000_000
MAX_POSITIONS = 3
HOLD_DAYS = 10
START_DATE = "2025-12-01"
PRICE_LOOKBACK_DAYS = 180

ETF_REPO = "4ru1013/united-etf-00981a-portfolio"
ETF_PREFIX = "00981A_holdings_"
ETF_CONTENTS_URL = f"https://api.github.com/repos/{ETF_REPO}/contents/data/out"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
OUTPUT_DIR = pathlib.Path("output")


@dataclass
class Position:
    code: str
    name: str
    setup: str
    signal_date: str
    buy_date: str
    buy_price: float
    shares: float
    capital: float
    holding_days_left: int
    trading_score: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text), dtype={"code": "string"})


def list_981_holdings_files() -> list[dict]:
    resp = requests.get(ETF_CONTENTS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    items = resp.json()
    pattern = re.compile(rf"^{re.escape(ETF_PREFIX)}(\d{{8}})\.csv$")
    files = []
    for item in items:
        name = item.get("name", "")
        m = pattern.match(name)
        if not m:
            continue
        date_str = m.group(1)
        files.append({
            "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
            "date_key": date_str,
            "name": name,
            "download_url": item.get("download_url"),
        })
    files = sorted(files, key=lambda x: x["date_key"])
    if not files:
        raise RuntimeError("No 00981A historical holdings files found.")
    return files


def load_holding_file(file_info: dict) -> pd.DataFrame:
    df = read_csv_from_url(file_info["download_url"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"code", "name", "shares"}.issubset(df.columns):
        raise ValueError(f"Invalid holdings columns in {file_info['name']}: {list(df.columns)}")
    df = df.copy()
    df["code"] = df["code"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce") if "weight" in df.columns else np.nan
    df = df.sort_values(["weight", "shares"], ascending=[False, False]).reset_index(drop=True)
    df["etf_rank"] = np.arange(1, len(df) + 1)
    df["is_top10"] = df["etf_rank"] <= 10
    df["signal_date"] = file_info["date"]
    return df[["signal_date", "code", "name", "shares", "weight", "etf_rank", "is_top10"]]


def yahoo_symbol_candidates(stock_id: str) -> list[str]:
    if stock_id == "TAIEX":
        return ["^TWII"]
    if stock_id == "00981A":
        return ["00981A.TW"]
    return [f"{stock_id}.TW", f"{stock_id}.TWO"]


def fetch_yahoo_chart(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc)
    params = {
        "period1": int(start_dt.timestamp()),
        "period2": int(end_dt.timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    resp = requests.get(YAHOO_CHART_URL.format(symbol=symbol), params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    error = payload.get("chart", {}).get("error")
    if error:
        raise RuntimeError(error)
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame()
    r = result[0]
    timestamps = r.get("timestamp") or []
    quote = (r.get("indicators", {}).get("quote") or [{}])[0]
    adj = (r.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
    if not timestamps or not adj:
        return pd.DataFrame()
    return pd.DataFrame({
        "date": [datetime.fromtimestamp(x, tz=timezone.utc).date().isoformat() for x in timestamps],
        "open": quote.get("open", [np.nan] * len(timestamps)),
        "close_raw": quote.get("close", [np.nan] * len(timestamps)),
        "close_adj": adj,
        "volume": quote.get("volume", [np.nan] * len(timestamps)),
        "yahoo_symbol": symbol,
    }).dropna(subset=["close_adj"])


def fetch_price(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    errors = []
    for symbol in yahoo_symbol_candidates(stock_id):
        try:
            df = fetch_yahoo_chart(symbol, start_date, end_date)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            continue
        if not df.empty:
            df["code"] = stock_id
            return df
    print(f"[WARN] price failed for {stock_id}: {' | '.join(errors)}")
    return pd.DataFrame()


def add_indicators_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    close = pd.to_numeric(df["close_adj"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["ret_20d"] = close / close.shift(20) - 1
    df["ret_60d"] = close / close.shift(60) - 1
    df["vol_ma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["vol_ma20"]
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    df["dif"] = ema8 - ema17
    df["macd"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["osc"] = df["dif"] - df["macd"]
    df["ma20_gt_ma60"] = df["ma20"] > df["ma60"]
    df["close_gt_ma60"] = close > df["ma60"]
    return df


def fetch_price_panel(codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    all_codes = sorted(set(codes + ["TAIEX", "00981A"]))
    for i, code in enumerate(all_codes, 1):
        print(f"[INFO] Fetch price {i}/{len(all_codes)}: {code}")
        df = fetch_price(code, start_date, end_date)
        if df.empty:
            continue
        frames.append(add_indicators_one(df))
    if not frames:
        raise RuntimeError("No price data fetched.")
    return pd.concat(frames, ignore_index=True)


def calc_trading_score(row: pd.Series) -> float:
    setup_score = {"A": 100, "B": 90, "C": 55, "D": 0}.get(str(row.get("setup")), 0)
    market_score = float(row.get("rs20_rank", 0) or 0)
    etf_score = 100 if bool(row.get("is_top10", False)) else 40
    return round(0.6 * setup_score + 0.3 * market_score + 0.1 * etf_score, 1)


def classify_setup(row: pd.Series) -> str:
    if not bool(row.get("ma20_gt_ma60", False)) or not bool(row.get("close_gt_ma60", False)):
        return "D"
    if pd.notna(row.get("osc")) and row.get("osc") > 0:
        return "A"
    # V1 approximation: if OSC is negative but close to zero relative to price, label B.
    # Full Daily Screen uses simulated OSC Flip Price. Backtest V1 uses this simpler proxy for speed.
    close = row.get("close_adj")
    osc = row.get("osc")
    if pd.notna(close) and pd.notna(osc) and abs(osc) / close <= 0.005:
        return "B"
    return "C"


def build_signal_table(holdings_by_date: dict[str, pd.DataFrame], price_panel: pd.DataFrame) -> pd.DataFrame:
    taiex = price_panel[price_panel["code"] == "TAIEX"][["date", "ret_20d", "ret_60d"]].rename(columns={"ret_20d": "taiex_ret20", "ret_60d": "taiex_ret60"})
    stock_px = price_panel[~price_panel["code"].isin(["TAIEX", "00981A"])]
    rows = []
    for signal_date, holdings in holdings_by_date.items():
        px_date = stock_px[stock_px["date"] == signal_date]
        if px_date.empty:
            continue
        merged = holdings.merge(px_date, on="code", how="inner")
        merged = merged.merge(taiex[taiex["date"] == signal_date], left_on="date", right_on="date", how="left")
        if merged.empty:
            continue
        merged["rs20"] = merged["ret_20d"] - merged["taiex_ret20"]
        merged["rs60"] = merged["ret_60d"] - merged["taiex_ret60"]
        merged["rs_accel"] = merged["rs20"] - merged["rs60"]
        merged["rs20_rank"] = merged["rs20"].rank(pct=True, ascending=True) * 100
        merged["setup"] = merged.apply(classify_setup, axis=1)
        merged["trading_score"] = merged.apply(calc_trading_score, axis=1)
        rows.append(merged)
    if not rows:
        raise RuntimeError("No signal table generated.")
    signals = pd.concat(rows, ignore_index=True)
    return signals.sort_values(["signal_date", "trading_score"], ascending=[True, False])


def get_next_trading_date(price_dates: list[str], date_str: str) -> str | None:
    for d in price_dates:
        if d > date_str:
            return d
    return None


def get_nth_trading_date(price_dates: list[str], start_date: str, n: int) -> str | None:
    future = [d for d in price_dates if d > start_date]
    if len(future) < n:
        return None
    return future[n - 1]


def price_at(price_panel: pd.DataFrame, code: str, date_str: str, column: str) -> float:
    row = price_panel[(price_panel["code"] == code) & (price_panel["date"] == date_str)]
    if row.empty:
        return np.nan
    value = row[column].iloc[0]
    return float(value) if pd.notna(value) else np.nan


def run_backtest(signals: pd.DataFrame, price_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_dates = sorted(price_panel[price_panel["code"] == "TAIEX"]["date"].unique().tolist())
    signal_dates = sorted(signals[signals["signal_date"] >= START_DATE]["signal_date"].unique().tolist())
    positions: list[Position] = []
    trades = []
    curve = []

    for signal_date in signal_dates:
        # Exit positions whose 10 trading days have elapsed at today's close.
        still_open = []
        for pos in positions:
            sell_date = get_nth_trading_date(price_dates, pos.buy_date, HOLD_DAYS)
            if sell_date == signal_date:
                sell_price = price_at(price_panel, pos.code, sell_date, "close_adj")
                if pd.notna(sell_price):
                    pnl = pos.shares * sell_price - pos.capital
                    trades.append({
                        "signal_date": pos.signal_date,
                        "buy_date": pos.buy_date,
                        "sell_date": sell_date,
                        "code": pos.code,
                        "name": pos.name,
                        "setup": pos.setup,
                        "trading_score": pos.trading_score,
                        "buy_price": pos.buy_price,
                        "sell_price": sell_price,
                        "capital": pos.capital,
                        "shares": pos.shares,
                        "return_pct": sell_price / pos.buy_price - 1,
                        "pnl": pnl,
                    })
                else:
                    still_open.append(pos)
            else:
                still_open.append(pos)
        positions = still_open

        # Add new positions after today's signal, entered next trading day's open.
        available_slots = MAX_POSITIONS - len(positions)
        if available_slots > 0:
            daily = signals[(signals["signal_date"] == signal_date) & (signals["setup"].isin(["A", "B"]))].sort_values("trading_score", ascending=False)
            held_codes = {p.code for p in positions}
            selected = []
            for _, row in daily.iterrows():
                code = str(row["code"])
                if code in held_codes:
                    continue
                selected.append(row)
                if len(selected) >= min(available_slots, MAX_POSITIONS):
                    break
            buy_date = get_next_trading_date(price_dates, signal_date)
            if buy_date:
                for row in selected:
                    code = str(row["code"])
                    buy_price = price_at(price_panel, code, buy_date, "open")
                    if not pd.notna(buy_price) or buy_price <= 0:
                        buy_price = price_at(price_panel, code, buy_date, "close_adj")
                    if not pd.notna(buy_price) or buy_price <= 0:
                        continue
                    positions.append(Position(
                        code=code,
                        name=str(row.get("name", "")),
                        setup=str(row.get("setup", "")),
                        signal_date=signal_date,
                        buy_date=buy_date,
                        buy_price=float(buy_price),
                        shares=POSITION_SIZE / float(buy_price),
                        capital=POSITION_SIZE,
                        holding_days_left=HOLD_DAYS,
                        trading_score=float(row.get("trading_score", 0)),
                    ))

        # Mark-to-market at close.
        position_value = 0.0
        for pos in positions:
            close_price = price_at(price_panel, pos.code, signal_date, "close_adj")
            if pd.notna(close_price):
                position_value += pos.shares * close_price
            else:
                position_value += pos.capital
        cash = INITIAL_CAPITAL - len(positions) * POSITION_SIZE
        curve.append({
            "date": signal_date,
            "strategy_value": cash + position_value,
            "cash": cash,
            "position_value": position_value,
            "open_positions": len(positions),
        })

    return pd.DataFrame(trades), pd.DataFrame(curve)


def calc_buy_and_hold_curve(price_panel: pd.DataFrame, code: str, benchmark_name: str, dates: list[str]) -> pd.Series:
    px = price_panel[price_panel["code"] == code].sort_values("date")
    px = px[px["date"].isin(dates)]
    if px.empty:
        return pd.Series(index=dates, dtype=float, name=benchmark_name)
    base = px[px["date"] >= START_DATE]
    if base.empty:
        return pd.Series(index=dates, dtype=float, name=benchmark_name)
    base_price = float(base.iloc[0]["close_adj"])
    series = px.set_index("date")["close_adj"] / base_price * INITIAL_CAPITAL
    return series.reindex(dates).ffill().rename(benchmark_name)


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    peak = series.cummax()
    dd = series / peak - 1
    return float(dd.min())


def summarize(trades: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    strategy = curve["strategy_value"]
    ret = strategy.pct_change().dropna()
    trade_returns = trades["return_pct"] if not trades.empty else pd.Series(dtype=float)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.nan
    total_return = strategy.iloc[-1] / strategy.iloc[0] - 1
    return pd.DataFrame([{
        "metric": "strategy",
        "trades": int(len(trades)),
        "total_return": total_return,
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else np.nan,
        "avg_trade_return": float(trade_returns.mean()) if len(trade_returns) else np.nan,
        "median_trade_return": float(trade_returns.median()) if len(trade_returns) else np.nan,
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
        "mdd": max_drawdown(strategy),
        "sharpe_daily": float(ret.mean() / ret.std() * math.sqrt(252)) if len(ret) > 1 and ret.std() != 0 else np.nan,
        "ending_value": float(strategy.iloc[-1]),
    }])


def setup_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["setup", "trades", "win_rate", "avg_return", "median_return", "total_pnl"])
    g = trades.groupby("setup")
    return g.agg(
        trades=("code", "count"),
        win_rate=("return_pct", lambda s: (s > 0).mean()),
        avg_return=("return_pct", "mean"),
        median_return=("return_pct", "median"),
        total_pnl=("pnl", "sum"),
    ).reset_index()


def main() -> None:
    ensure_dirs()
    files = list_981_holdings_files()
    files = [f for f in files if f["date"] >= "2025-11-28"]
    holdings_by_date = {f["date"]: load_holding_file(f) for f in files}
    all_codes = sorted(set(pd.concat(holdings_by_date.values())["code"].astype(str).tolist()))

    price_start = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=PRICE_LOOKBACK_DAYS)).date().isoformat()
    price_end = date.today().isoformat()
    price_panel = fetch_price_panel(all_codes, price_start, price_end)
    price_panel.to_csv(OUTPUT_DIR / "price_panel.csv", index=False, encoding="utf-8-sig")

    signals = build_signal_table(holdings_by_date, price_panel)
    signals.to_csv(OUTPUT_DIR / "signals.csv", index=False, encoding="utf-8-sig")

    trades, curve = run_backtest(signals, price_panel)
    dates = curve["date"].tolist()
    curve["taiex_value"] = calc_buy_and_hold_curve(price_panel, "TAIEX", "taiex_value", dates).values
    curve["etf981_value"] = calc_buy_and_hold_curve(price_panel, "00981A", "etf981_value", dates).values

    summary = summarize(trades, curve)
    setup = setup_analysis(trades)

    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    setup.to_csv(OUTPUT_DIR / "setup_analysis.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_DIR / "system_backtest.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        setup.to_excel(writer, sheet_name="Setup_Analysis", index=False)
        trades.to_excel(writer, sheet_name="Trades", index=False)
        curve.to_excel(writer, sheet_name="Equity_Curve", index=False)
        signals.to_excel(writer, sheet_name="Signals", index=False)

    print("[OK] Backtest complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
