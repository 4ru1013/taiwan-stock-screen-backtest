import math
import pathlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests

INITIAL_CAPITAL = 3_000_000.0
POSITION_SIZE = 1_000_000.0
MAX_POSITIONS = 3
HOLD_DAYS = 10
START_DATE = "2025-12-01"
LOOKBACK_DAYS = 180
OUTPUT_DIR = pathlib.Path("output")
ETF_CONTENTS_URL = "https://api.github.com/repos/4ru1013/united-etf-00981a-portfolio/contents/data/out"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass
class Position:
    code: str
    name: str
    setup: str
    signal_date: str
    buy_date: str
    sell_date: str
    buy_price: float
    shares: float
    capital: float
    score: float


@dataclass
class Order:
    code: str
    name: str
    setup: str
    signal_date: str
    buy_date: str
    score: float


def get_text(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r.text


def list_holdings_files() -> list[dict]:
    r = requests.get(ETF_CONTENTS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    pat = re.compile(r"^00981A_holdings_(\d{8})\.csv$")
    out = []
    for it in r.json():
        m = pat.match(it.get("name", ""))
        if m:
            d = m.group(1)
            out.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "key": d, "url": it["download_url"]})
    out.sort(key=lambda x: x["key"])
    return [x for x in out if x["date"] >= "2025-11-28"]


def load_holdings(file_info: dict) -> pd.DataFrame:
    df = pd.read_csv(StringIO(get_text(file_info["url"])), dtype={"code": "string"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["code"] = df["code"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce") if "weight" in df.columns else np.nan
    df = df.sort_values(["weight", "shares"], ascending=[False, False]).reset_index(drop=True)
    df["signal_date"] = file_info["date"]
    df["etf_rank"] = np.arange(1, len(df) + 1)
    df["is_top10"] = df["etf_rank"] <= 10
    return df[["signal_date", "code", "name", "shares", "weight", "etf_rank", "is_top10"]]


def symbols(code: str) -> list[str]:
    if code == "TAIEX":
        return ["^TWII"]
    if code == "00981A":
        return ["00981A.TW"]
    return [f"{code}.TW", f"{code}.TWO"]


def fetch_chart(sym: str, start: str, end: str) -> pd.DataFrame:
    p1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int((datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp())
    r = requests.get(YAHOO_URL.format(symbol=sym), params={"period1": p1, "period2": p2, "interval": "1d", "events": "history", "includeAdjustedClose": "true"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    j = r.json()["chart"]
    if j.get("error") or not j.get("result"):
        return pd.DataFrame()
    z = j["result"][0]
    ts = z.get("timestamp") or []
    q = (z.get("indicators", {}).get("quote") or [{}])[0]
    adj = (z.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
    if not ts or not adj:
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": [datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat() for t in ts],
        "open_raw": q.get("open", [np.nan] * len(ts)),
        "close_raw": q.get("close", [np.nan] * len(ts)),
        "close_adj": adj,
        "volume": q.get("volume", [np.nan] * len(ts)),
    }).dropna(subset=["close_adj"])
    rc = pd.to_numeric(df["close_raw"], errors="coerce")
    ro = pd.to_numeric(df["open_raw"], errors="coerce")
    ac = pd.to_numeric(df["close_adj"], errors="coerce")
    df["open_adj"] = np.where(rc > 0, ro * ac / rc, ro)
    return df


def fetch_price(code: str, start: str, end: str) -> pd.DataFrame:
    for s in symbols(code):
        try:
            df = fetch_chart(s, start, end)
        except Exception:
            continue
        if not df.empty:
            df["code"] = code
            return add_indicators(df)
    print(f"[WARN] no price: {code}")
    return pd.DataFrame()


def next_osc(close: pd.Series, px: float) -> float:
    s = pd.concat([close, pd.Series([px])], ignore_index=True)
    dif = s.ewm(span=8, adjust=False).mean() - s.ewm(span=17, adjust=False).mean()
    macd = dif.ewm(span=9, adjust=False).mean()
    return float((dif - macd).iloc[-1])


def flip_price(close: pd.Series, current_close: float, current_osc: float):
    if pd.isna(current_close) or pd.isna(current_osc):
        return np.nan
    if current_osc > 0:
        return "Already Positive"
    lo, hi = float(current_close), float(current_close) * 1.15
    if next_osc(close, hi) <= 0:
        return np.nan
    for _ in range(25):
        mid = (lo + hi) / 2
        if next_osc(close, mid) > 0:
            hi = mid
        else:
            lo = mid
    return round(hi, 2)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c = pd.to_numeric(df["close_adj"], errors="coerce")
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    df["ret_20d"] = c / c.shift(20) - 1
    df["ret_60d"] = c / c.shift(60) - 1
    dif = c.ewm(span=8, adjust=False).mean() - c.ewm(span=17, adjust=False).mean()
    df["dif"] = dif
    df["macd"] = dif.ewm(span=9, adjust=False).mean()
    df["osc"] = df["dif"] - df["macd"]
    df["ma20_gt_ma60"] = df["ma20"] > df["ma60"]
    df["close_gt_ma60"] = c > df["ma60"]
    flips = []
    for i in range(len(df)):
        if i < 20:
            flips.append(np.nan)
        else:
            flips.append(flip_price(c.iloc[: i + 1].dropna().reset_index(drop=True), c.iloc[i], df["osc"].iloc[i]))
    df["osc_flip_price"] = flips
    return df


def setup(row) -> str:
    if not bool(row.get("ma20_gt_ma60")) or not bool(row.get("close_gt_ma60")):
        return "D"
    if pd.notna(row.get("osc")) and row["osc"] > 0:
        return "A"
    fp = row.get("osc_flip_price")
    if isinstance(fp, (int, float, np.floating)) and pd.notna(fp) and fp <= row["close_adj"] * 1.05:
        return "B"
    return "C"


def build_signals(holdings: dict[str, pd.DataFrame], prices: pd.DataFrame) -> pd.DataFrame:
    bench = prices[prices.code == "TAIEX"][["date", "ret_20d", "ret_60d"]].rename(columns={"ret_20d": "b20", "ret_60d": "b60"})
    stocks = prices[~prices.code.isin(["TAIEX", "00981A"])]
    rows = []
    for d, h in holdings.items():
        px = stocks[stocks.date == d]
        if px.empty:
            continue
        m = h.merge(px, on="code", how="inner").merge(bench[bench.date == d], on="date", how="left")
        if m.empty:
            continue
        m["rs20"] = m["ret_20d"] - m["b20"]
        m["rs60"] = m["ret_60d"] - m["b60"]
        m["rs_accel"] = m["rs20"] - m["rs60"]
        m["rs20_rank"] = m["rs20"].rank(pct=True, ascending=True) * 100
        m["rs_accel_rank"] = m["rs_accel"].rank(pct=True, ascending=True) * 100
        m["market_component"] = 0.7 * m["rs20_rank"].fillna(0) + 0.3 * m["rs_accel_rank"].fillna(0)
        m["setup"] = m.apply(setup, axis=1)
        m["trading_score"] = (0.6 * m["setup"].map({"A": 100, "B": 90, "C": 55, "D": 0}).fillna(0) + 0.3 * m["market_component"] + 0.1 * np.where(m["is_top10"], 100, 40)).round(1)
        rows.append(m)
    return pd.concat(rows, ignore_index=True).sort_values(["signal_date", "trading_score"], ascending=[True, False])


def price_at(prices: pd.DataFrame, code: str, d: str, col: str) -> float:
    x = prices[(prices.code == code) & (prices.date == d)]
    if x.empty or col not in x:
        return np.nan
    v = x.iloc[0][col]
    return float(v) if pd.notna(v) else np.nan


def next_date(dates: list[str], d: str):
    return next((x for x in dates if x > d), None)


def nth_after(dates: list[str], d: str, n: int):
    f = [x for x in dates if x > d]
    return f[n - 1] if len(f) >= n else None


def run_bt(signals: pd.DataFrame, prices: pd.DataFrame):
    dates = sorted(prices[prices.code == "TAIEX"].date.unique().tolist())
    dates = [d for d in dates if d >= START_DATE]
    signal_dates = set(signals[signals.signal_date >= START_DATE].signal_date.unique())
    cash = INITIAL_CAPITAL
    positions: list[Position] = []
    orders: list[Order] = []
    trades, curve = [], []
    for d in dates:
        # buy pending orders
        keep_orders = []
        for o in orders:
            if o.buy_date != d:
                keep_orders.append(o)
                continue
            if cash < POSITION_SIZE or len(positions) >= MAX_POSITIONS or o.code in {p.code for p in positions}:
                continue
            px = price_at(prices, o.code, d, "open_adj")
            if not pd.notna(px) or px <= 0:
                px = price_at(prices, o.code, d, "close_adj")
            sd = nth_after(dates, d, HOLD_DAYS)
            if not sd or not pd.notna(px) or px <= 0:
                continue
            cash -= POSITION_SIZE
            positions.append(Position(o.code, o.name, o.setup, o.signal_date, d, sd, px, POSITION_SIZE / px, POSITION_SIZE, o.score))
        orders = keep_orders
        # sell due positions
        new_pos = []
        for p in positions:
            if p.sell_date == d:
                sp = price_at(prices, p.code, d, "close_adj")
                if pd.notna(sp) and sp > 0:
                    proceeds = p.shares * sp
                    cash += proceeds
                    trades.append({"signal_date": p.signal_date, "buy_date": p.buy_date, "sell_date": d, "code": p.code, "name": p.name, "setup": p.setup, "trading_score": p.score, "buy_price": p.buy_price, "sell_price": sp, "capital": p.capital, "shares": p.shares, "return_pct": sp / p.buy_price - 1, "pnl": proceeds - p.capital})
                else:
                    new_pos.append(p)
            else:
                new_pos.append(p)
        positions = new_pos
        # create next-day orders from only today's top 3; do not fill from lower ranks
        if d in signal_dates:
            bd = next_date(dates, d)
            if bd:
                top = signals[(signals.signal_date == d) & (signals.setup.isin(["A", "B"]))].sort_values("trading_score", ascending=False).head(MAX_POSITIONS)
                blocked = {p.code for p in positions} | {o.code for o in orders}
                for _, r in top.iterrows():
                    if r.code in blocked:
                        continue
                    orders.append(Order(r.code, r.name, r.setup, d, bd, float(r.trading_score)))
                    blocked.add(r.code)
        pv = 0.0
        for p in positions:
            cp = price_at(prices, p.code, d, "close_adj")
            pv += p.shares * cp if pd.notna(cp) and cp > 0 else p.capital
        curve.append({"date": d, "strategy_value": cash + pv, "cash": cash, "position_value": pv, "open_positions": len(positions), "pending_orders": len(orders)})
    return pd.DataFrame(trades), pd.DataFrame(curve)


def bh_curve(prices: pd.DataFrame, code: str, dates: list[str], name: str):
    px = prices[(prices.code == code) & (prices.date.isin(dates))].sort_values("date")
    base = px[px.date >= START_DATE]
    if base.empty:
        return pd.Series(index=dates, dtype=float, name=name)
    s = px.set_index("date")["close_adj"] / float(base.iloc[0].close_adj) * INITIAL_CAPITAL
    return s.reindex(dates).ffill().rename(name)


def mdd(s: pd.Series):
    return float((s / s.cummax() - 1).min()) if len(s) else np.nan


def summarize(trades: pd.DataFrame, curve: pd.DataFrame):
    rows = []
    for col, label in [("strategy_value", "strategy"), ("taiex_value", "taiex"), ("etf981_value", "etf981")]:
        if col not in curve or curve[col].dropna().empty:
            continue
        s = curve[col].dropna()
        r = s.pct_change().dropna()
        row = {"metric": label, "total_return": s.iloc[-1] / INITIAL_CAPITAL - 1, "mdd": mdd(s), "sharpe_daily": r.mean() / r.std() * math.sqrt(252) if len(r) > 1 and r.std() != 0 else np.nan, "ending_value": s.iloc[-1]}
        if label == "strategy":
            tr = trades.return_pct if not trades.empty else pd.Series(dtype=float)
            win = tr[tr > 0]
            loss = tr[tr <= 0]
            row.update({"trades": len(trades), "win_rate": (tr > 0).mean() if len(tr) else np.nan, "avg_trade_return": tr.mean() if len(tr) else np.nan, "median_trade_return": tr.median() if len(tr) else np.nan, "profit_factor": win.sum() / abs(loss.sum()) if abs(loss.sum()) > 0 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def setup_stats(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("setup").agg(trades=("code", "count"), win_rate=("return_pct", lambda x: (x > 0).mean()), avg_return=("return_pct", "mean"), median_return=("return_pct", "median"), total_pnl=("pnl", "sum")).reset_index()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = list_holdings_files()
    holdings = {f["date"]: load_holdings(f) for f in files}
    codes = sorted(set(pd.concat(holdings.values()).code.astype(str)))
    start = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    end = date.today().isoformat()
    prices = pd.concat([fetch_price(c, start, end) for c in sorted(set(codes + ["TAIEX", "00981A"]))], ignore_index=True)
    prices.to_csv(OUTPUT_DIR / "price_panel.csv", index=False, encoding="utf-8-sig")
    signals = build_signals(holdings, prices)
    signals.to_csv(OUTPUT_DIR / "signals.csv", index=False, encoding="utf-8-sig")
    trades, curve = run_bt(signals, prices)
    dates = curve.date.tolist()
    curve["taiex_value"] = bh_curve(prices, "TAIEX", dates, "taiex_value").values
    curve["etf981_value"] = bh_curve(prices, "00981A", dates, "etf981_value").values
    if (curve.cash < -1e-6).any() or (curve.open_positions > MAX_POSITIONS).any():
        raise RuntimeError("Portfolio accounting validation failed")
    summary = summarize(trades, curve)
    setup = setup_stats(trades)
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    setup.to_csv(OUTPUT_DIR / "setup_analysis.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT_DIR / "system_backtest.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        setup.to_excel(w, sheet_name="Setup_Analysis", index=False)
        trades.to_excel(w, sheet_name="Trades", index=False)
        curve.to_excel(w, sheet_name="Equity_Curve", index=False)
        signals.to_excel(w, sheet_name="Signals", index=False)
    print("[OK] Backtest v2 complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
