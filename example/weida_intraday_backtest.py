"""
韋大三大公式 — 分K近似回測（使用者補充參數版）

資料：加權指數每5秒（日盤 09:00–13:30）重採為 1/5/30 分K，作為台指期代理。
FinMind 期貨逐筆需付費；加權與近月台指高度相關。

使用者參數：
- 週期：30分 / 5分 / 1分K
- 停損：120 點
- 第1口停利：480 點
- 第2口停利：600 點（進場 2 口，分批出場）
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "weida_backtest_results"
SEC5_DIR = RESULTS / "taiex_5s"
OUT = RESULTS / "intraday_v2"
POINT_VALUE = 200
COST_PER_LOT = 2.0


@dataclass
class ExitParams:
    stop_loss: float = 120.0
    tp1: float = 480.0
    tp2: float = 600.0
    lots: int = 2


def load_ohlc(rule: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fp in sorted(SEC5_DIR.glob("*.pkl")):
        raw = pd.read_pickle(fp)
        raw["date"] = pd.to_datetime(raw["date"])
        s = raw.set_index("date")["TAIEX"].sort_index()
        ohlc = s.resample(rule).ohlc().dropna()
        if ohlc.empty:
            continue
        ohlc["session_date"] = ohlc.index.normalize()
        frames.append(ohlc)
    return pd.concat(frames).sort_index()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def simulate_two_lot(
    entry: float,
    direction: int,
    bars: pd.DataFrame,
    ep: ExitParams,
    entry_i: int,
) -> Tuple[float, str, int]:
    """2口：-120全停；+480平1口；+600平第2口；13:25後市價平。同棒停損優先。"""
    lots_left = ep.lots
    pnl = 0.0
    tp1_done = False
    reason = "time"
    j = entry_i

    for j in range(entry_i + 1, len(bars)):
        row = bars.iloc[j]
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        fav = (h - entry) if direction > 0 else (entry - l)
        adv = (entry - l) if direction > 0 else (h - entry)

        if adv >= ep.stop_loss:
            pnl += lots_left * (-ep.stop_loss)
            lots_left = 0
            reason = "stop"
            break

        if (not tp1_done) and fav >= ep.tp1 and lots_left > 0:
            pnl += ep.tp1
            lots_left -= 1
            tp1_done = True
            reason = "tp1"

        if tp1_done and fav >= ep.tp2 and lots_left > 0:
            pnl += ep.tp2
            lots_left -= 1
            reason = "tp2"
            break

        ts = bars.index[j]
        if hasattr(ts, "time") and ts.time() >= pd.Timestamp("13:25").time() and lots_left > 0:
            pnl += lots_left * ((c - entry) * direction)
            lots_left = 0
            reason = "time_1330"
            break

    if lots_left > 0:
        c = float(bars.iloc[-1]["close"])
        pnl += lots_left * ((c - entry) * direction)
        reason = "time_eod"

    pnl -= COST_PER_LOT * ep.lots
    return pnl, reason, max(1, j - entry_i)


def most_crossed_level(day_bars: pd.DataFrame, step: float = 5.0) -> float:
    if len(day_bars) < 2:
        return float(day_bars["close"].iloc[-1])
    lo = np.floor(day_bars["low"].min() / step) * step
    hi = np.ceil(day_bars["high"].max() / step) * step
    levels = np.arange(lo, hi + step, step)
    counts = np.zeros(len(levels))
    closes = day_bars["close"].to_numpy()
    for a, b in zip(closes[:-1], closes[1:]):
        lo_ab, hi_ab = (a, b) if a <= b else (b, a)
        counts += ((levels >= lo_ab) & (levels <= hi_ab)).astype(float)
    return float(levels[int(np.argmax(counts))])


def session_open_close_from_5s() -> pd.DataFrame:
    """
    FinMind 5秒第一筆常等於昨收（無真正跳空）。
    改用 09:00:30～09:01:00 均價當「開盤」，相對昨 13:30 收估跳空。
    """
    rows = []
    for fp in sorted(SEC5_DIR.glob("*.pkl")):
        raw = pd.read_pickle(fp)
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date")
        if raw.empty:
            continue
        d = raw["date"].iloc[0].normalize()
        close = float(raw["TAIEX"].iloc[-1])
        window = raw[
            (raw["date"].dt.time >= pd.Timestamp("09:00:30").time())
            & (raw["date"].dt.time <= pd.Timestamp("09:01:00").time())
        ]
        if window.empty:
            # fallback: 第2～12筆（約前1分鐘）
            open_px = float(raw["TAIEX"].iloc[1:12].mean())
        else:
            open_px = float(window["TAIEX"].mean())
        rows.append({"session_date": d, "open_px": open_px, "close_px": close})
    out = pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)
    out["prev_close"] = out["close_px"].shift(1)
    out["gap"] = out["open_px"] - out["prev_close"]
    return out


def backtest_f01(bars: pd.DataFrame, ep: ExitParams, gap_mult: float = 0.35) -> pd.DataFrame:
    daily = bars.groupby("session_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    daily["amp"] = daily["high"] - daily["low"]
    daily["avg_amp"] = daily["amp"].rolling(20, min_periods=10).mean()
    gaps = session_open_close_from_5s().set_index("session_date")
    daily = daily.join(gaps[["gap", "open_px"]])

    by_day = {d: g for d, g in bars.groupby("session_date")}
    trades: List[dict] = []
    for d, row in daily.dropna(subset=["gap", "avg_amp"]).iterrows():
        if abs(row["gap"]) <= row["avg_amp"] * gap_mult:
            continue
        day_bars = by_day.get(d)
        if day_bars is None or len(day_bars) < 3:
            continue
        direction = 1 if row["gap"] > 0 else -1
        entry_i = min(1, len(day_bars) - 1)
        entry = float(day_bars.iloc[entry_i]["open"])
        pnl, reason, held = simulate_two_lot(entry, direction, day_bars, ep, entry_i)
        trades.append(
            {
                "date": pd.Timestamp(d),
                "formula": "01_gap",
                "direction": "long" if direction > 0 else "short",
                "entry": entry,
                "pnl_pts": pnl,
                "reason": reason,
                "held_bars": held,
                "gap": float(row["gap"]),
            }
        )
    return pd.DataFrame(trades)


def backtest_f02(bars_5: pd.DataFrame, ep: ExitParams, retrace_max: float = 30.0) -> pd.DataFrame:
    days = sorted(bars_5["session_date"].unique())
    by_day = {d: g for d, g in bars_5.groupby("session_date")}
    trades: List[dict] = []
    for i in range(1, len(days)):
        prev_d, d = days[i - 1], days[i]
        key = most_crossed_level(by_day[prev_d])
        day_bars = by_day[d]
        touched = False
        for j in range(len(day_bars)):
            r = day_bars.iloc[j]
            h, l = float(r["high"]), float(r["low"])
            if not touched:
                if l <= key <= h and min(key - l, h - key) <= retrace_max:
                    touched = True
                continue
            direction = None
            entry = None
            if h >= key + 5 and float(r["close"]) > key:
                direction, entry = 1, max(key, float(r["open"]))
            elif l <= key - 5 and float(r["close"]) < key:
                direction, entry = -1, min(key, float(r["open"]))
            if direction is None:
                continue
            pnl, reason, held = simulate_two_lot(entry, direction, day_bars, ep, j)
            trades.append(
                {
                    "date": pd.Timestamp(d),
                    "formula": "02_cross",
                    "direction": "long" if direction > 0 else "short",
                    "entry": entry,
                    "pnl_pts": pnl,
                    "reason": reason,
                    "held_bars": held,
                    "key": key,
                }
            )
            break
    return pd.DataFrame(trades)


def backtest_f03(bars_5: pd.DataFrame, bars_30: pd.DataFrame, ep: ExitParams) -> pd.DataFrame:
    b30 = bars_30.copy()
    b30["macd_h_30"] = macd_hist(b30["close"])
    b5 = bars_5.copy()
    b5["macd_h"] = macd_hist(b5["close"])

    left = b5.copy()
    left["dt"] = left.index
    left = left.reset_index(drop=True).sort_values("dt")
    right = b30.copy()
    right["dt"] = right.index
    right = right.reset_index(drop=True)[["dt", "macd_h_30"]].sort_values("dt")
    merged = pd.merge_asof(left, right, on="dt", direction="backward")
    merged = merged.set_index("dt").sort_index()

    trades: List[dict] = []
    for d, g in merged.groupby("session_date"):
        g = g.copy()
        if len(g) < 30:
            continue
        taken = False
        for j in range(1, len(g)):
            if taken:
                break
            big = g.iloc[j]["macd_h_30"]
            prev_h, cur_h = g.iloc[j - 1]["macd_h"], g.iloc[j]["macd_h"]
            if pd.isna(big) or pd.isna(prev_h) or pd.isna(cur_h):
                continue
            big_dir = 1 if big > 0 else -1
            if big_dir > 0 and prev_h <= 0 < cur_h:
                direction = 1
            elif big_dir < 0 and prev_h >= 0 > cur_h:
                direction = -1
            else:
                continue
            entry = float(g.iloc[j]["close"])
            pnl, reason, held = simulate_two_lot(entry, direction, g, ep, j)
            trades.append(
                {
                    "date": pd.Timestamp(d),
                    "formula": "03_resonance",
                    "direction": "long" if direction > 0 else "short",
                    "entry": entry,
                    "pnl_pts": pnl,
                    "reason": reason,
                    "held_bars": held,
                }
            )
            taken = True
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame, name: str) -> Dict:
    if tr is None or len(tr) == 0:
        return {"strategy": name, "trades": 0}
    pnl = tr["pnl_pts"].astype(float)
    equity = pnl.cumsum()
    dd = equity - equity.cummax()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    return {
        "strategy": name,
        "trades": int(len(tr)),
        "win_rate": float((pnl > 0).mean()),
        "total_pnl_pts_2lots": float(pnl.sum()),
        "avg_pnl_pts_2lots": float(pnl.mean()),
        "total_pnl_twd": float(pnl.sum() * POINT_VALUE),
        "max_dd_pts": float(dd.min()) if len(dd) else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else None,
        "tp2_rate": float((tr["reason"] == "tp2").mean()),
        "tp1_only_rate": float((tr["reason"] == "tp1").mean()),
        "stop_rate": float((tr["reason"] == "stop").mean()),
    }


def plot_equity(trades_map: Dict[str, pd.DataFrame], path: Path) -> None:
    plt.figure(figsize=(11, 6))
    for name, tr in trades_map.items():
        if tr is None or len(tr) == 0:
            continue
        plt.plot(pd.to_datetime(tr["date"]), tr["pnl_pts"].cumsum(), label=name, lw=1.8)
    plt.axhline(0, color="#888", lw=0.8)
    plt.title("Weida intraday approx — SL120 / TP480+600 (2 lots, TAIEX proxy)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (pts, 2 lots combined)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep = ExitParams()

    print("Resampling ...")
    bars_1 = load_ohlc("1min")
    bars_5 = load_ohlc("5min")
    bars_30 = load_ohlc("30min")
    bars_5.to_pickle(OUT / "bars_5min.pkl")
    bars_30.to_pickle(OUT / "bars_30min.pkl")
    print("days", bars_5["session_date"].nunique(), "5m bars", len(bars_5))

    print("F01 5min...")
    t1 = backtest_f01(bars_5, ep)
    print("F01 30min...")
    t1_30 = backtest_f01(bars_30, ep)
    print("F01 1min...")
    t1_1 = backtest_f01(bars_1, ep)
    print("F02 5min...")
    t2 = backtest_f02(bars_5, ep)
    print("F03 30+5...")
    t3 = backtest_f03(bars_5, bars_30, ep)

    summary = pd.DataFrame(
        [
            summarize(t1, "01_gap_5min"),
            summarize(t1_30, "01_gap_30min"),
            summarize(t1_1, "01_gap_1min"),
            summarize(t2, "02_cross_5min"),
            summarize(t3, "03_resonance_30p5"),
        ]
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    t1.to_csv(OUT / "trades_01_5min.csv", index=False)
    t1_30.to_csv(OUT / "trades_01_30min.csv", index=False)
    t1_1.to_csv(OUT / "trades_01_1min.csv", index=False)
    t2.to_csv(OUT / "trades_02_5min.csv", index=False)
    t3.to_csv(OUT / "trades_03.csv", index=False)

    params = {
        "proxy": "TAIEX 5-second day session → 1/5/30min OHLC",
        "period": {
            "start": str(bars_5.index.min()),
            "end": str(bars_5.index.max()),
            "days": int(bars_5["session_date"].nunique()),
        },
        "exit": asdict(ep),
        "money": {"point_value_twd": POINT_VALUE, "cost_per_lot_pts": COST_PER_LOT},
    }
    (OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_equity(
        {"01_5m": t1, "01_30m": t1_30, "02_5m": t2, "03_30+5": t3},
        OUT / "equity_curves.png",
    )
    print(json.dumps(params, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
